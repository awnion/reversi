pub mod board;
pub mod engine;
pub mod eval;
pub mod utils;

use board::Board;
use engine::Deadline;
use engine::evaluate_all_moves;
use engine::evaluate_all_moves_until;
use engine::search_root_until;
use wasm_bindgen::prelude::*;
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Returns current time in milliseconds (WASM only; returns 0 on native).
#[inline]
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0.0
    }
}

/// Every move's score is a finished game: the position is solved.
fn solved(evals: &[(u64, i32)]) -> bool {
    evals.iter().all(|&(_, score)| score.abs() >= eval::WIN)
}

fn board_from_parts(bl: u32, bh: u32, wl: u32, wh: u32) -> Board {
    let black = (bh as u64) << 32 | (bl as u64);
    let white = (wh as u64) << 32 | (wl as u64);
    Board::new(black, white)
}

/// The position under analysis and the deepest search of it so far, kept
/// between calls so that each call carries on one level deeper.
struct Analysis {
    board: Board,
    is_black: bool,
    depth: u8,
    evals: Vec<(u64, i32)>,
    /// Time the next depth is expected to need, in milliseconds.
    next_depth_ms: f64,
    complete: bool,
}

/// Deepest analysis; past this a call returns the same result.
const ANALYSIS_MAX_DEPTH: u8 = 20;
/// A call that has to finish a depth may run over its budget, up to this; a
/// depth that needs longer ends the analysis.
const ANALYSIS_MAX_CALL_MS: f64 = 30_000.0;

#[wasm_bindgen]
pub struct MinimaxBot {
    analysis: std::cell::RefCell<Option<Analysis>>,
    analysis_complete: std::cell::Cell<bool>,
}

#[wasm_bindgen]
impl MinimaxBot {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MinimaxBot {
        crate::utils::set_panic_hook();
        MinimaxBot {
            analysis: std::cell::RefCell::new(None),
            analysis_complete: std::cell::Cell::new(false),
        }
    }
}

impl Default for MinimaxBot {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl MinimaxBot {
    /// Returns the best move index (0–63) or -1 if no moves available.
    /// Uses iterative deepening for the given time budget (milliseconds).
    pub fn choose_move(
        &self,
        board_black_low: u32,
        board_black_high: u32,
        board_white_low: u32,
        board_white_high: u32,
        is_black: bool,
        time_ms: f64,
    ) -> i32 {
        let board =
            board_from_parts(board_black_low, board_black_high, board_white_low, board_white_high);

        let legal = board.legal_moves(is_black);
        if legal == 0 {
            return -1;
        }
        // Nothing to think about.
        if legal.count_ones() == 1 {
            return legal.trailing_zeros() as i32;
        }

        // Deepen until the time is up. A depth that is cut short is discarded
        // and the move from the last finished depth is played.
        let mut deadline = Deadline::at(now_ms, now_ms() + time_ms);
        let mut best_move: Option<u64> = None;

        for depth in 1u8..=20 {
            let Some((m, _)) = search_root_until(board, is_black, depth, best_move, &mut deadline)
            else {
                break;
            };
            if m.is_some() {
                best_move = m;
            }
        }

        // Not even depth 1 finished (a tiny budget): play any legal move.
        let best_move = best_move.or(Some(1u64 << legal.trailing_zeros()));
        best_move.map(|m| m.trailing_zeros() as i32).unwrap_or(-1)
    }

    /// Evaluates every legal move, one depth deeper than the last call on the
    /// same position for as long as `time_ms` allows. A depth is never left
    /// half done: when the next one is expected to take longer than the
    /// budget, the call runs over (up to [`ANALYSIS_MAX_CALL_MS`]).
    /// Returns a flat Int32Array: [depth_reached, idx0, score0, idx1, score1, ...].
    /// Returns an empty array when there are no legal moves.
    pub fn analyze_position(
        &self,
        board_black_low: u32,
        board_black_high: u32,
        board_white_low: u32,
        board_white_high: u32,
        is_black: bool,
        time_ms: f64,
    ) -> Box<[i32]> {
        let board =
            board_from_parts(board_black_low, board_black_high, board_white_low, board_white_high);

        if board.legal_moves(is_black) == 0 {
            self.analysis_complete.set(true);
            return Box::new([]);
        }

        let mut slot = self.analysis.borrow_mut();
        let mut analysis = match slot.take() {
            Some(a) if a.board == board && a.is_black == is_black => a,
            // Depth 1 always completes, so there is always something to report.
            _ => Analysis {
                board,
                is_black,
                depth: 1,
                evals: evaluate_all_moves(board, is_black, 1),
                next_depth_ms: 0.0,
                complete: false,
            },
        };

        let budget = time_ms.max(analysis.next_depth_ms).min(ANALYSIS_MAX_CALL_MS);
        let mut deadline = Deadline::at(now_ms, now_ms() + budget);
        while !analysis.complete {
            let started = now_ms();
            let depth = analysis.depth + 1;
            let Some(evals) = evaluate_all_moves_until(board, is_black, depth, &mut deadline)
            else {
                // Cut short: give the next call twice as long, unless even the
                // longest call was not enough; then this is as deep as it gets.
                analysis.next_depth_ms = 2.0 * budget;
                analysis.complete = budget >= ANALYSIS_MAX_CALL_MS;
                break;
            };
            analysis.complete = depth >= ANALYSIS_MAX_DEPTH
                || solved(&evals)
                // Every line has reached the end of the game.
                || u32::from(depth) >= 2 * board.empty().count_ones();
            analysis.depth = depth;
            analysis.evals = evals;
            // Each depth takes a few times longer than the one before.
            analysis.next_depth_ms = 3.0 * (now_ms() - started);
        }

        // Pack into [depth, idx0, score0, idx1, score1, ...]
        let mut out = Vec::with_capacity(1 + analysis.evals.len() * 2);
        out.push(analysis.depth as i32);
        for (m, score) in &analysis.evals {
            out.push(m.trailing_zeros() as i32);
            out.push(*score);
        }
        self.analysis_complete.set(analysis.complete);
        *slot = Some(analysis);
        out.into_boxed_slice()
    }

    /// Whether the analysis of the current position is as deep as it goes;
    /// further calls return the same result.
    pub fn analysis_complete(&self) -> bool {
        self.analysis_complete.get()
    }
}
