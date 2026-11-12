pub mod board;
pub mod engine;
pub mod eval;
pub mod utils;

use board::Board;
use engine::Deadline;
use engine::MAX_SCORE;
use engine::MIN_SCORE;
use engine::evaluate_all_moves;
use engine::evaluate_all_moves_until;
use engine::search_until;
use wasm_bindgen::prelude::*;

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

fn board_from_parts(bl: u32, bh: u32, wl: u32, wh: u32) -> Board {
    let black = (bh as u64) << 32 | (bl as u64);
    let white = (wh as u64) << 32 | (wl as u64);
    Board::new(black, white)
}

#[wasm_bindgen]
pub struct MinimaxBot {}

#[wasm_bindgen]
impl MinimaxBot {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MinimaxBot {
        crate::utils::set_panic_hook();
        MinimaxBot {}
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
            let Some((m, _)) =
                search_until(board, is_black, depth, MIN_SCORE, MAX_SCORE, &mut deadline)
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

    /// Evaluates every legal move using iterative deepening within the time budget.
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
            return Box::new([]);
        }

        let mut deadline = Deadline::at(now_ms, now_ms() + time_ms);
        // Depth 1 always completes, so there is always something to report.
        let mut best_evals = evaluate_all_moves(board, is_black, 1);
        let mut best_depth = 1u8;

        for depth in 2u8..=20 {
            let Some(evals) = evaluate_all_moves_until(board, is_black, depth, &mut deadline)
            else {
                break;
            };
            best_evals = evals;
            best_depth = depth;
        }

        // Pack into [depth, idx0, score0, idx1, score1, ...]
        let mut out = Vec::with_capacity(1 + best_evals.len() * 2);
        out.push(best_depth as i32);
        for (m, score) in &best_evals {
            out.push(m.trailing_zeros() as i32);
            out.push(*score);
        }
        out.into_boxed_slice()
    }
}
