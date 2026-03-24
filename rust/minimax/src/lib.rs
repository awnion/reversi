pub mod utils;
pub mod board;
pub mod eval;
pub mod engine;

use wasm_bindgen::prelude::*;
use board::Board;
use engine::{search, evaluate_all_moves, MIN_SCORE, MAX_SCORE};

/// Returns current time in milliseconds (WASM only; returns 0 on native).
#[inline]
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    { js_sys::Date::now() }
    #[cfg(not(target_arch = "wasm32"))]
    { 0.0 }
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
        let board = board_from_parts(board_black_low, board_black_high, board_white_low, board_white_high);

        let deadline = now_ms() + time_ms;
        let mut best_move: Option<u64> = None;

        for depth in 1u8..=20 {
            let (m, _) = search(board, is_black, depth, MIN_SCORE, MAX_SCORE);
            if m.is_some() {
                best_move = m;
            }
            if now_ms() >= deadline {
                break;
            }
        }

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
        let board = board_from_parts(board_black_low, board_black_high, board_white_low, board_white_high);

        if board.legal_moves(is_black) == 0 {
            return Box::new([]);
        }

        let deadline = now_ms() + time_ms;
        let mut best_evals: Vec<(u64, i32)> = Vec::new();
        let mut best_depth = 0u8;

        for depth in 1u8..=20 {
            let evals = evaluate_all_moves(board, is_black, depth);
            best_evals = evals;
            best_depth = depth;
            if now_ms() >= deadline {
                break;
            }
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
