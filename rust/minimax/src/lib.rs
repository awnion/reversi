pub mod utils;
pub mod board;
pub mod eval;
pub mod engine;

use wasm_bindgen::prelude::*;
use board::Board;
use engine::search;

#[wasm_bindgen]
pub struct MinimaxBot {}

#[wasm_bindgen]
impl MinimaxBot {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MinimaxBot {
        crate::utils::set_panic_hook();
        MinimaxBot {}
    }

    pub fn choose_move(&self, board_black_low: u32, board_black_high: u32, board_white_low: u32, board_white_high: u32, is_black: bool, _time_ms: f64) -> i32 {
        let board_black = (board_black_high as u64) << 32 | (board_black_low as u64);
        let board_white = (board_white_high as u64) << 32 | (board_white_low as u64);
        
        let board = Board::new(board_black, board_white);
        let (best_move, _) = search(board, is_black, 5, -100000, 100000);
        
        if let Some(m) = best_move {
            m.trailing_zeros() as i32
        } else {
            -1
        }
    }
}
