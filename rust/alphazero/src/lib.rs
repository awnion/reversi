use reversi_mcts::EvalFn;
use reversi_mcts::MctsSearch;
use reversi_mcts::StaticEval;
use reversi_minimax::board::Board;
use reversi_nn::AlphaZeroNet;
use wasm_bindgen::prelude::*;

// Embed champion weights at compile time.
static CHAMPION_WEIGHTS: &[u8] = include_bytes!("../../../weights/champion.bin");

struct NnEval {
    net: AlphaZeroNet,
}

impl EvalFn for NnEval {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        // Build 3×8×8 channel-first planes
        let my_bits = if is_black { board.black } else { board.white };
        let opp_bits = if is_black { board.white } else { board.black };
        let mut planes = [0.0f32; 3 * 64];
        for i in 0..64usize {
            if (my_bits >> i) & 1 == 1 {
                planes[i] = 1.0;
            }
            if (opp_bits >> i) & 1 == 1 {
                planes[64 + i] = 1.0;
            }
            if (legal >> i) & 1 == 1 {
                planes[128 + i] = 1.0;
            }
        }
        self.net.forward(&planes)
    }
}

fn make_eval() -> Box<dyn EvalFn> {
    match AlphaZeroNet::load(CHAMPION_WEIGHTS) {
        Ok(net) => Box::new(NnEval { net }),
        Err(_) => Box::new(StaticEval),
    }
}

#[wasm_bindgen]
pub struct AlphaZeroBot {
    simulations: u32,
}

#[wasm_bindgen]
impl AlphaZeroBot {
    #[wasm_bindgen(constructor)]
    pub fn new(simulations: u32) -> AlphaZeroBot {
        AlphaZeroBot { simulations }
    }

    /// Choose best move. Returns bit index (0–63) or -1 if no legal moves.
    /// Board passed as four u32 (split from two u64).
    pub fn choose_move(&self, bl: u32, bh: u32, wl: u32, wh: u32, is_black: bool) -> i32 {
        let black = (bh as u64) << 32 | bl as u64;
        let white = (wh as u64) << 32 | wl as u64;
        let board = Board::new(black, white);

        let legal = board.legal_moves(is_black);
        if legal == 0 {
            return -1;
        }

        let eval = make_eval();
        let mut mcts = MctsSearch::new(board, is_black);
        for _ in 0..self.simulations {
            mcts.simulate(eval.as_ref());
        }

        mcts.best_move().map(|m| m.trailing_zeros() as i32).unwrap_or(-1)
    }

    /// Returns policy array (64 f32) as Float32Array.
    pub fn analyze_position(
        &self,
        bl: u32,
        bh: u32,
        wl: u32,
        wh: u32,
        is_black: bool,
    ) -> js_sys::Float32Array {
        let black = (bh as u64) << 32 | bl as u64;
        let white = (wh as u64) << 32 | wl as u64;
        let board = Board::new(black, white);

        let legal = board.legal_moves(is_black);
        let eval = make_eval();
        let mut mcts = MctsSearch::new(board, is_black);
        for _ in 0..self.simulations {
            mcts.simulate(eval.as_ref());
        }

        let policy = mcts.policy_target();
        // Zero out illegal squares
        let mut result = policy;
        for i in 0..64 {
            if (legal >> i) & 1 == 0 {
                result[i] = 0.0;
            }
        }
        js_sys::Float32Array::from(result.as_ref())
    }
}

impl Default for AlphaZeroBot {
    fn default() -> Self {
        Self::new(200)
    }
}
