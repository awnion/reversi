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

fn run_mcts_timed(mcts: &mut MctsSearch, eval: &dyn EvalFn, time_ms: f64) {
    let deadline = js_sys::Date::now() + time_ms;
    loop {
        for _ in 0..10 {
            mcts.simulate(eval);
        }
        if js_sys::Date::now() >= deadline {
            break;
        }
    }
}

#[wasm_bindgen]
pub struct AlphaZeroBot {
    eval: Box<dyn EvalFn>,
}

#[wasm_bindgen]
impl AlphaZeroBot {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AlphaZeroBot {
        AlphaZeroBot { eval: make_eval() }
    }

    /// Choose best move within `think_ms` milliseconds.
    /// Returns bit index (0–63) or -1 if no legal moves.
    pub fn choose_move(
        &self,
        bl: u32,
        bh: u32,
        wl: u32,
        wh: u32,
        is_black: bool,
        think_ms: f64,
    ) -> i32 {
        let black = (bh as u64) << 32 | bl as u64;
        let white = (wh as u64) << 32 | wl as u64;
        let board = Board::new(black, white);

        let legal = board.legal_moves(is_black);
        if legal == 0 {
            return -1;
        }

        let mut mcts = MctsSearch::new(board, is_black);
        run_mcts_timed(&mut mcts, self.eval.as_ref(), think_ms);

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
        think_ms: f64,
    ) -> js_sys::Float32Array {
        let black = (bh as u64) << 32 | bl as u64;
        let white = (wh as u64) << 32 | wl as u64;
        let board = Board::new(black, white);

        let legal = board.legal_moves(is_black);
        let mut mcts = MctsSearch::new(board, is_black);
        run_mcts_timed(&mut mcts, self.eval.as_ref(), think_ms);

        let policy = mcts.policy_target();
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
        Self::new()
    }
}
