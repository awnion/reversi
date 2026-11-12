use reversi_mcts::EvalFn;
use reversi_mcts::MctsSearch;
use reversi_minimax::board::Board;
use reversi_nn::AlphaZeroNet;
use wasm_bindgen::prelude::*;

// Embed champion weights at compile time.
static CHAMPION_WEIGHTS: &[u8] = include_bytes!("../../../weights/alphazero/champion.bin");

/// Facts about the embedded champion, written by the trainer next to
/// `champion.bin` (run, date, loss, tournament win rate).
static CHAMPION_META: &str = include_str!("../../../weights/alphazero/champion.json");

/// JSON describing this build: `{"build_date": "...", "champion": {...}}`.
#[wasm_bindgen]
pub fn meta() -> String {
    let build_date = option_env!("BUILD_DATE").unwrap_or("dev");
    format!(r#"{{"build_date":"{build_date}","champion":{}}}"#, CHAMPION_META.trim())
}

struct NnEval {
    net: AlphaZeroNet,
}

impl EvalFn for NnEval {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        let my_bits = if is_black { board.black } else { board.white };
        let opp_bits = if is_black { board.white } else { board.black };
        let mut planes = [0.0f32; 4 * 64];
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
        let phase = (board.black | board.white).count_ones() as f32 / 64.0;
        planes[192..].fill(phase);
        self.net.forward(&planes)
    }
}

fn make_eval() -> Box<dyn EvalFn> {
    // The weights are embedded at build time, so a load failure is a build
    // defect. Fail loudly: silently falling back to another evaluator once hid
    // a format mismatch and the bot never used its network.
    let net = AlphaZeroNet::load(CHAMPION_WEIGHTS)
        .unwrap_or_else(|err| panic!("embedded champion weights failed to load: {err}"));
    Box::new(NnEval { net })
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
        // Nothing to think about.
        if legal.count_ones() == 1 {
            return legal.trailing_zeros() as i32;
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
        for (i, r) in result.iter_mut().enumerate() {
            if (legal >> i) & 1 == 0 {
                *r = 0.0;
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
