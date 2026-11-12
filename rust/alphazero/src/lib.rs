use reversi_mcts::EvalFn;
use reversi_mcts::SharedSearch;
use reversi_minimax::board::Board;
use reversi_nn::AlphaZeroNet;
use wasm_bindgen::prelude::*;

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

fn make_eval() -> Box<dyn EvalFn + Sync> {
    // The weights are embedded at build time, so a load failure is a build
    // defect. Fail loudly: silently falling back to another evaluator once hid
    // a format mismatch and the bot never used its network.
    let net = AlphaZeroNet::load(CHAMPION_WEIGHTS)
        .unwrap_or_else(|err| panic!("embedded champion weights failed to load: {err}"));
    Box::new(NnEval { net })
}

/// Simulate on this thread until the deadline (milliseconds, `Date.now()`).
fn simulate_until(search: &SharedSearch, eval: &dyn EvalFn, deadline: f64) {
    loop {
        for _ in 0..4 {
            search.simulate(eval);
        }
        if js_sys::Date::now() >= deadline {
            break;
        }
    }
}

/// Search for `time_ms` milliseconds on every thread of the pool, all working
/// on the same tree.
#[cfg(feature = "threads")]
fn run_mcts_timed(search: &SharedSearch, eval: &(dyn EvalFn + Sync), time_ms: f64) {
    let deadline = js_sys::Date::now() + time_ms;
    rayon::scope(|scope| {
        for _ in 0..rayon::current_num_threads() {
            scope.spawn(|_| simulate_until(search, eval, deadline));
        }
    });
}

/// Search for `time_ms` milliseconds on the calling thread.
#[cfg(not(feature = "threads"))]
fn run_mcts_timed(search: &SharedSearch, eval: &(dyn EvalFn + Sync), time_ms: f64) {
    simulate_until(search, eval, js_sys::Date::now() + time_ms);
}

/// Start the worker threads of the search; resolves once they are running.
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Number of threads the search runs on (1 in the single-threaded build).
#[wasm_bindgen]
pub fn search_threads() -> usize {
    #[cfg(feature = "threads")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "threads"))]
    {
        1
    }
}

#[wasm_bindgen]
pub struct AlphaZeroBot {
    eval: Box<dyn EvalFn + Sync>,
    /// Simulations made by the most recent search.
    last_simulations: std::cell::Cell<u32>,
}

#[wasm_bindgen]
impl AlphaZeroBot {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AlphaZeroBot {
        AlphaZeroBot { eval: make_eval(), last_simulations: std::cell::Cell::new(0) }
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

        let search = SharedSearch::new(board, is_black);
        run_mcts_timed(&search, self.eval.as_ref(), think_ms);

        let tree = search.into_inner();
        self.last_simulations.set(tree.visits());
        tree.best_move().map(|m| m.trailing_zeros() as i32).unwrap_or(-1)
    }

    /// Simulations made by the most recent `choose_move`/`analyze_position`.
    pub fn last_simulations(&self) -> u32 {
        self.last_simulations.get()
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
        let search = SharedSearch::new(board, is_black);
        run_mcts_timed(&search, self.eval.as_ref(), think_ms);

        let tree = search.into_inner();
        self.last_simulations.set(tree.visits());
        let policy = tree.policy_target();
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
