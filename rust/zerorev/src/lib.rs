use reversi_mcts::EvalFn;
use reversi_mcts::MctsSearch;
use reversi_mcts::SharedSearch;
use reversi_minimax::board::Board;
use reversi_nn::AlphaZeroNet;
use wasm_bindgen::prelude::*;

static CHAMPION_WEIGHTS: &[u8] = include_bytes!("../../../weights/zerorev/champion.bin");

/// Facts about the embedded champion, written by the trainer next to
/// `champion.bin` (run, date, loss, tournament win rate).
static CHAMPION_META: &str = include_str!("../../../weights/zerorev/champion.json");

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

/// An analysis stops growing its tree at this many nodes (a few hundred MB).
const ANALYSIS_MAX_NODES: usize = 4_000_000;

/// The position under analysis and its search tree, kept between calls so
/// that each call carries on where the last one stopped.
struct Analysis {
    board: Board,
    is_black: bool,
    search: SharedSearch,
}

#[wasm_bindgen]
pub struct ZeroRevBot {
    eval: Box<dyn EvalFn + Sync>,
    /// Simulations made by the most recent search.
    last_simulations: std::cell::Cell<u32>,
    last_value: std::cell::Cell<f32>,
    analysis: std::cell::RefCell<Option<Analysis>>,
    analysis_complete: std::cell::Cell<bool>,
}

#[wasm_bindgen]
impl ZeroRevBot {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ZeroRevBot {
        ZeroRevBot {
            eval: make_eval(),
            last_simulations: std::cell::Cell::new(0),
            last_value: std::cell::Cell::new(0.0),
            analysis: std::cell::RefCell::new(None),
            analysis_complete: std::cell::Cell::new(false),
        }
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

    /// The most recent `analyze_position`'s verdict for the side to move,
    /// from −1 (a loss) to +1 (a win).
    pub fn last_value(&self) -> f32 {
        self.last_value.get()
    }

    /// Whether the analysis of the current position has grown as far as it
    /// may; further calls return the same result.
    pub fn analysis_complete(&self) -> bool {
        self.analysis_complete.get()
    }

    /// Search the position for `think_ms` more milliseconds and return the
    /// share of the search each square got (64 f32). Asked about the same
    /// position again, it carries on growing the same tree.
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

        let mut slot = self.analysis.borrow_mut();
        let analysis = match slot.take() {
            Some(a) if a.board == board && a.is_black == is_black => a,
            // A new position: a new tree; the old one is dropped.
            _ => Analysis { board, is_black, search: SharedSearch::new(board, is_black) },
        };
        let search = &analysis.search;
        if search.with(MctsSearch::nodes) < ANALYSIS_MAX_NODES {
            run_mcts_timed(search, self.eval.as_ref(), think_ms);
        }
        let (mut policy, visits, value, nodes) = search
            .with(|tree| (tree.policy_target(), tree.visits(), tree.root_value(), tree.nodes()));
        *slot = Some(analysis);

        self.last_simulations.set(visits);
        self.last_value.set(value);
        self.analysis_complete.set(nodes >= ANALYSIS_MAX_NODES);
        for (i, r) in policy.iter_mut().enumerate() {
            if (legal >> i) & 1 == 0 {
                *r = 0.0;
            }
        }
        js_sys::Float32Array::from(policy.as_ref())
    }
}

impl Default for ZeroRevBot {
    fn default() -> Self {
        Self::new()
    }
}
