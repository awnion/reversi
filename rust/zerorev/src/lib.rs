use reversi_mcts::EvalFn;
use reversi_mcts::MctsSearch;
use reversi_mcts::SharedSearch;
use reversi_mcts::batch::ENDGAME_SOLVE_EMPTIES;
use reversi_mcts::endgame::solve;
use reversi_minimax::board::Board;
use reversi_nn::AlphaZeroNet;
use wasm_bindgen::prelude::*;

static CHAMPION_WEIGHTS: &[u8] = include_bytes!("../../../weights/zerorev/champion.bin");

/// Facts about the embedded champion, written by the trainer next to
/// `champion.bin` (run, date, loss, tournament win rate).
static CHAMPION_META: &str = include_str!("../../../weights/zerorev/champion.json");

/// Gladiator0: a selected earlier champion of the same architecture, kept as
/// an opponent of its own.
static GLADIATOR0_WEIGHTS: &[u8] = include_bytes!("../../../weights/gladiator0/gladiator0.bin");
static GLADIATOR0_META: &str = include_str!("../../../weights/gladiator0/gladiator0.json");

/// Ramo0: the same architecture, trained on another machine (run `ramo0`).
static RAMO0_WEIGHTS: &[u8] = include_bytes!("../../../weights/ramo0/champion.bin");
static RAMO0_META: &str = include_str!("../../../weights/ramo0/champion.json");

/// ZeroRev6x48: a 6-block, 48-channel network trained with supervision on
/// Ramo0's self-play positions (`python -m train.supervised`).
static ZEROREV6X48_WEIGHTS: &[u8] = include_bytes!("../../../weights/zerorev6x48/champion.bin");
static ZEROREV6X48_META: &str = include_str!("../../../weights/zerorev6x48/champion.json");

/// The embedded network called `network`: `"gladiator0"`, `"ramo0"`,
/// `"zerorev6x48"`, or the champion for any other name. Returns its weights
/// and the facts about them.
fn network(network: &str) -> (&'static [u8], &'static str) {
    match network {
        "gladiator0" => (GLADIATOR0_WEIGHTS, GLADIATOR0_META),
        "ramo0" => (RAMO0_WEIGHTS, RAMO0_META),
        "zerorev6x48" => (ZEROREV6X48_WEIGHTS, ZEROREV6X48_META),
        _ => (CHAMPION_WEIGHTS, CHAMPION_META),
    }
}

/// JSON describing this build and the network called `network` (see
/// `ZeroRevBot::new`): `{"build_date": "...", "champion": {...}}`.
#[wasm_bindgen]
pub fn meta(network: &str) -> String {
    let build_date = option_env!("BUILD_DATE").unwrap_or("dev");
    let (_, meta) = self::network(network);
    format!(r#"{{"build_date":"{build_date}","champion":{}}}"#, meta.trim())
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

fn make_eval(weights: &[u8]) -> Box<dyn EvalFn + Sync> {
    // The weights are embedded at build time, so a load failure is a build
    // defect. Fail loudly: silently falling back to another evaluator once hid
    // a format mismatch and the bot never used its network.
    let net = AlphaZeroNet::load(weights)
        .unwrap_or_else(|err| panic!("embedded network weights failed to load: {err}"));
    Box::new(NnEval { net })
}

/// When to stop a timed search.
#[derive(Clone, Copy)]
struct Limit {
    /// `Date.now()` milliseconds.
    started: f64,
    deadline: f64,
    /// Stop before the deadline once the move can no longer change.
    stop_when_settled: bool,
}

impl Limit {
    fn new(time_ms: f64, stop_when_settled: bool) -> Self {
        let started = js_sys::Date::now();
        Limit { started, deadline: started + time_ms, stop_when_settled }
    }
}

/// Simulate on this thread until the limit.
fn simulate_until(search: &SharedSearch, eval: &dyn EvalFn, limit: Limit) {
    loop {
        for _ in 0..4 {
            search.simulate(eval);
        }
        let now = js_sys::Date::now();
        if now >= limit.deadline {
            break;
        }
        if limit.stop_when_settled && search.settled(now - limit.started, limit.deadline - now) {
            break;
        }
    }
}

/// Search up to the limit on every thread of the pool, all working on the
/// same tree.
#[cfg(feature = "threads")]
fn run_mcts_timed(search: &SharedSearch, eval: &(dyn EvalFn + Sync), limit: Limit) {
    rayon::scope(|scope| {
        for _ in 0..rayon::current_num_threads() {
            scope.spawn(|_| simulate_until(search, eval, limit));
        }
    });
}

/// Search up to the limit on the calling thread.
#[cfg(not(feature = "threads"))]
fn run_mcts_timed(search: &SharedSearch, eval: &(dyn EvalFn + Sync), limit: Limit) {
    simulate_until(search, eval, limit);
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
/// ... or at this many simulations. Near the end of a game most simulations
/// reach a finished position and add no node, so the tree would never reach
/// its node limit and the search would run forever learning nothing new.
const ANALYSIS_MAX_SIMULATIONS: u32 = 50_000_000;

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
    /// A bot playing with the embedded network called `network`:
    /// `"gladiator0"`, `"ramo0"`, `"zerorev6x48"`, or the champion for any
    /// other name.
    #[wasm_bindgen(constructor)]
    pub fn new(network: &str) -> ZeroRevBot {
        let (weights, _) = self::network(network);
        ZeroRevBot {
            eval: make_eval(weights),
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

        // The last moves are played exactly, and at once.
        if (!(black | white)).count_ones() <= ENDGAME_SOLVE_EMPTIES
            && let (Some(best), _) = solve(board, is_black)
        {
            self.last_simulations.set(0);
            return best.trailing_zeros() as i32;
        }

        let search = SharedSearch::new(board, is_black);
        run_mcts_timed(&search, self.eval.as_ref(), Limit::new(think_ms, true));

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
        let done = |tree: &MctsSearch| {
            tree.nodes() >= ANALYSIS_MAX_NODES || tree.visits() >= ANALYSIS_MAX_SIMULATIONS
        };
        if !search.with(done) {
            run_mcts_timed(search, self.eval.as_ref(), Limit::new(think_ms, false));
        }
        let (mut policy, visits, value, complete) = search
            .with(|tree| (tree.policy_target(), tree.visits(), tree.root_value(), done(tree)));
        *slot = Some(analysis);

        self.last_simulations.set(visits);
        self.last_value.set(value);
        self.analysis_complete.set(complete);
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
        Self::new("champion")
    }
}
