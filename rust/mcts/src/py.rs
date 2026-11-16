use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use reversi_minimax::board::Board;

use crate::batch::Budget;
use crate::batch::PLANES_LEN;
use crate::batch::SelfPlayPool;
use crate::batch::write_planes;
use crate::game::GameRecord;
use crate::game::play_game;
use crate::game::play_match;
use crate::game::play_match_from;
use crate::game::random_opening;
use crate::search::EvalFn;
use crate::search::StaticEval;

// ── PyEvalFn ──────────────────────────────────────────────────────────────

/// Wraps a Python callable as an `EvalFn`.
///
/// Expected Python signature:
/// ```python
/// def eval_fn(board_black: int, board_white: int, is_black: bool, legal: int
///             ) -> tuple[list[float], float]:
///     ...
/// ```
/// Returns `(policy: list[float] of length 64, value: float in [-1, 1])`.
struct PyEvalFn {
    callable: PyObject,
}

impl EvalFn for PyEvalFn {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        Python::with_gil(|py| {
            let call_result = self.callable.call1(py, (board.black, board.white, is_black, legal));

            let result = match call_result {
                Ok(r) => r,
                Err(e) => {
                    // Python exception from the eval callable (e.g. Metal OOM).
                    // Print and fall back to a uniform prior + zero value so the
                    // game can continue rather than hard-crashing.
                    e.print(py);
                    return StaticEval.evaluate(board, is_black, legal);
                }
            };

            match result.extract::<(Vec<f32>, f32)>(py) {
                Ok((policy_vec, value)) => {
                    let mut policy = [0.0f32; 64];
                    for (i, p) in policy_vec.into_iter().enumerate().take(64) {
                        policy[i] = p;
                    }
                    (policy, value)
                }
                Err(_) => StaticEval.evaluate(board, is_black, legal),
            }
        })
    }
}

// ── MctsWorker ────────────────────────────────────────────────────────────

#[pyclass]
pub struct MctsWorker {
    simulations: u32,
}

#[pymethods]
impl MctsWorker {
    #[new]
    pub fn new(simulations: u32) -> Self {
        MctsWorker { simulations }
    }

    /// Run one complete self-play game.
    ///
    /// `eval_fn` — optional Python callable:
    ///   `(board_black: int, board_white: int, is_black: bool, legal: int)
    ///    -> (policy: list[float], value: float)`
    ///
    /// When `None` (or omitted), falls back to the minimax static evaluator.
    ///
    /// Returns a list of dicts, one per move:
    ///   `{board_black, board_white, is_black, legal, mcts_policy, outcome,
    ///     margin, policy_weight, value_weight}`
    #[pyo3(signature = (eval_fn=None))]
    pub fn run_game<'py>(
        &self,
        py: Python<'py>,
        eval_fn: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyList>> {
        // Release the GIL for the duration of the MCTS loop so that multiple
        // worker threads can run their trees in parallel.  PyEvalFn re-acquires
        // it (via Python::with_gil) only for the per-leaf callback.
        let record = match eval_fn {
            Some(callable) => {
                py.allow_threads(|| play_game(&PyEvalFn { callable }, self.simulations))
            }
            None => py.allow_threads(|| play_game(&StaticEval, self.simulations)),
        };

        record_to_py(py, &record)
    }
}

/// One game as a list of per-position dicts (see [`MctsWorker::run_game`]).
fn record_to_py<'py>(py: Python<'py>, record: &GameRecord) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for pos in &record.positions {
        let d = PyDict::new(py);
        d.set_item("board_black", pos.board.black)?;
        d.set_item("board_white", pos.board.white)?;
        d.set_item("is_black", pos.is_black)?;
        d.set_item("legal", pos.legal)?;
        d.set_item("mcts_policy", pos.mcts_policy.to_vec())?;
        d.set_item("outcome", pos.outcome)?;
        d.set_item("margin", pos.margin)?;
        d.set_item("policy_weight", pos.policy_weight)?;
        d.set_item("value_weight", pos.value_weight)?;
        list.append(d)?;
    }
    Ok(list)
}

// ── BatchedSelfPlay ───────────────────────────────────────────────────────

/// Many self-play games advanced in lockstep, with one network call per round.
///
/// ```python
/// pool = BatchedSelfPlay(games=512, simulations=200)
/// while True:
///     planes = np.frombuffer(pool.collect(), np.float32).reshape(-1, 4, 8, 8)
///     policy, value = model(planes)          # policy: probabilities, (N, 64)
///     pool.apply(policy.tobytes(), value.tobytes())
///     for game in pool.take_finished(): ...
/// ```
#[pyclass]
pub struct BatchedSelfPlay {
    pool: SelfPlayPool,
    planes: Vec<f32>,
    policies: Vec<f32>,
    values: Vec<f32>,
}

/// Reinterpret native-endian `f32` bytes, reusing `out`'s allocation.
fn read_f32s(bytes: &[u8], out: &mut Vec<f32>) -> PyResult<()> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(PyValueError::new_err("buffer length is not a multiple of 4"));
    }
    out.clear();
    out.extend(chunks.iter().map(|c| f32::from_ne_bytes(*c)));
    Ok(())
}

#[pymethods]
impl BatchedSelfPlay {
    /// `stagger` spreads the game starts over about one game length so that
    /// games finish at a steady rate (batches are smaller during that ramp-up).
    /// `fast_simulations` and `full_fraction` turn on playout cap
    /// randomization (see [`Budget`]); by default every move is a full search.
    #[new]
    #[pyo3(signature = (games, simulations, stagger=true, fast_simulations=None, full_fraction=1.0))]
    pub fn new(
        games: usize,
        simulations: u32,
        stagger: bool,
        fast_simulations: Option<u32>,
        full_fraction: f32,
    ) -> Self {
        let budget =
            Budget::capped(simulations, fast_simulations.unwrap_or(simulations), full_fraction);
        let pool = if stagger {
            SelfPlayPool::staggered(games, budget)
        } else {
            SelfPlayPool::new(games, budget)
        };
        BatchedSelfPlay { pool, planes: Vec::new(), policies: Vec::new(), values: Vec::new() }
    }

    /// Advance every game to its next evaluation. Returns the network input
    /// as native-endian float32 bytes, shape `(n, 4, 8, 8)` with `n <= games`.
    pub fn collect<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let requests = self.pool.collect();
        self.planes.resize(requests.len() * PLANES_LEN, 0.0);
        for (request, out) in requests.iter().zip(self.planes.as_chunks_mut::<PLANES_LEN>().0) {
            write_planes(request, out);
        }
        let bytes: Vec<u8> = self.planes.iter().flat_map(|v| v.to_ne_bytes()).collect();
        PyBytes::new(py, &bytes)
    }

    /// Apply the evaluations for the last `collect()`: float32 bytes of shape
    /// `(n, 64)` (move probabilities) and `(n,)` (values).
    pub fn apply(&mut self, policies: &[u8], values: &[u8]) -> PyResult<()> {
        read_f32s(policies, &mut self.policies)?;
        read_f32s(values, &mut self.values)?;
        if self.policies.len() != self.values.len() * 64 {
            return Err(PyValueError::new_err("expected 64 policy entries per value"));
        }
        self.pool.apply(&self.policies, &self.values);
        Ok(())
    }

    /// Play `games` games from now on (see `SelfPlayPool::set_games`).
    pub fn set_games(&mut self, games: usize) {
        self.pool.set_games(games);
    }

    /// Games in play once the last `set_games` has settled.
    #[getter]
    pub fn games(&self) -> usize {
        self.pool.games()
    }

    /// Games finished since the last call, each as a list of position dicts.
    pub fn take_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let games = PyList::empty(py);
        for record in self.pool.take_finished() {
            games.append(record_to_py(py, &record)?)?;
        }
        Ok(games)
    }
}

/// Play one game between two Python eval callables.
/// Returns +1.0 if black wins, -1.0 if white wins, 0.0 if draw.
#[pyfunction]
#[pyo3(signature = (eval_black, eval_white, simulations=50))]
pub fn run_match(
    py: Python,
    eval_black: PyObject,
    eval_white: PyObject,
    simulations: u32,
) -> PyResult<f32> {
    let eb = PyEvalFn { callable: eval_black };
    let ew = PyEvalFn { callable: eval_white };
    Ok(py.allow_threads(|| play_match(&eb, &ew, simulations)))
}

/// Run the Rust inference path (the one the browser bots use) on weights in
/// the exported `.bin` format. `planes` is one position, 4×8×8 channel-first.
/// Returns `(policy, value)`; used to check parity with the MLX model.
#[pyfunction]
pub fn nn_forward(weights: &[u8], planes: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
    if planes.len() != PLANES_LEN {
        return Err(PyValueError::new_err("planes must hold 4*8*8 floats"));
    }
    let net = reversi_nn::AlphaZeroNet::load(weights).map_err(PyValueError::new_err)?;
    let (policy, value) = net.forward(&planes);
    Ok((policy.to_vec(), value))
}

// ── Tournament ────────────────────────────────────────────────────────────

/// [`EvalFn`] backed by the CPU inference path that the browser bots use.
struct NetEval<'a>(&'a reversi_nn::AlphaZeroNet);

impl EvalFn for NetEval<'_> {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        let (mine, theirs) =
            if is_black { (board.black, board.white) } else { (board.white, board.black) };
        let mut planes = [0.0f32; PLANES_LEN];
        for i in 0..64 {
            planes[i] = ((mine >> i) & 1) as f32;
            planes[64 + i] = ((theirs >> i) & 1) as f32;
            planes[128 + i] = ((legal >> i) & 1) as f32;
        }
        planes[192..].fill((board.black | board.white).count_ones() as f32 / 64.0);
        self.0.forward(&planes)
    }
}

/// Round-robin between networks given as exported `.bin` blobs.
///
/// Every pairing plays `games_per_pair` games: each random opening
/// (`opening_plies` random moves, derived from `seed`) is played twice with
/// colours swapped. Games run on `threads` CPU threads with the Rust inference
/// path and without the GIL, so training keeps the GPU.
///
/// Returns the points per entrant (win 1, draw 0.5), in input order.
#[pyfunction]
#[pyo3(signature = (weights, games_per_pair, simulations, opening_plies=4, seed=0, threads=1))]
pub fn run_tournament(
    py: Python,
    weights: Vec<Vec<u8>>,
    games_per_pair: u32,
    simulations: u32,
    opening_plies: u32,
    seed: u64,
    threads: usize,
) -> PyResult<Vec<f32>> {
    let nets = weights
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            reversi_nn::AlphaZeroNet::load(bytes)
                .map_err(|err| PyValueError::new_err(format!("entrant {i}: {err}")))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // (black, white, opening seed)
    let mut games = Vec::new();
    for a in 0..nets.len() {
        for b in a + 1..nets.len() {
            for g in 0..games_per_pair {
                let opening = seed
                    .wrapping_add((a * nets.len() + b) as u64 * 1_000_003)
                    .wrapping_add(u64::from(g / 2));
                games.push(if g % 2 == 0 { (a, b, opening) } else { (b, a, opening) });
            }
        }
    }

    let next = AtomicUsize::new(0);
    let scores = py.allow_threads(|| {
        let play = || {
            let mut scores = vec![0.0f32; nets.len()];
            while let Some(&(black, white, opening)) =
                games.get(next.fetch_add(1, Ordering::Relaxed))
            {
                let (board, black_to_move) = random_opening(opening_plies, opening);
                let outcome = play_match_from(
                    board,
                    black_to_move,
                    &NetEval(&nets[black]),
                    &NetEval(&nets[white]),
                    simulations,
                );
                scores[black] += (1.0 + outcome) / 2.0;
                scores[white] += (1.0 - outcome) / 2.0;
            }
            scores
        };
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.max(1)).map(|_| scope.spawn(play)).collect();
            let mut total = vec![0.0f32; nets.len()];
            for worker in workers {
                let scores = worker.join().expect("tournament worker panicked");
                for (t, s) in total.iter_mut().zip(scores) {
                    *t += s;
                }
            }
            total
        })
    });
    Ok(scores)
}

/// Game pairs between two networks (exported `.bin` blobs), for a sequential
/// test: pair `i` plays one random opening (`opening_plies` moves, derived
/// from `seed + i`) twice, with colours swapped.
///
/// Returns the points of `a` per pair, from 0 to 2 (win 1, draw 0.5), in pair
/// order. Games run on `threads` CPU threads without the GIL.
#[pyfunction]
#[pyo3(signature = (a, b, pairs, simulations, opening_plies=4, seed=0, threads=1))]
#[allow(clippy::too_many_arguments)]
pub fn run_pairs(
    py: Python,
    a: Vec<u8>,
    b: Vec<u8>,
    pairs: usize,
    simulations: u32,
    opening_plies: u32,
    seed: u64,
    threads: usize,
) -> PyResult<Vec<f32>> {
    let load = |bytes: &[u8], name: &str| {
        reversi_nn::AlphaZeroNet::load(bytes)
            .map_err(|err| PyValueError::new_err(format!("{name}: {err}")))
    };
    let nets = [load(&a, "a")?, load(&b, "b")?];
    let next = AtomicUsize::new(0);
    let points = py.allow_threads(|| {
        let play = || {
            let mut points = Vec::new();
            loop {
                let game = next.fetch_add(1, Ordering::Relaxed);
                if game >= 2 * pairs {
                    break points;
                }
                let (pair, a_is_black) = (game / 2, game % 2 == 0);
                let (board, black_to_move) =
                    random_opening(opening_plies, seed.wrapping_add(pair as u64));
                let (black, white) = if a_is_black { (0, 1) } else { (1, 0) };
                let outcome = play_match_from(
                    board,
                    black_to_move,
                    &NetEval(&nets[black]),
                    &NetEval(&nets[white]),
                    simulations,
                );
                let for_a = if a_is_black { outcome } else { -outcome };
                points.push((pair, (1.0 + for_a) / 2.0));
            }
        };
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.max(1)).map(|_| scope.spawn(play)).collect();
            let mut total = vec![0.0f32; pairs];
            for worker in workers {
                for (pair, p) in worker.join().expect("match worker panicked") {
                    total[pair] += p;
                }
            }
            total
        })
    });
    Ok(points)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MctsWorker>()?;
    m.add_class::<BatchedSelfPlay>()?;
    m.add_function(wrap_pyfunction!(run_match, m)?)?;
    m.add_function(wrap_pyfunction!(nn_forward, m)?)?;
    m.add_function(wrap_pyfunction!(run_tournament, m)?)?;
    m.add_function(wrap_pyfunction!(run_pairs, m)?)?;
    Ok(())
}
