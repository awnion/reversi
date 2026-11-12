use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use reversi_minimax::board::Board;

use crate::batch::PLANES_LEN;
use crate::batch::SelfPlayPool;
use crate::batch::write_planes;
use crate::game::GameRecord;
use crate::game::play_game;
use crate::game::play_match;
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
    ///     policy_weight, value_weight}`
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
    #[new]
    #[pyo3(signature = (games, simulations, stagger=true))]
    pub fn new(games: usize, simulations: u32, stagger: bool) -> Self {
        let pool = if stagger {
            SelfPlayPool::staggered(games, simulations)
        } else {
            SelfPlayPool::new(games, simulations)
        };
        BatchedSelfPlay { pool, planes: Vec::new(), policies: Vec::new(), values: Vec::new() }
    }

    /// Advance every game to its next evaluation. Returns the network input
    /// as native-endian float32 bytes, shape `(n, 4, 8, 8)` with `n <= games`.
    pub fn collect<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let requests = self.pool.collect();
        self.planes.resize(requests.len() * PLANES_LEN, 0.0);
        for (request, out) in requests.iter().zip(self.planes.chunks_exact_mut(PLANES_LEN)) {
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MctsWorker>()?;
    m.add_class::<BatchedSelfPlay>()?;
    m.add_function(wrap_pyfunction!(run_match, m)?)?;
    Ok(())
}
