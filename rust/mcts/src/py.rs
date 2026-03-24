use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use reversi_minimax::board::Board;

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
    ///   `{board_black, board_white, is_black, mcts_policy, outcome}`
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

        let list = PyList::empty(py);
        for pos in &record.positions {
            let d = PyDict::new(py);
            d.set_item("board_black", pos.board.black)?;
            d.set_item("board_white", pos.board.white)?;
            d.set_item("is_black", pos.is_black)?;
            d.set_item("mcts_policy", pos.mcts_policy.to_vec())?;
            d.set_item("outcome", pos.outcome)?;
            list.append(d)?;
        }
        Ok(list)
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
    m.add_function(wrap_pyfunction!(run_match, m)?)?;
    Ok(())
}
