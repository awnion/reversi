pub mod game;
#[cfg(feature = "python")]
pub mod py;
pub mod search;

pub use game::GameRecord;
pub use game::PositionRecord;
pub use game::play_game;
#[cfg(feature = "python")]
use pyo3::prelude::*;
pub use search::EvalFn;
pub use search::MctsSearch;
pub use search::StaticEval;

#[cfg(feature = "python")]
#[pymodule]
fn reversi_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)
}
