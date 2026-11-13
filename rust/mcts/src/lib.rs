pub mod arena;
pub mod batch;
pub mod endgame;
pub mod game;
pub mod parallel;
#[cfg(feature = "python")]
pub mod py;
pub mod search;

pub use arena::MatchPool;
pub use batch::Budget;
pub use batch::PLANES_LEN;
pub use batch::SelfPlayPool;
pub use batch::write_planes;
pub use batch::write_position_planes;
pub use game::GameRecord;
pub use game::PositionRecord;
pub use game::play_game;
pub use game::play_match_from;
pub use game::random_opening;
pub use parallel::SharedSearch;
#[cfg(not(target_arch = "wasm32"))]
pub use parallel::run_threads;
#[cfg(feature = "python")]
use pyo3::prelude::*;
pub use search::EvalFn;
pub use search::EvalRequest;
pub use search::MctsSearch;
pub use search::StaticEval;

#[cfg(feature = "python")]
#[pymodule]
fn reversi_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)
}
