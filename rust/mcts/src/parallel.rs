//! One search tree shared between threads.
//!
//! The network evaluation is by far the most expensive part of a simulation,
//! so the tree itself is simply kept behind a mutex: a thread locks it to
//! pick a leaf, releases it while the network runs, and locks it again to
//! store the result. In-flight simulations are marked on their path (virtual
//! loss), which sends concurrent threads down different branches.

use std::sync::Mutex;

use reversi_minimax::board::Board;

use crate::search::EvalFn;
use crate::search::MctsSearch;

pub struct SharedSearch {
    tree: Mutex<MctsSearch>,
}

impl SharedSearch {
    pub fn new(board: Board, is_black: bool) -> Self {
        SharedSearch { tree: Mutex::new(MctsSearch::new(board, is_black)) }
    }

    /// Run one simulation. Safe to call from any number of threads at once.
    pub fn simulate(&self, eval: &dyn EvalFn) {
        let request = self.lock().select_leaf_shared();
        if let Some(request) = request {
            // The lock is not held while the network runs.
            let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
            self.lock().apply_eval(&request, &policy, value);
        }
    }

    /// Whether a timed search of a fresh tree may stop early, `elapsed_ms`
    /// into it with `left_ms` to go: at the pace so far, the simulations still
    /// to come could not make another move the most visited one.
    pub fn settled(&self, elapsed_ms: f64, left_ms: f64) -> bool {
        /// Too few simulations say little about the pace.
        const MIN_VISITS: u32 = 32;
        self.with(|tree| {
            let visits = tree.visits();
            if visits < MIN_VISITS || elapsed_ms <= 0.0 {
                return false;
            }
            let to_come = (f64::from(visits) * left_ms.max(0.0) / elapsed_ms).ceil() as u32;
            tree.settled(to_come.saturating_add(tree.in_flight()))
        })
    }

    /// Read the tree, e.g. for `best_move()` once the threads are done.
    pub fn with<R>(&self, read: impl FnOnce(&MctsSearch) -> R) -> R {
        read(&self.lock())
    }

    pub fn into_inner(self) -> MctsSearch {
        self.tree.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MctsSearch> {
        // A panic in another thread leaves the tree usable for reading.
        self.tree.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Simulate on `threads` OS threads until `keep_going` returns false.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_threads(
    search: &SharedSearch,
    eval: &(dyn EvalFn + Sync),
    threads: usize,
    keep_going: &(dyn Fn() -> bool + Sync),
) {
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                while keep_going() {
                    search.simulate(eval);
                }
            });
        }
    });
}
