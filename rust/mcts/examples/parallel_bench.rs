//! How the shared search scales with threads when each evaluation costs about
//! as much as the network does in the browser (~0.7 ms):
//! `cargo run --release -p reversi_mcts --example parallel_bench`

use std::time::Duration;
use std::time::Instant;

use reversi_mcts::EvalFn;
use reversi_mcts::SharedSearch;
use reversi_mcts::StaticEval;
use reversi_mcts::run_threads;
use reversi_minimax::board::Board;

/// The static evaluator, slowed down to the cost of a network forward pass.
struct SlowEval(Duration);

impl EvalFn for SlowEval {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        let start = Instant::now();
        while start.elapsed() < self.0 {
            std::hint::spin_loop();
        }
        StaticEval.evaluate(board, is_black, legal)
    }
}

fn main() {
    let board = Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36));
    let eval = SlowEval(Duration::from_micros(700));
    let window = Duration::from_secs(2);
    let mut single = 0.0;
    for threads in [1, 2, 4, 8] {
        let search = SharedSearch::new(board, true);
        let start = Instant::now();
        run_threads(&search, &eval, threads, &|| start.elapsed() < window);
        let rate = search.into_inner().visits() as f64 / start.elapsed().as_secs_f64();
        if threads == 1 {
            single = rate;
        }
        println!("{threads} threads: {rate:6.0} simulations/s  ({:.1}x)", rate / single);
    }
}
