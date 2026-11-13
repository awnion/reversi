//! Time the exact endgame solver by number of empty squares:
//! `cargo run --release -p reversi_mcts --example endgame_bench`
use std::time::Instant;

use reversi_mcts::endgame::solve;
use reversi_mcts::random_opening;

fn main() {
    for empties in [10u32, 12, 14, 16] {
        let mut times = Vec::new();
        for seed in 0..40u64 {
            let (board, is_black) = random_opening(60 - empties, seed * 7919 + 1);
            if 64 - (board.black | board.white).count_ones() != empties {
                continue;
            }
            let started = Instant::now();
            std::hint::black_box(solve(board, is_black));
            times.push(started.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(f64::total_cmp);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "{empties} empties: mean {mean:.2} ms, median {:.2} ms, max {:.2} ms ({} positions)",
            times[times.len() / 2],
            times[times.len() - 1],
            times.len()
        );
    }
}
