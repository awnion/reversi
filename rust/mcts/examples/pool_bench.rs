//! Single-thread throughput of the batched self-play core, without a network:
//! `cargo run --release -p reversi_mcts --example pool_bench [games] [seconds]`

use std::time::Duration;
use std::time::Instant;

use reversi_mcts::EvalFn;
use reversi_mcts::PLANES_LEN;
use reversi_mcts::SelfPlayPool;
use reversi_mcts::StaticEval;
use reversi_mcts::write_planes;

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(512);
    let seconds: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(5);

    let eval = StaticEval;
    let mut pool = SelfPlayPool::new(games, 200);
    let mut planes = vec![0.0_f32; games * PLANES_LEN];
    let mut policies = vec![0.0_f32; games * 64];
    let mut values = vec![0.0_f32; games];

    let (mut evals, mut positions) = (0u64, 0u64);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) {
        let requests = pool.collect();
        for (i, request) in requests.iter().enumerate() {
            write_planes(request, &mut planes[i * PLANES_LEN..(i + 1) * PLANES_LEN]);
            let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
            policies[i * 64..(i + 1) * 64].copy_from_slice(&policy);
            values[i] = value;
        }
        evals += requests.len() as u64;
        positions += pool.take_finished().iter().map(|r| r.positions.len() as u64).sum::<u64>();
        pool.apply(&policies, &values);
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "games={games}: {:.0} evals/s, {:.0} finished positions/s (single thread)",
        evals as f64 / secs,
        positions as f64 / secs,
    );
}
