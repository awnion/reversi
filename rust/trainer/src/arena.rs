//! `reversi-selfplay match`: two networks play pairs of games on the GPU.
//!
//! Search threads each own a [`MatchPool`] over a share of the pairs and send
//! every round's leaves, split by network, to the GPU thread (here: the
//! calling thread). It waits until every live worker has sent its round (or
//! `MAX_WAIT`), runs each network once over all of its positions and hands
//! the rows back. Without the cache and the pipelining of self-play: a match
//! is short, and the virtual-loss leaves already make the batches large.

use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use reversi_mcts::MatchPool;
use reversi_mcts::all_openings;
use reversi_mcts::random_opening;
use reversi_minimax::board::Board;

use crate::net::FoldedWeights;
use crate::net::NetOptions;
use crate::net::TorchNet;
use crate::net::pack;
use crate::pipeline::bucket;

/// After the first request of a round, wait at most this long for the rest.
const MAX_WAIT: Duration = Duration::from_millis(5);

pub struct MatchConfig {
    pub pairs: usize,
    pub simulations: u32,
    pub opening_plies: u32,
    pub seed: u64,
    /// Play every distinct position after `opening_plies` moves once (both
    /// colours) instead of `pairs` random openings.
    pub all_openings: bool,
    pub workers: usize,
    pub leaves_per_round: u32,
}

enum Message {
    /// A round's positions, packed as in [`pack`], one list per network.
    Eval { worker: usize, packed: [Vec<i64>; 2] },
    /// The worker's games are over: points of network 0 per pair.
    Done(Vec<(usize, f32)>),
}

/// One worker's games, played to the end.
fn run_worker(
    id: usize,
    mut pool: MatchPool,
    requests: &Sender<Message>,
    replies: &Receiver<[Vec<f32>; 2]>,
) {
    let mut policies = Vec::new();
    let mut values = Vec::new();
    let mut round_nets: Vec<u8> = Vec::new();
    loop {
        let mut packed = [Vec::new(), Vec::new()];
        let (batch, nets) = pool.collect();
        if batch.is_empty() {
            let _ = requests.send(Message::Done(pool.pair_points()));
            return;
        }
        round_nets.clear();
        round_nets.extend_from_slice(nets);
        for (request, &net) in batch.iter().zip(nets) {
            packed[net as usize].extend(pack(request.board, request.is_black, request.legal));
        }
        if requests.send(Message::Eval { worker: id, packed }).is_err() {
            return;
        }
        let Ok(rows) = replies.recv() else { return };
        policies.clear();
        values.clear();
        let mut next = [0usize; 2];
        for &net in &round_nets {
            let net = net as usize;
            let row = &rows[net][next[net] * 65..(next[net] + 1) * 65];
            next[net] += 1;
            policies.extend_from_slice(&row[..64]);
            values.push(row[64]);
        }
        pool.apply(&policies, &values);
    }
}

/// Points of `a` per pair (0 to 2): pair `i` from opening `seed + i`, or from
/// the `i`-th of [`all_openings`] with `config.all_openings`.
pub fn run_match(
    a: &FoldedWeights,
    b: &FoldedWeights,
    options: NetOptions,
    config: &MatchConfig,
) -> Vec<f32> {
    let _no_grad = tch::no_grad_guard();
    if options.device.is_cuda() {
        tch::Cuda::cudnn_set_benchmark(true);
    }
    let nets = [TorchNet::new(a, options), TorchNet::new(b, options)];
    let openings: Vec<(usize, (Board, bool))> = if config.all_openings {
        all_openings(config.opening_plies).into_iter().enumerate().collect()
    } else {
        (0..config.pairs)
            .map(|i| (i, random_opening(config.opening_plies, config.seed.wrapping_add(i as u64))))
            .collect()
    };
    let pairs = openings.len();
    let workers = config.workers.clamp(1, pairs.max(1));
    let (request_tx, request_rx) = crossbeam_channel::unbounded();
    let mut points = vec![0.0f32; pairs];

    std::thread::scope(|scope| {
        let mut reply_txs = Vec::new();
        for id in 0..workers {
            let (reply_tx, reply_rx) = crossbeam_channel::unbounded();
            reply_txs.push(reply_tx);
            let share = openings[pairs * id / workers..pairs * (id + 1) / workers].to_vec();
            let pool = MatchPool::from_openings(share, config.simulations, config.leaves_per_round);
            let requests = request_tx.clone();
            scope.spawn(move || run_worker(id, pool, &requests, &reply_rx));
        }
        drop(request_tx);

        let mut live = workers;
        let mut round: Vec<(usize, [Vec<i64>; 2])> = Vec::new();
        let mut packed: Vec<i64> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        while live > 0 {
            // Gather one round: every live worker's request, or what came
            // within MAX_WAIT of the first.
            let mut deadline: Option<Instant> = None;
            while round.len() < live {
                let message = match deadline {
                    None => request_rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
                    Some(at) => {
                        request_rx.recv_timeout(at.saturating_duration_since(Instant::now()))
                    }
                };
                match message {
                    Ok(Message::Eval { worker, packed }) => {
                        round.push((worker, packed));
                        deadline.get_or_insert_with(|| Instant::now() + MAX_WAIT);
                    }
                    Ok(Message::Done(pair_points)) => {
                        live -= 1;
                        for (pair, p) in pair_points {
                            points[pair] = p;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => {
                        live = 0;
                        break;
                    }
                }
            }
            if round.is_empty() {
                continue;
            }

            // Each network once over all of its positions.
            let mut rows: Vec<[Vec<f32>; 2]> =
                round.iter().map(|_| [Vec::new(), Vec::new()]).collect();
            for (net_index, net) in nets.iter().enumerate() {
                packed.clear();
                for (_, request) in &round {
                    packed.extend_from_slice(&request[net_index]);
                }
                let total = packed.len() / 4;
                if total == 0 {
                    continue;
                }
                let padded = bucket(total);
                packed.resize(padded * 4, 0);
                let output = net.forward(&net.unpack(&packed));
                out.resize(padded * 65, 0.0);
                output.copy_data(&mut out, padded * 65);
                let mut offset = 0;
                for (slot, (_, request)) in rows.iter_mut().zip(&round) {
                    let n = request[net_index].len() / 4;
                    slot[net_index] = out[offset * 65..(offset + n) * 65].to_vec();
                    offset += n;
                }
            }
            for ((worker, _), rows) in round.drain(..).zip(rows) {
                let _ = reply_txs[worker].send(rows);
            }
        }
    });
    points
}
