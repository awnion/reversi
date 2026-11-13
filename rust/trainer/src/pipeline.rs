//! The self-play pipeline.
//!
//! - **Workers** (CPU threads) each own a few [`SelfPlayPool`]s. A pool's
//!   `collect()` becomes one [`Request`] tagged with the worker and pool; the
//!   pool then waits for its [`Reply`] before it is touched again, so every
//!   pool has at most one outstanding batch and `apply()` always gets the
//!   evaluations of its own last `collect()`. While one pool waits, the worker
//!   searches in the others.
//! - **The GPU thread** owns all of libtorch/CUDA: weight uploads included. It
//!   gathers requests from all pools into one large batch (up to the batch
//!   target), pads it to a bucket size (few distinct shapes for cuDNN's
//!   autotuner), evaluates it and sends each pool its slice back.
//! - **The writer** turns finished games into replay shards.
//!
//! Every blocking wait also selects on the stop channel, so a stop wakes all
//! stages at once.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use crossbeam_channel::TryRecvError;
use crossbeam_channel::select;
use reversi_mcts::GameRecord;
use reversi_mcts::SelfPlayPool;

use crate::control::LoadedWeights;
use crate::control::PanicGuard;
use crate::control::Stop;
use crate::log_selfplay;
use crate::net::NetOptions;
use crate::net::TorchNet;
use crate::net::pack;
use crate::shard::ShardWriter;
use crate::thermal::Pacer;

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub workers: usize,
    pub pools_per_worker: usize,
    pub games_per_pool: usize,
    pub simulations: u32,
    /// Spread game starts over one game length (steady output, but small
    /// batches for the first ~60 × simulations rounds).
    pub stagger: bool,
    /// Positions per network call; requests are never split.
    pub batch_target: usize,
    /// Wait up to `max_wait` for at least this many positions.
    pub min_batch: usize,
    pub max_wait: Duration,
    /// Network outputs kept per [`EvalCache`] generation (0 = no cache).
    pub cache_entries: usize,
}

/// Counters shared with the progress log and the benchmark.
#[derive(Default)]
pub struct Stats {
    pub evals: AtomicU64,
    pub padded_evals: AtomicU64,
    pub batches: AtomicU64,
    pub last_batch: AtomicU64,
    /// Nanoseconds from upload to completed readback, summed over batches.
    pub gpu_ns: AtomicU64,
    pub games: AtomicU64,
    pub positions: AtomicU64,
    /// Positions answered from the [`EvalCache`] or by a duplicate in the
    /// same batch, without a network evaluation.
    pub cache_hits: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub at: Option<Instant>,
    pub evals: u64,
    pub padded_evals: u64,
    pub batches: u64,
    pub gpu_ns: u64,
    pub games: u64,
    pub positions: u64,
    pub cache_hits: u64,
}

impl Stats {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            at: Some(Instant::now()),
            evals: self.evals.load(Ordering::Relaxed),
            padded_evals: self.padded_evals.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            gpu_ns: self.gpu_ns.load(Ordering::Relaxed),
            games: self.games.load(Ordering::Relaxed),
            positions: self.positions.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
        }
    }
}

/// One pool's positions to evaluate, packed as in [`pack`].
pub struct Request {
    worker: usize,
    pool: usize,
    packed: Vec<i64>,
}

impl Request {
    fn len(&self) -> usize {
        self.packed.len() / 4
    }
}

/// Evaluations for one [`Request`], in request order.
pub struct Reply {
    pool: usize,
    policies: Vec<f32>,
    values: Vec<f32>,
    /// The request's buffer, handed back for reuse.
    packed: Vec<i64>,
}

// ── Workers ───────────────────────────────────────────────────────────────

pub fn run_worker(
    id: usize,
    config: &PipelineConfig,
    requests: &Sender<Request>,
    replies: &Receiver<Reply>,
    games: &Sender<GameRecord>,
    stop: &Stop,
) {
    let _guard = PanicGuard { stop: stop.clone(), stage: "worker" };
    let new_pool = || {
        if config.stagger {
            SelfPlayPool::staggered(config.games_per_pool, config.simulations)
        } else {
            SelfPlayPool::new(config.games_per_pool, config.simulations)
        }
    };
    let mut pools: Vec<SelfPlayPool> = (0..config.pools_per_worker).map(|_| new_pool()).collect();
    let mut spare: Vec<Vec<i64>> = Vec::new();

    let submit = |pool: usize, pools: &mut [SelfPlayPool], spare: &mut Vec<Vec<i64>>| {
        let mut packed = spare.pop().unwrap_or_default();
        packed.clear();
        for request in pools[pool].collect() {
            packed.extend(pack(request.board, request.is_black, request.legal));
        }
        requests.send(Request { worker: id, pool, packed }).is_ok()
    };

    for pool in 0..pools.len() {
        if !submit(pool, &mut pools, &mut spare) {
            return;
        }
    }
    loop {
        select! {
            recv(replies) -> reply => {
                let Ok(reply) = reply else { return };
                let pool = &mut pools[reply.pool];
                pool.apply(&reply.policies, &reply.values);
                for record in pool.take_finished() {
                    if games.send(record).is_err() {
                        return;
                    }
                }
                spare.push(reply.packed);
                if stop.is_set() || !submit(reply.pool, &mut pools, &mut spare) {
                    return;
                }
            }
            recv(stop.receiver()) -> _ => return,
        }
    }
}

// ── Evaluation cache ──────────────────────────────────────────────────────

/// A position as the network sees it: own discs, opponent's discs (colour
/// does not enter the input, and the rest of the planes follows from these).
type CacheKey = (i64, i64);

/// One network output row: 64 priors and the value.
type Row = [f32; 65];

/// Hasher for [`CacheKey`]: the discs are already well mixed bitboards.
#[derive(Default)]
struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("only i64 keys are hashed")
    }

    fn write_i64(&mut self, n: i64) {
        self.0 = (self.0.rotate_left(29) ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type KeyMap<V> = std::collections::HashMap<CacheKey, V, std::hash::BuildHasherDefault<KeyHasher>>;

/// Network outputs for positions seen recently. The same positions come up
/// again and again: openings across games and transpositions inside a search.
/// Two generations bound the memory: when the current one is full it becomes
/// the previous one and the older entries are dropped. Cleared on every
/// weights update, so answers always come from the current network.
pub struct EvalCache {
    current: KeyMap<Row>,
    previous: KeyMap<Row>,
    capacity: usize,
}

impl EvalCache {
    pub fn new(capacity: usize) -> Self {
        EvalCache { current: KeyMap::default(), previous: KeyMap::default(), capacity }
    }

    fn get(&self, key: &CacheKey) -> Option<&Row> {
        self.current.get(key).or_else(|| self.previous.get(key))
    }

    fn insert(&mut self, key: CacheKey, row: Row) {
        if self.capacity == 0 {
            return;
        }
        if self.current.len() >= self.capacity {
            self.previous = std::mem::take(&mut self.current);
        }
        self.current.insert(key, row);
    }

    fn clear(&mut self) {
        self.current.clear();
        self.previous.clear();
    }
}

// ── GPU thread ────────────────────────────────────────────────────────────

/// Batch sizes are padded to a few buckets, so that cuDNN's autotuner sees a
/// bounded set of shapes: powers of two up to 512, then multiples of 512.
pub fn bucket(n: usize) -> usize {
    if n <= 512 { n.next_power_of_two().max(16) } else { n.div_ceil(512) * 512 }
}

pub struct GpuThread {
    pub config: PipelineConfig,
    pub options: NetOptions,
    pub requests: Receiver<Request>,
    pub replies: Vec<Sender<Reply>>,
    pub weights: Receiver<LoadedWeights>,
    pub paused: Arc<AtomicBool>,
    /// Thermal throttle (f64 bits), see `thermal::Governor`.
    pub throttle: Arc<AtomicU64>,
    pub stats: Arc<Stats>,
    pub stop: Stop,
}

impl GpuThread {
    pub fn run(self) {
        let _guard = PanicGuard { stop: self.stop.clone(), stage: "gpu" };
        let _no_grad = tch::no_grad_guard();
        tch::set_num_threads(1);
        if self.options.device.is_cuda() {
            tch::Cuda::cudnn_set_benchmark(true);
        }

        let mut net = select! {
            recv(self.weights) -> loaded => {
                let Ok(loaded) = loaded else { return };
                let net = TorchNet::new(&loaded.weights, self.options);
                log_selfplay!("weights-loaded", process = 0, source = loaded.source);
                net
            }
            recv(self.stop.receiver()) -> _ => return,
        };

        let config = &self.config;
        let mut carry: Option<Request> = None;
        let mut batch: Vec<Request> = Vec::new();
        let mut packed: Vec<i64> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        let mut rows: Vec<f32> = Vec::new();
        let mut cache = EvalCache::new(config.cache_entries);
        let mut unique: KeyMap<usize> = KeyMap::default();
        let mut pending: Vec<(usize, usize)> = Vec::new();
        let mut pacer = Pacer::default();
        let mut paused_since: Option<Instant> = None;

        while !self.stop.is_set() {
            // Newest weights first, uploaded between batches.
            let mut newest = None;
            while let Ok(loaded) = self.weights.try_recv() {
                newest = Some(loaded);
            }
            if let Some(loaded) = newest {
                net = TorchNet::new(&loaded.weights, self.options);
                cache.clear();
                log_selfplay!("weights-reloaded", process = 0, source = loaded.source);
            }

            // The learner asks everyone to rest after a tournament.
            if self.paused.load(Ordering::SeqCst) {
                if paused_since.is_none() {
                    paused_since = Some(Instant::now());
                    log_selfplay!("cooldown-pause", process = 0);
                }
                self.stop.sleep(Duration::from_millis(250));
                continue;
            }
            if let Some(since) = paused_since.take() {
                log_selfplay!(
                    "cooldown-resume",
                    process = 0,
                    seconds = format!("{:.0}", since.elapsed().as_secs_f64()),
                );
            }

            // Gather a batch.
            let first = match carry.take() {
                Some(request) => request,
                None => select! {
                    recv(self.requests) -> request => match request {
                        Ok(request) => request,
                        Err(_) => return,
                    },
                    recv(self.stop.receiver()) -> _ => return,
                    default(Duration::from_millis(250)) => continue,
                },
            };
            let mut total = first.len();
            batch.push(first);
            let gather_started = Instant::now();
            while total < config.batch_target {
                let next = match self.requests.try_recv() {
                    Ok(request) => request,
                    Err(TryRecvError::Disconnected) => break,
                    Err(TryRecvError::Empty) => {
                        let waited = gather_started.elapsed();
                        if total >= config.min_batch || waited >= config.max_wait {
                            break;
                        }
                        match self.requests.recv_timeout(config.max_wait - waited) {
                            Ok(request) => request,
                            Err(_) => break,
                        }
                    }
                };
                if total + next.len() > config.batch_target {
                    carry = Some(next);
                    break;
                }
                total += next.len();
                batch.push(next);
            }

            // Answer what the cache knows, evaluate each other position once.
            rows.clear();
            rows.resize(total * 65, 0.0);
            packed.clear();
            unique.clear();
            pending.clear();
            let mut row = 0;
            for request in &batch {
                for position in request.packed.as_chunks::<4>().0 {
                    let key = (position[0], position[1]);
                    if let Some(cached) = cache.get(&key) {
                        rows[row * 65..(row + 1) * 65].copy_from_slice(cached);
                    } else {
                        let next = unique.len();
                        let slot = *unique.entry(key).or_insert(next);
                        if slot == next {
                            packed.extend_from_slice(position);
                        }
                        pending.push((row, slot));
                    }
                    row += 1;
                }
            }
            let evaluated = unique.len();
            let started = Instant::now();
            let mut padded = 0;
            if evaluated > 0 {
                padded = bucket(evaluated);
                packed.resize(padded * 4, 0);
                let output = net.forward(&net.unpack(&packed));
                out.resize(padded * 65, 0.0);
                // Blocks until the batch has completed on the device.
                output.copy_data(&mut out, padded * 65);
                for &(row, slot) in &pending {
                    rows[row * 65..(row + 1) * 65]
                        .copy_from_slice(&out[slot * 65..(slot + 1) * 65]);
                }
                for (key, slot) in unique.drain() {
                    let mut entry = [0.0; 65];
                    entry.copy_from_slice(&out[slot * 65..(slot + 1) * 65]);
                    cache.insert(key, entry);
                }
            }
            let work = started.elapsed();

            // Hand every pool its slice.
            let mut offset = 0;
            for request in batch.drain(..) {
                let n = request.len();
                let rows = &rows[offset * 65..(offset + n) * 65];
                let mut policies = Vec::with_capacity(n * 64);
                let mut values = Vec::with_capacity(n);
                for row in rows.as_chunks::<65>().0 {
                    policies.extend_from_slice(&row[..64]);
                    values.push(row[64]);
                }
                offset += n;
                // A worker that is gone is shutting down; nothing to do.
                let _ = self.replies[request.worker].send(Reply {
                    pool: request.pool,
                    policies,
                    values,
                    packed: request.packed,
                });
            }

            let stats = &self.stats;
            stats.evals.fetch_add(evaluated as u64, Ordering::Relaxed);
            stats.cache_hits.fetch_add((total - evaluated) as u64, Ordering::Relaxed);
            stats.padded_evals.fetch_add(padded as u64, Ordering::Relaxed);
            stats.batches.fetch_add(1, Ordering::Relaxed);
            stats.last_batch.store(evaluated as u64, Ordering::Relaxed);
            stats.gpu_ns.fetch_add(work.as_nanos() as u64, Ordering::Relaxed);

            // Slow down while the machine is too hot, in proportion to the
            // completed work.
            let throttle = f64::from_bits(self.throttle.load(Ordering::Relaxed));
            let pause = pacer.pause_after(work.as_secs_f64(), throttle);
            if pause > 0.0 {
                self.stop.sleep(Duration::from_secs_f64(pause));
            }
        }
    }
}

// ── Writer ────────────────────────────────────────────────────────────────

/// After a stop, keep draining finished games for at most this long.
const DRAIN_AFTER_STOP: Duration = Duration::from_secs(2);

/// Writes finished games until every worker is gone (or shortly after a
/// stop), then flushes the partial shard. `writer` is `None` for benchmarks.
pub fn run_writer(
    games: &Receiver<GameRecord>,
    mut writer: Option<ShardWriter>,
    stats: &Stats,
    max_games: Option<u64>,
    stop: &Stop,
) {
    let _guard = PanicGuard { stop: stop.clone(), stage: "writer" };
    loop {
        match games.recv_timeout(Duration::from_millis(100)) {
            Ok(record) => {
                let count = match writer.as_mut() {
                    Some(writer) => match writer.add_record(&record) {
                        Ok(count) => count,
                        Err(error) => {
                            stop.fail("writer", &error);
                            return;
                        }
                    },
                    None => record.positions.len(),
                };
                stats.positions.fetch_add(count as u64, Ordering::Relaxed);
                let games = stats.games.fetch_add(1, Ordering::Relaxed) + 1;
                if max_games.is_some_and(|max| games >= max) {
                    stop.trigger();
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if stop.elapsed().is_some_and(|t| t >= DRAIN_AFTER_STOP) {
            break;
        }
    }
    if let Some(writer) = writer.as_mut()
        && let Err(error) = writer.flush()
    {
        stop.fail("writer", &error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets() {
        assert_eq!(bucket(1), 16);
        assert_eq!(bucket(100), 128);
        assert_eq!(bucket(512), 512);
        assert_eq!(bucket(513), 1024);
        assert_eq!(bucket(8192), 8192);
        assert_eq!(bucket(8000), 8192);
    }
}
