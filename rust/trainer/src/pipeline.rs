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
use reversi_mcts::Budget;
use reversi_mcts::GameRecord;
use reversi_mcts::SelfPlayPool;

use crate::control::LoadedWeights;
use crate::control::PanicGuard;
use crate::control::Stop;
use crate::control::Tuning;
use crate::graph::GraphedForward;
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
    /// Simulations per move: `full` on a random `full_fraction` of the moves
    /// (recorded), `fast` on the rest (only played).
    pub budget: Budget,
    /// Spread game starts over one game length (steady output, but small
    /// batches for the first ~60 × simulations rounds).
    pub stagger: bool,
    /// Positions per network call; requests are never split.
    pub batch_target: usize,
    /// Wait up to `max_wait` for at least this many positions.
    pub min_batch: usize,
    pub max_wait: Duration,
    /// Network outputs the [`SharedCache`] keeps (twice that at most; 0 = off).
    pub cache_entries: usize,
    /// Run the network through CUDA graphs ([`GraphedForward`]).
    pub cuda_graphs: bool,
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
    /// Positions answered from the [`SharedCache`] or by a duplicate in the
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

/// One pool's positions to evaluate, packed as in [`pack`]: only the ones
/// the cache did not know, each once.
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

/// Network outputs for one [`Request`], 65 floats per position (64 priors,
/// the value), in request order.
pub struct Reply {
    pool: usize,
    rows: Vec<f32>,
    /// The [`SharedCache`] generation of the weights that computed them.
    generation: u64,
    /// The request's buffer, handed back for reuse.
    packed: Vec<i64>,
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

const CACHE_SHARDS: usize = 64;

/// One lock's worth of the cache. Two generations bound the memory: when the
/// current map is full it becomes the previous one and the older entries go.
struct CacheShard {
    generation: u64,
    current: KeyMap<Row>,
    previous: KeyMap<Row>,
}

impl CacheShard {
    /// Forget everything computed by older weights.
    fn catch_up(&mut self, generation: u64) {
        if generation > self.generation {
            self.current.clear();
            self.previous.clear();
            self.generation = generation;
        }
    }
}

/// Network outputs for positions seen recently, shared by the workers. The
/// same positions come up again and again: openings across games and
/// transpositions inside a search. The workers look positions up before
/// asking the GPU, so the GPU thread only runs the network.
///
/// Every weights update starts a new generation; answers of older weights
/// are neither returned nor stored.
pub struct SharedCache {
    shards: Vec<std::sync::Mutex<CacheShard>>,
    capacity_per_shard: usize,
    generation: AtomicU64,
}

impl SharedCache {
    pub fn new(entries: usize) -> Self {
        let shard = || {
            std::sync::Mutex::new(CacheShard {
                generation: 0,
                current: KeyMap::default(),
                previous: KeyMap::default(),
            })
        };
        SharedCache {
            shards: (0..CACHE_SHARDS).map(|_| shard()).collect(),
            capacity_per_shard: entries / CACHE_SHARDS,
            generation: AtomicU64::new(0),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// New weights: everything cached so far is stale. Returns the new
    /// generation.
    pub fn invalidate(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn shard(&self, key: &CacheKey) -> std::sync::MutexGuard<'_, CacheShard> {
        let mixed =
            (key.0 as u64 ^ (key.1 as u64).rotate_left(23)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let index = (mixed >> 58) as usize % CACHE_SHARDS;
        self.shards[index].lock().unwrap_or_else(|e| e.into_inner())
    }

    fn get(&self, key: &CacheKey, generation: u64) -> Option<Row> {
        if self.capacity_per_shard == 0 {
            return None;
        }
        let mut shard = self.shard(key);
        shard.catch_up(generation);
        if shard.generation != generation {
            return None;
        }
        shard.current.get(key).or_else(|| shard.previous.get(key)).copied()
    }

    fn insert(&self, key: CacheKey, row: Row, generation: u64) {
        if self.capacity_per_shard == 0 || generation != self.generation() {
            return;
        }
        let mut shard = self.shard(&key);
        shard.catch_up(generation);
        if shard.generation != generation {
            return;
        }
        if shard.current.len() >= self.capacity_per_shard {
            shard.previous = std::mem::take(&mut shard.current);
        }
        shard.current.insert(key, row);
    }
}

// ── Workers ───────────────────────────────────────────────────────────────

/// A pool's batch while its request is with the GPU.
#[derive(Default)]
struct Pending {
    /// 65 floats per position of the batch; cache hits already filled in.
    rows: Vec<f32>,
    /// `(position, slot)`: position of the batch answered by `slot` of the reply.
    misses: Vec<(u32, u32)>,
    /// Key of each slot of the request.
    keys: Vec<CacheKey>,
}

/// What a worker shares with the rest of the pipeline.
pub struct WorkerLinks {
    pub cache: Arc<SharedCache>,
    pub stats: Arc<Stats>,
    pub requests: Sender<Request>,
    pub replies: Receiver<Reply>,
    pub games: Sender<GameRecord>,
    pub tuning: Arc<Tuning>,
    pub stop: Stop,
}

/// A worker's pools and buffers.
struct Worker {
    id: usize,
    links: WorkerLinks,
    pools: Vec<SelfPlayPool>,
    pending: Vec<Pending>,
    spare: Vec<Vec<i64>>,
    slots: KeyMap<u32>,
    policies: Vec<f32>,
    values: Vec<f32>,
    /// Pools of all workers, and the games target last applied.
    all_pools: usize,
    applied_games: u64,
}

impl Worker {
    /// This worker's share of the games target, spread over its pools.
    fn apply_games_target(&mut self) {
        let target = self.links.tuning.games.load(Ordering::Relaxed);
        if target == self.applied_games {
            return;
        }
        self.applied_games = target;
        let (target, all) = (target as usize, self.all_pools);
        let first = self.id * self.pools.len();
        for (i, pool) in self.pools.iter_mut().enumerate() {
            let index = first + i;
            pool.set_games(target / all + usize::from(index < target % all));
        }
    }

    /// Apply a pool's completed batch (`pending[pool].rows`) and hand its
    /// finished games on.
    fn finish(&mut self, pool: usize) -> bool {
        self.policies.clear();
        self.values.clear();
        for row in self.pending[pool].rows.as_chunks::<65>().0 {
            self.policies.extend_from_slice(&row[..64]);
            self.values.push(row[64]);
        }
        let pool = &mut self.pools[pool];
        pool.apply(&self.policies, &self.values);
        pool.take_finished().into_iter().all(|record| self.links.games.send(record).is_ok())
    }

    /// Collect the pool's next batch, answer what the cache knows and send
    /// the rest to the GPU, each position once. A batch the cache answers
    /// completely is applied right away and the next one collected.
    fn submit(&mut self, pool: usize) -> bool {
        loop {
            if self.links.stop.is_set() {
                return false;
            }
            self.apply_games_target();
            let cache = &self.links.cache;
            let generation = cache.generation();
            let state = &mut self.pending[pool];
            let mut packed = self.spare.pop().unwrap_or_default();
            packed.clear();
            state.misses.clear();
            state.keys.clear();
            self.slots.clear();
            let batch = self.pools[pool].collect();
            state.rows.clear();
            state.rows.resize(batch.len() * 65, 0.0);
            let mut hits = 0u64;
            for (i, request) in batch.iter().enumerate() {
                let position = pack(request.board, request.is_black, request.legal);
                let key = (position[0], position[1]);
                if let Some(row) = cache.get(&key, generation) {
                    state.rows[i * 65..(i + 1) * 65].copy_from_slice(&row);
                    hits += 1;
                    continue;
                }
                let next = state.keys.len() as u32;
                let slot = *self.slots.entry(key).or_insert(next);
                if slot == next {
                    packed.extend_from_slice(&position);
                    state.keys.push(key);
                } else {
                    hits += 1;
                }
                state.misses.push((i as u32, slot));
            }
            self.links.stats.cache_hits.fetch_add(hits, Ordering::Relaxed);
            if !packed.is_empty() {
                let request = Request { worker: self.id, pool, packed };
                return self.links.requests.send(request).is_ok();
            }
            self.spare.push(packed);
            if !self.finish(pool) {
                return false;
            }
        }
    }

    /// The GPU's answer for a pool: fill in the rows, remember them, go on.
    fn receive(&mut self, reply: Reply) -> bool {
        let state = &mut self.pending[reply.pool];
        for &(position, slot) in &state.misses {
            let (position, slot) = (position as usize * 65, slot as usize * 65);
            state.rows[position..position + 65].copy_from_slice(&reply.rows[slot..slot + 65]);
        }
        for (slot, &key) in state.keys.iter().enumerate() {
            let mut row = [0.0; 65];
            row.copy_from_slice(&reply.rows[slot * 65..(slot + 1) * 65]);
            self.links.cache.insert(key, row, reply.generation);
        }
        self.spare.push(reply.packed);
        self.finish(reply.pool) && self.submit(reply.pool)
    }
}

pub fn run_worker(id: usize, config: &PipelineConfig, links: WorkerLinks) {
    let _guard = PanicGuard { stop: links.stop.clone(), stage: "worker" };
    let new_pool = || {
        if config.stagger {
            SelfPlayPool::staggered(config.games_per_pool, config.budget)
        } else {
            SelfPlayPool::new(config.games_per_pool, config.budget)
        }
    };
    let pools: Vec<SelfPlayPool> = (0..config.pools_per_worker).map(|_| new_pool()).collect();
    let mut worker = Worker {
        id,
        pending: (0..pools.len()).map(|_| Pending::default()).collect(),
        pools,
        links,
        spare: Vec::new(),
        slots: KeyMap::default(),
        policies: Vec::new(),
        values: Vec::new(),
        all_pools: config.workers * config.pools_per_worker,
        applied_games: (config.workers * config.pools_per_worker * config.games_per_pool) as u64,
    };
    for pool in 0..worker.pools.len() {
        if !worker.submit(pool) {
            return;
        }
    }
    let (replies, stop) = (worker.links.replies.clone(), worker.links.stop.clone());
    loop {
        select! {
            recv(replies) -> reply => {
                let Ok(reply) = reply else { return };
                if !worker.receive(reply) {
                    return;
                }
            }
            recv(stop.receiver()) -> _ => return,
        }
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
    pub cache: Arc<SharedCache>,
    pub paused: Arc<AtomicBool>,
    /// Thermal throttle (f64 bits), see `thermal::Governor`.
    pub throttle: Arc<AtomicU64>,
    /// The learner's share of pauses ([`Tuning::duty`]); the larger of the
    /// two applies.
    pub tuning: Arc<Tuning>,
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

        let (mut net, mut generation) = select! {
            recv(self.weights) -> loaded => {
                let Ok(loaded) = loaded else { return };
                let net = TorchNet::new(&loaded.weights, self.options);
                log_selfplay!("weights-loaded", process = 0, source = loaded.source);
                (net, self.cache.invalidate())
            }
            recv(self.stop.receiver()) -> _ => return,
        };

        let config = &self.config;
        let mut graphed = GraphedForward::new(config.cuda_graphs && self.options.device.is_cuda());
        let mut carry: Option<Request> = None;
        let mut batch: Vec<Request> = Vec::new();
        let mut packed: Vec<i64> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        let mut pacer = Pacer::default();
        let mut paused_since: Option<Instant> = None;

        while !self.stop.is_set() {
            // Newest weights first, uploaded between batches.
            let mut newest = None;
            while let Ok(loaded) = self.weights.try_recv() {
                newest = Some(loaded);
            }
            if let Some(loaded) = newest {
                // Into the same tensors, so that the CUDA graphs stay valid.
                let mut fresh = TorchNet::new(&loaded.weights, self.options);
                if !net.copy_weights_from(&mut fresh) {
                    net = fresh;
                    graphed.reset();
                }
                generation = self.cache.invalidate();
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

            // Evaluate.
            let padded = bucket(total);
            packed.clear();
            for request in &batch {
                packed.extend_from_slice(&request.packed);
            }
            packed.resize(padded * 4, 0);
            let started = Instant::now();
            out.resize(padded * 65, 0.0);
            // Blocks until the batch has completed on the device.
            graphed.run(&net, &packed, &mut out);
            let work = started.elapsed();

            // Hand every pool its rows.
            let mut offset = 0;
            for request in batch.drain(..) {
                let n = request.len();
                let rows = out[offset * 65..(offset + n) * 65].to_vec();
                offset += n;
                // A worker that is gone is shutting down; nothing to do.
                let _ = self.replies[request.worker].send(Reply {
                    pool: request.pool,
                    rows,
                    generation,
                    packed: request.packed,
                });
            }

            let stats = &self.stats;
            stats.evals.fetch_add(total as u64, Ordering::Relaxed);
            stats.padded_evals.fetch_add(padded as u64, Ordering::Relaxed);
            stats.batches.fetch_add(1, Ordering::Relaxed);
            stats.last_batch.store(total as u64, Ordering::Relaxed);
            stats.gpu_ns.fetch_add(work.as_nanos() as u64, Ordering::Relaxed);

            // Slow down while the machine is too hot, in proportion to the
            // completed work.
            let throttle = f64::from_bits(self.throttle.load(Ordering::Relaxed));
            let throttle = throttle.max(self.tuning.duty());
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
