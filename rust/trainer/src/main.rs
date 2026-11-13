//! `reversi-selfplay`: batched AlphaZero self-play on libtorch (CUDA), a
//! drop-in replacement for the Python self-play processes of
//! `python/train/loop.py` (`REVERSI_SELFPLAY=rust`).
//!
//! ```text
//! reversi-selfplay selfplay --run-dir weights/<run> [options]   # replay shards
//! reversi-selfplay bench --weights <file.bin> [options]         # throughput
//! reversi-selfplay check [--weights <file.bin>] [--allow-cpu]   # libtorch + CUDA
//! reversi-selfplay match --a <a.bin> --b <b.bin> [options]      # a against b
//! reversi-selfplay infer --weights <file.bin> --input <f32> --output <f32>
//! ```

mod arena;
mod control;
mod graph;
mod log;
mod net;
mod pipeline;
mod shard;
mod thermal;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use reversi_mcts::Budget;
use tch::Device;
use tch::Tensor;

use crate::control::Stop;
use crate::control::Watcher;
use crate::control::WeightsSource;
use crate::net::FoldedWeights;
use crate::net::NetOptions;
use crate::net::TorchNet;
use crate::pipeline::GpuThread;
use crate::pipeline::PipelineConfig;
use crate::pipeline::Snapshot;
use crate::pipeline::Stats;
use crate::shard::ShardWriter;

const USAGE: &str = "\
usage: reversi-selfplay <command> [options]

commands:
  selfplay  --run-dir DIR       self-play into DIR/replay_spool (weights: DIR/selfplay_latest.bin,
                                else DIR/champion.bin; pauses while DIR/cooldown exists)
  bench     --weights FILE      self-play throughput, games discarded
  check     [--weights FILE]    libtorch loads, CUDA is available, one forward pass works
            [--allow-cpu]       succeed without CUDA
  match     --a FILE --b FILE   pairs of games, a random opening each, each network black
                                once; prints one JSON line: a's points per pair (0 to 2)
            [--pairs N]         (100)
            [--simulations N]   per move (800)
            [--opening-plies N] random moves before the networks take over (4)
            [--seed N]          pair i starts from opening seed + i (0)
            [--workers N]       search threads (6)
            [--leaves N]        leaves per game and round, apart by virtual loss (8)
  infer     --weights FILE --input PLANES.f32 --output OUT.f32
                                (N, 4, 8, 8) float32 in, (N, 65) float32 out: 64 move
                                probabilities and the value per position

self-play options (selfplay, bench):
  --simulations N         MCTS simulations per recorded move (800)
  --fast-simulations N    playout cap randomization: simulations on the moves
                          that are only played, not recorded (= --simulations: off)
  --full-fraction F       share of moves searched with --simulations and recorded (1)
  --workers N             search threads (12)
  --pools-per-worker N    game pools per search thread (2)
  --games-per-pool N      concurrent games per pool (256)
  --batch-target N        positions per network call (8192)
  --min-batch N           wait up to --max-wait-ms for this many positions (0)
  --max-wait-ms N         (2)
  --cache-entries N       network outputs kept in the cache, 0 = off (1000000)
  --no-cuda-graphs        run the network without CUDA graphs
  --no-stagger            start all games at once (default for bench)
  --stagger               spread game starts over one game length (default for selfplay)
  --positions-per-shard N (4096)
  --process-index N       for the producer id rs<N>_pid<pid> and log lines (0)
  --parent-pid PID        exit when this process is no longer the parent
  --max-seconds S         stop after S seconds
  --max-games N           stop after N finished games
  --log-seconds S         progress log interval (30)
  --seconds S             bench: measured seconds after warm-up (60)
  --warmup S              bench: seconds before measuring (20)
  --abort-temp C          bench: stop when the GPU reaches C degrees (78)

network options (all commands):
  --device auto|cuda|cpu  (auto: cuda when available)
  --fp32                  full precision on CUDA, without TF32 (default: half)
  --nchw                  channel-first activations (default on CUDA: channels-last)
";

struct Args {
    command: String,
    values: HashMap<String, String>,
    flags: Vec<String>,
}

const FLAGS: &[&str] =
    &["fp32", "nchw", "no-stagger", "stagger", "allow-cpu", "no-cuda-graphs", "help"];

impl Args {
    fn parse() -> Result<Args, String> {
        let mut raw = std::env::args().skip(1);
        let command = raw.next().ok_or("missing command")?;
        let mut values = HashMap::new();
        let mut flags = Vec::new();
        while let Some(arg) = raw.next() {
            let key =
                arg.strip_prefix("--").ok_or_else(|| format!("unexpected argument {arg:?}"))?;
            if FLAGS.contains(&key) {
                flags.push(key.to_owned());
            } else {
                let value = raw.next().ok_or_else(|| format!("--{key} needs a value"))?;
                values.insert(key.to_owned(), value);
            }
        }
        Ok(Args { command, values, flags })
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    fn string(&mut self, name: &str) -> Option<String> {
        self.values.remove(name)
    }

    fn path(&mut self, name: &str) -> Result<PathBuf, String> {
        self.string(name).map(PathBuf::from).ok_or_else(|| format!("--{name} is required"))
    }

    fn num<T: std::str::FromStr>(&mut self, name: &str, default: T) -> Result<T, String> {
        match self.values.remove(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{name}: invalid value {v:?}")),
        }
    }

    fn opt_num<T: std::str::FromStr>(&mut self, name: &str) -> Result<Option<T>, String> {
        self.values
            .remove(name)
            .map(|v| v.parse().map_err(|_| format!("--{name}: invalid value {v:?}")))
            .transpose()
    }

    /// Error on options nobody asked for (typos).
    fn finish(&self) -> Result<(), String> {
        match self.values.keys().next() {
            Some(key) => Err(format!("unknown option --{key}")),
            None => Ok(()),
        }
    }

    fn net_options(&mut self) -> Result<NetOptions, String> {
        let device = match self.string("device").as_deref().unwrap_or("auto") {
            "auto" => Device::cuda_if_available(),
            "cuda" if tch::Cuda::is_available() => Device::Cuda(0),
            "cuda" => return Err("--device cuda: CUDA is not available".into()),
            "cpu" => Device::Cpu,
            other => return Err(format!("--device: unknown device {other:?}")),
        };
        let cuda = device.is_cuda();
        let fp32 = self.flag("fp32");
        Ok(NetOptions {
            device,
            half: cuda && !fp32,
            channels_last: cuda && !self.flag("nchw"),
            tf32: false,
        })
    }
}

fn main() -> ExitCode {
    let mut args = match Args::parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = match args.command.as_str() {
        "selfplay" => selfplay(&mut args, false),
        "bench" => selfplay(&mut args, true),
        "check" => check(&mut args),
        "infer" => infer(&mut args),
        "match" => play_match(&mut args),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command {other:?}")),
    };
    match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Removes the pid file on exit, if it is still ours.
struct PidFile(PathBuf);

impl PidFile {
    fn create(path: PathBuf) -> Result<Self, String> {
        std::fs::write(&path, format!("{}\n", std::process::id()))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(PidFile(path))
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let ours = std::fs::read_to_string(&self.0)
            .is_ok_and(|s| s.trim() == std::process::id().to_string());
        if ours {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

fn selfplay(args: &mut Args, bench: bool) -> Result<ExitCode, String> {
    let options = args.net_options()?;
    let stagger = if bench { args.flag("stagger") } else { !args.flag("no-stagger") };
    let config = PipelineConfig {
        // 12 × 2 × 256 = 6 144 games: at 800 simulations the search trees of
        // 12k concurrent games took over 12 GB and the kernel killed the
        // process. Two pools per thread: one searches while the other waits
        // for the GPU. `bench` on the laptop: 6 threads with one pool of 1 024
        // 105 games/s; 8 × 2 × 384 174; with CUDA graphs, which make the GPU
        // thread wait for the searches, 182; 12 threads 195.
        workers: args.num("workers", 12usize)?.max(1),
        pools_per_worker: args.num("pools-per-worker", 2usize)?.max(1),
        games_per_pool: args.num("games-per-pool", 256usize)?.max(1),
        budget: {
            let full = args.num("simulations", 800u32)?;
            let fast = args.num("fast-simulations", full)?;
            Budget::capped(full, fast, args.num("full-fraction", 1.0f32)?)
        },
        stagger,
        batch_target: args.num("batch-target", 8192usize)?.max(1),
        min_batch: args.num("min-batch", 0usize)?,
        max_wait: Duration::from_millis(args.num("max-wait-ms", 2u64)?),
        cache_entries: args.num("cache-entries", 1_000_000usize)?,
        cuda_graphs: !args.flag("no-cuda-graphs"),
    };
    let process_index: usize = args.num("process-index", 0)?;
    let positions_per_shard: usize = args.num("positions-per-shard", 4096)?;
    let parent_pid: Option<u32> = args.opt_num("parent-pid")?;
    let max_seconds: Option<f64> = args.opt_num("max-seconds")?;
    let max_games: Option<u64> = args.opt_num("max-games")?;
    let log_every = Duration::from_secs_f64(args.num("log-seconds", 30.0f64)?);
    let bench_seconds: f64 = args.num("seconds", 60.0)?;
    let bench_warmup: f64 = args.num("warmup", 20.0)?;
    let abort_temp: f64 = args.num("abort-temp", 78.0)?;

    let (source, run_dir) = if bench {
        (WeightsSource::File(args.path("weights")?), None)
    } else {
        let run_dir = args.path("run-dir")?;
        (WeightsSource::RunDir(run_dir.clone()), Some(run_dir))
    };
    args.finish()?;

    let stop = Stop::new();
    control::install_signal_handlers(&stop)?;
    if parent_pid.is_some() {
        control::die_with_parent();
    }
    let _pid_file = match &run_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            Some(PidFile::create(dir.join("selfplay-rs.pid"))?)
        }
        None => None,
    };
    let writer = run_dir.as_ref().map(|dir| {
        ShardWriter::new(
            dir.join("replay_spool"),
            format!("rs{process_index}_pid{}", std::process::id()),
            positions_per_shard,
        )
    });

    log_selfplay!(
        "start",
        process = process_index,
        pid = std::process::id(),
        device = format!("{:?}", options.device),
        half = options.half,
        channels_last = options.channels_last,
        workers = config.workers,
        pools = config.workers * config.pools_per_worker,
        games = config.workers * config.pools_per_worker * config.games_per_pool,
        simulations = config.budget.full,
        fast_simulations = config.budget.fast,
        full_fraction = config.budget.full_fraction,
        batch_target = config.batch_target,
        stagger = config.stagger,
        cuda_graphs = config.cuda_graphs && options.device.is_cuda(),
    );

    let stats = Arc::new(Stats::default());
    let paused = Arc::new(AtomicBool::new(false));
    let throttle = Arc::new(AtomicU64::new(0f64.to_bits()));
    let temps = Arc::new(Mutex::new(String::new()));
    let (weights_tx, weights_rx) = crossbeam_channel::unbounded();
    let (request_tx, request_rx) = crossbeam_channel::unbounded();
    let (games_tx, games_rx) = crossbeam_channel::unbounded();

    let spawn = |name: String, f: Box<dyn FnOnce() + Send>| {
        std::thread::Builder::new().name(name).spawn(f).map_err(|e| e.to_string())
    };

    // Temperatures: polled on their own thread (nvidia-smi can take a while).
    let mut governor = thermal::Governor::new();
    let thermal_available = governor.available();
    if thermal_available {
        let (stop, throttle, temps) = (stop.clone(), throttle.clone(), temps.clone());
        spawn(
            "thermal".into(),
            Box::new(move || {
                loop {
                    if governor.poll() {
                        log_selfplay!(
                            "thermal-throttle",
                            temp = governor.temperatures.describe(),
                            limits = governor.describe_limits(),
                            slowdown = format!("{:.0}%", governor.throttle * 100.0),
                        );
                    }
                    throttle.store(governor.throttle.to_bits(), Ordering::Relaxed);
                    *temps.lock().unwrap_or_else(|e| e.into_inner()) =
                        governor.temperatures.describe();
                    if bench && governor.temperatures.gpu.is_some_and(|t| t >= abort_temp) {
                        log_selfplay!("bench-abort", temp = governor.temperatures.describe());
                        stop.fail("thermal", &format!("GPU at or above {abort_temp} C"));
                    }
                    if stop.sleep(thermal::POLL) {
                        return;
                    }
                }
            }),
        )?;
    }

    let watcher = Watcher {
        source,
        cooldown_flag: run_dir.as_ref().map(|d| d.join("cooldown")),
        paused: paused.clone(),
        weights_tx,
        parent_pid,
        stop: stop.clone(),
    };
    spawn("control".into(), Box::new(move || watcher.run()))?;

    let writer_thread = {
        let (stats, stop) = (stats.clone(), stop.clone());
        spawn(
            "writer".into(),
            Box::new(move || pipeline::run_writer(&games_rx, writer, &stats, max_games, &stop)),
        )?
    };

    let cache = Arc::new(pipeline::SharedCache::new(config.cache_entries));
    let mut reply_txs = Vec::new();
    for id in 0..config.workers {
        let (reply_tx, reply_rx) = crossbeam_channel::unbounded();
        reply_txs.push(reply_tx);
        let links = pipeline::WorkerLinks {
            cache: cache.clone(),
            stats: stats.clone(),
            requests: request_tx.clone(),
            replies: reply_rx,
            games: games_tx.clone(),
            stop: stop.clone(),
        };
        let config = config.clone();
        spawn(format!("worker{id}"), Box::new(move || pipeline::run_worker(id, &config, links)))?;
    }
    drop((request_tx, games_tx));

    let gpu = GpuThread {
        config: config.clone(),
        options,
        requests: request_rx,
        replies: reply_txs,
        weights: weights_rx,
        cache,
        paused,
        throttle,
        stats: stats.clone(),
        stop: stop.clone(),
    };
    spawn("gpu".into(), Box::new(move || gpu.run()))?;

    // Progress log, limits and the benchmark window, until stopped.
    let started = Instant::now();
    let mut last = stats.snapshot();
    let mut last_log = Instant::now();
    let mut window: Option<Snapshot> = None;
    let mut window_end: Option<Snapshot> = None;
    while !stop.sleep(Duration::from_millis(200)) {
        let elapsed = started.elapsed().as_secs_f64();
        if max_seconds.is_some_and(|max| elapsed >= max) {
            stop.trigger();
        }
        if bench {
            if window.is_none() && elapsed >= bench_warmup {
                window = Some(stats.snapshot());
                log_selfplay!("bench-warmup-done", seconds = format!("{elapsed:.0}"));
            }
            if window.is_some() && elapsed >= bench_warmup + bench_seconds {
                window_end = Some(stats.snapshot());
                stop.trigger();
            }
        }
        if last_log.elapsed() >= log_every {
            let now = stats.snapshot();
            let seconds = now.at.zip(last.at).map_or(1.0, |(a, b)| (a - b).as_secs_f64());
            let rate = (now.evals - last.evals) as f64 / seconds.max(1e-9);
            let asked = (now.evals - last.evals) + (now.cache_hits - last.cache_hits);
            let hits = (now.cache_hits - last.cache_hits) as f64 / asked.max(1) as f64;
            let temp = temps.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let temp = if thermal_available { format!(" temp={temp}") } else { String::new() };
            // `log_event` field order of the Python self-play processes.
            crate::log::event(
                "selfplay",
                &format!(
                    "progress process={process_index} games={} positions={} batch={} evals_per_s={rate:.0} cache_hits={:.0}%{temp}",
                    now.games,
                    now.positions,
                    stats.last_batch.load(Ordering::Relaxed),
                    hits * 100.0,
                ),
                &[],
            );
            last = now;
            last_log = Instant::now();
        }
    }

    // Stopped: the writer drains and flushes (bounded), everything else is
    // simply left behind by the exit.
    let _ = writer_thread.join();

    if bench {
        let (Some(a), Some(b)) = (window, window_end) else {
            log_selfplay!(
                "bench-incomplete",
                seconds = format!("{:.0}", started.elapsed().as_secs_f64())
            );
            return Ok(ExitCode::FAILURE);
        };
        let seconds = b.at.zip(a.at).map_or(1.0, |(b, a)| (b - a).as_secs_f64());
        let batches = (b.batches - a.batches).max(1);
        let temp = temps.lock().unwrap_or_else(|e| e.into_inner()).clone();
        log_selfplay!(
            "bench",
            seconds = format!("{seconds:.1}"),
            evals_per_s = format!("{:.0}", (b.evals - a.evals) as f64 / seconds),
            mean_batch = format!("{:.0}", (b.evals - a.evals) as f64 / batches as f64),
            mean_padded =
                format!("{:.0}", (b.padded_evals - a.padded_evals) as f64 / batches as f64),
            batches_per_s = format!("{:.1}", batches as f64 / seconds),
            gpu_busy = format!("{:.0}%", (b.gpu_ns - a.gpu_ns) as f64 / 1e9 / seconds * 100.0),
            games_per_s = format!("{:.1}", (b.games - a.games) as f64 / seconds),
            positions_per_s = format!("{:.0}", (b.positions - a.positions) as f64 / seconds),
            evals_per_position = format!(
                "{:.1}",
                (b.evals - a.evals) as f64 / (b.positions - a.positions).max(1) as f64
            ),
            cache_hits = format!(
                "{:.0}%",
                (b.cache_hits - a.cache_hits) as f64 * 100.0
                    / ((b.evals - a.evals) + (b.cache_hits - a.cache_hits)).max(1) as f64
            ),
            temp = temp,
        );
    }
    let total = stats.snapshot();
    log_selfplay!(
        "exit",
        process = process_index,
        games = total.games,
        positions = total.positions,
        failed = stop.failed(),
    );
    Ok(if stop.failed() { ExitCode::FAILURE } else { ExitCode::SUCCESS })
}

/// libtorch loads, CUDA is there (unless `--allow-cpu`), and the network
/// gives finite probabilities on a small batch.
fn check(args: &mut Args) -> Result<ExitCode, String> {
    let allow_cpu = args.flag("allow-cpu");
    let weights = args.string("weights").map(PathBuf::from);
    let options = args.net_options()?;
    args.finish()?;
    let cuda = tch::Cuda::is_available();
    println!(
        "libtorch ok: cuda={cuda} devices={} cudnn={} device={:?} half={} channels_last={}",
        tch::Cuda::device_count(),
        tch::Cuda::cudnn_is_available(),
        options.device,
        options.half,
        options.channels_last,
    );
    if !cuda && !allow_cpu {
        eprintln!("error: CUDA is not available");
        return Ok(ExitCode::from(3));
    }
    let bytes = match &weights {
        Some(path) => std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
        None => net::random_bin(1),
    };
    let folded = FoldedWeights::from_bin(&bytes)?;
    let _guard = tch::no_grad_guard();
    let mut net = TorchNet::new(&folded, options);
    // 64 different positions: random openings of 0 to 31 plies.
    let packed: Vec<i64> = (0..64u64)
        .flat_map(|i| {
            let (board, mut is_black) = reversi_mcts::random_opening((i % 32) as u32, i);
            let mut legal = board.legal_moves(is_black);
            if legal == 0 {
                is_black = !is_black;
                legal = board.legal_moves(is_black);
            }
            net::pack(board, is_black, legal)
        })
        .collect();
    let out = to_vec(&net.forward(&net.unpack(&packed)));
    for row in out.as_chunks::<65>().0 {
        let sum: f32 = row[..64].iter().sum();
        if !row.iter().all(|v| v.is_finite()) || (sum - 1.0).abs() > 1e-3 {
            eprintln!("error: bad network output (policy sum {sum})");
            return Ok(ExitCode::FAILURE);
        }
    }
    println!("forward ok: value={:.4}", out[64]);
    if options.device.is_cuda() {
        // The same positions through a CUDA graph: same outputs.
        let mut graphed = graph::GraphedForward::new(true);
        let mut via_graph = vec![0.0f32; out.len()];
        for _ in 0..2 {
            graphed.run(&net, &packed, &mut via_graph);
        }
        let diff = out.iter().zip(&via_graph).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        if diff > 1e-3 {
            eprintln!("error: CUDA graph output differs by {diff}");
            return Ok(ExitCode::FAILURE);
        }
        // New weights copied into the captured tensors: the graph follows.
        let mut other = TorchNet::new(&FoldedWeights::from_bin(&net::random_bin(2))?, options);
        let expected = to_vec(&other.forward(&other.unpack(&packed)));
        if !net.copy_weights_from(&mut other) {
            eprintln!("error: weights of the same network do not fit");
            return Ok(ExitCode::FAILURE);
        }
        graphed.run(&net, &packed, &mut via_graph);
        let after = expected.iter().zip(&via_graph).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        if after > 1e-3 {
            eprintln!("error: CUDA graph kept the old weights (difference {after})");
            return Ok(ExitCode::FAILURE);
        }
        println!("cuda graph ok: max difference {diff:.1e}, after new weights {after:.1e}");
    }
    Ok(ExitCode::SUCCESS)
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    let mut out = vec![0.0; t.numel()];
    t.copy_data(&mut out, t.numel());
    out
}

/// Two networks play pairs of games; one JSON line with a's points per pair.
fn play_match(args: &mut Args) -> Result<ExitCode, String> {
    let options = args.net_options()?;
    let mut load = |name: &str| -> Result<FoldedWeights, String> {
        let path = args.path(name)?;
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        FoldedWeights::from_bin(&bytes)
    };
    let (a, b) = (load("a")?, load("b")?);
    let config = arena::MatchConfig {
        pairs: args.num("pairs", 100usize)?,
        simulations: args.num("simulations", 800u32)?,
        opening_plies: args.num("opening-plies", 4u32)?,
        seed: args.num("seed", 0u64)?,
        workers: args.num("workers", 6usize)?,
        leaves_per_round: args.num("leaves", 8u32)?,
    };
    args.finish()?;
    let started = Instant::now();
    let points = arena::run_match(&a, &b, options, &config);
    let points: Vec<String> = points.iter().map(|p| p.to_string()).collect();
    println!(
        "{{\"points\": [{}], \"seconds\": {:.1}}}",
        points.join(", "),
        started.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

/// Planes file in, policies and values out (for parity tests).
fn infer(args: &mut Args) -> Result<ExitCode, String> {
    let options = args.net_options()?;
    let weights = args.path("weights")?;
    let input = args.path("input")?;
    let output = args.path("output")?;
    let batch: usize = args.num("batch", 4096)?;
    args.finish()?;
    let bytes = std::fs::read(&weights).map_err(|e| format!("{}: {e}", weights.display()))?;
    let net_weights = FoldedWeights::from_bin(&bytes)?;
    let raw = std::fs::read(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    let (chunks, rest) = raw.as_chunks::<4>();
    if !rest.is_empty() || !chunks.len().is_multiple_of(reversi_mcts::PLANES_LEN) {
        return Err("input is not a whole number of (4, 8, 8) float32 planes".into());
    }
    let planes: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let _guard = tch::no_grad_guard();
    let net = TorchNet::new(&net_weights, options);
    let mut out = Vec::with_capacity(planes.len() / reversi_mcts::PLANES_LEN * 65);
    for chunk in planes.chunks(batch.max(1) * reversi_mcts::PLANES_LEN) {
        out.extend(to_vec(&net.forward(&net.planes(chunk))));
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&output, bytes).map_err(|e| format!("{}: {e}", output.display()))?;
    Ok(ExitCode::SUCCESS)
}
