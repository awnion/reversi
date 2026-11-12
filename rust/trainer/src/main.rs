//! `reversi-selfplay`: batched AlphaZero self-play on libtorch (CUDA), a
//! drop-in replacement for the Python self-play processes of
//! `python/train/loop.py` (`REVERSI_SELFPLAY=rust`).
//!
//! ```text
//! reversi-selfplay selfplay --run-dir weights/<run> [options]   # replay shards
//! reversi-selfplay bench --weights <file.bin> [options]         # throughput
//! reversi-selfplay check [--weights <file.bin>] [--allow-cpu]   # libtorch + CUDA
//! reversi-selfplay infer --weights <file.bin> --input <f32> --output <f32>
//! ```

mod control;
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
  infer     --weights FILE --input PLANES.f32 --output OUT.f32
                                (N, 4, 8, 8) float32 in, (N, 65) float32 out: 64 move
                                probabilities and the value per position

self-play options (selfplay, bench):
  --simulations N         MCTS simulations per move (200)
  --workers N             search threads (6)
  --pools-per-worker N    game pools per search thread (2)
  --games-per-pool N      concurrent games per pool (1024)
  --batch-target N        positions per network call (8192)
  --min-batch N           wait up to --max-wait-ms for this many positions (0)
  --max-wait-ms N         (2)
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

const FLAGS: &[&str] = &["fp32", "nchw", "no-stagger", "stagger", "allow-cpu", "help"];

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
        workers: args.num("workers", 6usize)?.max(1),
        pools_per_worker: args.num("pools-per-worker", 2usize)?.max(1),
        games_per_pool: args.num("games-per-pool", 1024usize)?.max(1),
        simulations: args.num("simulations", 200u32)?,
        stagger,
        batch_target: args.num("batch-target", 8192usize)?.max(1),
        min_batch: args.num("min-batch", 0usize)?,
        max_wait: Duration::from_millis(args.num("max-wait-ms", 2u64)?),
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
        simulations = config.simulations,
        batch_target = config.batch_target,
        stagger = config.stagger,
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

    let mut reply_txs = Vec::new();
    for id in 0..config.workers {
        let (reply_tx, reply_rx) = crossbeam_channel::unbounded();
        reply_txs.push(reply_tx);
        let (config, request_tx, games_tx, stop) =
            (config.clone(), request_tx.clone(), games_tx.clone(), stop.clone());
        spawn(
            format!("worker{id}"),
            Box::new(move || {
                pipeline::run_worker(id, &config, &request_tx, &reply_rx, &games_tx, &stop)
            }),
        )?;
    }
    drop((request_tx, games_tx));

    let gpu = GpuThread {
        config: config.clone(),
        options,
        requests: request_rx,
        replies: reply_txs,
        weights: weights_rx,
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
            let temp = temps.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let temp = if thermal_available { format!(" temp={temp}") } else { String::new() };
            // `log_event` field order of the Python self-play processes.
            crate::log::event(
                "selfplay",
                &format!(
                    "progress process={process_index} games={} positions={} batch={} evals_per_s={rate:.0}{temp}",
                    now.games,
                    now.positions,
                    stats.last_batch.load(Ordering::Relaxed),
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
    let net = TorchNet::new(&folded, options);
    let board = reversi_minimax::board::Board::new((1 << 28) | (1 << 35), (1 << 27) | (1 << 36));
    let legal = board.legal_moves(true);
    let packed: Vec<i64> = (0..64).flat_map(|_| net::pack(board, true, legal)).collect();
    let out = to_vec(&net.forward(&net.unpack(&packed)));
    for row in out.as_chunks::<65>().0 {
        let sum: f32 = row[..64].iter().sum();
        if !row.iter().all(|v| v.is_finite()) || (sum - 1.0).abs() > 1e-3 {
            eprintln!("error: bad network output (policy sum {sum})");
            return Ok(ExitCode::FAILURE);
        }
    }
    println!("forward ok: value={:.4}", out[64]);
    Ok(ExitCode::SUCCESS)
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    let mut out = vec![0.0; t.numel()];
    t.copy_data(&mut out, t.numel());
    out
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
