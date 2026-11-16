//! Process control: the stop signal every stage waits on, signal handling,
//! parent supervision, weight reloads, the learner's cooldown flag and its
//! control file (games in play, share of time paused).

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;

use crate::log_selfplay;
use crate::net::FoldedWeights;

/// How often the weights file is checked for a new version (as in Python).
pub const WEIGHT_REFRESH: Duration = Duration::from_secs(5);

/// A stop request shared by all threads. Waiting on [`Stop::receiver`] in a
/// `select!` wakes up as soon as the stop is triggered: the channel never
/// carries a message, it is disconnected.
#[derive(Clone)]
pub struct Stop {
    inner: Arc<StopInner>,
}

struct StopInner {
    stopped: AtomicBool,
    failed: AtomicBool,
    sender: Mutex<Option<Sender<()>>>,
    receiver: Receiver<()>,
    since: Mutex<Option<Instant>>,
}

impl Stop {
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam_channel::bounded(0);
        Stop {
            inner: Arc::new(StopInner {
                stopped: AtomicBool::new(false),
                failed: AtomicBool::new(false),
                sender: Mutex::new(Some(sender)),
                receiver,
                since: Mutex::new(None),
            }),
        }
    }

    pub fn trigger(&self) {
        let mut since = self.inner.since.lock().unwrap_or_else(|e| e.into_inner());
        since.get_or_insert_with(Instant::now);
        self.inner.stopped.store(true, Ordering::SeqCst);
        drop(self.inner.sender.lock().unwrap_or_else(|e| e.into_inner()).take());
    }

    /// Stop because of an error; the process exits with a failure code.
    pub fn fail(&self, what: &str, error: &str) {
        log_selfplay!("fatal", process = 0, stage = what, error = format!("{error:?}"));
        self.inner.failed.store(true, Ordering::SeqCst);
        self.trigger();
    }

    pub fn is_set(&self) -> bool {
        self.inner.stopped.load(Ordering::SeqCst)
    }

    pub fn failed(&self) -> bool {
        self.inner.failed.load(Ordering::SeqCst)
    }

    pub fn receiver(&self) -> &Receiver<()> {
        &self.inner.receiver
    }

    /// Time since the stop was triggered.
    pub fn elapsed(&self) -> Option<Duration> {
        self.inner.since.lock().unwrap_or_else(|e| e.into_inner()).map(|t| t.elapsed())
    }

    /// Sleep for `duration` unless stopped first; true when stopped.
    pub fn sleep(&self, duration: Duration) -> bool {
        match self.inner.receiver.recv_timeout(duration) {
            Err(RecvTimeoutError::Timeout) => self.is_set(),
            _ => true,
        }
    }
}

/// Fails the process when the owning thread panics, so that a dead stage
/// cannot leave the others (and the learner) waiting forever.
pub struct PanicGuard {
    pub stop: Stop,
    pub stage: &'static str,
}

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.stop.fail(self.stage, "panicked");
        }
    }
}

/// SIGTERM/SIGINT trigger the stop; a second one exits at once.
pub fn install_signal_handlers(stop: &Stop) -> Result<(), String> {
    use signal_hook::consts::SIGINT;
    use signal_hook::consts::SIGTERM;
    let mut signals =
        signal_hook::iterator::Signals::new([SIGTERM, SIGINT]).map_err(|e| e.to_string())?;
    let stop = stop.clone();
    std::thread::Builder::new()
        .name("signals".into())
        .spawn(move || {
            for signum in signals.forever() {
                log_selfplay!("signal", process = 0, signum = signum);
                if stop.is_set() {
                    std::process::exit(128 + signum);
                }
                stop.trigger();
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Ask the kernel to send SIGTERM when the parent dies (Linux).
pub fn die_with_parent() {
    #[cfg(target_os = "linux")]
    // SAFETY: plain prctl call with integer arguments.
    unsafe {
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
    }
}

pub fn parent_alive(parent_pid: u32) -> bool {
    // SAFETY: getppid has no preconditions.
    let ppid = unsafe { libc::getppid() };
    ppid as u32 == parent_pid
}

/// New weights, parsed and folded on the CPU, for the GPU thread to upload.
pub struct LoadedWeights {
    pub weights: FoldedWeights,
    pub source: String,
}

/// The weights self-play should use: `selfplay_latest.bin`, else
/// `champion.bin` (as `_resolve_selfplay_source` in Python, minus `.npz`
/// checkpoints: the learner publishes `selfplay_latest.bin` at startup).
pub fn resolve_weights(run_dir: &Path) -> Option<PathBuf> {
    ["selfplay_latest.bin", "champion.bin"].iter().map(|n| run_dir.join(n)).find(|p| p.is_file())
}

fn mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// What the learner sets for this process through its control file.
pub struct Tuning {
    /// Games in play, over all pools.
    pub games: AtomicU64,
    /// Share of the time the GPU thread sleeps (f64 bits), so that the
    /// learner on the same GPU keeps up.
    pub duty: AtomicU64,
}

impl Tuning {
    pub fn new(games: u64) -> Self {
        Tuning { games: AtomicU64::new(games), duty: AtomicU64::new(0f64.to_bits()) }
    }

    pub fn duty(&self) -> f64 {
        f64::from_bits(self.duty.load(Ordering::Relaxed))
    }

    /// Apply `games=N duty=F` (either may be missing).
    fn apply(&self, text: &str) {
        for field in text.split_whitespace() {
            match field.split_once('=') {
                Some(("games", v)) => {
                    if let Ok(games) = v.parse::<u64>() {
                        self.games.store(games.max(1), Ordering::Relaxed);
                    }
                }
                Some(("duty", v)) => {
                    if let Ok(duty) = v.parse::<f64>() {
                        self.duty.store(duty.clamp(0.0, 0.95).to_bits(), Ordering::Relaxed);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Watches the weights file, the cooldown flag and the control file.
pub struct Watcher {
    /// A run directory to poll, or one fixed weights file.
    pub source: WeightsSource,
    pub cooldown_flag: Option<PathBuf>,
    pub paused: Arc<AtomicBool>,
    pub control_file: Option<PathBuf>,
    pub tuning: Arc<Tuning>,
    pub weights_tx: Sender<LoadedWeights>,
    pub parent_pid: Option<u32>,
    pub stop: Stop,
}

pub enum WeightsSource {
    RunDir(PathBuf),
    File(PathBuf),
}

impl Watcher {
    /// Runs until stopped. Checks for weights every 0.5 s until the first
    /// ones arrive, then every [`WEIGHT_REFRESH`].
    pub fn run(self) {
        let _guard = PanicGuard { stop: self.stop.clone(), stage: "control" };
        let mut loaded: Option<(PathBuf, Option<SystemTime>)> = None;
        let mut last_check: Option<Instant> = None;
        let mut waiting_logged = false;
        let mut failed_logged: Option<(PathBuf, Option<SystemTime>)> = None;
        let mut control_seen: Option<SystemTime> = None;
        loop {
            if let Some(file) = &self.control_file
                && let Some(modified) = mtime(file)
                && control_seen != Some(modified)
                && let Ok(text) = std::fs::read_to_string(file)
            {
                control_seen = Some(modified);
                self.tuning.apply(&text);
            }
            if let Some(parent) = self.parent_pid
                && !parent_alive(parent)
            {
                log_selfplay!("parent-exited", process = 0, parent_pid = parent);
                self.stop.trigger();
                return;
            }
            if let Some(flag) = &self.cooldown_flag {
                self.paused.store(flag.exists(), Ordering::SeqCst);
            }
            let due = match last_check {
                None => true,
                Some(t) => loaded.is_none() || t.elapsed() >= WEIGHT_REFRESH,
            };
            if due {
                last_check = Some(Instant::now());
                let path = match &self.source {
                    WeightsSource::RunDir(dir) => resolve_weights(dir),
                    WeightsSource::File(file) => Some(file.clone()).filter(|f| f.is_file()),
                };
                match path {
                    None if !waiting_logged => {
                        log_selfplay!("waiting-for-weights", process = 0);
                        waiting_logged = true;
                    }
                    None => {}
                    Some(path) => {
                        let version = (path.clone(), mtime(&path));
                        if loaded.as_ref() != Some(&version)
                            && failed_logged.as_ref() != Some(&version)
                        {
                            match std::fs::read(&path)
                                .map_err(|e| e.to_string())
                                .and_then(|b| FoldedWeights::from_bin(&b))
                            {
                                Ok(weights) => {
                                    let source = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned())
                                        .unwrap_or_default();
                                    if self
                                        .weights_tx
                                        .send(LoadedWeights { weights, source })
                                        .is_err()
                                    {
                                        return;
                                    }
                                    loaded = Some(version);
                                }
                                Err(error) => {
                                    log_selfplay!(
                                        "weights-load-failed",
                                        process = 0,
                                        source = path.display(),
                                        error = format!("{error:?}"),
                                    );
                                    failed_logged = Some(version);
                                }
                            }
                        }
                    }
                }
            }
            if self.stop.sleep(Duration::from_millis(500)) {
                return;
            }
        }
    }
}
