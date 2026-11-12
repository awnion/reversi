//! Temperature-aware pacing, a port of `python/train/thermal.py`.
//!
//! The same sensors (hwmon `k10temp`/`zenpower`/`coretemp`/`cpu_thermal` for
//! the CPU, `nvidia-smi` for the GPU), limits (GPU 70/74 °C, CPU 85/90 °C,
//! overridable with `REVERSI_{GPU,CPU}_TEMP_{SOFT,HARD}`) and throttle steps.
//! The throttle is the share of wall time to spend sleeping; the GPU thread
//! applies it to the time its batches took to *complete* (see
//! [`Pacer::pause_after`]).

use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

pub const POLL: Duration = Duration::from_secs(5);
const THROTTLE_STEP_UP: f64 = 0.15;
const THROTTLE_STEP_DOWN: f64 = 0.05;
const MAX_THROTTLE: f64 = 0.9;
const MIN_SLEEP_SECONDS: f64 = 0.02;
const CPU_HWMON_NAMES: [&str; 4] = ["k10temp", "zenpower", "coretemp", "cpu_thermal"];

fn limit(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Temperatures {
    pub cpu: Option<f64>,
    pub gpu: Option<f64>,
}

impl Temperatures {
    pub fn is_empty(&self) -> bool {
        self.cpu.is_none() && self.gpu.is_none()
    }

    /// `cpu82/gpu71`, or `n/a`.
    pub fn describe(&self) -> String {
        let parts: Vec<String> = [("cpu", self.cpu), ("gpu", self.gpu)]
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| format!("{k}{v:.0}")))
            .collect();
        if parts.is_empty() { "n/a".to_owned() } else { parts.join("/") }
    }
}

pub struct Governor {
    /// (soft, hard) per sensor.
    pub cpu_limits: (f64, f64),
    pub gpu_limits: (f64, f64),
    pub throttle: f64,
    pub temperatures: Temperatures,
    cpu_dirs: Vec<PathBuf>,
    has_gpu: bool,
}

fn cpu_hwmon_dirs() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            std::fs::read_to_string(p.join("name"))
                .is_ok_and(|n| CPU_HWMON_NAMES.contains(&n.trim()))
        })
        .collect();
    dirs.sort();
    dirs
}

fn read_cpu(dirs: &[PathBuf]) -> Option<f64> {
    let mut best: Option<f64> = None;
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !(name.starts_with("temp") && name.ends_with("_input")) {
                continue;
            }
            if let Some(t) = std::fs::read_to_string(entry.path())
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
            {
                let t = t / 1000.0;
                best = Some(best.map_or(t, |b| b.max(t)));
            }
        }
    }
    best
}

/// `nvidia-smi` on the PATH.
fn has_nvidia_smi() -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| dir.join("nvidia-smi").is_file())
    })
}

/// Hottest GPU according to `nvidia-smi`, with a 3 s timeout.
pub fn read_gpu() -> Option<f64> {
    let mut child = Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu", "--format=csv,noheader"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let mut out = String::new();
    std::io::Read::read_to_string(&mut child.stdout.take()?, &mut out).ok()?;
    out.split_whitespace().filter_map(|s| s.parse::<f64>().ok()).reduce(f64::max)
}

impl Governor {
    pub fn new() -> Self {
        let cpu_dirs = cpu_hwmon_dirs();
        Governor {
            cpu_limits: (
                limit("REVERSI_CPU_TEMP_SOFT", 85.0),
                limit("REVERSI_CPU_TEMP_HARD", 90.0),
            ),
            gpu_limits: (
                limit("REVERSI_GPU_TEMP_SOFT", 70.0),
                limit("REVERSI_GPU_TEMP_HARD", 74.0),
            ),
            throttle: 0.0,
            temperatures: Temperatures::default(),
            cpu_dirs,
            has_gpu: has_nvidia_smi(),
        }
    }

    pub fn available(&self) -> bool {
        !self.cpu_dirs.is_empty() || self.has_gpu
    }

    pub fn read(&mut self) -> Temperatures {
        let cpu = if self.cpu_dirs.is_empty() { None } else { read_cpu(&self.cpu_dirs) };
        let gpu = if self.has_gpu { read_gpu() } else { None };
        self.temperatures = Temperatures { cpu, gpu };
        self.temperatures
    }

    fn sensors(&self) -> impl Iterator<Item = (f64, (f64, f64))> + '_ {
        [(self.temperatures.cpu, self.cpu_limits), (self.temperatures.gpu, self.gpu_limits)]
            .into_iter()
            .filter_map(|(t, l)| t.map(|t| (t, l)))
    }

    /// `cpu85-90/gpu70-74`.
    pub fn describe_limits(&self) -> String {
        format!(
            "cpu{:.0}-{:.0}/gpu{:.0}-{:.0}",
            self.cpu_limits.0, self.cpu_limits.1, self.gpu_limits.0, self.gpu_limits.1
        )
    }

    /// Read the sensors and update the throttle (call every [`POLL`]).
    /// Returns true when the throttle changed.
    pub fn poll(&mut self) -> bool {
        if !self.available() {
            return false;
        }
        let previous = self.throttle;
        if self.read().is_empty() {
            self.throttle = 0.0;
        } else if self.sensors().any(|(t, (_, hard))| t >= hard) {
            self.throttle = (self.throttle + THROTTLE_STEP_UP).min(MAX_THROTTLE);
        } else if self.sensors().all(|(t, (soft, _))| t < soft) {
            self.throttle = (self.throttle - THROTTLE_STEP_DOWN).max(0.0);
        }
        self.throttle != previous
    }
}

/// Turns the throttle into sleeps, on the thread that does the work.
#[derive(Default)]
pub struct Pacer {
    owed: f64,
}

impl Pacer {
    /// Seconds to sleep after `work_seconds` of completed work; small pauses
    /// accumulate and are paid out in chunks of at least 20 ms.
    pub fn pause_after(&mut self, work_seconds: f64, throttle: f64) -> f64 {
        if throttle <= 0.0 {
            self.owed = 0.0;
            return 0.0;
        }
        self.owed += work_seconds * throttle / (1.0 - throttle);
        if self.owed < MIN_SLEEP_SECONDS {
            return 0.0;
        }
        std::mem::take(&mut self.owed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacer_matches_python() {
        let mut pacer = Pacer::default();
        assert_eq!(pacer.pause_after(1.0, 0.0), 0.0);
        // 50 % throttle: as long a pause as the work, paid out in chunks.
        assert_eq!(pacer.pause_after(0.01, 0.5), 0.0);
        let pause = pacer.pause_after(0.01, 0.5);
        assert!((pause - 0.02).abs() < 1e-12, "{pause}");
    }

    #[test]
    fn throttle_steps() {
        let mut g = Governor::new();
        g.temperatures = Temperatures { cpu: Some(50.0), gpu: Some(81.0) };
        assert!(g.sensors().any(|(t, (_, hard))| t >= hard));
        assert_eq!(Temperatures { cpu: Some(81.6), gpu: Some(70.2) }.describe(), "cpu82/gpu70");
        assert_eq!(Temperatures::default().describe(), "n/a");
    }
}
