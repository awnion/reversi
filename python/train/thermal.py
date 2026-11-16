"""Temperature-aware pacing for long training runs.

`ThermalGovernor` reads the CPU and GPU temperatures every few seconds and
turns them into a throttle: the fraction of wall time the caller should spend
sleeping. Each sensor has its own soft and hard limit. While every sensor is
below its soft limit nothing is slowed down; once any sensor reaches its hard
limit the throttle is raised step by step until the machine cools; in between
it is held.

The GPU is kept cool (70/74 °C). Laptop CPUs run hot under load, so the CPU
limits are higher (85/90 °C); tournaments are not throttled at all and may push
the CPU further for a short while.

Sensors (Linux): `nvidia-smi` for the GPU, or NVML where `nvidia-smi` is not
installed; hwmon (`k10temp`, `coretemp`, ...) for the CPU. macOS needs root to
read temperatures, but reports its thermal pressure level to anyone
(`notifyutil`): the "pressure" sensor, held at moderate (1) and slowed down
from heavy (2) on. Where no sensor is available the governor is a no-op.

Limits are in °C and can be overridden with `REVERSI_GPU_TEMP_SOFT`,
`REVERSI_GPU_TEMP_HARD`, `REVERSI_CPU_TEMP_SOFT` and `REVERSI_CPU_TEMP_HARD`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import hardware


def _limit(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# sensor → (soft, hard)
LIMITS: dict[str, tuple[float, float]] = {
    "gpu": (_limit("REVERSI_GPU_TEMP_SOFT", 70), _limit("REVERSI_GPU_TEMP_HARD", 74)),
    "cpu": (_limit("REVERSI_CPU_TEMP_SOFT", 85), _limit("REVERSI_CPU_TEMP_HARD", 90)),
    # macOS thermal pressure: 0 nominal, 1 moderate, 2 heavy, 3 trapping.
    "pressure": (1, 2),
}
POLL_SECONDS = 5.0
# Throttle = share of wall time spent sleeping.
THROTTLE_STEP_UP = 0.15
THROTTLE_STEP_DOWN = 0.05
MAX_THROTTLE = 0.9
# Sleep in chunks of at least this long instead of after every tiny work unit.
MIN_SLEEP_SECONDS = 0.02

# hwmon drivers that report the CPU package/die temperature.
_CPU_HWMON_NAMES = {"k10temp", "zenpower", "coretemp", "cpu_thermal"}


def _cpu_hwmon_dirs() -> list[Path]:
    dirs = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hwmon / "name").read_text().strip() in _CPU_HWMON_NAMES:
                dirs.append(hwmon)
        except OSError:
            continue
    return dirs


def _read_cpu(dirs: list[Path]) -> float | None:
    readings = []
    for hwmon in dirs:
        for sensor in hwmon.glob("temp*_input"):
            try:
                readings.append(int(sensor.read_text()) / 1000.0)
            except (OSError, ValueError):
                continue
    return max(readings) if readings else None


def _read_gpu() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout
        return max(float(line) for line in out.split() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _read_gpu_nvml() -> float | None:
    lib = hardware.nvml()
    if lib is None or lib.count() == 0:
        return None
    return float(max(lib.temperature(i) for i in range(lib.count())))


class ThermalGovernor:
    def __init__(self, limits: dict[str, tuple[float, float]] | None = None) -> None:
        self.limits = dict(limits or LIMITS)
        self.throttle = 0.0
        self.temperatures: dict[str, float] = {}
        self._cpu_dirs = _cpu_hwmon_dirs()
        self._has_gpu = shutil.which("nvidia-smi") is not None
        nvml = hardware.nvml()
        self._has_nvml = not self._has_gpu and nvml is not None and nvml.count() > 0
        self._has_pressure = (
            sys.platform == "darwin" and hardware.mac_thermal_pressure() is not None
        )
        self.available = (
            bool(self._cpu_dirs)
            or self._has_gpu
            or self._has_nvml
            or self._has_pressure
        )
        self._last_poll = float("-inf")
        self._owed = 0.0

    def read(self) -> dict[str, float]:
        """Read the sensors now. Empty when none is available."""
        temperatures = {}
        if self._cpu_dirs and (cpu := _read_cpu(self._cpu_dirs)) is not None:
            temperatures["cpu"] = cpu
        if self._has_gpu and (gpu := _read_gpu()) is not None:
            temperatures["gpu"] = gpu
        if self._has_nvml and (gpu := _read_gpu_nvml()) is not None:
            temperatures["gpu"] = gpu
        if (
            self._has_pressure
            and (level := hardware.mac_thermal_pressure()) is not None
        ):
            temperatures["pressure"] = float(level)
        self.temperatures = temperatures
        return temperatures

    def _any_at_hard_limit(self) -> bool:
        return any(t >= self.limits[k][1] for k, t in self.temperatures.items())

    def _all_below_soft_limit(self) -> bool:
        return all(t < self.limits[k][0] for k, t in self.temperatures.items())

    def sensors(self) -> list[str]:
        """Names of the sensors this machine has."""
        names = ["cpu"] if self._cpu_dirs else []
        names += ["gpu"] if self._has_gpu or self._has_nvml else []
        return names + (["pressure"] if self._has_pressure else [])

    def describe(self) -> str:
        """Current readings, e.g. ``cpu82/gpu71``."""
        if not self.temperatures:
            return "n/a"
        return "/".join(f"{k}{v:.0f}" for k, v in sorted(self.temperatures.items()))

    def describe_limits(self) -> str:
        """Soft-hard limits per sensor, e.g. ``cpu85-90/gpu70-74``."""
        return "/".join(
            f"{k}{soft:.0f}-{hard:.0f}"
            for k, (soft, hard) in sorted(self.limits.items())
        )

    def poll(self) -> bool:
        """Refresh the throttle if the poll interval has passed.

        Returns True when the throttle level changed.
        """
        if not self.available:
            return False
        now = time.monotonic()
        if now - self._last_poll < POLL_SECONDS:
            return False
        self._last_poll = now
        previous = self.throttle
        if not self.read():
            self.throttle = 0.0
        elif self._any_at_hard_limit():
            self.throttle = min(MAX_THROTTLE, self.throttle + THROTTLE_STEP_UP)
        elif self._all_below_soft_limit():
            self.throttle = max(0.0, self.throttle - THROTTLE_STEP_DOWN)
        return self.throttle != previous

    def pause_after(self, work_seconds: float) -> float:
        """Seconds to sleep after `work_seconds` of work (0 when not throttled).

        Small pauses are accumulated and paid out in chunks.
        """
        if self.throttle <= 0.0:
            self._owed = 0.0
            return 0.0
        self._owed += work_seconds * self.throttle / (1.0 - self.throttle)
        if self._owed < MIN_SLEEP_SECONDS:
            return 0.0
        owed, self._owed = self._owed, 0.0
        return owed

    def cooled(self) -> bool:
        """Read the sensors; True once every one is below its soft limit."""
        if not self.available:
            return True
        self.read()
        return self._all_below_soft_limit()
