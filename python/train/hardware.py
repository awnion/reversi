"""What this machine has and how training spreads over it.

`probe()` finds the backend, the accelerators, the memory, the CPUs that
search well and the temperature sensors without importing a framework:
NVIDIA GPUs through NVML (the driver's library, via ctypes), Apple Silicon
through `sysctl`. `make_plan()` turns that into one self-play process per
accelerator (the learner's included), the CPUs of each process and the size
of the replay buffer. Environment variables override parts of the plan:

- ``REVERSI_SELFPLAY_GPUS`` (``"1,2"``): the GPUs that get a self-play process;
- ``REVERSI_LEARNER_CPUS`` / ``REVERSI_SELFPLAY_CPUS`` (``"0,1,24-25"``): one
  CPU set for the learner, one shared by all self-play processes;
- ``REVERSI_REPLAY_SIZE``: positions in the replay buffer.

`Usage` samples how busy each accelerator is, for the tuner's log line.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Bytes per replay position: float32 planes (4x8x8), policy (64) and four
# scalars.
REPLAY_BYTES_PER_POSITION = 4 * (4 * 64 + 64 + 4)
REPLAY_MAX = 8_000_000
# Share of the machine's memory the replay buffer may take.
REPLAY_MEMORY_SHARE = 0.25
# Physical cores kept for the learner (with their SMT siblings).
LEARNER_CORES = 2


# --- NVML ----------------------------------------------------------------------


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class Nvml:
    """The few NVML calls needed here; `None` from `load()` without a driver."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            raise OSError("nvmlDeviceGetCount failed")
        self._handles = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))
            self._handles.append(handle)

    @classmethod
    def load(cls) -> Nvml | None:
        for name in ("libnvidia-ml.so.1", "nvml.dll"):
            try:
                lib = ctypes.CDLL(name)
                if lib.nvmlInit_v2() == 0:
                    return cls(lib)
            except (OSError, AttributeError):
                continue
        return None

    def count(self) -> int:
        return len(self._handles)

    def name(self, i: int) -> str:
        buf = ctypes.create_string_buffer(96)
        self._lib.nvmlDeviceGetName(self._handles[i], buf, 96)
        return buf.value.decode(errors="replace")

    def memory_gb(self, i: int) -> float:
        mem = _Memory()
        self._lib.nvmlDeviceGetMemoryInfo(self._handles[i], ctypes.byref(mem))
        return mem.total / 2**30

    def utilization(self, i: int) -> int:
        """Percent of the last sample period a kernel was running."""
        util = _Utilization()
        self._lib.nvmlDeviceGetUtilizationRates(self._handles[i], ctypes.byref(util))
        return int(util.gpu)

    def temperature(self, i: int) -> int:
        temp = ctypes.c_uint()
        self._lib.nvmlDeviceGetTemperature(self._handles[i], 0, ctypes.byref(temp))
        return int(temp.value)


_nvml: Nvml | None | bool = False


def nvml() -> Nvml | None:
    global _nvml
    if _nvml is False:
        _nvml = Nvml.load() if sys.platform != "darwin" else None
    return _nvml  # type: ignore[return-value]


def visible_gpus() -> list[int]:
    """NVML indices of the GPUs CUDA may use (CUDA_VISIBLE_DEVICES), in
    PCI order like NVML (`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set at startup)."""
    lib = nvml()
    if lib is None:
        return []
    all_gpus = list(range(lib.count()))
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return all_gpus
    listed = [int(p) for p in raw.split(",") if p.strip().isdigit()]
    return [g for g in listed if g in all_gpus]


# --- macOS -----------------------------------------------------------------------


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def mac_thermal_pressure() -> int | None:
    """The system's thermal pressure level without root: 0 nominal,
    1 moderate, 2 heavy, 3 trapping, 4 sleeping (`notifyutil`)."""
    try:
        out = subprocess.run(
            ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout
        return int(out.split()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def mac_gpu_utilization() -> int | None:
    """Apple GPU 'Device Utilization %' from the IOAccelerator statistics."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(match[1]) if match else None


# --- CPUs ------------------------------------------------------------------------


def _linux_cpus() -> list[int]:
    """CPUs this process may run on; on hybrid Intel chips only the
    performance cores."""
    cpus = sorted(os.sched_getaffinity(0))
    performance = Path("/sys/devices/cpu_core/cpus")
    if performance.exists():
        wanted = set(_parse_cpus(performance.read_text()))
        cpus = [c for c in cpus if c in wanted] or cpus
    return cpus


def _parse_cpus(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part.strip():
            continue
        first, _, last = part.strip().partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _cores(cpus: list[int]) -> list[list[int]]:
    """`cpus` grouped by physical core (SMT siblings together), in order."""
    seen: set[int] = set()
    cores = []
    for cpu in cpus:
        if cpu in seen:
            continue
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings = [c for c in _parse_cpus(path.read_text()) if c in cpus]
        except OSError:
            siblings = [cpu]
        siblings = sorted(set(siblings) | {cpu})
        seen.update(siblings)
        cores.append(siblings)
    return cores


def _memory_bytes() -> int:
    if sys.platform == "darwin":
        return int(_sysctl("hw.memsize") or 0)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


# --- probe and plan ----------------------------------------------------------------


@dataclass
class Hardware:
    backend: str
    host: str
    accelerators: list[str]
    accelerator_memory_gb: list[float]
    # NVML indices of the accelerators (CUDA), empty otherwise.
    gpu_indices: list[int]
    memory_gb: float
    # Performance cores, each a list of logical CPUs (one per core on macOS).
    cores: list[list[int]]
    can_pin: bool
    sensors: list[str]

    def describe(self) -> str:
        accel = ",".join(
            f"{name}({mem:.0f}GB)"
            for name, mem in zip(
                self.accelerators, self.accelerator_memory_gb, strict=True
            )
        )
        logical = sum(len(c) for c in self.cores)
        return (
            f"backend={self.backend} host={self.host} "
            f"accelerators={len(self.accelerators)}[{accel}] "
            f"memory={self.memory_gb:.0f}GB "
            f"performance_cores={len(self.cores)} logical_cpus={logical} "
            f"pinning={'yes' if self.can_pin else 'no'} "
            f"sensors={','.join(self.sensors) or 'none'}"
        )


def probe(backend: str, sensors: list[str]) -> Hardware:
    host = platform.node().split(".")[0]
    memory_gb = _memory_bytes() / 2**30
    if sys.platform == "darwin":
        chip = _sysctl("machdep.cpu.brand_string") or "Apple GPU"
        performance = int(
            _sysctl("hw.perflevel0.logicalcpu") or _sysctl("hw.ncpu") or 1
        )
        return Hardware(
            backend=backend,
            host=host,
            accelerators=[chip] if backend == "mlx" else [],
            # Unified memory: the GPU shares the machine's.
            accelerator_memory_gb=[memory_gb] if backend == "mlx" else [],
            gpu_indices=[],
            memory_gb=memory_gb,
            cores=[[i] for i in range(performance)],
            can_pin=False,
            sensors=sensors,
        )
    gpus = visible_gpus()
    lib = nvml()
    return Hardware(
        backend=backend,
        host=host,
        accelerators=[lib.name(g) for g in gpus] if lib else [],
        accelerator_memory_gb=[lib.memory_gb(g) for g in gpus] if lib else [],
        gpu_indices=gpus,
        memory_gb=memory_gb,
        cores=_cores(_linux_cpus()),
        can_pin=hasattr(os, "sched_setaffinity"),
        sensors=sensors,
    )


@dataclass
class SelfPlaySlot:
    index: int
    # NVML index of its GPU (CUDA), None for the default device (MLX, CPU).
    gpu: int | None
    # Shares the accelerator with the learner: the learner goes first.
    shared: bool
    cpus: set[int] | None
    workers: int


@dataclass
class Plan:
    slots: list[SelfPlaySlot]
    learner_cpus: set[int] | None
    learner_gpu: int | None
    replay_size: int
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        slots = " ".join(
            f"sp{s.index}:gpu={'default' if s.gpu is None else s.gpu}"
            f"{'(shared)' if s.shared else ''}"
            f",cpus={len(s.cpus) if s.cpus else 'all'},workers={s.workers}"
            for s in self.slots
        )
        learner = len(self.learner_cpus) if self.learner_cpus else "all"
        gpu = "default" if self.learner_gpu is None else self.learner_gpu
        return (
            f"selfplay_processes={len(self.slots)} {slots} "
            f"learner_gpu={gpu} "
            f"learner_cpus={learner} replay_size={self.replay_size}"
            + (f" overrides={','.join(self.notes)}" if self.notes else "")
        )


def _env_cpus(name: str) -> set[int] | None:
    raw = os.environ.get(name, "").strip()
    return set(_parse_cpus(raw)) if raw else None


def make_plan(hw: Hardware) -> Plan:
    notes = []
    # Accelerators: every visible GPU, or the one default device.
    gpus: list[int | None] = list(hw.gpu_indices) or [None]
    learner_gpu = gpus[0]
    raw = os.environ.get("REVERSI_SELFPLAY_GPUS", "").strip()
    if raw:
        gpus = [int(g) for g in raw.split(",") if g.strip()]
        notes.append("REVERSI_SELFPLAY_GPUS")

    # CPUs: LEARNER_CORES cores for the learner, the rest split evenly.
    learner_cpus: set[int] | None = None
    shares: list[set[int] | None] = [None] * len(gpus)
    workers = [max(1, (sum(len(c) for c in hw.cores) - 2) // len(gpus))] * len(gpus)
    if hw.can_pin and len(hw.cores) > LEARNER_CORES:
        learner_cpus = {c for core in hw.cores[:LEARNER_CORES] for c in core}
        rest = hw.cores[LEARNER_CORES:]
        n = len(gpus)
        groups = [rest[len(rest) * i // n : len(rest) * (i + 1) // n] for i in range(n)]
        shares = [{c for core in g for c in core} or None for g in groups]
        workers = [len(s) if s else 1 for s in shares]
    if (env := _env_cpus("REVERSI_LEARNER_CPUS")) is not None:
        learner_cpus = env
        notes.append("REVERSI_LEARNER_CPUS")
    if (env := _env_cpus("REVERSI_SELFPLAY_CPUS")) is not None:
        shares = [env] * len(gpus)
        workers = [max(1, len(env) // len(gpus))] * len(gpus)
        notes.append("REVERSI_SELFPLAY_CPUS")

    replay = int(hw.memory_gb * 2**30 * REPLAY_MEMORY_SHARE / REPLAY_BYTES_PER_POSITION)
    replay = max(100_000, min(REPLAY_MAX, replay // 100_000 * 100_000))
    if os.environ.get("REVERSI_REPLAY_SIZE"):
        replay = int(os.environ["REVERSI_REPLAY_SIZE"])
        notes.append("REVERSI_REPLAY_SIZE")

    slots = [
        SelfPlaySlot(
            index=i,
            gpu=gpu,
            shared=gpu == learner_gpu,
            cpus=shares[i],
            workers=workers[i],
        )
        for i, gpu in enumerate(gpus)
    ]
    return Plan(slots, learner_cpus, learner_gpu, replay, notes)


# --- usage -----------------------------------------------------------------------


class Usage:
    """Accelerator utilization averaged since the last `take()`, and the CPU
    time of the self-play processes."""

    def __init__(self, hw: Hardware) -> None:
        self._hw = hw
        self._sums = [0.0] * max(1, len(hw.accelerators))
        self._samples = 0
        self._cpu_seconds: dict[int, float] = {}

    def sample(self) -> None:
        lib = nvml()
        if self._hw.gpu_indices and lib is not None:
            for i, g in enumerate(self._hw.gpu_indices):
                self._sums[i] += lib.utilization(g)
            self._samples += 1
        elif sys.platform == "darwin" and (u := mac_gpu_utilization()) is not None:
            self._sums[0] += u
            self._samples += 1

    def take(self) -> list[float] | None:
        """Mean percent per accelerator since the last call."""
        if not self._samples:
            return None
        means = [s / self._samples for s in self._sums]
        self._sums = [0.0] * len(self._sums)
        self._samples = 0
        return means

    def cpu_seconds(self, pids: list[int]) -> float:
        """CPU seconds the processes used since the last call (all threads)."""
        total = 0.0
        for pid in pids:
            now = _process_cpu_seconds(pid)
            if now is None:
                continue
            total += max(0.0, now - self._cpu_seconds.get(pid, now))
            self._cpu_seconds[pid] = now
        return total


def _process_cpu_seconds(pid: int) -> float | None:
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, ValueError):
            return None
    try:
        out = subprocess.run(
            ["ps", "-o", "time=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    # [[dd-]hh:]mm:ss.cc
    days, _, rest = out.rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds + (int(days) * 86400 if days else 0)
