"""Fits self-play to the machine while training runs.

The KPI is the rate of positions that are both written by self-play and
trained on: min(positions added per second, steps per second x TRAIN_RATIO).

- Games in play: every TUNE_SECONDS the number of concurrent games per
  self-play process is moved by TUNE_STEP (alternating direction after a
  miss); the change stays when the KPI rose by more than TUNE_GAIN over the
  window before it, else it is undone. After a miss in both directions the
  value is kept for RETUNE_SECONDS. Never upwards while the thermal limiter
  or a cooldown is active. The value found is saved in the run directory
  with the network size and the machine (TUNE_FILE) and used on restart.
- Learner first on a shared accelerator: the self-play process there sleeps
  a share of its time (its "duty") that grows while the learner falls behind
  (steps owed) and shrinks while the learner waits for data.
- Every TUNE_SECONDS one log line: KPI, steps/s, accelerator utilization,
  self-play CPU use and what limits the KPI now.

Self-play processes read their settings from `selfplay<i>.control` in the
run directory (`games=N duty=F`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .hardware import Hardware, Plan, Usage

TUNE_SECONDS = 60.0
# Measured part of a window: the first seconds after a change are skipped.
SETTLE_SECONDS = 15.0
TUNE_STEP = 0.25
TUNE_GAIN = 0.03
RETUNE_SECONDS = 30 * 60
# Not before self-play has ramped up (staggered game starts).
TUNE_START_SECONDS = 180.0
MIN_GAMES = 256
MAX_GAMES = 65_536
DEFAULT_GAMES = 6_144
DUTY_SECONDS = 2.0
DUTY_MAX = 0.9
# Steps owed: above HIGH the shared self-play slows down, below LOW (the
# learner is waiting for data) it speeds up again.
BACKLOG_HIGH = 100
BACKLOG_LOW = 20
TUNE_FILE = "autotune.json"


@dataclass
class _Mark:
    at: float
    step: int
    added: int
    busy: float


class Tuner:
    def __init__(
        self,
        run_dir: Path,
        hw: Hardware,
        plan: Plan,
        net: str,
        train_ratio: int,
        log,
        children: list,
        default_games: int = DEFAULT_GAMES,
    ) -> None:
        self.run_dir = run_dir
        self.hw = hw
        self.plan = plan
        self.net = net
        self.train_ratio = train_ratio
        self.log = log
        self.usage = Usage(hw)
        self.games = self._saved_games() or default_games
        self.duty = 0.0
        self.started = time.monotonic()
        # Hill climbing state.
        self.phase = "base"  # "base": measuring the current value; "trial"
        self.direction = 1
        self.misses = 0
        self.settled_until = 0.0
        self.base_kpi: float | None = None
        self.trial_from: int | None = None
        self.window_start: _Mark | None = None
        self.window_measure: _Mark | None = None
        self.window_hot = False
        # Why the window was not a clean measurement.
        self.window_pauses: set[str] = set()
        self.last_duty = 0.0
        self.last_sample = 0.0
        # Self-play processes (Popen), for their CPU time.
        self.children = children
        self._write_controls()

    # --- persistence ---------------------------------------------------------

    def _saved_games(self) -> int | None:
        try:
            saved = json.loads((self.run_dir / TUNE_FILE).read_text())
        except (OSError, ValueError):
            return None
        if saved.get("host") == self.hw.host and saved.get("net") == self.net:
            return int(saved["games"])
        return None

    def _save(self, kpi: float) -> None:
        state = {
            "host": self.hw.host,
            "net": self.net,
            "games": self.games,
            "kpi": round(kpi),
            "selfplay_processes": len(self.plan.slots),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / TUNE_FILE).write_text(json.dumps(state) + "\n")

    def _write_controls(self) -> None:
        for slot in self.plan.slots:
            duty = self.duty if slot.shared else 0.0
            path = self.run_dir / f"selfplay{slot.index}.control"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(f"games={self.games} duty={duty:.3f}\n")
            tmp.replace(path)

    # --- per-iteration -----------------------------------------------------------

    def tick(
        self,
        step: int,
        added: int,
        backlog: float,
        busy_seconds: float,
        hot: bool,
        match_running: bool,
        saving: bool = False,
    ) -> None:
        """Call often from the training loop. `added`: positions ever added
        to the replay; `backlog`: steps owed; `busy_seconds`: total time spent
        in training steps; `hot`: the thermal limiter or a cooldown is on."""
        now = time.monotonic()
        if now - self.last_sample >= 1.0:
            self.last_sample = now
            self.usage.sample()
        # A replay save stalls ingestion and the learner: the steps owed then
        # say nothing about the learner's speed.
        if now - self.last_duty >= DUTY_SECONDS and not saving:
            self.last_duty = now
            self._adjust_duty(backlog)
        mark = _Mark(now, step, added, busy_seconds)
        if self.window_start is None:
            self.window_start = mark
            self.usage.cpu_seconds(self._pids())
            return
        self.window_hot = self.window_hot or hot or match_running or saving
        for reason, on in (
            ("temperature", hot),
            ("match", match_running),
            ("save", saving),
        ):
            if on:
                self.window_pauses.add(reason)
        if self.window_measure is None and now - self.window_start.at >= SETTLE_SECONDS:
            self.window_measure = mark
        if now - self.window_start.at < SETTLE_SECONDS + TUNE_SECONDS:
            return
        self._end_window(mark, backlog, hot)

    def _pids(self) -> list[int]:
        return [c.pid for c in self.children if c.poll() is None]

    def note_hot(self) -> None:
        """A match ended (and maybe a cooldown) outside `tick`."""
        self.window_hot = True
        self.window_pauses.add("match")

    def _adjust_duty(self, backlog: float) -> None:
        if not any(s.shared for s in self.plan.slots):
            return
        old = self.duty
        if backlog > BACKLOG_HIGH:
            self.duty = min(DUTY_MAX, self.duty + 0.05)
        elif backlog < BACKLOG_LOW:
            self.duty = max(0.0, self.duty - 0.05)
        if self.duty != old:
            self._write_controls()

    def _end_window(self, end: _Mark, backlog: float, hot: bool) -> None:
        start = self.window_measure or self.window_start
        assert start is not None
        seconds = max(end.at - start.at, 1e-9)
        added = (end.added - start.added) / seconds
        steps = (end.step - start.step) / seconds
        consumed = steps * self.train_ratio
        kpi = min(added, consumed)
        busy = (end.busy - start.busy) / seconds
        util = self.usage.take()
        cores = self.usage.cpu_seconds(self._pids()) / max(
            end.at - (self.window_start.at if self.window_start else end.at), 1e-9
        )
        allotted = sum(
            len(s.cpus) if s.cpus else sum(len(c) for c in self.hw.cores)
            for s in self.plan.slots
        )
        limit = self._limit(util, cores, allotted, busy, backlog, self.window_pauses)
        action = self._tune(kpi, hot, self.window_hot)
        accel = ",".join(f"{u:.0f}%" for u in util) if util is not None else "n/a"
        self.log(
            "tune",
            "minute",
            kpi=f"{kpi:.0f}/s",
            added=f"{added:.0f}/s",
            trained=f"{consumed:.0f}/s",
            steps_per_s=f"{steps:.1f}",
            learner_busy=f"{busy:.0%}",
            backlog=f"{backlog:.0f}",
            accelerators=accel,
            selfplay_cpu=f"{cores:.1f}/{allotted}",
            games=self.games,
            duty=f"{self.duty:.0%}",
            limit=limit,
            tuning=action,
        )
        self.window_start = end
        self.window_measure = None
        self.window_hot = False
        self.window_pauses = set()

    def _limit(
        self,
        util: list[float] | None,
        cores: float,
        allotted: int,
        busy: float,
        backlog: float,
        pauses: set[str],
    ) -> str:
        if "temperature" in pauses:
            return "temperature"
        if pauses:
            return "paused:" + "+".join(sorted(pauses))
        if busy >= 0.9 or backlog >= 4 * BACKLOG_HIGH:
            return "training"
        if util is not None:
            dedicated = [
                util[self.hw.gpu_indices.index(s.gpu)]
                for s in self.plan.slots
                if not s.shared and s.gpu in self.hw.gpu_indices
            ]
            selfplay = dedicated or util
            if sum(selfplay) / len(selfplay) >= 85:
                return "accelerator"
        if allotted and cores >= 0.85 * allotted:
            return "cpu"
        return "none"

    def _tune(self, kpi: float, hot: bool, window_hot: bool) -> str:
        now = time.monotonic()
        if now - self.started < TUNE_START_SECONDS:
            return "ramp-up"
        if window_hot:
            # A match, cooldown or throttling in the window: the KPI says
            # nothing about the games; undo a trial and measure again.
            if self.phase == "trial" and self.trial_from is not None:
                self.games = self.trial_from
                self._write_controls()
            self.phase, self.base_kpi = "base", None
            return "skipped"
        if self.phase == "trial":
            assert self.base_kpi is not None and self.trial_from is not None
            if kpi > self.base_kpi * (1 + TUNE_GAIN):
                self.base_kpi = kpi
                self.misses = 0
                self._save(kpi)
                result = f"kept {self.trial_from}->{self.games}"
                self.phase = "base"  # next window: try further from here
                return result + self._propose(hot, now, measured=True)
            result = f"undone {self.games}->{self.trial_from}"
            self.games = self.trial_from
            self.direction = -self.direction
            self.misses += 1
            self._write_controls()
            self.phase, self.base_kpi = "base", None
            if self.misses >= 2:
                self.settled_until = now + RETUNE_SECONDS
                self.misses = 0
                self._save(kpi)
                return result + ", settled"
            return result
        # Base window: the KPI at the current value.
        self.base_kpi = kpi
        return "measured" + self._propose(hot, now, measured=True)

    def _propose(self, hot: bool, now: float, measured: bool) -> str:
        if now < self.settled_until:
            return ""
        if self.direction > 0 and hot:
            return ", no increase while hot"
        new = round(self.games * (1 + TUNE_STEP * self.direction))
        new = max(MIN_GAMES, min(MAX_GAMES, new))
        if new == self.games:
            self.direction = -self.direction
            return ""
        self.trial_from, self.games = self.games, new
        self.phase = "trial"
        self._write_controls()
        return f", trying {self.trial_from}->{new}"
