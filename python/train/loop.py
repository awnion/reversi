"""
Run with: uv run python -m train.loop

Architecture:
- One learner process owns optimizer state, checkpoints, tournaments and the
  in-memory replay buffer.
- N self-play processes generate game records into replay shard files.
- Each self-play process runs its own SyncBatchEval thread and MCTS workers.
- The learner asynchronously ingests shard files, trains, and periodically
  exports fresh weights for self-play workers to hot-reload.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import weights as weights_io
from .backend import Model, backend_name, create_model
from .backend.common import BASE_LEARNING_RATE
from .replay import ReplayBuffer
from .replay_spool import ReplayShardWriter, ingest_replay_shards
from .thermal import ThermalGovernor
from .weights import Weights

# --- hyper-parameters --------------------------------------------------------

# Self-play search per move, as in AlphaZero. Raised from 200 once the Rust
# self-play (tree reuse, endgame solver, eval cache) outran the learner: the
# spare speed buys sharper policy targets instead of positions nobody trains
# on.
SIMULATIONS = 800
# Playout cap randomization (KataGo): only FULL_FRACTION of the moves get the
# full search and become training positions; the others are played after a
# FAST_SIMULATIONS search without root noise. About 2.5x more games for the
# same search work, which is what the value head is short of.
FAST_SIMULATIONS = 150
FULL_FRACTION = 0.25
BATCH_SIZE = 512
# Require this many positions before training starts.
# Prevents overfitting to a tiny replay buffer.
MIN_REPLAY = 30_000
# Only do 1 gradient step per TRAIN_RATIO new positions added. With batches
# of BATCH_SIZE this is the sample reuse: each position is trained on about
# BATCH_SIZE / TRAIN_RATIO times (512 / 100 ≈ 5). Far more, and the network
# memorises the latest games instead of learning from them.
TRAIN_RATIO = 100
# Cap on training steps owed to already-ingested positions.
MAX_PENDING_STEPS = 1_000
CHECKPOINT_EVERY = 500
# Exponential moving average of the weights, with a time constant of
# EMA_STEPS steps, updated every EMA_EVERY steps. Self-play, the champion
# match and the champion use the average, which plays better than the
# weights of any single step; training continues from the raw weights.
EMA_STEPS = 10_000
EMA_EVERY = 10
KEEP_CHECKPOINTS = 20
LOG_EVERY = 100
# Learning-rate schedule: a short warmup, then a few long constant stages,
# counted from the step at which this schedule started for the run (kept in
# LR_ORIGIN_FILE, so restarts continue it). The old step decay halved the
# rate every 5k steps and sat at its floor for almost the whole run.
LR_WARMUP_STEPS = 1_000
LR_STAGES = ((0, BASE_LEARNING_RATE), (300_000, 1e-4), (800_000, 3e-5))
_WEIGHTS_BASE = Path(__file__).parent.parent.parent / "weights"

WEIGHTS_DIR = _WEIGHTS_BASE / "alphazero"  # replaced by _apply_run_name() before use
CHAMPION_BIN = WEIGHTS_DIR / "champion.bin"

# Concurrent self-play games. They advance in lockstep inside Rust and their
# leaf positions are evaluated as one batch per round, so this is also the
# inference batch size. The GPU saturates at roughly 1000-2000.
SELFPLAY_GAMES = 1024

# Tournament settings
# Self-play always uses the weight average; the champion (what the browser
# plays, what `just snapshot` fetches) is that average, replaced every
# TOURNAMENT_EVERY_SECONDS of training once the network has had at least
# TOURNAMENT_MIN_STEPS steps since. By the clock rather than by steps, because
# the step rate follows the self-play speed and the thermal throttling.
# No gating, as in AlphaZero: telling +10 Elo apart takes ~4 000 games, and an
# SPRT match of 400 at 800 simulations took 30 minutes and stayed inconclusive.
TOURNAMENT_EVERY_SECONDS = 30 * 60
TOURNAMENT_MIN_STEPS = 5_000
# Before the replacement, pairs of games against a fixed reference
# (reference.bin; the champion of the moment if missing), from one random
# opening each, each colour once: an Elo trend for the log, not a decision.
# On the GPU (`reversi-selfplay match`, with Rust self-play): 50 pairs at
# 2 000 simulations, closer to how the browser bot plays (±55 Elo), each game
# selecting TOURNAMENT_LEAVES leaves per round so that the batches stay large.
# On the CPU (`run_pairs`): 25 pairs at the self-play 800 (±90 Elo, 4 min).
TOURNAMENT_PAIRS_GPU = 50
TOURNAMENT_SIMS_GPU = 2_000
TOURNAMENT_LEAVES = 16
TOURNAMENT_PAIRS_CPU = 25
TOURNAMENT_SIMS_CPU = SIMULATIONS
TOURNAMENT_OPENING_PLIES = 4  # random moves before the engines take over
# Tournament games run on CPU threads; leave cores for self-play and training.
TOURNAMENT_THREADS = max(1, (os.cpu_count() or 2) // 2)
CHAMPION_HISTORY = 5  # keep this many past champion snapshots

# Replay persistence
REPLAY_PATH = WEIGHTS_DIR / "replay_buffer.npz"
# Positions kept for training: several minutes of self-play, so that the
# network does not chase the last minute of games. ~1.3 KB each (2.6 GB).
REPLAY_SIZE = 2_000_000
# Save the replay buffer every N training steps. Saving blocks the learner
# (about 20 s per million positions), so not too often.
REPLAY_SAVE_EVERY = 20_000
PID_FILE = WEIGHTS_DIR / "train.pid"
REPLAY_SPOOL_DIR = WEIGHTS_DIR / "replay_spool"
SELFPLAY_MODEL_BIN = WEIGHTS_DIR / "selfplay_latest.bin"
# Present while the learner lets the machine cool down after a tournament;
# self-play processes pause for as long as it exists.
COOLDOWN_FLAG = WEIGHTS_DIR / "cooldown"
COOLDOWN_MAX_SECONDS = 300.0
# The step at which the learning-rate schedule started (see LR_STAGES).
LR_ORIGIN_FILE = WEIGHTS_DIR / "lr_schedule.json"
# Facts about champion.bin (run, date, loss, win rate, training step).
CHAMPION_META = WEIGHTS_DIR / "champion.json"
# The weight average (EMA_STEPS), saved with every checkpoint.
EMA_PATH = WEIGHTS_DIR / "ema.npz"
# The fixed opponent of the champion matches.
REFERENCE_BIN = WEIGHTS_DIR / "reference.bin"


def _apply_run_name(name: str) -> None:
    """Point all path globals at weights/<name>/."""
    global WEIGHTS_DIR, CHAMPION_BIN, REPLAY_PATH, PID_FILE
    global REPLAY_SPOOL_DIR, SELFPLAY_MODEL_BIN, COOLDOWN_FLAG, CHAMPION_META
    global LR_ORIGIN_FILE, EMA_PATH, REFERENCE_BIN
    WEIGHTS_DIR = _WEIGHTS_BASE / name
    CHAMPION_BIN = WEIGHTS_DIR / "champion.bin"
    REPLAY_PATH = WEIGHTS_DIR / "replay_buffer.npz"
    PID_FILE = WEIGHTS_DIR / "train.pid"
    REPLAY_SPOOL_DIR = WEIGHTS_DIR / "replay_spool"
    SELFPLAY_MODEL_BIN = WEIGHTS_DIR / "selfplay_latest.bin"
    COOLDOWN_FLAG = WEIGHTS_DIR / "cooldown"
    CHAMPION_META = WEIGHTS_DIR / "champion.json"
    LR_ORIGIN_FILE = WEIGHTS_DIR / "lr_schedule.json"
    EMA_PATH = WEIGHTS_DIR / "ema.npz"
    REFERENCE_BIN = WEIGHTS_DIR / "reference.bin"


# A self-play process alternates between the search trees (one CPU core) and
# the network (GPU), so one process leaves the GPU idle most of the time.
# Several processes, each with its own full batch, keep it busy. On Apple
# Silicon the CPU and GPU share one chip and one process is enough.
DEFAULT_SELFPLAY_PROCESSES = 1 if backend_name() == "mlx" else 4
DEFAULT_SHARD_POSITIONS = 4_096
WEIGHT_REFRESH_SECONDS = 5.0  # self-play: poll for new weights
SELFPLAY_LOG_SECONDS = 30.0

# Self-play implementation, from REVERSI_SELFPLAY:
# - "python" (default): the self-play processes below;
# - "rust": one `reversi-selfplay` process (rust/trainer, `just build-selfplay`),
#   libtorch on CUDA with all CPU cores searching. Falls back to Python when
#   its `check` fails or it keeps crashing;
# - "auto": the same as "rust" (a separate name for scripts that opt in).
# Extra arguments for the binary: REVERSI_SELFPLAY_RUST_ARGS, e.g.
# "--workers 8 --batch-target 8192".
SELFPLAY_IMPLEMENTATIONS = ("python", "rust", "auto")
RUST_SELFPLAY_BIN = Path(
    os.environ.get(
        "REVERSI_SELFPLAY_BIN",
        Path(__file__).resolve().parent.parent.parent
        / "rust"
        / "trainer"
        / "target"
        / "release"
        / "reversi-selfplay",
    )
)
# Restarts after unexpected exits before falling back to Python self-play; a
# process that ran this long resets the count.
RUST_SELFPLAY_RESTARTS = 3
RUST_SELFPLAY_STABLE_SECONDS = 600.0


def log_event(component: str, message: str, **fields) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"{stamp} [{component}] {message}{suffix}", flush=True)


def _learning_rate_at(steps: int) -> float:
    """Learning rate after `steps` steps of the current schedule."""
    rate = LR_STAGES[0][1]
    for start, stage_rate in LR_STAGES:
        if steps >= start:
            rate = stage_rate
    if steps < LR_WARMUP_STEPS:
        rate *= 0.1 + 0.9 * steps / LR_WARMUP_STEPS
    return rate


def _lr_origin(step: int) -> int:
    """The step this run's schedule started at; recorded on first use."""
    try:
        return int(json.loads(LR_ORIGIN_FILE.read_text())["origin"])
    except (OSError, ValueError, KeyError, TypeError):
        LR_ORIGIN_FILE.write_text(json.dumps({"origin": step}) + "\n")
        return step


def _champion_meta() -> dict:
    try:
        return json.loads(CHAMPION_META.read_text())
    except (OSError, ValueError):
        return {}


class WeightAverage:
    """Exponential moving average of the weights (see EMA_STEPS)."""

    def __init__(self, weights: Weights) -> None:
        self.weights: Weights = {name: arr.copy() for name, arr in weights.items()}

    @classmethod
    def resume(cls, weights: Weights) -> WeightAverage:
        """The saved average if it fits these weights, else start from them."""
        try:
            saved = weights_io.load(EMA_PATH)
        except Exception:
            saved = {}
        fits = saved.keys() == weights.keys() and all(
            saved[name].shape == arr.shape for name, arr in weights.items()
        )
        log_event("ema", "resume" if fits else "start", path=EMA_PATH.name)
        return cls(saved if fits else weights)

    def update(self, weights: Weights, steps: int) -> None:
        """Fold in `weights`, which stand for the last `steps` steps."""
        keep = math.exp(-steps / EMA_STEPS)
        for name, arr in weights.items():
            avg = self.weights.get(name)
            if avg is None or avg.shape != arr.shape or "num_batches" in name:
                self.weights[name] = arr.copy()
                continue
            avg *= keep
            avg += (1.0 - keep) * arr

    def snapshot(self) -> Weights:
        return {name: arr.copy() for name, arr in self.weights.items()}

    def save(self) -> None:
        tmp = EMA_PATH.with_name("ema.tmp.npz")
        weights_io.save_npz(self.weights, tmp)
        tmp.replace(EMA_PATH)


# --- temperature control -----------------------------------------------------


def _log_throttle(component: str, governor: ThermalGovernor) -> None:
    log_event(
        component,
        "thermal-throttle",
        temp=governor.describe(),
        limits=governor.describe_limits(),
        slowdown=f"{governor.throttle:.0%}",
    )


async def _cool_down(governor: ThermalGovernor, stop_event: asyncio.Event) -> None:
    """Pause training and self-play until every sensor is below its soft limit.

    Tournaments may run hot; this is the rest afterwards.
    """
    if governor.cooled():
        return
    started = time.monotonic()
    log_event("thermal", "cooldown-start", temp=governor.describe())
    COOLDOWN_FLAG.touch()
    try:
        while not stop_event.is_set():
            if time.monotonic() - started >= COOLDOWN_MAX_SECONDS:
                break
            await asyncio.sleep(2.0)
            if governor.cooled():
                break
    finally:
        COOLDOWN_FLAG.unlink(missing_ok=True)
    log_event(
        "thermal",
        "cooldown-end",
        temp=governor.describe(),
        seconds=f"{time.monotonic() - started:.0f}",
    )


# --- training loop -----------------------------------------------------------


async def train_loop(
    model: Model,
    replay: ReplayBuffer,
    stop_event: asyncio.Event,
    lock: threading.Lock,
    next_tournament_step: list,
) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    step, best_loss = _load_best_checkpoint(model)
    # A resumed run continues its schedule instead of starting over.
    lr_origin = _lr_origin(step)
    learning_rate = _learning_rate_at(step - lr_origin)
    model.set_learning_rate(learning_rate)
    log_event("train", "lr", step=step, origin=lr_origin, lr=f"{learning_rate:.2e}")
    # Matches are counted from here, not from step 0.
    next_tournament_step[0] = step + TOURNAMENT_MIN_STEPS
    next_tournament_at = time.monotonic() + TOURNAMENT_EVERY_SECONDS
    replay_at_last_step = replay.total_added
    ema = WeightAverage.resume(model.weights())
    _publish_selfplay_weights(ema.weights)

    tournament_task: asyncio.Task | None = None
    governor = ThermalGovernor()
    log_event(
        "thermal",
        "governor",
        sensors=governor.describe() if governor.read() else "unavailable",
        limits=governor.describe_limits(),
    )
    warmup_log_t = time.monotonic()
    warmup_pos0 = len(replay)
    step_log_t = time.monotonic()

    while not stop_event.is_set():
        ingest_replay_shards(replay, REPLAY_SPOOL_DIR)

        # Wait for enough data before starting
        if len(replay) < MIN_REPLAY:
            now = time.monotonic()
            if now - warmup_log_t >= 10.0:
                elapsed = now - warmup_log_t
                gained = len(replay) - warmup_pos0
                rate = gained / elapsed if elapsed > 0 else 0
                remaining = MIN_REPLAY - len(replay)
                eta = int(remaining / rate) if rate > 0 else 0
                log_event(
                    "train",
                    "warmup",
                    positions=len(replay),
                    needed=MIN_REPLAY,
                    pct=f"{len(replay) / MIN_REPLAY * 100:.0f}%",
                    rate=f"{rate:.0f}/s",
                    eta=f"{eta}s",
                )
                warmup_log_t = now
                warmup_pos0 = len(replay)

            await asyncio.sleep(0.1)
            continue

        # Rate-limit: one step per TRAIN_RATIO new positions. Data arrives in
        # shards of thousands of positions, so spend the allowance one step at
        # a time instead of once per shard.
        new_positions = replay.total_added - replay_at_last_step
        if new_positions < TRAIN_RATIO:
            await asyncio.sleep(0.005)
            continue
        replay_at_last_step += TRAIN_RATIO
        # If self-play outruns the learner, drop the excess allowance rather
        # than letting it grow without bound.
        max_backlog = TRAIN_RATIO * MAX_PENDING_STEPS
        if replay.total_added - replay_at_last_step > max_backlog:
            replay_at_last_step = replay.total_added - max_backlog

        step_started = time.monotonic()
        batch = replay.sample(BATCH_SIZE)
        with lock:
            metrics = model.train_step(*batch)
        loss_val = metrics["loss"]
        step += 1
        step_seconds = time.monotonic() - step_started

        new_lr = _learning_rate_at(step - lr_origin)
        if new_lr != learning_rate:
            learning_rate = new_lr
            model.set_learning_rate(learning_rate)
            if step - lr_origin >= LR_WARMUP_STEPS:
                log_event("train", "lr", step=step, lr=f"{learning_rate:.2e}")

        if math.isnan(loss_val):
            log_event("train", "nan-loss", step=step)
            continue

        if step % EMA_EVERY == 0:
            with lock:
                raw = model.weights()
            ema.update(raw, EMA_EVERY)

        now = time.monotonic()
        if step % LOG_EVERY == 0 or now - step_log_t >= 30.0:
            elapsed = now - step_log_t
            sps = LOG_EVERY / elapsed if elapsed > 0 and step % LOG_EVERY == 0 else None
            log_event(
                "train",
                "step",
                step=step,
                loss=f"{loss_val:.4f}",
                # The part of the policy loss that can still go down.
                policy_kl=f"{metrics['policy_kl']:.4f}",
                value=f"{metrics['value']:.4f}",
                margin=f"{metrics['margin']:.4f}",
                lr=f"{learning_rate:.1e}",
                replay=len(replay),
                **({"sps": f"{sps:.1f}"} if sps is not None else {}),
                **({"temp": governor.describe()} if governor.available else {}),
            )
            step_log_t = now

        save = step % CHECKPOINT_EVERY == 0
        if loss_val < best_loss:
            best_loss = loss_val
            save = True

        if save:
            path = WEIGHTS_DIR / f"iter_{step:06d}_loss{loss_val:.4f}.npz"
            with lock:
                current = model.weights()
            _save_checkpoint(current, path)
            ema.save()
            _publish_selfplay_weights(ema.weights)
            log_event("train", "checkpoint", step=step, path=path.name)
            _prune_checkpoints()

        # Periodically persist replay buffer so restarts don't lose data
        if step % REPLAY_SAVE_EVERY == 0:
            replay.save(REPLAY_PATH)

        # A finished tournament may have heated the machine: rest before
        # going on.
        if tournament_task is not None and tournament_task.done():
            tournament_task = None
            await _cool_down(governor, stop_event)

        # Slow down while the machine is too hot.
        if governor.poll():
            _log_throttle("train", governor)
        pause = governor.pause_after(step_seconds)
        if pause > 0:
            await asyncio.sleep(pause)

        # Trigger tournament when step crosses the next scheduled boundary.
        # A tournament can outlast the interval; never run two at once.
        tournament_idle = tournament_task is None
        if (
            tournament_idle
            and step >= next_tournament_step[0]
            and time.monotonic() >= next_tournament_at
        ):
            next_tournament_step[0] = step + TOURNAMENT_MIN_STEPS
            next_tournament_at = time.monotonic() + TOURNAMENT_EVERY_SECONDS
            tournament_task = asyncio.create_task(
                _run_champion_match(ema.snapshot(), best_loss, step)
            )
            tournament_task.add_done_callback(
                lambda t: (
                    not t.cancelled()
                    and t.exception()
                    and log_event("tournament", "task-error", error=repr(t.exception()))
                )
            )

        await asyncio.sleep(0)


# --- tournament --------------------------------------------------------------


async def _run_champion_match(
    challenger: Weights,
    current_loss: float = float("inf"),
    current_step: int = 0,
) -> None:
    """Play the weight average against the reference for the log, then make
    it the champion."""
    if not CHAMPION_BIN.exists():
        log_event("tournament", "bootstrap-champion")
        weights_io.save_bin(challenger, CHAMPION_BIN)
        _record_champion(challenger, win_rate=1.0, loss=current_loss, step=current_step)
        log_event("tournament", "promoted", path=CHAMPION_BIN.name, mode="bootstrap")
        return
    if not REFERENCE_BIN.exists():
        shutil.copyfile(CHAMPION_BIN, REFERENCE_BIN)
        log_event("tournament", "reference-set", source=CHAMPION_BIN.name)

    started = time.monotonic()
    seed = time.time_ns() & 0xFFFF_FFFF
    points = None
    if _selfplay_implementation() != "python" and RUST_SELFPLAY_BIN.exists():
        points = await _match_on_gpu(challenger, seed)
    if points is None:
        points = await _match_on_cpu(challenger, seed)
    if points is None:
        log_event("tournament", "aborted")
        return

    n = len(points)
    score = sum(points) / (2 * n)
    # Two standard errors of the mean pair score, in Elo around `score`.
    spread = 2 * math.sqrt(sum((p / 2 - score) ** 2 for p in points) / n / n)
    low, high = _score_to_elo(score - spread), _score_to_elo(score + spread)
    log_event(
        "tournament",
        "result",
        score=f"{sum(points):.1f}/{2 * n}",
        win_rate=f"{score:.0%}",
        elo=f"{_score_to_elo(score):+.0f}",
        elo_range=f"[{low:+.0f},{high:+.0f}]",
        seconds=f"{time.monotonic() - started:.0f}",
        opponent=REFERENCE_BIN.name,
    )
    weights_io.save_bin(challenger, CHAMPION_BIN)
    _record_champion(challenger, score, current_loss, step=current_step)
    log_event("tournament", "promoted", path=CHAMPION_BIN.name, step=current_step)


async def _match_on_gpu(challenger: Weights, seed: int) -> list[float] | None:
    """`reversi-selfplay match`: points of `challenger` per pair, or None."""
    challenger_path = WEIGHTS_DIR / "challenger.tmp.bin"
    weights_io.save_bin(challenger, challenger_path)
    pairs = TOURNAMENT_PAIRS_GPU
    log_event(
        "tournament", "start", device="gpu", sims=TOURNAMENT_SIMS_GPU, games=2 * pairs
    )
    cmd = [
        str(RUST_SELFPLAY_BIN),
        "match",
        "--a",
        str(challenger_path),
        "--b",
        str(REFERENCE_BIN),
        "--pairs",
        str(pairs),
        "--simulations",
        str(TOURNAMENT_SIMS_GPU),
        "--leaves",
        str(TOURNAMENT_LEAVES),
        "--opening-plies",
        str(TOURNAMENT_OPENING_PLIES),
        "--seed",
        str(seed),
    ]
    try:
        child = await asyncio.create_subprocess_exec(
            *cmd,
            env=_rust_selfplay_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await child.communicate()
        if child.returncode != 0:
            raise RuntimeError(
                f"exit {child.returncode}: {err.decode().strip()[-500:]}"
            )
        points = json.loads(out.decode().strip().splitlines()[-1])["points"]
        if len(points) != pairs:
            raise RuntimeError(f"{len(points)} pairs instead of {pairs}")
        return points
    except Exception as e:  # noqa: BLE001 - fall back to the CPU match
        log_event("tournament", "gpu-match-failed", error=repr(e))
        return None
    finally:
        challenger_path.unlink(missing_ok=True)


async def _match_on_cpu(challenger: Weights, seed: int) -> list[float] | None:
    """`run_pairs` on CPU threads: points of `challenger` per pair, or None."""
    import traceback

    import reversi_mcts

    try:
        reference_blob = REFERENCE_BIN.read_bytes()
    except OSError as e:
        log_event("tournament", "load-failed", error=repr(e))
        return None
    challenger_blob = weights_io.to_bytes(challenger)
    pairs = TOURNAMENT_PAIRS_CPU
    log_event(
        "tournament", "start", device="cpu", sims=TOURNAMENT_SIMS_CPU, games=2 * pairs
    )

    def play():
        return reversi_mcts.run_pairs(
            challenger_blob,
            reference_blob,
            pairs,
            TOURNAMENT_SIMS_CPU,
            opening_plies=TOURNAMENT_OPENING_PLIES,
            seed=seed,
            threads=TOURNAMENT_THREADS,
        )

    try:
        return await asyncio.get_running_loop().run_in_executor(None, play)
    except Exception:
        log_event("tournament", "run-error", error=traceback.format_exc().strip())
        return None


def _score_to_elo(score: float) -> float:
    score = min(max(score, 1e-3), 1.0 - 1e-3)
    return -400.0 * math.log10(1.0 / score - 1.0)


def _record_champion(
    weights: Weights,
    win_rate: float,
    loss: float = float("inf"),
    step: int | None = None,
) -> None:
    """Save a snapshot of the new champion and update champions/history.json."""
    champions_dir = WEIGHTS_DIR / "champions"
    champions_dir.mkdir(parents=True, exist_ok=True)
    history_file = champions_dir / "history.json"

    # Load existing history
    if history_file.exists():
        try:
            history: list[dict] = json.loads(history_file.read_text())
        except Exception:
            history = []
    else:
        history = []

    # Save model snapshot as .npz (keeps weights for future use)
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    loss_tag = f"{loss:.4f}" if loss != float("inf") else "unknown"
    snap_path = champions_dir / f"champion_{stamp}_loss{loss_tag}.npz"
    _save_checkpoint(weights, snap_path)

    entry = {
        "date": now.isoformat(),
        "file": snap_path.name,
        "loss": round(loss, 4) if loss != float("inf") else None,
        "win_rate": round(win_rate, 3),
    }
    history.append(entry)
    log_event(
        "champion",
        "recorded",
        file=snap_path.name,
        loss=loss_tag,
        win_rate=f"{win_rate:.0%}",
    )

    # Prune to keep only the most recent CHAMPION_HISTORY entries
    if len(history) > CHAMPION_HISTORY:
        to_remove = history[:-CHAMPION_HISTORY]
        history = history[-CHAMPION_HISTORY:]
        for old in to_remove:
            old_path = champions_dir / old["file"]
            if old_path.exists():
                old_path.unlink()
                log_event("champion", "pruned", file=old["file"])

    with history_file.open("w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")

    # Facts about the champion that travel with champion.bin: the browser bots
    # embed this file and show it next to the strategy name.
    with CHAMPION_META.open("w") as f:
        json.dump(
            {
                "run": WEIGHTS_DIR.name,
                "date": now.date().isoformat(),
                "loss": entry["loss"],
                "win_rate": entry["win_rate"],
                # Training resumes from the champion and continues from here.
                "step": step,
                "backend": backend_name(),
            },
            f,
            indent=2,
        )
        f.write("\n")


# --- checkpoint helpers ------------------------------------------------------


def _load_best_checkpoint(model: Model) -> tuple[int, float]:
    """Load the weights to continue training from; returns (step, loss).

    The newest checkpoint, if it is newer than the champion: the champion
    changes only after a long match, and resuming from it would throw away
    the training since. Otherwise the champion, then any checkpoint.
    """

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    checkpoints = sorted(WEIGHTS_DIR.glob("iter_*.npz"), key=_step)
    latest = checkpoints[-1] if checkpoints else None
    champion_step = int(_champion_meta().get("step") or 0)

    if CHAMPION_BIN.exists() and (latest is None or _step(latest) <= champion_step):
        try:
            model.load_weights(weights_io.load(CHAMPION_BIN))
            log_event("train", "resume", source="champion.bin", step=champion_step)
            return champion_step, float("inf")
        except Exception as e:
            log_event("train", "resume-failed", source="champion.bin", error=repr(e))

    if latest is not None:
        weights = weights_io.load(latest)
        applied = model.load_weights(weights)
        step, loss = max(_step(latest), 0), _loss(latest)
        log_event(
            "train",
            "resume",
            source=latest.name,
            step=step,
            loss=f"{loss:.4f}",
            tensors=f"{applied}/{len(weights)}",
        )
        return step, loss

    return 0, float("inf")


def _save_checkpoint(weights: Weights, path: Path) -> None:
    weights_io.save_npz(weights, path)
    log_event("weights", "saved-checkpoint", tensors=len(weights), path=path.name)


def _publish_selfplay_weights(weights: Weights) -> None:
    tmp = SELFPLAY_MODEL_BIN.with_suffix(".bin.tmp")
    weights_io.save_bin(weights, tmp)
    tmp.replace(SELFPLAY_MODEL_BIN)
    log_event("weights", "published-selfplay", path=SELFPLAY_MODEL_BIN.name)


def _prune_checkpoints() -> None:
    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints = list(WEIGHTS_DIR.glob("iter_*.npz"))
    best_by_loss = {p.name for p in sorted(checkpoints, key=_loss)[:KEEP_CHECKPOINTS]}
    best_by_step = {p.name for p in sorted(checkpoints, key=_step)[-KEEP_CHECKPOINTS:]}
    keep = best_by_loss | best_by_step
    for p in checkpoints:
        if p.name not in keep:
            p.unlink()
            log_event("train", "pruned-checkpoint", file=p.name)


# --- entry point -------------------------------------------------------------


def _resolve_selfplay_source() -> Path | None:
    if SELFPLAY_MODEL_BIN.exists():
        return SELFPLAY_MODEL_BIN
    if CHAMPION_BIN.exists():
        return CHAMPION_BIN
    checkpoints = sorted(
        WEIGHTS_DIR.glob("iter_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0,
    )
    return checkpoints[-1] if checkpoints else None


def _load_weights_into_model(model: Model, path: Path) -> None:
    weights = weights_io.load(path)
    applied = model.load_weights(weights)
    log_event(
        "weights",
        "loaded-compatible",
        source=path.name,
        applied=applied,
        total=len(weights),
    )


def _spawn_selfplay_processes(
    count: int,
    workers_per_process: int,
    positions_per_shard: int,
    run_name: str,
) -> list[subprocess.Popen]:
    children: list[subprocess.Popen] = []
    for process_index in range(count):
        cmd = [
            sys.executable,
            "-m",
            "train.loop",
            "--role",
            "selfplay",
            "--process-index",
            str(process_index),
            "--selfplay-workers",
            str(workers_per_process),
            "--positions-per-shard",
            str(positions_per_shard),
            "--run-name",
            run_name,
        ]
        child = subprocess.Popen(cmd)
        children.append(child)
        log_event(
            "system",
            "spawn-selfplay",
            process=process_index,
            pid=child.pid,
            workers=workers_per_process,
        )
    return children


def _selfplay_implementation() -> str:
    name = os.environ.get("REVERSI_SELFPLAY", "python")
    if name not in SELFPLAY_IMPLEMENTATIONS:
        raise ValueError(
            f"REVERSI_SELFPLAY must be one of {SELFPLAY_IMPLEMENTATIONS}, got {name!r}"
        )
    return name


def _rust_selfplay_env() -> dict[str, str]:
    """The environment for `reversi-selfplay`: libtorch from this venv's torch
    on the library path (found without importing torch)."""
    env = dict(os.environ)
    spec = importlib.util.find_spec("torch")
    if spec is not None and spec.submodule_search_locations:
        lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
        if lib.is_dir():
            paths = [str(lib), env.get("LD_LIBRARY_PATH", "")]
            env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in paths if p)
    return env


def _rust_selfplay_works(env: dict[str, str]) -> bool:
    """`reversi-selfplay check`: libtorch loads and CUDA works."""
    if not RUST_SELFPLAY_BIN.exists():
        log_event(
            "system",
            "selfplay-rust-unavailable",
            reason="binary-missing",
            path=RUST_SELFPLAY_BIN,
        )
        return False
    try:
        result = subprocess.run(
            [str(RUST_SELFPLAY_BIN), "check"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log_event("system", "selfplay-rust-unavailable", reason=repr(e))
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        log_event(
            "system",
            "selfplay-rust-unavailable",
            reason=f"check-exit-{result.returncode}",
            detail=repr(detail[-1] if detail else ""),
        )
        return False
    log_event("system", "selfplay-rust-check", result=repr(result.stdout.strip()))
    return True


def _spawn_rust_selfplay(
    positions_per_shard: int, env: dict[str, str]
) -> subprocess.Popen:
    cmd = [
        str(RUST_SELFPLAY_BIN),
        "selfplay",
        "--run-dir",
        str(WEIGHTS_DIR.resolve()),
        "--simulations",
        str(SIMULATIONS),
        "--fast-simulations",
        str(FAST_SIMULATIONS),
        "--full-fraction",
        str(FULL_FRACTION),
        "--positions-per-shard",
        str(positions_per_shard),
        "--parent-pid",
        str(os.getpid()),
        *shlex.split(os.environ.get("REVERSI_SELFPLAY_RUST_ARGS", "")),
    ]
    child = subprocess.Popen(cmd, env=env)
    log_event("system", "spawn-selfplay-rust", pid=child.pid, args=repr(cmd[2:]))
    return child


async def _supervise_rust_selfplay(
    children: list[subprocess.Popen],
    child: subprocess.Popen,
    spawn_rust,
    spawn_python,
    stop_event: asyncio.Event,
) -> None:
    """Restart the Rust self-play process when it exits on its own; after
    RUST_SELFPLAY_RESTARTS quick failures, switch to Python self-play."""
    restarts = 0
    started = time.monotonic()
    while not stop_event.is_set():
        await asyncio.sleep(2.0)
        code = child.poll()
        if code is None or stop_event.is_set():
            continue
        ran = time.monotonic() - started
        log_event(
            "system",
            "selfplay-rust-exited",
            pid=child.pid,
            code=code,
            seconds=f"{ran:.0f}",
        )
        children.remove(child)
        if ran >= RUST_SELFPLAY_STABLE_SECONDS:
            restarts = 0
        if restarts >= RUST_SELFPLAY_RESTARTS:
            log_event("system", "selfplay-rust-fallback", restarts=restarts)
            children.extend(spawn_python())
            return
        restarts += 1
        await asyncio.sleep(5.0 * restarts)
        if stop_event.is_set():
            return
        try:
            child = spawn_rust()
        except Exception as e:
            # Without a fallback here the learner would wait for data forever.
            log_event("system", "selfplay-rust-fallback", error=repr(e))
            children.extend(spawn_python())
            return
        children.append(child)
        started = time.monotonic()
        log_event("system", "selfplay-rust-restarted", restarts=restarts)


def _stop_child_processes(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()


async def _run_selfplay_process(
    process_index: int,
    selfplay_games: int,
    positions_per_shard: int,
) -> None:
    """Self-play producer: one batched search loop feeding the replay spool.

    `selfplay_games` games advance in lockstep inside Rust. Each round collects
    one pending leaf position per game, evaluates them in a single network
    call and hands the results back; finished games go to the shard writer.
    """
    import reversi_mcts

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    model = create_model()
    weights_mtime_ns = None
    source = _resolve_selfplay_source()
    if source is not None:
        _load_weights_into_model(model, source)
        weights_mtime_ns = source.stat().st_mtime_ns
        log_event(
            "selfplay",
            "weights-loaded",
            process=process_index,
            source=source.name,
        )
    else:
        log_event("selfplay", "weights-random-init", process=process_index)

    shard_writer = ReplayShardWriter(
        REPLAY_SPOOL_DIR,
        producer_id=f"sp{process_index}_pid{os.getpid()}",
        positions_per_shard=positions_per_shard,
    )
    pool = reversi_mcts.BatchedSelfPlay(
        selfplay_games,
        SIMULATIONS,
        fast_simulations=FAST_SIMULATIONS,
        full_fraction=FULL_FRACTION,
    )
    stop_flag = [False]

    def _shutdown(signum, frame):
        log_event("selfplay", "signal", process=process_index, signum=signum)
        stop_flag[0] = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    games = positions = evals = 0
    evals_at_log = 0
    last_refresh = last_log = time.monotonic()
    governor = ThermalGovernor()
    try:
        while not stop_flag[0]:
            round_started = time.monotonic()
            planes = np.frombuffer(pool.collect(), dtype=np.float32)
            planes = planes.reshape(-1, 4, 8, 8)
            # MCTS priors are probabilities, as in the Rust inference path.
            policies, values = model.infer(planes)
            pool.apply(
                np.ascontiguousarray(policies, dtype=np.float32).tobytes(),
                np.ascontiguousarray(values, dtype=np.float32).tobytes(),
            )
            batch = planes.shape[0]
            evals += batch

            for record in pool.take_finished():
                positions += shard_writer.add_record(record)
                games += 1

            now = time.monotonic()

            # Slow down while the machine is too hot.
            if governor.poll():
                _log_throttle("selfplay", governor)
            pause = governor.pause_after(now - round_started)
            if pause > 0:
                time.sleep(pause)

            if now - last_refresh >= WEIGHT_REFRESH_SECONDS:
                last_refresh = now
                # The learner asks everyone to rest after a tournament.
                while COOLDOWN_FLAG.exists() and not stop_flag[0]:
                    time.sleep(0.5)
                source = _resolve_selfplay_source()
                if source is not None:
                    mtime_ns = source.stat().st_mtime_ns
                    if mtime_ns != weights_mtime_ns:
                        _load_weights_into_model(model, source)
                        weights_mtime_ns = mtime_ns
                        log_event(
                            "selfplay",
                            "weights-reloaded",
                            process=process_index,
                            source=source.name,
                        )
            if now - last_log >= SELFPLAY_LOG_SECONDS:
                rate = (evals - evals_at_log) / (now - last_log)
                log_event(
                    "selfplay",
                    "progress",
                    process=process_index,
                    games=games,
                    positions=positions,
                    batch=batch,
                    evals_per_s=f"{rate:.0f}",
                    **({"temp": governor.describe()} if governor.available else {}),
                )
                last_log = now
                evals_at_log = evals
    finally:
        shard_writer.flush()


async def _run_learner(
    selfplay_processes: int,
    selfplay_workers: int,
    positions_per_shard: int,
    run_name: str,
) -> None:
    # Guard against duplicate instances
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(old_pid, 0)  # check if process exists
            log_event("system", "already-running", pid=old_pid)
            return
        except ProcessLookupError:
            pass  # stale pid file, proceed
    PID_FILE.write_text(str(os.getpid()))
    COOLDOWN_FLAG.unlink(missing_ok=True)

    model = create_model()
    replay = ReplayBuffer(max_size=REPLAY_SIZE)
    stop_event = asyncio.Event()
    lock = threading.Lock()
    children: list[subprocess.Popen] = []

    # Restore replay buffer from previous run if available
    replay.load(REPLAY_PATH)

    selfplay = _selfplay_implementation()
    log_event(
        "system",
        "startup",
        backend=backend_name(),
        selfplay=selfplay,
        selfplay_processes=selfplay_processes,
        simulations=SIMULATIONS,
        fast_simulations=FAST_SIMULATIONS,
        full_fraction=FULL_FRACTION,
        min_replay=MIN_REPLAY,
        train_ratio=TRAIN_RATIO,
        tournament_every=f"{TOURNAMENT_EVERY_SECONDS}s",
    )
    next_tournament_step = [TOURNAMENT_MIN_STEPS]

    def _shutdown(signum, frame):
        """SIGTERM handler: save state and stop gracefully."""
        # Only flag the stop here: saving from the handler raced with itself
        # when two signals arrived, and the `finally` below saves anyway.
        log_event("system", "signal", signum=signum, action="save-and-stop")
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def spawn_python() -> list[subprocess.Popen]:
        return _spawn_selfplay_processes(
            selfplay_processes, selfplay_workers, positions_per_shard, run_name
        )

    supervisor: asyncio.Task | None = None
    if selfplay_processes > 0:
        rust_env = _rust_selfplay_env()
        if selfplay != "python" and _rust_selfplay_works(rust_env):
            child = _spawn_rust_selfplay(positions_per_shard, rust_env)
            children.append(child)
            supervisor = asyncio.create_task(
                _supervise_rust_selfplay(
                    children,
                    child,
                    lambda: _spawn_rust_selfplay(positions_per_shard, rust_env),
                    spawn_python,
                    stop_event,
                )
            )
        else:
            children.extend(spawn_python())

    try:
        await train_loop(model, replay, stop_event, lock, next_tournament_step)
    except KeyboardInterrupt:
        log_event("system", "keyboard-interrupt")
        replay.save(REPLAY_PATH)
    finally:
        stop_event.set()
        if supervisor is not None:
            supervisor.cancel()
        ingest_replay_shards(replay, REPLAY_SPOOL_DIR, limit=10_000)
        replay.save(REPLAY_PATH)
        _stop_child_processes(children)
        COOLDOWN_FLAG.unlink(missing_ok=True)
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)


def _default_selfplay_workers(processes: int) -> int:
    """Concurrent games per self-play process: a full batch each, so that
    more processes add GPU work instead of splitting it into small batches."""
    return SELFPLAY_GAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("learner", "selfplay"),
        default="learner",
    )
    parser.add_argument(
        "--selfplay-processes",
        type=int,
        default=DEFAULT_SELFPLAY_PROCESSES,
    )
    parser.add_argument(
        "--selfplay-workers",
        type=int,
        default=None,
        help="Concurrent games (= inference batch size) per self-play process",
    )
    parser.add_argument(
        "--positions-per-shard",
        type=int,
        default=DEFAULT_SHARD_POSITIONS,
    )
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Named run; weights go to weights/<run-name>/",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    _apply_run_name(args.run_name)
    workers = args.selfplay_workers
    if workers is None:
        workers = _default_selfplay_workers(args.selfplay_processes)
    if args.role == "selfplay":
        await _run_selfplay_process(
            process_index=args.process_index,
            selfplay_games=workers,
            positions_per_shard=args.positions_per_shard,
        )
        return
    await _run_learner(
        selfplay_processes=args.selfplay_processes,
        selfplay_workers=workers,
        positions_per_shard=args.positions_per_shard,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    asyncio.run(main())
