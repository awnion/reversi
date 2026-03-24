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
import json
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .eval_server import SyncBatchEval, board_to_planes
from .export import MAGIC, VERSION, export_model
from .loss import compute_loss
from .model import AlphaZeroNet
from .replay import ReplayBuffer
from .replay_spool import ReplayShardWriter, ingest_replay_shards

# --- hyper-parameters --------------------------------------------------------

SIMULATIONS = 300
BATCH_SIZE = 512
# Require this many positions before training starts.
# Prevents overfitting to a tiny replay buffer.
MIN_REPLAY = 30_000
# Only do 1 gradient step per TRAIN_RATIO new positions added.
TRAIN_RATIO = 10
CHECKPOINT_EVERY = 500
KEEP_CHECKPOINTS = 20
LOG_EVERY = 100
BASE_LR = 3e-4
LR_DECAY_EVERY = 5_000  # multiply LR by LR_DECAY_FACTOR every N steps
LR_DECAY_FACTOR = 0.5
LR_MIN = 1e-5
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
CHAMPION_BIN = WEIGHTS_DIR / "champion.bin"

# Workers block on threading.Event.wait() (GIL released) most of the time,
# so we can run many more than CPU cores.
N_WORKERS = (os.cpu_count() or 4) * 4

# Larger batches = better GPU utilisation.
EVAL_BATCH_SIZE = 96

# Tournament settings
TOURNAMENT_EVERY = 3_000  # run a mini-tournament every N training steps
TOURNAMENT_POOL_CHECKPOINTS = 2  # recent checkpoints to include with champion/current
TOURNAMENT_GAMES_PER_PAIR = 2  # games for each pairing inside the tournament pool
TOURNAMENT_SIMS = 200  # MCTS simulations per move in tournament
WIN_THRESHOLD = 0.55  # win rate required to crown a new champion
CHAMPION_HISTORY = 5  # keep this many past champion snapshots

# Replay persistence
REPLAY_PATH = WEIGHTS_DIR / "replay_buffer.npz"
REPLAY_SAVE_EVERY = 1_000  # save replay buffer every N training steps
PID_FILE = WEIGHTS_DIR / "train.pid"
REPLAY_SPOOL_DIR = WEIGHTS_DIR / "replay_spool"
SELFPLAY_MODEL_BIN = WEIGHTS_DIR / "selfplay_latest.bin"
DEFAULT_SELFPLAY_PROCESSES = max(1, os.cpu_count() // 2)
DEFAULT_SHARD_POSITIONS = 4_096
WEIGHT_REFRESH_GAMES = 5


def log_event(component: str, message: str, **fields) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"{stamp} [{component}] {message}{suffix}", flush=True)


# --- self-play worker --------------------------------------------------------


def _self_play_worker(
    worker_id: int,
    batch_eval: SyncBatchEval,
    sink,  # ReplayBuffer (inline) or ReplayShardWriter (subprocess)
    games_counter: list,
    stop_flag: list,
    refresh_weights=None,
) -> None:
    """Runs in a ThreadPoolExecutor thread."""
    import reversi_mcts

    w = reversi_mcts.MctsWorker(SIMULATIONS)
    while not stop_flag[0]:
        record = w.run_game(batch_eval.evaluate)
        if isinstance(sink, ReplayBuffer):
            sink.add(record)
            positions = len(record)
        else:
            positions = sink.add_record(record)
        games_counter[0] += 1
        if refresh_weights is not None and games_counter[0] % WEIGHT_REFRESH_GAMES == 0:
            refresh_weights()
        if games_counter[0] == 1 or games_counter[0] % 25 == 0:
            log_event(
                "selfplay",
                "worker-progress",
                worker=worker_id,
                games=games_counter[0],
                positions=positions,
            )


# --- training loop -----------------------------------------------------------


async def train_loop(
    model: AlphaZeroNet,
    replay: ReplayBuffer,
    stop_event: asyncio.Event,
    lock: threading.Lock,
    next_tournament_step: list,
) -> None:
    opt = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    step, best_loss = _load_best_checkpoint(model)
    replay_at_last_step = replay.total_added
    _publish_selfplay_weights(model)

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

        # Rate-limit: don't train faster than data comes in
        new_positions = replay.total_added - replay_at_last_step
        if new_positions < TRAIN_RATIO:
            await asyncio.sleep(0.005)
            continue
        replay_at_last_step = replay.total_added

        (
            planes_np,
            policies_np,
            outcomes_np,
            policy_weights_np,
            value_weights_np,
        ) = replay.sample(BATCH_SIZE)

        # Convert numpy → MLX outside the lock to minimize lock hold time
        x = mx.array(np.asarray(planes_np, dtype=np.float32))
        phase = x[:, 3, 0, 0]
        policy_targets = 0.97 * mx.array(policies_np) + 0.03 / 64.0
        value_targets = mx.array(outcomes_np)
        policy_weights_mx = mx.array(policy_weights_np)
        value_weights_mx = mx.array(value_weights_np)

        with lock:
            loss, grads = loss_and_grad(
                model,
                x,
                policy_targets,
                value_targets,
                policy_weights_mx,
                value_weights_mx,
                phase,
            )
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)

        step += 1
        loss_val = loss.item()

        # Step-based LR decay
        if step % LR_DECAY_EVERY == 0:
            decay = LR_DECAY_FACTOR ** (step // LR_DECAY_EVERY)
            new_lr = max(BASE_LR * decay, LR_MIN)
            opt.learning_rate = mx.array(new_lr)
            log_event("train", "lr-decay", step=step, lr=f"{new_lr:.2e}")

        if math.isnan(loss_val):
            log_event("train", "nan-loss", step=step)
            continue

        now = time.monotonic()
        if step % LOG_EVERY == 0 or now - step_log_t >= 30.0:
            elapsed = now - step_log_t
            sps = LOG_EVERY / elapsed if elapsed > 0 and step % LOG_EVERY == 0 else None
            log_event(
                "train",
                "step",
                step=step,
                loss=f"{loss_val:.4f}",
                replay=len(replay),
                **({"sps": f"{sps:.1f}"} if sps is not None else {}),
            )
            step_log_t = now

        save = step % CHECKPOINT_EVERY == 0
        if loss_val < best_loss:
            best_loss = loss_val
            save = True

        if save:
            path = WEIGHTS_DIR / f"iter_{step:06d}_loss{loss_val:.4f}.npz"
            _save_checkpoint(model, path)
            _publish_selfplay_weights(model)
            log_event("train", "checkpoint", step=step, path=path.name)
            _prune_checkpoints()

        # Periodically persist replay buffer so restarts don't lose data
        if step % REPLAY_SAVE_EVERY == 0:
            replay.save(REPLAY_PATH)

        # Trigger tournament when step crosses the next scheduled boundary
        if step >= next_tournament_step[0]:
            next_tournament_step[0] = step + TOURNAMENT_EVERY
            task = asyncio.create_task(
                _run_mini_tournament(model, lock, best_loss, None)
            )
            task.add_done_callback(
                lambda t: (
                    t.exception()
                    and log_event("tournament", "task-error", error=repr(t.exception()))
                )
            )

        await asyncio.sleep(0)


# --- tournament --------------------------------------------------------------


def _load_model_from_bin(bin_path: Path) -> AlphaZeroNet:
    weights = []
    with open(bin_path, "rb") as f:
        magic, version = struct.unpack("<IH", f.read(6))
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"Unrecognised .bin format: {bin_path}")
        (count,) = struct.unpack("<H", f.read(2))
        for _ in range(count):
            (name_len,) = struct.unpack("<B", f.read(1))
            name = f.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<B", f.read(1))
            shape = struct.unpack(f"<{ndim}I", f.read(ndim * 4))
            n_elems = 1
            for d in shape:
                n_elems *= d
            arr = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).reshape(shape)
            weights.append((name, mx.array(arr)))
    m = AlphaZeroNet()
    applied = _load_compatible_weights(m, weights)
    mx.eval(m.parameters())
    log_event(
        "weights",
        "loaded-compatible",
        source=bin_path.name,
        applied=applied,
        total=len(weights),
    )
    m.eval()
    return m


def _load_model_from_checkpoint(checkpoint: Path) -> AlphaZeroNet:
    data = np.load(str(checkpoint))
    weights = [(k, mx.array(data[k])) for k in data.files]
    model = AlphaZeroNet()
    applied = _load_compatible_weights(model, weights)
    mx.eval(model.parameters())
    log_event(
        "weights",
        "loaded-compatible",
        source=checkpoint.name,
        applied=applied,
        total=len(weights),
    )
    model.eval()
    return model


def _make_eval_fn(model: AlphaZeroNet, lock: threading.Lock | None = None):
    """Return a thread-safe eval callback for MCTS."""

    def evaluate(
        board_black: int, board_white: int, is_black: bool, legal: int
    ) -> tuple[list[float], float]:
        planes = board_to_planes(board_black, board_white, is_black, legal)
        x = mx.array(planes[None])
        if lock is not None:
            with lock:
                model.eval()
                p, v = model(x)
                mx.eval(p, v)
                model.train()
        else:
            with mx.stream(mx.cpu):
                p, v = model(x)
                mx.eval(p, v)
        return p[0].tolist(), float(v[0])

    return evaluate


async def _run_mini_tournament(
    current_model: AlphaZeroNet,
    lock: threading.Lock,
    current_loss: float = float("inf"),
    batch_eval: SyncBatchEval | None = None,
) -> None:
    """Run a small round-robin over champion/current/recent checkpoints."""
    import traceback

    import reversi_mcts

    # Take a snapshot of current model weights to avoid interference from
    # training updates while the tournament games are running.
    snapshot = AlphaZeroNet()
    with lock:
        snapshot.load_weights(
            [(k, mx.array(v)) for k, v in _iter_params(current_model)],
            strict=False,
        )
        mx.eval(snapshot.parameters())
    snapshot.eval()

    if not CHAMPION_BIN.exists():
        log_event("tournament", "bootstrap-champion")
        with lock:
            export_model(snapshot, CHAMPION_BIN)
            _record_champion(snapshot, win_rate=1.0, loss=current_loss)
        log_event("tournament", "promoted", path=CHAMPION_BIN.name, mode="bootstrap")
        return

    try:
        champion_model = _load_model_from_bin(CHAMPION_BIN)
    except Exception as e:
        log_event("tournament", "load-failed", error=repr(e))
        return

    def _loss_from_path(path: Path) -> float:
        try:
            return float(path.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    def _step_from_path(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    recent_paths = sorted(
        WEIGHTS_DIR.glob("iter_*.npz"),
        key=_step_from_path,
        reverse=True,
    )[:TOURNAMENT_POOL_CHECKPOINTS]

    entrants: list[dict] = [
        {
            "name": "champion.bin",
            "source": CHAMPION_BIN,
            "model": champion_model,
            "loss": float("inf"),
            "is_champion": True,
            "is_current": False,
        },
        {
            "name": "current",
            "source": None,
            "model": snapshot,
            "loss": current_loss,
            "is_champion": False,
            "is_current": True,
        },
    ]
    for path in recent_paths:
        try:
            entrants.append(
                {
                    "name": path.name,
                    "source": path,
                    "model": _load_model_from_checkpoint(path),
                    "loss": _loss_from_path(path),
                    "is_champion": False,
                    "is_current": False,
                }
            )
        except Exception as e:
            log_event("tournament", "checkpoint-skip", file=path.name, error=repr(e))

    log_event(
        "tournament",
        "start",
        entrants=len(entrants),
        games_per_pair=TOURNAMENT_GAMES_PER_PAIR,
        sims=TOURNAMENT_SIMS,
    )

    eval_fns = {
        entrant["name"]: _make_eval_fn(entrant["model"]) for entrant in entrants
    }

    def run_games():
        if batch_eval is not None:
            batch_eval.pause()
            log_event("tournament", "pause-selfplay")
        try:
            scores = {entrant["name"]: 0.0 for entrant in entrants}
            pairings = [
                (entrants[i], entrants[j])
                for i in range(len(entrants))
                for j in range(i + 1, len(entrants))
            ]
            total_games = len(pairings) * TOURNAMENT_GAMES_PER_PAIR
            done = 0
            for left, right in pairings:
                left_eval = eval_fns[left["name"]]
                right_eval = eval_fns[right["name"]]
                for i in range(TOURNAMENT_GAMES_PER_PAIR):
                    black_eval, white_eval = (
                        (left_eval, right_eval)
                        if i % 2 == 0
                        else (right_eval, left_eval)
                    )
                    outcome = reversi_mcts.run_match(
                        black_eval, white_eval, TOURNAMENT_SIMS
                    )
                    if i % 2 == 0:
                        black_name, white_name = left["name"], right["name"]
                    else:
                        black_name, white_name = right["name"], left["name"]
                    if outcome > 0:
                        scores[black_name] += 1.0
                    elif outcome < 0:
                        scores[white_name] += 1.0
                    else:
                        scores[black_name] += 0.5
                        scores[white_name] += 0.5
                    done += 1
                if done % max(1, total_games // 4) == 0 or done == total_games:
                    log_event(
                        "tournament",
                        "progress",
                        games=f"{done}/{total_games}",
                    )
            ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return ranking, total_games
        except Exception:
            log_event("tournament", "run-error", error=traceback.format_exc().strip())
            return None, 0
        finally:
            if batch_eval is not None:
                batch_eval.resume()
                log_event("tournament", "resume-selfplay")

    loop = asyncio.get_running_loop()
    ranking, total_games = await loop.run_in_executor(None, run_games)
    if ranking is None:
        log_event("tournament", "aborted")
        return
    winner_name, winner_score = ranking[0]
    entrants_by_name = {entrant["name"]: entrant for entrant in entrants}
    winner = entrants_by_name[winner_name]
    max_points = (len(entrants) - 1) * TOURNAMENT_GAMES_PER_PAIR
    win_rate = winner_score / max_points if max_points > 0 else 0.0
    log_event(
        "tournament",
        "result",
        winner=winner_name,
        score=f"{winner_score:.1f}/{max_points}",
        win_rate=f"{win_rate:.0%}",
    )

    if not winner["is_champion"] and win_rate >= WIN_THRESHOLD:
        log_event("tournament", "promote", threshold=f"{WIN_THRESHOLD:.0%}")
        with lock:
            export_model(winner["model"], CHAMPION_BIN)
            _record_champion(winner["model"], win_rate, winner["loss"])
        log_event("tournament", "promoted", path=CHAMPION_BIN.name, winner=winner_name)
    else:
        log_event("tournament", "retain-champion")


def _iter_params(model: AlphaZeroNet):
    """Yield (flat_key, array) pairs from a model."""
    arrays: dict = {}
    _collect_params(model.parameters(), "", arrays)
    yield from arrays.items()


def _iter_param_shapes(model: AlphaZeroNet):
    """Yield (flat_key, shape) pairs without materializing arrays."""
    shapes: dict = {}
    _collect_param_shapes(model.parameters(), "", shapes)
    yield from shapes.items()


def _load_compatible_weights(
    model: AlphaZeroNet,
    weights: list[tuple[str, mx.array]],
) -> int:
    """Load only tensors whose names and shapes match the current model."""
    current_shapes = dict(_iter_param_shapes(model))
    compatible: list[tuple[str, mx.array]] = []
    for name, arr in weights:
        target_shape = current_shapes.get(name)
        if target_shape is None:
            continue
        if tuple(arr.shape) != tuple(target_shape):
            continue
        compatible.append((name, arr))
    if compatible:
        model.load_weights(compatible, strict=False)
    return len(compatible)


def _record_champion(
    model: AlphaZeroNet,
    win_rate: float,
    loss: float = float("inf"),
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
    _save_checkpoint(model, snap_path)

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

    history_file.write_text(json.dumps(history, indent=2))


# --- checkpoint helpers ------------------------------------------------------


def _load_best_checkpoint(model: AlphaZeroNet) -> tuple[int, float]:
    """Load the strongest known checkpoint for continued training.
    Prefer champion.bin because promotion is strength-based; fall back to the
    latest .npz checkpoint only when no deployed champion exists.
    Returns (step, loss).
    """

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    if CHAMPION_BIN.exists():
        try:
            loaded = _load_model_from_bin(CHAMPION_BIN)
            _load_compatible_weights(
                model, [(k, mx.array(v)) for k, v in _iter_params(loaded)]
            )
            mx.eval(model.parameters())
            log_event("train", "resume", source="champion.bin", step=0)
            return 0, float("inf")
        except Exception as e:
            log_event("train", "resume-failed", source="champion.bin", error=repr(e))

    checkpoints = sorted(
        WEIGHTS_DIR.glob("iter_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0,
    )
    if checkpoints:
        latest = checkpoints[-1]
        data = np.load(str(latest))
        weights = [(k, mx.array(data[k])) for k in data.files]
        applied = _load_compatible_weights(model, weights)
        mx.eval(model.parameters())
        try:
            step = int(latest.stem.split("_")[1])
            loss = _loss(latest)
        except (IndexError, ValueError):
            step, loss = 0, float("inf")
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


def _collect_params(obj, prefix: str, arrays: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_params(v, f"{prefix}.{k}" if prefix else k, arrays)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect_params(v, f"{prefix}.{i}" if prefix else str(i), arrays)
    elif hasattr(obj, "shape"):
        arrays[prefix] = np.array(obj)


def _collect_param_shapes(obj, prefix: str, shapes: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_param_shapes(v, f"{prefix}.{k}" if prefix else k, shapes)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect_param_shapes(v, f"{prefix}.{i}" if prefix else str(i), shapes)
    elif hasattr(obj, "shape"):
        shapes[prefix] = tuple(obj.shape)


def _save_checkpoint(model: AlphaZeroNet, path: Path) -> None:
    arrays: dict = {}
    _collect_params(model.parameters(), "", arrays)
    np.savez(str(path), **arrays)
    log_event("weights", "saved-checkpoint", tensors=len(arrays), path=path.name)


def _publish_selfplay_weights(model: AlphaZeroNet) -> None:
    tmp = SELFPLAY_MODEL_BIN.with_suffix(".bin.tmp")
    export_model(model, tmp)
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


def _load_weights_into_model(
    model: AlphaZeroNet,
    path: Path,
    lock: threading.Lock | None = None,
) -> None:
    loaded = (
        _load_model_from_bin(path)
        if path.suffix == ".bin"
        else _load_model_from_checkpoint(path)
    )
    weights = [(k, mx.array(v)) for k, v in _iter_params(loaded)]
    if lock is None:
        _load_compatible_weights(model, weights)
        mx.eval(model.parameters())
    else:
        with lock:
            _load_compatible_weights(model, weights)
            mx.eval(model.parameters())


def _spawn_selfplay_processes(
    count: int,
    workers_per_process: int,
    positions_per_shard: int,
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
    selfplay_workers: int,
    positions_per_shard: int,
) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    model = AlphaZeroNet()
    lock = threading.Lock()
    source = _resolve_selfplay_source()
    if source is not None:
        _load_weights_into_model(model, source, lock)
        log_event(
            "selfplay",
            "weights-loaded",
            process=process_index,
            source=source.name,
        )
    else:
        log_event("selfplay", "weights-random-init", process=process_index)

    batch_eval = SyncBatchEval(model, batch_size=EVAL_BATCH_SIZE, lock=lock)
    shard_writer = ReplayShardWriter(
        REPLAY_SPOOL_DIR,
        producer_id=f"sp{process_index}_pid{os.getpid()}",
        positions_per_shard=positions_per_shard,
    )
    stop_event = asyncio.Event()
    stop_flag = [False]
    event_loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=selfplay_workers)
    games_counters = [[0] for _ in range(selfplay_workers)]
    weight_state = {"mtime_ns": None}
    refresh_lock = threading.Lock()

    def refresh_weights() -> None:
        source = _resolve_selfplay_source()
        if source is None:
            return
        stat = source.stat()
        mtime_ns = stat.st_mtime_ns
        if weight_state["mtime_ns"] == mtime_ns:
            return
        if not refresh_lock.acquire(blocking=False):
            return
        try:
            if weight_state["mtime_ns"] == mtime_ns:
                return
            _load_weights_into_model(model, source, lock)
            weight_state["mtime_ns"] = mtime_ns
            log_event(
                "selfplay",
                "weights-reloaded",
                process=process_index,
                source=source.name,
            )
        finally:
            refresh_lock.release()

    if source is not None:
        weight_state["mtime_ns"] = source.stat().st_mtime_ns

    def _shutdown(signum, frame):
        log_event("selfplay", "signal", process=process_index, signum=signum)
        stop_flag[0] = True
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    worker_futures = [
        event_loop.run_in_executor(
            executor,
            _self_play_worker,
            i,
            batch_eval,
            shard_writer,
            games_counters[i],
            stop_flag,
            refresh_weights,
        )
        for i in range(selfplay_workers)
    ]

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
    finally:
        stop_flag[0] = True
        shard_writer.flush()
        batch_eval.stop()
        executor.shutdown(wait=False)
        for future in worker_futures:
            future.cancel()


async def _run_learner(
    selfplay_processes: int,
    selfplay_workers: int,
    positions_per_shard: int,
    inline_workers: int = 0,
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

    model = AlphaZeroNet()
    replay = ReplayBuffer(max_size=500_000)
    stop_event = asyncio.Event()
    lock = threading.Lock()
    children: list[subprocess.Popen] = []

    # Restore replay buffer from previous run if available
    replay.load(REPLAY_PATH)

    log_event(
        "system",
        "startup",
        selfplay_processes=selfplay_processes,
        inline_workers=inline_workers,
        simulations=SIMULATIONS,
        eval_batch=EVAL_BATCH_SIZE,
        min_replay=MIN_REPLAY,
        train_ratio=TRAIN_RATIO,
        tournament_every=TOURNAMENT_EVERY,
    )
    next_tournament_step = [TOURNAMENT_EVERY]

    inline_executor = None
    inline_batch_eval = None
    if inline_workers > 0:
        inline_batch_eval = SyncBatchEval(model, batch_size=EVAL_BATCH_SIZE, lock=lock)
        stop_flag = [False]
        games_counters = [[0] for _ in range(inline_workers)]
        inline_executor = ThreadPoolExecutor(max_workers=inline_workers)
        event_loop = asyncio.get_event_loop()
        for i in range(inline_workers):
            event_loop.run_in_executor(
                inline_executor,
                _self_play_worker,
                i,
                inline_batch_eval,
                replay,
                games_counters[i],
                stop_flag,
            )

    def _shutdown(signum, frame):
        """SIGTERM handler: save state and stop gracefully."""
        log_event("system", "signal", signum=signum, action="save-and-stop")
        replay.save(REPLAY_PATH)
        stop_event.set()
        if inline_workers > 0:
            stop_flag[0] = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if selfplay_processes > 0:
        children = _spawn_selfplay_processes(
            selfplay_processes, selfplay_workers, positions_per_shard
        )

    try:
        await train_loop(model, replay, stop_event, lock, next_tournament_step)
    except KeyboardInterrupt:
        log_event("system", "keyboard-interrupt")
        replay.save(REPLAY_PATH)
    finally:
        stop_event.set()
        if inline_workers > 0:
            stop_flag[0] = True
            inline_batch_eval.stop()
            inline_executor.shutdown(wait=False)
        ingest_replay_shards(replay, REPLAY_SPOOL_DIR, limit=10_000)
        replay.save(REPLAY_PATH)
        _stop_child_processes(children)
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)


def _default_selfplay_workers(processes: int) -> int:
    total_workers = N_WORKERS
    return max(1, total_workers // max(1, processes))


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
    parser.add_argument("--selfplay-workers", type=int, default=None)
    parser.add_argument("--inline-workers", type=int, default=0)
    parser.add_argument(
        "--positions-per-shard",
        type=int,
        default=DEFAULT_SHARD_POSITIONS,
    )
    parser.add_argument("--process-index", type=int, default=0)
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    workers = args.selfplay_workers
    if workers is None:
        workers = _default_selfplay_workers(args.selfplay_processes)
    if args.role == "selfplay":
        await _run_selfplay_process(
            process_index=args.process_index,
            selfplay_workers=workers,
            positions_per_shard=args.positions_per_shard,
        )
        return
    await _run_learner(
        selfplay_processes=args.selfplay_processes,
        selfplay_workers=workers,
        positions_per_shard=args.positions_per_shard,
        inline_workers=args.inline_workers,
    )


if __name__ == "__main__":
    asyncio.run(main())
