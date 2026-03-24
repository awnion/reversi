"""
Run with: uv run python -m train.loop

Architecture:
- N_WORKERS self-play threads generate game records into the replay buffer.
- One background model-thread (SyncBatchEval) batches leaf evaluations → GPU.
- The asyncio train loop runs gradient steps, rate-limited to TRAIN_RATIO new
  positions per step so it can't outrun data generation.
- Every TOURNAMENT_EVERY training steps a mini-tournament pits the current
  model against the stored champion; if it wins, it becomes the new champion.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import struct
import threading
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
TOURNAMENT_GAMES_PER_PAIR = 8  # games for each pairing inside the tournament pool
TOURNAMENT_SIMS = 200  # MCTS simulations per move in tournament
WIN_THRESHOLD = 0.55  # win rate required to crown a new champion
CHAMPION_HISTORY = 5  # keep this many past champion snapshots

# Replay persistence
REPLAY_PATH = WEIGHTS_DIR / "replay_buffer.npz"
REPLAY_SAVE_EVERY = 1_000  # save replay buffer every N training steps
PID_FILE = WEIGHTS_DIR / "train.pid"


def log_event(component: str, message: str, **fields) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"{stamp} [{component}] {message}{suffix}", flush=True)


# --- self-play worker --------------------------------------------------------


def _self_play_worker(
    worker_id: int,
    batch_eval: SyncBatchEval,
    replay: ReplayBuffer,
    games_counter: list,
    stop_flag: list,
) -> None:
    """Runs in a ThreadPoolExecutor thread."""
    import reversi_mcts

    w = reversi_mcts.MctsWorker(SIMULATIONS)
    while not stop_flag[0]:
        record = w.run_game(batch_eval.evaluate)
        replay.add(record)
        games_counter[0] += 1
        if games_counter[0] % 25 == 0:
            log_event(
                "selfplay",
                "worker-progress",
                worker=worker_id,
                games=games_counter[0],
                replay=len(replay),
            )


# --- training loop -----------------------------------------------------------


async def train_loop(
    model: AlphaZeroNet,
    replay: ReplayBuffer,
    stop_event: asyncio.Event,
    lock: threading.Lock,
    next_tournament_step: list,
    batch_eval: SyncBatchEval,
) -> None:
    opt = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    step, best_loss = _load_best_checkpoint(model)
    replay_at_last_step = replay.total_added

    while not stop_event.is_set():
        # Wait for enough data before starting
        if len(replay) < MIN_REPLAY:
            await asyncio.sleep(0.1)
            continue

        # Rate-limit: don't train faster than data comes in
        new_positions = replay.total_added - replay_at_last_step
        if new_positions < TRAIN_RATIO:
            await asyncio.sleep(0.005)
            continue
        replay_at_last_step = replay.total_added

        (
            planes,
            policies,
            outcomes,
            policy_weights,
            value_weights,
        ) = replay.sample(BATCH_SIZE)
        with lock:
            loss, grads = loss_and_grad(
                model,
                planes,
                policies,
                outcomes,
                policy_weights,
                value_weights,
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

        if step % LOG_EVERY == 0:
            log_event(
                "train",
                "step",
                step=step,
                loss=f"{loss_val:.4f}",
                replay=len(replay),
            )

        save = step % CHECKPOINT_EVERY == 0
        if loss_val < best_loss:
            best_loss = loss_val
            save = True

        if save:
            path = WEIGHTS_DIR / f"iter_{step:06d}_loss{loss_val:.4f}.npz"
            _save_checkpoint(model, path)
            log_event("train", "checkpoint", step=step, path=path.name)
            _prune_checkpoints()

        # Periodically persist replay buffer so restarts don't lose data
        if step % REPLAY_SAVE_EVERY == 0:
            replay.save(REPLAY_PATH)

        # Trigger tournament when step crosses the next scheduled boundary
        if step >= next_tournament_step[0]:
            next_tournament_step[0] = step + TOURNAMENT_EVERY
            task = asyncio.create_task(
                _run_mini_tournament(model, lock, best_loss, batch_eval)
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
        entrant["name"]: _make_eval_fn(entrant["model"], lock) for entrant in entrants
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


async def main() -> None:
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

    # Restore replay buffer from previous run if available
    replay.load(REPLAY_PATH)

    batch_eval = SyncBatchEval(model, batch_size=EVAL_BATCH_SIZE, lock=lock)

    log_event(
        "system",
        "startup",
        workers=N_WORKERS,
        simulations=SIMULATIONS,
        eval_batch=EVAL_BATCH_SIZE,
        min_replay=MIN_REPLAY,
        train_ratio=TRAIN_RATIO,
        tournament_every=TOURNAMENT_EVERY,
    )

    stop_flag = [False]
    games_counters = [[0] for _ in range(N_WORKERS)]
    executor = ThreadPoolExecutor(max_workers=N_WORKERS)
    event_loop = asyncio.get_event_loop()
    next_tournament_step = [TOURNAMENT_EVERY]

    def _shutdown(signum, frame):
        """SIGTERM handler: save state and stop gracefully."""
        log_event("system", "signal", signum=signum, action="save-and-stop")
        replay.save(REPLAY_PATH)
        stop_flag[0] = True
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)

    worker_futures = [
        event_loop.run_in_executor(
            executor,
            _self_play_worker,
            i,
            batch_eval,
            replay,
            games_counters[i],
            stop_flag,
        )
        for i in range(N_WORKERS)
    ]

    try:
        await asyncio.gather(
            train_loop(
                model, replay, stop_event, lock, next_tournament_step, batch_eval
            ),
            *worker_futures,
        )
    except KeyboardInterrupt:
        log_event("system", "keyboard-interrupt")
        replay.save(REPLAY_PATH)
    finally:
        stop_flag[0] = True
        stop_event.set()
        batch_eval.stop()
        executor.shutdown(wait=False)
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
