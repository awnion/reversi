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

SIMULATIONS = 50
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
LR_DECAY_FACTOR = 0.1
LR_MIN = 1e-5
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
CHAMPION_BIN = WEIGHTS_DIR / "champion.bin"

# Workers block on threading.Event.wait() (GIL released) most of the time,
# so we can run many more than CPU cores.
N_WORKERS = (os.cpu_count() or 4) * 4

# Larger batches = better GPU utilisation.
EVAL_BATCH_SIZE = 64

# Tournament settings
TOURNAMENT_EVERY = 3_000  # run a mini-tournament every N training steps
TOURNAMENT_GAMES = 40  # games per tournament (use even number)
TOURNAMENT_SIMS = 50  # MCTS simulations per move in tournament
WIN_THRESHOLD = 0.55  # win rate required to crown a new champion
CHAMPION_HISTORY = 5  # keep this many past champion snapshots

# Replay persistence
REPLAY_PATH = WEIGHTS_DIR / "replay_buffer.npz"
REPLAY_SAVE_EVERY = 1_000  # save replay buffer every N training steps
PID_FILE = WEIGHTS_DIR / "train.pid"


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
        if games_counter[0] % 10 == 0:
            print(
                f"[worker {worker_id}] games={games_counter[0]}, replay={len(replay)}"
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

        bb, bw, ib, legal, policies, outcomes = replay.sample(BATCH_SIZE)
        with lock:
            loss, grads = loss_and_grad(model, bb, bw, ib, legal, policies, outcomes)
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
            print(f"[train] LR → {new_lr:.2e} at step={step}")

        if math.isnan(loss_val):
            print(f"[train] NaN loss at step={step}, skipping")
            continue

        if step % LOG_EVERY == 0:
            print(f"[train] step={step}, loss={loss_val:.4f}, replay={len(replay)}")

        save = step % CHECKPOINT_EVERY == 0
        if loss_val < best_loss:
            best_loss = loss_val
            save = True

        if save:
            path = WEIGHTS_DIR / f"iter_{step:06d}_loss{loss_val:.4f}.npz"
            _save_checkpoint(model, path)
            print(f"[train] checkpoint → {path}")
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
                    t.exception() and print(f"[tournament] ERROR: {t.exception()}")
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
    m.load_weights(weights, strict=False)
    mx.eval(m.parameters())
    m.eval()
    return m


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
    """Run TOURNAMENT_GAMES games vs the stored champion.bin.
    Runs in the background without blocking the training loop.
    If the current model achieves WIN_THRESHOLD win rate, it becomes champion.
    """
    import traceback

    import reversi_mcts

    if not CHAMPION_BIN.exists():
        print("[tournament] no champion.bin yet — skipping")
        return

    print(
        f"[tournament] starting {TOURNAMENT_GAMES} games "
        f"(sims={TOURNAMENT_SIMS}) vs champion"
    )
    try:
        champion_model = _load_model_from_bin(CHAMPION_BIN)
    except Exception as e:
        print(f"[tournament] failed to load champion: {e}")
        return

    # Use the same lock for champion eval to avoid concurrent GPU access
    champion_eval = _make_eval_fn(champion_model, lock)

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
    current_eval = _make_eval_fn(snapshot, lock)

    def run_games() -> float:
        if batch_eval is not None:
            batch_eval.pause()
            print("[tournament] self-play paused for GPU exclusivity")
        try:
            score = 0.0
            for i in range(TOURNAMENT_GAMES):
                # Alternate colours to remove first-mover advantage
                if i % 2 == 0:
                    outcome = reversi_mcts.run_match(
                        current_eval, champion_eval, TOURNAMENT_SIMS
                    )
                    score += 1.0 if outcome > 0 else (0.5 if outcome == 0 else 0.0)
                else:
                    outcome = reversi_mcts.run_match(
                        champion_eval, current_eval, TOURNAMENT_SIMS
                    )
                    score += 1.0 if outcome < 0 else (0.5 if outcome == 0 else 0.0)
                g, t = i + 1, TOURNAMENT_GAMES
                print(f"[tournament] game {g}/{t} done, score={score:.1f}")
            return score
        except Exception:
            print(f"[tournament] run_games exception:\n{traceback.format_exc()}")
            return -1.0
        finally:
            if batch_eval is not None:
                batch_eval.resume()
                print("[tournament] self-play resumed")

    loop = asyncio.get_running_loop()
    score = await loop.run_in_executor(None, run_games)
    if score < 0:
        print("[tournament] aborted due to error")
        return
    win_rate = score / TOURNAMENT_GAMES
    print(
        f"[tournament] score={score:.1f}/{TOURNAMENT_GAMES} "
        f"(win_rate={win_rate:.0%}) vs champion"
    )

    if win_rate >= WIN_THRESHOLD:
        print("[tournament] New champion! Exporting champion.bin...")
        with lock:
            export_model(current_model, CHAMPION_BIN)
            _record_champion(snapshot, win_rate, current_loss)
        print("[tournament] champion.bin updated")
    else:
        print("[tournament] champion holds")


def _iter_params(model: AlphaZeroNet):
    """Yield (flat_key, array) pairs from a model."""
    arrays: dict = {}
    _collect_params(model.parameters(), "", arrays)
    yield from arrays.items()


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
    print(
        f"[champion] recorded {snap_path.name} "
        f"(loss={loss_tag}, win_rate={win_rate:.0%})"
    )

    # Prune to keep only the most recent CHAMPION_HISTORY entries
    if len(history) > CHAMPION_HISTORY:
        to_remove = history[:-CHAMPION_HISTORY]
        history = history[-CHAMPION_HISTORY:]
        for old in to_remove:
            old_path = champions_dir / old["file"]
            if old_path.exists():
                old_path.unlink()
                print(f"[champion] pruned {old['file']}")

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
            model.load_weights(
                [(k, mx.array(v)) for k, v in _iter_params(loaded)], strict=False
            )
            mx.eval(model.parameters())
            print("[train] resumed from champion.bin (step=0)")
            return 0, float("inf")
        except Exception as e:
            print(f"[train] failed to load champion.bin: {e}; starting fresh")

    checkpoints = sorted(
        WEIGHTS_DIR.glob("iter_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0,
    )
    if checkpoints:
        latest = checkpoints[-1]
        data = np.load(str(latest))
        weights = [(k, mx.array(data[k])) for k in data.files]
        model.load_weights(weights, strict=False)
        mx.eval(model.parameters())
        try:
            step = int(latest.stem.split("_")[1])
            loss = _loss(latest)
        except (IndexError, ValueError):
            step, loss = 0, float("inf")
        print(f"[train] resumed from {latest.name} (step={step}, loss={loss:.4f})")
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


def _save_checkpoint(model: AlphaZeroNet, path: Path) -> None:
    arrays: dict = {}
    _collect_params(model.parameters(), "", arrays)
    np.savez(str(path), **arrays)
    print(f"[train] saved {len(arrays)} tensors → {path}")


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
            print(f"[train] pruned {p.name}")


# --- entry point -------------------------------------------------------------


async def main() -> None:
    # Guard against duplicate instances
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(old_pid, 0)  # check if process exists
            print(
                f"[train] ERROR: another instance is already running (PID {old_pid}). "
                "Stop it first with: kill -TERM {old_pid}"
            )
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

    print(
        f"Starting {N_WORKERS} self-play workers "
        f"(SIMULATIONS={SIMULATIONS}, eval_batch={EVAL_BATCH_SIZE})"
    )
    print(
        f"MIN_REPLAY={MIN_REPLAY}, TRAIN_RATIO={TRAIN_RATIO}, "
        f"tournament every {TOURNAMENT_EVERY} steps"
    )

    stop_flag = [False]
    games_counters = [[0] for _ in range(N_WORKERS)]
    executor = ThreadPoolExecutor(max_workers=N_WORKERS)
    event_loop = asyncio.get_event_loop()
    next_tournament_step = [TOURNAMENT_EVERY]

    def _shutdown(signum, frame):
        """SIGTERM handler: save state and stop gracefully."""
        print(f"\n[train] received signal {signum}, saving state...")
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
        print("\nStopping...")
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
