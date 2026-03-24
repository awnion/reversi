"""
Run with: uv run python -m train.loop
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .eval_server import SyncBatchEval
from .loss import compute_loss
from .model import AlphaZeroNet
from .replay import ReplayBuffer

SIMULATIONS = 50
BATCH_SIZE = 512
MIN_REPLAY = 5_000
CHECKPOINT_EVERY = 200
KEEP_CHECKPOINTS = 20
LOG_EVERY = 50
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"

# Workers block on threading.Event.wait() (GIL released) most of the time,
# so we can run many more than CPU cores.
N_WORKERS = (os.cpu_count() or 4) * 4

# Larger batches = better GPU utilisation.
EVAL_BATCH_SIZE = 64


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


async def train_loop(
    model: AlphaZeroNet,
    replay: ReplayBuffer,
    stop_event: asyncio.Event,
    lock,
) -> None:
    opt = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    step, best_loss = _load_best_checkpoint(model)

    while not stop_event.is_set():
        if len(replay) < MIN_REPLAY:
            await asyncio.sleep(0.1)
            continue

        bb, bw, ib, policies, outcomes = replay.sample(BATCH_SIZE)
        with lock:
            loss, grads = loss_and_grad(model, bb, bw, ib, policies, outcomes)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)

        step += 1
        loss_val = loss.item()

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

        await asyncio.sleep(0)


def _load_best_checkpoint(model: AlphaZeroNet) -> tuple[int, float]:
    """Load the checkpoint with the lowest loss. Returns (step, loss)."""

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    checkpoints = sorted(WEIGHTS_DIR.glob("iter_*.npz"), key=_loss)
    if not checkpoints:
        return 0, float("inf")

    best = checkpoints[0]
    data = np.load(str(best))
    weights = [(k, mx.array(data[k])) for k in data.files]
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())

    try:
        step = int(best.stem.split("_")[1])
        loss = _loss(best)
    except (IndexError, ValueError):
        step, loss = 0, float("inf")

    print(f"[train] resumed from {best.name} (step={step}, loss={loss:.4f})")
    return step, loss


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


async def main() -> None:
    model = AlphaZeroNet()
    replay = ReplayBuffer(max_size=500_000)
    stop_event = asyncio.Event()

    import threading

    lock = threading.Lock()

    batch_eval = SyncBatchEval(model, batch_size=EVAL_BATCH_SIZE, lock=lock)

    print(
        f"Starting {N_WORKERS} self-play workers "
        f"(SIMULATIONS={SIMULATIONS}, eval_batch={EVAL_BATCH_SIZE})"
    )
    print("MCTS runs in parallel threads; leaves batched → one GPU call per batch")

    stop_flag = [False]
    games_counters = [[0] for _ in range(N_WORKERS)]
    executor = ThreadPoolExecutor(max_workers=N_WORKERS)
    loop = asyncio.get_event_loop()

    worker_futures = [
        loop.run_in_executor(
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
            train_loop(model, replay, stop_event, lock),
            *worker_futures,
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_flag[0] = True
        stop_event.set()
        batch_eval.stop()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
