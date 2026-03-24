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
BATCH_SIZE = 256
MIN_REPLAY = 1_000
CHECKPOINT_EVERY = 200
LOG_EVERY = 100
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"

# Number of parallel MCTS workers.  Each runs in its own thread; MCTS is pure
# Rust (GIL released), so they run truly in parallel on all CPU cores.
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Batch size for the model thread.  Workers fill this before GPU dispatch.
EVAL_BATCH_SIZE = min(N_WORKERS, 16)


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
        if games_counter[0] % 5 == 0:
            print(
                f"[worker {worker_id}] games={games_counter[0]}, replay={len(replay)}"
            )


async def train_loop(
    model: AlphaZeroNet,
    replay: ReplayBuffer,
    stop_event: asyncio.Event,
) -> None:
    opt = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    step = 0
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    while not stop_event.is_set():
        if len(replay) < MIN_REPLAY:
            await asyncio.sleep(0.1)
            continue

        bb, bw, ib, policies, outcomes = replay.sample(BATCH_SIZE)
        loss, grads = loss_and_grad(model, bb, bw, ib, policies, outcomes)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        step += 1
        if step % LOG_EVERY == 0:
            print(f"[train] step={step}, loss={loss.item():.4f}, replay={len(replay)}")

        if step % CHECKPOINT_EVERY == 0:
            path = WEIGHTS_DIR / f"iter_{step:06d}.npz"
            _save_checkpoint(model, path)
            print(f"[train] checkpoint → {path}")

        await asyncio.sleep(0)


def _save_checkpoint(model: AlphaZeroNet, path: Path) -> None:
    flat = model.parameters()
    arrays: dict = {}

    def _collect(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _collect(v, key)
            elif hasattr(v, "shape"):
                arrays[key] = np.array(v)

    _collect(flat)
    np.savez(str(path), **arrays)
    print(f"[train] saved {len(arrays)} tensors → {path}")


async def main() -> None:
    model = AlphaZeroNet()
    replay = ReplayBuffer(max_size=500_000)
    stop_event = asyncio.Event()

    batch_eval = SyncBatchEval(model, batch_size=EVAL_BATCH_SIZE)

    print(
        f"Starting {N_WORKERS} self-play workers "
        f"(SIMULATIONS={SIMULATIONS}, eval_batch={EVAL_BATCH_SIZE})"
    )
    print("MCTS runs in parallel threads; leaves batched → one GPU call per batch")

    stop_flag = [False]
    games_counters = [[0] for _ in range(N_WORKERS)]
    executor = ThreadPoolExecutor(max_workers=N_WORKERS)
    loop = asyncio.get_event_loop()

    # Launch all worker threads via run_in_executor so the asyncio loop stays
    # free to run train_loop between batches.
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
            train_loop(model, replay, stop_event),
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
