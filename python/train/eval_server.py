import asyncio
import queue
import threading

import mlx.core as mx
import numpy as np

from .model import AlphaZeroNet

_SHIFTS = np.arange(64, dtype=np.uint64)


def board_to_planes(
    board_black: int, board_white: int, is_black: bool, legal: int
) -> np.ndarray:
    """Convert bitboard ints to 3×8×8 float32 planes (vectorised)."""
    my_bits = board_black if is_black else board_white
    opp_bits = board_white if is_black else board_black
    bits = np.array([my_bits, opp_bits, legal], dtype=np.uint64)
    return (
        ((bits[:, None] >> _SHIFTS) & np.uint64(1)).astype(np.float32).reshape(3, 8, 8)
    )


class LeafEvalServer:
    def __init__(
        self,
        model: AlphaZeroNet,
        batch_size: int = 64,
        timeout_ms: float = 2.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def evaluate(
        self,
        board_black: int,
        board_white: int,
        is_black: bool,
        legal: int,
    ) -> tuple[list[float], float]:
        """Submit a position for evaluation. Suspends until the batch is processed."""
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        planes = board_to_planes(board_black, board_white, is_black, legal)
        await self._queue.put((planes, fut))
        return await fut

    async def run(self):
        """Main loop: drain queue in batches, run inference, resolve futures."""
        self._running = True
        while self._running:
            # Collect up to batch_size items, with timeout
            items = []
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self.timeout)
                items.append(item)
            except TimeoutError:
                await asyncio.sleep(0)
                continue

            # Drain remaining without blocking
            while len(items) < self.batch_size:
                try:
                    items.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if not items:
                await asyncio.sleep(0.001)
                continue

            # In rare cases, queue.get() might return None if we put a sentinel
            if items[0] is None:
                break

            planes_list, futures = zip(*items, strict=False)
            batch = mx.array(np.stack(planes_list))  # (N, 3, 8, 8)
            policy_logits, values = self.model(batch)
            mx.eval(policy_logits, values)

            # Distribute results back
            for fut, p_log, v in zip(futures, policy_logits, values, strict=False):
                if not fut.done():
                    fut.set_result((p_log, v))

    def stop(self):
        self._running = False


class SyncBatchEval:
    """
    Thread-safe batched leaf evaluator.

    Architecture
    ------------
    * N MCTS worker threads call ``evaluate()`` at each leaf node.
    * ``evaluate()`` enqueues the request and calls ``threading.Event.wait()``,
      which **releases the GIL** while blocking — so other workers can keep
      running their own MCTS trees in parallel.
    * A single background model-thread drains the queue, assembles a batch,
      runs one Metal forward pass, and sets each worker's event.
    * The GIL naturally serialises model access: workers hold it only during
      the tiny queue-put call; the model thread holds it during inference.

    Result: N-way MCTS parallelism + batched GPU inference, no async needed.
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        batch_size: int = 16,
        timeout: float = 0.005,
        lock=None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.lock = lock or threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._running = True
        self._paused = threading.Event()  # set = paused, clear = running
        self._thread = threading.Thread(target=self._model_loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pause GPU inference (tournament uses GPU exclusively)."""
        self._paused.set()

    def resume(self) -> None:
        """Resume GPU inference after tournament."""
        self._paused.clear()

    def _model_loop(self) -> None:
        """
        Dedicated inference thread: batches requests → one GPU call → notify workers.
        """
        while self._running:
            if self._paused.is_set():
                import time

                time.sleep(0.01)
                continue
            items: list = []
            try:
                items.append(self._q.get(timeout=self.timeout))
            except queue.Empty:
                continue

            while len(items) < self.batch_size:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break

            planes_batch = np.stack([it[0] for it in items])  # (N, 3, 8, 8)
            x = mx.array(planes_batch)

            with self.lock:
                self.model.eval()
                policy_logits, values = self.model(x)
                mx.eval(policy_logits, values)
                self.model.train()

            policies_np = np.array(policy_logits)
            values_np = np.array(values)

            for i, (_, event, result_box) in enumerate(items):
                result_box.append((policies_np[i].tolist(), float(values_np[i])))
                event.set()

    def evaluate(
        self, board_black: int, board_white: int, is_black: bool, legal: int
    ) -> tuple[list[float], float]:
        """
        Called from worker threads inside Rust MCTS via PyO3 callback.
        ``threading.Event.wait()`` releases the GIL while blocking.
        """
        planes = board_to_planes(board_black, board_white, is_black, legal)
        result_box: list = []
        event = threading.Event()
        self._q.put((planes, event, result_box))
        event.wait()
        return result_box[0]

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
