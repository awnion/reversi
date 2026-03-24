from __future__ import annotations

from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self._boards_black = np.zeros(max_size, dtype=np.uint64)
        self._boards_white = np.zeros(max_size, dtype=np.uint64)
        self._is_black = np.zeros(max_size, dtype=bool)
        self._policies = np.zeros((max_size, 64), dtype=np.float32)
        self._outcomes = np.zeros(max_size, dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._total_added = 0

    def add(self, record) -> None:
        """Add all positions from a GameRecord (list of dicts from MctsWorker)."""
        for pos in record:
            i = self._ptr
            self._boards_black[i] = pos["board_black"]
            self._boards_white[i] = pos["board_white"]
            self._is_black[i] = pos["is_black"]
            self._policies[i] = pos["mcts_policy"]
            self._outcomes[i] = pos["outcome"]
            self._ptr = (self._ptr + 1) % self.max_size
            self._size = min(self._size + 1, self.max_size)
            self._total_added += 1

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            self._boards_black[idx],
            self._boards_white[idx],
            self._is_black[idx],
            self._policies[idx],
            self._outcomes[idx],
        )

    def save(self, path: Path) -> None:
        """Persist buffer to disk (npz). Safe to call while workers are adding."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            str(tmp),
            boards_black=self._boards_black[: self._size],
            boards_white=self._boards_white[: self._size],
            is_black=self._is_black[: self._size],
            policies=self._policies[: self._size],
            outcomes=self._outcomes[: self._size],
            ptr=np.array(self._ptr % self._size if self._size else 0),
        )
        tmp.replace(path)
        print(f"[replay] saved {self._size} positions → {path}")

    def load(self, path: Path) -> None:
        """Restore buffer from disk. Called once at startup before workers start."""
        path = Path(path)
        if not path.exists():
            return
        data = np.load(str(path))
        n = len(data["boards_black"])
        if n == 0:
            return
        n = min(n, self.max_size)
        self._boards_black[:n] = data["boards_black"][:n]
        self._boards_white[:n] = data["boards_white"][:n]
        self._is_black[:n] = data["is_black"][:n]
        self._policies[:n] = data["policies"][:n]
        self._outcomes[:n] = data["outcomes"][:n]
        self._size = n
        self._ptr = int(data["ptr"]) % n if n else 0
        print(f"[replay] loaded {n} positions from {path}")

    @property
    def total_added(self) -> int:
        return self._total_added

    def __len__(self) -> int:
        return self._size
