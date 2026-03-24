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

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            self._boards_black[idx],
            self._boards_white[idx],
            self._is_black[idx],
            self._policies[idx],
            self._outcomes[idx],
        )

    def __len__(self) -> int:
        return self._size
