from __future__ import annotations

import zlib
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def log_event(component: str, message: str, **fields) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"{stamp} [{component}] {message}{suffix}", flush=True)


def materialize_record(record) -> tuple[np.ndarray, ...]:
    """Convert a raw self-play record into replay tensor batches."""
    from .eval_server import board_to_planes

    planes = []
    policies = []
    outcomes = []
    policy_weights = []
    value_weights = []
    for pos in record:
        planes.append(
            board_to_planes(
                pos["board_black"],
                pos["board_white"],
                pos["is_black"],
                pos["legal"],
            )
        )
        policies.append(pos["mcts_policy"])
        outcomes.append(pos["outcome"])
        policy_weights.append(pos.get("policy_weight", 1.0))
        value_weights.append(pos.get("value_weight", 1.0))
    return (
        np.asarray(planes, dtype=np.float32),
        np.asarray(policies, dtype=np.float32),
        np.asarray(outcomes, dtype=np.float32),
        np.asarray(policy_weights, dtype=np.float32),
        np.asarray(value_weights, dtype=np.float32),
    )


class ReplayBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self._policies = np.zeros((max_size, 64), dtype=np.float32)
        self._outcomes = np.zeros(max_size, dtype=np.float32)
        self._policy_weights = np.ones(max_size, dtype=np.float32)
        self._value_weights = np.ones(max_size, dtype=np.float32)
        # Precomputed input planes: (4, 8, 8) float32 per entry
        # planes[3] = phase plane (popcount/64 repeated spatially)
        self._planes = np.zeros((max_size, 4, 8, 8), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._total_added = 0

    def add(self, record) -> None:
        """Add all positions from a GameRecord (list of dicts from MctsWorker)."""
        self.add_samples(*materialize_record(record))

    def add_samples(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        policy_weights: np.ndarray,
        value_weights: np.ndarray,
    ) -> None:
        """Add a batch of already-materialized samples."""
        n = int(len(planes))
        if n == 0:
            return
        idx = np.arange(self._ptr, self._ptr + n) % self.max_size
        self._planes[idx] = planes
        self._policies[idx] = policies
        self._outcomes[idx] = outcomes
        self._policy_weights[idx] = policy_weights
        self._value_weights[idx] = value_weights
        self._ptr = int((self._ptr + n) % self.max_size)
        self._size = min(self._size + n, self.max_size)
        self._total_added += n

    def ingest_shard(self, path: Path, *, unlink: bool = True) -> int:
        """Load one replay shard written by a self-play producer."""
        path = Path(path)
        with np.load(str(path)) as data:
            required = {
                "planes",
                "policies",
                "outcomes",
                "policy_weights",
                "value_weights",
            }
            if not required.issubset(data.files):
                raise ValueError(f"Replay shard missing fields: {path}")
            planes = np.asarray(data["planes"], dtype=np.float32)
            count = int(len(planes))
            self.add_samples(
                planes,
                np.asarray(data["policies"], dtype=np.float32),
                np.asarray(data["outcomes"], dtype=np.float32),
                np.asarray(data["policy_weights"], dtype=np.float32),
                np.asarray(data["value_weights"], dtype=np.float32),
            )
        if unlink:
            path.unlink(missing_ok=True)
        return count

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            self._planes[idx],
            self._policies[idx],
            self._outcomes[idx],
            self._policy_weights[idx],
            self._value_weights[idx],
        )

    def save(self, path: Path) -> None:
        """Persist buffer to disk (npz). Safe to call while workers are adding."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                policies=self._policies[: self._size],
                outcomes=self._outcomes[: self._size],
                policy_weights=self._policy_weights[: self._size],
                value_weights=self._value_weights[: self._size],
                planes=self._planes[: self._size],
                ptr=np.array(self._ptr % self._size if self._size else 0),
            )
        tmp.replace(path)
        log_event("replay", "saved", positions=self._size, path=path.name)

    def load(self, path: Path) -> None:
        """Restore buffer from disk. Called once at startup before workers start."""
        path = Path(path)
        if not path.exists():
            return
        try:
            with np.load(str(path)) as data:
                if "planes" not in data:
                    log_event("replay", "ignore-stale-schema", path=path.name)
                    return
                n = len(data["planes"])
                if n == 0:
                    return
                n = min(n, self.max_size)
                loaded_planes = data["planes"][:n]
                if loaded_planes.shape[1:] == (3, 8, 8):
                    # Old format: (N, 3, 8, 8) uint8 + separate phase → convert
                    log_event("replay", "convert-planes-3-to-4", positions=n)
                    spatial = loaded_planes.astype(np.float32)
                    if "phase" in data:
                        phase_vals = data["phase"][:n].astype(np.float32)
                    else:
                        # No phase data; reconstruct zeros.
                        phase_vals = np.zeros(n, dtype=np.float32)
                    phase_plane = np.repeat(
                        phase_vals[:, None, None], 64, axis=-1
                    ).reshape(n, 1, 8, 8)
                    self._planes[:n] = np.concatenate([spatial, phase_plane], axis=1)
                elif loaded_planes.shape[1:] == (4, 8, 8):
                    self._planes[:n] = loaded_planes
                else:
                    log_event(
                        "replay",
                        "ignore-bad-planes-shape",
                        shape=loaded_planes.shape,
                    )
                    return
                self._policies[:n] = data["policies"][:n]
                self._outcomes[:n] = data["outcomes"][:n]
                self._policy_weights[:n] = data["policy_weights"][:n]
                self._value_weights[:n] = data["value_weights"][:n]
                self._size = n
                self._ptr = int(data["ptr"]) % n if n else 0
        except (OSError, ValueError, BadZipFile, EOFError, RuntimeError, zlib.error):
            bad_path = path.with_suffix(path.suffix + ".bad")
            path.replace(bad_path)
            log_event(
                "replay",
                "corrupt-buffer",
                path=path.name,
                quarantined=bad_path.name,
            )
            return
        log_event("replay", "loaded", positions=n, path=path.name)

    @property
    def total_added(self) -> int:
        return self._total_added

    def __len__(self) -> int:
        return self._size
