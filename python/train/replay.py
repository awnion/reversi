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
    from .planes import board_to_planes

    planes = []
    policies = []
    outcomes = []
    margins = []
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
        # Records from before the margin was recorded have none: NaN, and the
        # margin loss skips them.
        margins.append(pos.get("margin", np.nan))
        policy_weights.append(pos.get("policy_weight", 1.0))
        value_weights.append(pos.get("value_weight", 1.0))
    return (
        np.asarray(planes, dtype=np.float32),
        np.asarray(policies, dtype=np.float32),
        np.asarray(outcomes, dtype=np.float32),
        np.asarray(margins, dtype=np.float32),
        np.asarray(policy_weights, dtype=np.float32),
        np.asarray(value_weights, dtype=np.float32),
    )


def random_symmetries(
    planes: np.ndarray, policies: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Each position in one of the 8 symmetries of the board, chosen at random.

    The rules do not change under rotations and reflections of the board, so
    a position and its policy can be turned together; the value and the
    margin stay the same. Square ``i`` is row ``i // 8``, column ``i % 8`` in
    both the planes and the policy.
    """
    n = len(planes)
    grid = policies.reshape(n, 8, 8)
    out_planes = np.empty_like(planes)
    out_grid = np.empty_like(grid)
    kinds = rng.integers(0, 8, size=n)
    for kind in range(8):
        rows = np.flatnonzero(kinds == kind)
        if rows.size == 0:
            continue
        p, g = planes[rows], grid[rows]
        if kind & 4:  # reflect in the main diagonal
            p, g = p.swapaxes(2, 3), g.swapaxes(1, 2)
        turns = kind & 3
        out_planes[rows] = np.rot90(p, turns, axes=(2, 3))
        out_grid[rows] = np.rot90(g, turns, axes=(1, 2))
    return out_planes, out_grid.reshape(n, 64)


class ReplayBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self._policies = np.zeros((max_size, 64), dtype=np.float32)
        self._outcomes = np.zeros(max_size, dtype=np.float32)
        self._margins = np.full(max_size, np.nan, dtype=np.float32)
        self._policy_weights = np.ones(max_size, dtype=np.float32)
        self._value_weights = np.ones(max_size, dtype=np.float32)
        # Precomputed input planes: (4, 8, 8) float32 per entry
        # planes[3] = phase plane (popcount/64 repeated spatially)
        self._planes = np.zeros((max_size, 4, 8, 8), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._total_added = 0
        self._rng = np.random.default_rng()

    def add(self, record) -> None:
        """Add all positions from a GameRecord (list of dicts from MctsWorker)."""
        self.add_samples(*materialize_record(record))

    def add_samples(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        margins: np.ndarray,
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
        self._margins[idx] = margins
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
            margins = (
                np.asarray(data["margins"], dtype=np.float32)
                if "margins" in data.files
                else np.full(count, np.nan, dtype=np.float32)
            )
            self.add_samples(
                planes,
                np.asarray(data["policies"], dtype=np.float32),
                np.asarray(data["outcomes"], dtype=np.float32),
                margins,
                np.asarray(data["policy_weights"], dtype=np.float32),
                np.asarray(data["value_weights"], dtype=np.float32),
            )
        if unlink:
            path.unlink(missing_ok=True)
        return count

    def sample(self, batch_size: int, *, symmetries: bool = True):
        """A random batch: planes, policies, outcomes, margins and the two
        sample weights; each position turned to a random symmetry."""
        idx = self._rng.choice(self._size, batch_size, replace=False)
        planes, policies = self._planes[idx], self._policies[idx]
        if symmetries:
            planes, policies = random_symmetries(planes, policies, self._rng)
        return (
            planes,
            policies,
            self._outcomes[idx],
            self._margins[idx],
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
                margins=self._margins[: self._size],
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
                if "margins" in data:
                    self._margins[:n] = data["margins"][:n]
                self._policy_weights[:n] = data["policy_weights"][:n]
                self._value_weights[:n] = data["value_weights"][:n]
                self._size = n
                # A buffer that is not full yet (also one loaded into a larger
                # capacity) continues after its last entry; a full one at the
                # oldest entry.
                self._ptr = int(data["ptr"]) % n if n == self.max_size else n
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
