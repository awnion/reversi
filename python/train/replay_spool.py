from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import numpy as np

from .replay import ReplayBuffer, log_event, materialize_record


class ReplayShardWriter:
    """Accumulate self-play samples and flush them into atomic shard files."""

    def __init__(
        self,
        spool_dir: Path,
        producer_id: str,
        positions_per_shard: int = 4_096,
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.producer_id = producer_id
        self.positions_per_shard = positions_per_shard
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._planes: list[np.ndarray] = []
        self._policies: list[np.ndarray] = []
        self._outcomes: list[np.ndarray] = []
        self._policy_weights: list[np.ndarray] = []
        self._value_weights: list[np.ndarray] = []
        self._positions = 0
        self._lock = Lock()

    def add_record(self, record) -> int:
        planes, policies, outcomes, policy_weights, value_weights = materialize_record(
            record
        )
        count = int(len(planes))
        if count == 0:
            return 0
        should_flush = False
        with self._lock:
            self._planes.append(planes)
            self._policies.append(policies)
            self._outcomes.append(outcomes)
            self._policy_weights.append(policy_weights)
            self._value_weights.append(value_weights)
            self._positions += count
            should_flush = self._positions >= self.positions_per_shard
        if should_flush:
            self.flush()
        return count

    def flush(self) -> int:
        with self._lock:
            if self._positions == 0:
                return 0
            count = self._positions
            planes = np.concatenate(self._planes, axis=0)
            policies = np.concatenate(self._policies, axis=0)
            outcomes = np.concatenate(self._outcomes, axis=0)
            policy_weights = np.concatenate(self._policy_weights, axis=0)
            value_weights = np.concatenate(self._value_weights, axis=0)
            self._planes.clear()
            self._policies.clear()
            self._outcomes.clear()
            self._policy_weights.clear()
            self._value_weights.clear()
            self._positions = 0
        now_ns = time.time_ns()
        basename = f"{self.producer_id}_{now_ns}_{count}"
        tmp_path = self.spool_dir / f"{basename}.tmp"
        final_path = self.spool_dir / f"{basename}.npz"
        with tmp_path.open("wb") as f:
            np.savez_compressed(
                f,
                planes=planes,
                policies=policies,
                outcomes=outcomes,
                policy_weights=policy_weights,
                value_weights=value_weights,
            )
        tmp_path.replace(final_path)
        log_event(
            "replay",
            "shard-written",
            path=final_path.name,
            positions=count,
            producer=self.producer_id,
        )
        return count


def ingest_replay_shards(
    replay: ReplayBuffer,
    spool_dir: Path,
    *,
    limit: int = 32,
) -> int:
    """Import up to `limit` pending shard files into the learner replay buffer."""
    spool_dir = Path(spool_dir)
    if not spool_dir.exists():
        return 0
    imported = 0
    for path in sorted(spool_dir.glob("*.npz"))[:limit]:
        try:
            imported += replay.ingest_shard(path)
        except Exception as exc:
            bad_path = path.with_suffix(path.suffix + ".bad")
            path.replace(bad_path)
            log_event(
                "replay",
                "shard-corrupt",
                path=path.name,
                quarantined=bad_path.name,
                error=type(exc).__name__,
            )
    if imported:
        log_event("replay", "ingested-shards", positions=imported, replay=len(replay))
    return imported
