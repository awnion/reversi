"""Train a network of any size on saved self-play positions (no self-play).

Answers one question: is the 4×32 network the limit, or the data? The newest
positions are held out; the network trained on the rest is compared on them
with the self-play champion (``--baseline``), which has not seen them either.
Positions come from a saved replay buffer (``--replay``) or from a directory
of self-play shards (``--shards``, e.g. a ``reversi-selfplay selfplay`` run
with a fixed champion).

    uv run python -m train.supervised --shards ../weights/collect/replay_spool \\
        --blocks 6 --channels 48 --out ../weights/sup-6x48 \\
        --baseline ../weights/ramo0/champion.bin

``--q-weight`` sets the share of the search value in the value target
(self-play records 0.2; shards with ``search_values`` can be remixed).
Writes ``best.bin`` / ``best.npz`` (lowest held-out loss) and ``log.jsonl``
to ``--out``. PyTorch only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import weights as weights_io
from .backend.torch_backend import TorchModel
from .replay import ReplayBuffer, log_event

# The search-value share self-play mixes into `outcomes`
# (`VALUE_TARGET_SEARCH_WEIGHT` in rust/mcts/src/batch.rs).
RECORDED_Q_WEIGHT = 0.2
SCALARS = ("outcomes", "margins", "policy_weights", "value_weights")


def symmetry_permutations() -> np.ndarray:
    """(8, 64): square ``i`` of the turned board is square ``perm[i]`` of the
    original; the same 8 turns as `replay.random_symmetries`."""
    grid = np.arange(64).reshape(8, 8)
    perms = []
    for kind in range(8):
        g = grid.T if kind & 4 else grid
        perms.append(np.rot90(g, kind & 3).reshape(64))
    return np.stack(perms)


def pack(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compact form for the GPU, ~170 bytes a position: the three 0/1 planes
    as bitboards, the disc count, policies in half precision."""
    planes = arrays["planes"]
    bits = np.packbits(
        planes[:, :3].reshape(-1, 3, 64) > 0.5, axis=2, bitorder="little"
    )
    packed = {
        "bits": bits.view(np.int64).reshape(-1, 3),
        "discs": np.rint(planes[:, 3, 0, 0] * 64).astype(np.uint8),
        "policies": arrays["policies"].astype(np.float16),
    }
    for name in (*SCALARS, "search_values"):
        packed[name] = arrays[name].astype(np.float32)
    return packed


def from_replay(path: Path) -> dict[str, np.ndarray]:
    """A saved buffer, oldest position first."""
    replay = ReplayBuffer(max_size=2_000_000)
    replay.load(path)
    size = len(replay)
    ptr = replay._ptr if size == replay.max_size else size
    order = (ptr + np.arange(size)) % size
    arrays = {
        "planes": replay._planes[order],
        "policies": replay._policies[order],
        "outcomes": replay._outcomes[order],
        "margins": replay._margins[order],
        "policy_weights": replay._policy_weights[order],
        "value_weights": replay._value_weights[order],
        "search_values": np.full(size, np.nan, dtype=np.float32),
    }
    return pack(arrays)


def from_shards(directory: Path) -> dict[str, np.ndarray]:
    """Every shard in `directory`, oldest file first."""
    parts: list[dict[str, np.ndarray]] = []
    for path in sorted(directory.glob("*.npz"), key=lambda p: p.stat().st_mtime):
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        if "search_values" not in arrays:
            arrays["search_values"] = np.full(
                len(arrays["outcomes"]), np.nan, np.float32
            )
        parts.append(pack(arrays))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def value_targets(data: dict[str, np.ndarray], q_weight: float) -> np.ndarray:
    """Outcomes remixed to `q_weight` search value where it was recorded."""
    q, mixed = data["search_values"], data["outcomes"]
    known = np.isfinite(q)
    z = np.where(
        known,
        (mixed - RECORDED_Q_WEIGHT * np.nan_to_num(q)) / (1 - RECORDED_Q_WEIGHT),
        mixed,
    )
    return np.where(
        known, (1 - q_weight) * z + q_weight * np.nan_to_num(q), mixed
    ).astype(np.float32)


class DeviceData:
    """All positions on the GPU, the newest `holdout` held out. Held-out
    value targets stay as recorded, so losses compare across `q_weight`."""

    def __init__(
        self, data: dict[str, np.ndarray], holdout: int, q_weight: float, device
    ):
        n = len(data["outcomes"])
        cut = n - holdout
        train_values = value_targets(data, q_weight)[:cut]

        def put(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        self.train = {k: put(v[:cut]) for k, v in data.items() if k != "search_values"}
        self.train["outcomes"] = put(train_values)
        self.held = {k: put(v[cut:]) for k, v in data.items() if k != "search_values"}
        self.train_size, self.held_size = cut, holdout
        self.perms = put(symmetry_permutations())
        self.shifts = torch.arange(64, device=device)
        self.device = device

    def _planes(self, part: dict, idx, perm=None) -> torch.Tensor:
        bits = (part["bits"][idx].unsqueeze(-1) >> self.shifts) & 1  # (B, 3, 64)
        n = bits.shape[0]
        phase = (part["discs"][idx].float() / 64).view(n, 1, 1).expand(n, 1, 64)
        planes = torch.cat([bits.float(), phase], 1)
        if perm is not None:
            planes = planes.gather(2, perm.unsqueeze(1).expand(n, 4, 64))
        return planes.view(n, 4, 8, 8).contiguous(memory_format=torch.channels_last)

    def _rest(self, part: dict, idx, perm=None) -> list[torch.Tensor]:
        policies = part["policies"][idx].float()
        if perm is not None:
            policies = policies.gather(1, perm)
        return [policies, *(part[k][idx] for k in SCALARS)]

    def batch(self, size: int) -> list[torch.Tensor]:
        idx = torch.randint(self.train_size, (size,), device=self.device)
        perm = self.perms[torch.randint(8, (size,), device=self.device)]
        return [self._planes(self.train, idx, perm), *self._rest(self.train, idx, perm)]

    def held_out(self, start: int, stop: int) -> list[torch.Tensor]:
        idx = torch.arange(start, min(stop, self.held_size), device=self.device)
        return [self._planes(self.held, idx), *self._rest(self.held, idx)]


def evaluate(model: TorchModel, data: DeviceData, chunk: int = 16_384) -> dict:
    """Held-out losses, averaged over chunks by their size."""
    totals: dict[str, float] = {}
    for start in range(0, data.held_size, chunk):
        part = data.held_out(start, start + chunk)
        metrics = model.evaluate_tensors(*part)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * len(part[0])
    return {name: round(total / data.held_size, 4) for name, total in totals.items()}


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine down to 1 % of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path, help="saved replay buffer (.npz)")
    source.add_argument("--shards", type=Path, help="directory of self-play shards")
    parser.add_argument("--q-weight", type=float, default=RECORDED_Q_WEIGHT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--holdout", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--baseline", type=Path, help="checkpoint to compare with")
    parser.add_argument(
        "--init", type=Path, help="start from these weights (sets the size)"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    def record(**entry) -> None:
        log_event("supervised", entry.pop("event"), **entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    arrays = from_replay(args.replay) if args.replay else from_shards(args.shards)
    init = weights_io.load(args.init) if args.init else None
    if init is not None:
        args.channels = init["stem.layers.0.weight"].shape[0]
        args.blocks = sum(name.endswith(".conv1.weight") for name in init)
    model = TorchModel(args.channels, args.blocks)
    if init is not None:
        model.load_weights(init)
    data = DeviceData(arrays, args.holdout, args.q_weight, model.device)
    del arrays
    params = sum(p.numel() for p in model.net.parameters())
    record(
        event="start",
        size=f"{args.blocks}x{args.channels}",
        params=params,
        train=data.train_size,
        held_out=args.holdout,
        q_weight=args.q_weight,
    )
    if args.baseline:
        baseline = TorchModel()
        loaded = baseline.load_weights(weights_io.load(args.baseline))
        record(
            event="baseline",
            path=args.baseline.name,
            tensors=loaded,
            **evaluate(baseline, data),
        )
        del baseline

    best = math.inf
    started = time.monotonic()
    for step in range(args.steps):
        model.set_learning_rate(learning_rate(step, args.steps, args.lr, args.warmup))
        train = model.train_tensors(*data.batch(args.batch))
        if (step + 1) % args.eval_every and step + 1 != args.steps:
            continue
        held = evaluate(model, data)
        improved = held["loss"] < best
        if improved:
            best = held["loss"]
            weights = model.weights()
            weights_io.save_bin(weights, args.out / "best.bin")
            weights_io.save_npz(weights, args.out / "best.npz")
        record(
            event="eval",
            step=step + 1,
            steps_per_s=round((step + 1) / (time.monotonic() - started), 1),
            train_loss=round(train["loss"], 4),
            best=improved,
            **held,
        )


if __name__ == "__main__":
    main()
