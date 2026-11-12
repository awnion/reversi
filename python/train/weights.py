"""Model weights on disk, independent of the ML framework.

A set of weights is a flat ``{name: float32 array}`` dict in the *canonical*
layout, which is the MLX parameter tree: names such as
``stem.layers.0.weight``, convolution weights as ``(out, kH, kW, in)``, and
the policy head flattened as ``(H, W, C)``. Every backend converts to and from
this layout, so checkpoints are interchangeable between machines.

Two containers hold the same data:

- ``.npz``: training checkpoints;
- ``.bin``: the export read by the Rust inference code (browser bots,
  tournaments) and used to hand weights to self-play processes.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x52455653  # "REVS"
VERSION = 1

Weights = dict[str, np.ndarray]


def write_bin(f, weights: Weights) -> None:
    f.write(struct.pack("<IH", MAGIC, VERSION))
    f.write(struct.pack("<H", len(weights)))
    for name, arr in weights.items():
        arr = np.asarray(arr, dtype=np.float32)
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<B", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<B", arr.ndim))
        for dim in arr.shape:
            f.write(struct.pack("<I", dim))
        f.write(arr.tobytes())


def to_bytes(weights: Weights) -> bytes:
    """Serialise weights to the ``.bin`` format."""
    buffer = io.BytesIO()
    write_bin(buffer, weights)
    return buffer.getvalue()


def save_bin(weights: Weights, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        write_bin(f, weights)


def save_npz(weights: Weights, path: Path) -> None:
    np.savez(str(path), **weights)


def _read_bin(path: Path) -> Weights:
    weights: Weights = {}
    with open(path, "rb") as f:
        magic, version = struct.unpack("<IH", f.read(6))
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"Unrecognised .bin format: {path}")
        (count,) = struct.unpack("<H", f.read(2))
        for _ in range(count):
            (name_len,) = struct.unpack("<B", f.read(1))
            name = f.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<B", f.read(1))
            shape = struct.unpack(f"<{ndim}I", f.read(ndim * 4))
            n_elems = int(np.prod(shape, dtype=np.int64)) if shape else 1
            data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32)
            weights[name] = data.reshape(shape).copy()
    return weights


def load(path: Path) -> Weights:
    """Read weights from a ``.bin`` export or an ``.npz`` checkpoint."""
    path = Path(path)
    if path.suffix == ".bin":
        return _read_bin(path)
    with np.load(str(path)) as data:
        return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
