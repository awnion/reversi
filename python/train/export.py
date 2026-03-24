"""
Export MLX model weights to binary format for Rust inference.

Usage: uv run python -m train.export \
    --checkpoint weights/iter_001000.npz --output weights/champion.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x52455653  # "REVS"
VERSION = 1


def _write_tensors(f, tensors: dict):
    f.write(struct.pack("<IH", MAGIC, VERSION))
    f.write(struct.pack("<H", len(tensors)))
    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<B", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<B", arr.ndim))
        for dim in arr.shape:
            f.write(struct.pack("<I", dim))
        f.write(arr.tobytes())


def export(checkpoint_path: Path, output_path: Path):
    data = np.load(str(checkpoint_path))
    tensors = {k: data[k] for k in data.files}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        _write_tensors(f, tensors)
    size_kb = output_path.stat().st_size / 1024
    print(f"Exported {len(tensors)} tensors → {output_path} ({size_kb:.1f} KB)")


def export_model(model, output_path: Path):
    """Export an in-memory MLX model to .bin without re-reading from disk."""
    from .loop import _collect_params

    arrays: dict = {}
    _collect_params(model.parameters(), "", arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        _write_tensors(f, arrays)
    size_kb = output_path.stat().st_size / 1024
    print(f"Exported {len(arrays)} tensors → {output_path} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
