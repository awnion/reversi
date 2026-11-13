"""
Export a training checkpoint to the binary format read by the Rust inference
code (browser bots, tournaments).

Usage: uv run python -m train.export \
    --checkpoint ../weights/<run>/iter_<step>_loss<loss>.npz \
    --output ../weights/<run>/champion.bin

Only the .bin is written; champion.json is left as it is.

The format itself is described in `train/weights.py`.
"""

import argparse
from datetime import datetime
from pathlib import Path

from . import weights as weights_io


def _log(component: str, message: str, **fields) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"{stamp} [{component}] {message}{suffix}", flush=True)


def export(checkpoint_path: Path, output_path: Path) -> None:
    weights = weights_io.load(checkpoint_path)
    weights_io.save_bin(weights, output_path)
    size_kb = output_path.stat().st_size / 1024
    _log(
        "export",
        "saved",
        tensors=len(weights),
        path=output_path.name,
        kb=f"{size_kb:.1f}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
