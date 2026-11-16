"""ML backends for the training loop.

The loop only needs a handful of operations on the network; each backend
implements them with the framework that is fastest on its platform:

- ``mlx``: Apple Silicon (Metal);
- ``torch``: NVIDIA GPUs (CUDA), also the fallback elsewhere.

Both exchange weights in the canonical layout described in
``train/weights.py``, so checkpoints written on one machine continue on the
other. Override the automatic choice with ``REVERSI_BACKEND=mlx|torch``.
"""

from __future__ import annotations

import os
import sys
from typing import Protocol

import numpy as np

from ..weights import Weights


class Model(Protocol):
    """One network instance (weights plus, when training, its optimizer)."""

    def weights(self) -> Weights:
        """Current weights in the canonical layout."""

    def load_weights(self, weights: Weights) -> int:
        """Load every tensor whose name and shape match; return how many."""

    def infer(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``(N, 4, 8, 8)`` positions in inference mode.

        Returns move probabilities ``(N, 64)`` and values ``(N,)``, float32.
        """

    def train_step(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        margins: np.ndarray,
        policy_weights: np.ndarray,
        value_weights: np.ndarray,
    ) -> dict[str, float]:
        """One optimizer step on a batch.

        Returns the total ``loss`` and its parts: ``policy`` (cross-entropy),
        ``policy_kl`` (the same minus the entropy of the targets, the part
        that can still go down), ``value`` and ``margin``.
        """

    def set_learning_rate(self, learning_rate: float) -> None: ...


def backend_name() -> str:
    name = os.environ.get("REVERSI_BACKEND")
    if name:
        if name not in ("mlx", "torch"):
            raise ValueError(f"REVERSI_BACKEND must be 'mlx' or 'torch', got {name!r}")
        return name
    return "mlx" if sys.platform == "darwin" else "torch"


def create_model(channels: int = 32, blocks: int = 4) -> Model:
    """A freshly initialised network of this size on this machine's backend."""
    if backend_name() == "mlx":
        from .mlx_backend import MlxModel

        return MlxModel(channels, blocks)
    from .torch_backend import TorchModel

    return TorchModel(channels, blocks)
