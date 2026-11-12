"""MLX backend (Apple Silicon)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from ..loss import compute_loss
from ..model import AlphaZeroNet
from ..weights import Weights
from .common import BASE_LEARNING_RATE, GRAD_CLIP_NORM, LABEL_SMOOTHING, WEIGHT_DECAY


class MlxModel:
    def __init__(self) -> None:
        self.net = AlphaZeroNet()
        mx.eval(self.net.parameters())
        self._training = True
        self._optimizer = optim.AdamW(
            learning_rate=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self._loss_and_grad = nn.value_and_grad(self.net, compute_loss)

    # The MLX parameter tree *is* the canonical layout.
    def weights(self) -> Weights:
        return {k: np.array(v) for k, v in tree_flatten(self.net.parameters())}

    def load_weights(self, weights: Weights) -> int:
        shapes = {k: tuple(v.shape) for k, v in tree_flatten(self.net.parameters())}
        compatible = [
            (name, mx.array(arr))
            for name, arr in weights.items()
            if shapes.get(name) == tuple(arr.shape)
        ]
        if compatible:
            self.net.load_weights(compatible, strict=False)
            mx.eval(self.net.parameters())
        return len(compatible)

    def _set_training(self, training: bool) -> None:
        if self._training != training:
            self.net.train(training)
            self._training = training

    def infer(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._set_training(False)
        logits, values, _ = self.net(mx.array(planes))
        policies = mx.softmax(logits, axis=-1)
        mx.eval(policies, values)
        return np.array(policies), np.array(values)

    def train_step(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        margins: np.ndarray,
        policy_weights: np.ndarray,
        value_weights: np.ndarray,
    ) -> dict[str, float]:
        self._set_training(True)
        x = mx.array(np.asarray(planes, dtype=np.float32))
        phase = x[:, 3, 0, 0]
        policy_targets = (1.0 - LABEL_SMOOTHING) * mx.array(
            policies
        ) + LABEL_SMOOTHING / 64.0
        policy_w = mx.array(policy_weights)
        (loss, policy_loss, value_loss, margin_loss), grads = self._loss_and_grad(
            self.net,
            x,
            policy_targets,
            mx.array(outcomes),
            mx.array(margins),
            policy_w,
            mx.array(value_weights),
            phase,
        )
        grads, _ = optim.clip_grad_norm(grads, max_norm=GRAD_CLIP_NORM)
        self._optimizer.update(self.net, grads)
        # The part of the policy loss that can go down: the cross-entropy
        # minus the entropy of the targets themselves (the KL divergence).
        entropy = -mx.sum(policy_targets * mx.log(policy_targets), axis=-1)
        target_entropy = mx.sum(entropy * policy_w) / mx.maximum(
            mx.sum(policy_w), mx.array(1.0)
        )
        mx.eval(
            self.net.parameters(),
            self._optimizer.state,
            loss,
            policy_loss,
            value_loss,
            margin_loss,
            target_entropy,
        )
        return {
            "loss": loss.item(),
            "policy": policy_loss.item(),
            "policy_kl": policy_loss.item() - target_entropy.item(),
            "value": value_loss.item(),
            "margin": margin_loss.item(),
        }

    def set_learning_rate(self, learning_rate: float) -> None:
        self._optimizer.learning_rate = mx.array(learning_rate)
