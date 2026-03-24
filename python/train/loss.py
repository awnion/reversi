import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import AlphaZeroNet


def compute_loss(
    model: AlphaZeroNet,
    planes_np,
    policies,
    outcomes,
    policy_weights,
    value_weights,
):
    """
    planes_np: (batch_size, 4, 8, 8) numpy float32 — precomputed input planes
    policies: (batch_size, 64) numpy — MCTS visit-count targets
    outcomes: (batch_size,) numpy — game outcomes from current player's POV
    policy_weights, value_weights: per-sample numpy weights

    Returns scalar loss.
    """
    x = mx.array(np.asarray(planes_np, dtype=np.float32))
    policy_logits, values = model(x)  # (batch_size, 64), (batch_size,)

    # Policy loss: cross-entropy with label smoothing to prevent memorization
    # of peaked MCTS distributions (ε=0.1 mixes with uniform over 64 moves)
    log_probs = nn.log_softmax(policy_logits, axis=-1)
    policy_targets = mx.array(policies)
    policy_targets = 0.97 * policy_targets + 0.03 / 64.0
    policy_weights = mx.array(policy_weights)
    policy_loss_per = -mx.sum(policy_targets * log_probs, axis=-1)
    policy_norm = mx.maximum(mx.sum(policy_weights), mx.array(1.0))
    policy_loss = mx.sum(policy_loss_per * policy_weights) / policy_norm

    # Value loss: MSE with phase-weighted samples
    value_targets = mx.array(outcomes)
    phase = mx.array(np.asarray(planes_np[:, 3, 0, 0], dtype=np.float32))
    value_weights = mx.array(value_weights) * (0.5 + 1.5 * phase)
    value_loss_per = (values - value_targets) ** 2
    value_norm = mx.maximum(mx.sum(value_weights), mx.array(1.0))
    value_loss = mx.sum(value_loss_per * value_weights) / value_norm

    return policy_loss + value_loss
