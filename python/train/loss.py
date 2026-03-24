import mlx.core as mx
import mlx.nn as nn

from .model import AlphaZeroNet


def compute_loss(
    model: AlphaZeroNet,
    x,
    policy_targets,
    value_targets,
    policy_weights,
    value_weights,
    phase,
):
    """
    All inputs are pre-converted mlx.arrays.
    x: (batch_size, 4, 8, 8) — input planes
    policy_targets: (batch_size, 64) — smoothed MCTS targets
    value_targets: (batch_size,) — game outcomes
    policy_weights, value_weights: (batch_size,) — per-sample weights
    phase: (batch_size,) — game progress 0..1

    Returns scalar loss.
    """
    policy_logits, values = model(x)  # (batch_size, 64), (batch_size,)

    # Policy loss: cross-entropy with label smoothing
    log_probs = nn.log_softmax(policy_logits, axis=-1)
    policy_loss_per = -mx.sum(policy_targets * log_probs, axis=-1)
    policy_norm = mx.maximum(mx.sum(policy_weights), mx.array(1.0))
    policy_loss = mx.sum(policy_loss_per * policy_weights) / policy_norm

    # Value loss: MSE with phase-weighted samples
    value_weights = value_weights * (0.5 + 1.5 * phase)
    value_loss_per = (values - value_targets) ** 2
    value_norm = mx.maximum(mx.sum(value_weights), mx.array(1.0))
    value_loss = mx.sum(value_loss_per * value_weights) / value_norm

    return policy_loss + value_loss
