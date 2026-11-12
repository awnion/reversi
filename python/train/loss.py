import mlx.core as mx
import mlx.nn as nn

from .backend.common import MARGIN_LOSS_WEIGHT
from .model import AlphaZeroNet


def _mean(per_sample, weights):
    return mx.sum(per_sample * weights) / mx.maximum(mx.sum(weights), mx.array(1.0))


def compute_loss(
    model: AlphaZeroNet,
    x,
    policy_targets,
    value_targets,
    margin_targets,
    policy_weights,
    value_weights,
    phase,
):
    """
    All inputs are pre-converted mlx.arrays.
    x: (batch_size, 4, 8, 8) — input planes
    policy_targets: (batch_size, 64) — smoothed MCTS targets
    value_targets: (batch_size,) — game outcomes
    margin_targets: (batch_size,) — final disc margin / 64, NaN when unknown
    policy_weights, value_weights: (batch_size,) — per-sample weights
    phase: (batch_size,) — game progress 0..1

    Returns the total loss, then the policy, value and margin losses.
    """
    policy_logits, values, margins = model(x)

    # Policy loss: cross-entropy with label smoothing
    log_probs = nn.log_softmax(policy_logits, axis=-1)
    policy_loss = _mean(-mx.sum(policy_targets * log_probs, axis=-1), policy_weights)

    # Value loss: MSE with phase-weighted samples
    value_weights = value_weights * (0.5 + 1.5 * phase)
    value_loss = _mean((values - value_targets) ** 2, value_weights)

    # Margin loss, on the samples whose margin is known
    known = mx.logical_not(mx.isnan(margin_targets))
    margin_targets = mx.where(known, margin_targets, 0.0)
    margin_loss = _mean((margins - margin_targets) ** 2, value_weights * known)

    total = policy_loss + value_loss + MARGIN_LOSS_WEIGHT * margin_loss
    return total, policy_loss, value_loss, margin_loss
