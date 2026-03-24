import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .eval_server import board_to_planes
from .model import AlphaZeroNet


def compute_loss(
    model: AlphaZeroNet,
    boards_black,
    boards_white,
    is_black_arr,
    legal_arr,
    policies,
    outcomes,
    policy_weights,
    value_weights,
):
    """
    boards_black, boards_white: (B,) uint64 numpy arrays
    boards_black, boards_white: (batch_size,) uint64 numpy arrays
    is_black_arr: (batch_size,) bool numpy array
    legal_arr: (batch_size,) uint64 numpy array
    policies: (batch_size, 64) float32 — MCTS visit-count targets
    outcomes: (batch_size,) float32 — game outcomes from current player's POV
    policy_weights, value_weights: per-sample weights

    Returns scalar loss.
    """
    batch_size = len(boards_black)
    # Build input planes (batch_size, 4, 8, 8)
    planes = np.stack(
        [
            board_to_planes(
                int(boards_black[i]),
                int(boards_white[i]),
                bool(is_black_arr[i]),
                int(legal_arr[i]),
            )
            for i in range(batch_size)
        ]
    )
    x = mx.array(planes)

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

    # Value loss: MSE
    value_targets = mx.array(outcomes)
    phase = mx.array(
        np.array(
            [
                ((int(boards_black[i]) | int(boards_white[i])).bit_count()) / 64.0
                for i in range(batch_size)
            ],
            dtype=np.float32,
        )
    )
    value_weights = mx.array(value_weights) * (0.5 + 1.5 * phase)
    value_loss_per = (values - value_targets) ** 2
    value_norm = mx.maximum(mx.sum(value_weights), mx.array(1.0))
    value_loss = mx.sum(value_loss_per * value_weights) / value_norm

    return policy_loss + value_loss
