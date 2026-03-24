import mlx.core as mx
import numpy as np

from .eval_server import board_to_planes
from .model import AlphaZeroNet


def compute_loss(
    model: AlphaZeroNet, boards_black, boards_white, is_black_arr, policies, outcomes
):
    """
    boards_black, boards_white: (B,) uint64 numpy arrays
    boards_black, boards_white: (batch_size,) uint64 numpy arrays
    is_black_arr: (batch_size,) bool numpy array
    policies: (batch_size, 64) float32 — MCTS visit-count targets
    outcomes: (batch_size,) float32 — game outcomes from current player's POV

    Returns scalar loss.
    """
    batch_size = len(boards_black)
    # Build input planes (batch_size, 3, 8, 8)
    planes = np.stack(
        [
            board_to_planes(
                int(boards_black[i]), int(boards_white[i]), bool(is_black_arr[i]), 0
            )
            for i in range(batch_size)
        ]
    )
    x = mx.array(planes)

    policy_logits, values = model(x)  # (batch_size, 64), (batch_size,)

    # Policy loss: cross-entropy (log-softmax vs target distribution)
    log_probs = policy_logits - mx.log(
        mx.sum(mx.exp(policy_logits), axis=-1, keepdims=True)
    )
    policy_targets = mx.array(policies)
    policy_loss = -mx.mean(mx.sum(policy_targets * log_probs, axis=-1))

    # Value loss: MSE
    value_targets = mx.array(outcomes)
    value_loss = mx.mean((values - value_targets) ** 2)

    return policy_loss + value_loss
