"""Smoke tests — run with: uv run python -m train.test_smoke"""

import mlx.core as mx
import numpy as np

from train.eval_server import board_to_planes
from train.model import AlphaZeroNet
from train.replay import ReplayBuffer


def test_model_forward():
    net = AlphaZeroNet()
    x = mx.random.normal((4, 3, 8, 8))
    policy, value = net(x)
    mx.eval(policy, value)
    assert policy.shape == (4, 64), f"bad policy shape: {policy.shape}"
    assert value.shape == (4,), f"bad value shape: {value.shape}"
    vals = np.array(value)
    assert np.all(vals >= -1) and np.all(vals <= 1), "value out of [-1,1]"
    print("✓ model forward pass")


def test_board_planes():
    # Standard starting position
    black = (1 << 28) | (1 << 35)
    white = (1 << 27) | (1 << 36)
    legal = 0  # just test shape
    planes = board_to_planes(black, white, True, legal)
    assert planes.shape == (3, 8, 8)
    assert planes[0].sum() == 2  # 2 black discs
    assert planes[1].sum() == 2  # 2 white discs
    print("✓ board_to_planes")


def test_replay_buffer():
    buf = ReplayBuffer(max_size=100)
    fake_record = [
        {
            "board_black": 1,
            "board_white": 2,
            "is_black": True,
            "mcts_policy": np.ones(64) / 64,
            "outcome": 1.0,
        }
    ]
    buf.add(fake_record)
    assert len(buf) == 1
    sample = buf.sample(1)
    assert len(sample) == 5
    print("✓ replay buffer")


def test_compute_loss():
    from train.loss import compute_loss

    net = AlphaZeroNet()
    B = 4  # noqa: N806
    boards_black = np.array([1, 2, 3, 4], dtype=np.uint64)
    boards_white = np.array([8, 16, 32, 64], dtype=np.uint64)
    is_black_arr = np.array([True, False, True, False])
    policies = np.ones((B, 64), dtype=np.float32) / 64
    outcomes = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    loss = compute_loss(
        net, boards_black, boards_white, is_black_arr, policies, outcomes
    )
    mx.eval(loss)
    loss_val = float(loss.item())
    assert np.isfinite(loss_val), f"loss is not finite: {loss_val}"
    assert loss_val > 0, f"loss should be positive: {loss_val}"
    print(f"✓ compute_loss (loss={loss_val:.4f})")


if __name__ == "__main__":
    test_board_planes()
    test_replay_buffer()
    test_model_forward()
    test_compute_loss()
    print("All smoke tests passed!")
