"""Smoke tests — run with: uv run python -m train.test_smoke"""

import mlx.core as mx
import numpy as np

from train.eval_server import board_to_planes
from train.model import AlphaZeroNet
from train.replay import ReplayBuffer


def test_model_forward():
    net = AlphaZeroNet()
    x = mx.random.normal((4, 4, 8, 8))
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
    assert planes.shape == (4, 8, 8)
    assert planes[0].sum() == 2  # 2 black discs
    assert planes[1].sum() == 2  # 2 white discs
    assert np.allclose(planes[3], 4 / 64)
    print("✓ board_to_planes")


def test_replay_buffer():
    buf = ReplayBuffer(max_size=100)
    fake_record = [
        {
            "board_black": 1,
            "board_white": 2,
            "is_black": True,
            "legal": 4,
            "mcts_policy": np.ones(64) / 64,
            "outcome": 1.0,
            "policy_weight": 1.0,
            "value_weight": 1.0,
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
    legal_arr = np.array([4, 8, 16, 32], dtype=np.uint64)
    policies = np.ones((B, 64), dtype=np.float32) / 64
    outcomes = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    policy_weights = np.ones(B, dtype=np.float32)
    value_weights = np.ones(B, dtype=np.float32)
    # Precompute planes (phase is inside planes[:,3])
    all_planes = np.stack(
        [
            board_to_planes(
                int(boards_black[i]),
                int(boards_white[i]),
                bool(is_black_arr[i]),
                int(legal_arr[i]),
            )
            for i in range(B)
        ]
    )
    # Convert to MLX (as loop does outside the lock)
    x = mx.array(all_planes)
    phase = x[:, 3, 0, 0]
    policy_targets = 0.97 * mx.array(policies) + 0.03 / 64.0
    value_targets = mx.array(outcomes)
    policy_weights_mx = mx.array(policy_weights)
    value_weights_mx = mx.array(value_weights)
    loss = compute_loss(
        net,
        x,
        policy_targets,
        value_targets,
        policy_weights_mx,
        value_weights_mx,
        phase,
    )
    mx.eval(loss)
    loss_val = float(loss.item())
    assert np.isfinite(loss_val), f"loss is not finite: {loss_val}"
    assert loss_val > 0, f"loss should be positive: {loss_val}"
    print(f"✓ compute_loss (loss={loss_val:.4f})")


def test_rust_inference_matches_mlx():
    """The exported .bin must give the same outputs in Rust (browser bots)."""
    import tempfile
    from pathlib import Path

    import reversi_mcts

    from train.export import export_model

    net = AlphaZeroNet()
    # A few training-mode passes so the BatchNorm running stats are non-trivial.
    for _ in range(3):
        mx.eval(net(mx.random.normal((16, 4, 8, 8))))
    mx.eval(net.parameters())
    net.eval()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.bin"
        export_model(net, path)
        raw = path.read_bytes()

    rng = np.random.default_rng(0)
    for _ in range(10):
        cells = rng.integers(0, 3, size=64)
        black = sum(1 << i for i in range(64) if cells[i] == 1)
        white = sum(1 << i for i in range(64) if cells[i] == 2)
        legal = int(rng.integers(0, 2**63))
        planes = board_to_planes(black, white, bool(rng.integers(0, 2)), legal)
        logits, value = net(mx.array(planes[None]))
        want_policy = np.array(mx.softmax(logits, axis=-1))[0]
        got_policy, got_value = reversi_mcts.nn_forward(
            raw, planes.reshape(-1).tolist()
        )
        assert np.allclose(got_policy, want_policy, atol=1e-4), "policy mismatch"
        assert abs(got_value - float(value[0])) < 1e-4, "value mismatch"
    print("✓ rust inference matches mlx")


if __name__ == "__main__":
    test_board_planes()
    test_replay_buffer()
    test_model_forward()
    test_compute_loss()
    test_rust_inference_matches_mlx()
    print("All smoke tests passed!")
