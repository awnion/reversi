"""Smoke tests — run with: uv run python -m train.test_smoke

They exercise whichever backend this machine uses (see `train/backend`).
"""

import numpy as np

from train import weights as weights_io
from train.backend import backend_name, create_model
from train.planes import board_to_planes
from train.replay import ReplayBuffer


def _random_planes(rng: np.random.Generator, n: int) -> np.ndarray:
    planes = []
    for _ in range(n):
        cells = rng.integers(0, 3, size=64)
        black = sum(1 << i for i in range(64) if cells[i] == 1)
        white = sum(1 << i for i in range(64) if cells[i] == 2)
        legal = int(rng.integers(0, 2**63))
        planes.append(board_to_planes(black, white, bool(rng.integers(0, 2)), legal))
    return np.stack(planes)


def _train_batch(rng: np.random.Generator, n: int):
    policies = rng.random((n, 64), dtype=np.float32)
    policies /= policies.sum(axis=1, keepdims=True)
    outcomes = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), size=n)
    ones = np.ones(n, dtype=np.float32)
    return _random_planes(rng, n), policies, outcomes, ones, ones


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


def test_model_infer():
    model = create_model()
    policies, values = model.infer(_random_planes(np.random.default_rng(0), 4))
    assert policies.shape == (4, 64), f"bad policy shape: {policies.shape}"
    assert values.shape == (4,), f"bad value shape: {values.shape}"
    assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-3), "policies must sum to 1"
    assert np.all(values >= -1) and np.all(values <= 1), "value out of [-1,1]"
    print("✓ model inference")


def test_train_step():
    model = create_model()
    rng = np.random.default_rng(1)
    batch = _train_batch(rng, 64)
    losses = [model.train_step(*batch) for _ in range(30)]
    assert all(np.isfinite(loss) for loss in losses), f"non-finite loss: {losses}"
    assert losses[-1] < losses[0], f"loss did not go down on a fixed batch: {losses}"
    print(f"✓ train step (loss {losses[0]:.3f} -> {losses[-1]:.3f})")


def test_weights_roundtrip():
    """Weights survive the canonical layout and both file formats."""
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(2)
    source = create_model()
    # A few training steps so the BatchNorm running stats are non-trivial.
    for _ in range(3):
        source.train_step(*_train_batch(rng, 32))
    weights = source.weights()
    assert weights["stem.layers.0.weight"].shape == (32, 3, 3, 4), (
        "canonical conv layout"
    )
    assert weights["policy_fc.weight"].shape == (64, 128)

    planes = _random_planes(rng, 8)
    want_policy, want_value = source.infer(planes)

    with tempfile.TemporaryDirectory() as tmp:
        for name, save in (
            ("w.npz", weights_io.save_npz),
            ("w.bin", weights_io.save_bin),
        ):
            path = Path(tmp) / name
            save(weights, path)
            target = create_model()
            applied = target.load_weights(weights_io.load(path))
            assert applied == len(weights), f"{name}: loaded {applied}/{len(weights)}"
            got_policy, got_value = target.infer(planes)
            assert np.allclose(got_policy, want_policy, atol=2e-3), f"{name}: policy"
            assert np.allclose(got_value, want_value, atol=2e-3), f"{name}: value"
    print("✓ weights roundtrip (.npz and .bin)")


def test_rust_inference_matches_backend():
    """The exported .bin must give the same outputs in Rust (browser bots)."""
    import reversi_mcts

    rng = np.random.default_rng(3)
    model = create_model()
    for _ in range(3):
        model.train_step(*_train_batch(rng, 32))
    raw = weights_io.to_bytes(model.weights())

    planes = _random_planes(rng, 10)
    want_policy, want_value = model.infer(planes)
    for i in range(len(planes)):
        got_policy, got_value = reversi_mcts.nn_forward(
            raw, planes[i].reshape(-1).tolist()
        )
        # Loose enough for half-precision inference on CUDA.
        assert np.allclose(got_policy, want_policy[i], atol=2e-3), "policy mismatch"
        assert abs(got_value - float(want_value[i])) < 5e-3, "value mismatch"
    print("✓ rust inference matches the backend")


if __name__ == "__main__":
    print(f"backend: {backend_name()}")
    test_board_planes()
    test_replay_buffer()
    test_model_infer()
    test_train_step()
    test_weights_roundtrip()
    test_rust_inference_matches_backend()
    print("All smoke tests passed!")
