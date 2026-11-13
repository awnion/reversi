"""Checks for the Rust self-play binary (rust/trainer, `just build-selfplay`).

Run with: uv run python -m train.test_selfplay_rs

- the binary's inference matches this machine's backend (``model.infer``) and,
  in full precision, the Rust CPU reference;
- its replay shards load through ``ReplayBuffer.ingest_shard`` and hold sane
  samples, terminal value-only samples included;
- SIGTERM stops it within 5 s with the partial shard flushed;
- its games look like the Python self-play games on the same weights.

The binary is ``REVERSI_SELFPLAY_BIN`` or rust/trainer/target/release/.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from train import weights as weights_io
from train.backend import backend_name, create_model
from train.replay import ReplayBuffer, materialize_record

ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = Path(
    os.environ.get(
        "REVERSI_SELFPLAY_BIN",
        ROOT / "rust" / "trainer" / "target" / "release" / "reversi-selfplay",
    )
)

# Half precision on CUDA: the binary folds BatchNorm into half-precision
# convolutions, PyTorch's autocast keeps it separate; both round differently.
HALF_POLICY_ATOL = 5e-3
HALF_VALUE_ATOL = 1e-2
FULL_ATOL = 1e-4


def _run(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BINARY), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def _cuda() -> bool:
    out = _run("check", "--allow-cpu").stdout
    return "cuda=true" in out


def _random_planes(rng: np.random.Generator, n: int) -> np.ndarray:
    from train.planes import board_to_planes

    planes = []
    for _ in range(n):
        cells = rng.integers(0, 3, size=64)
        black = sum(1 << i for i in range(64) if cells[i] == 1)
        white = sum(1 << i for i in range(64) if cells[i] == 2)
        empty = sum(1 << i for i in range(64) if cells[i] == 0)
        legal = empty & int(rng.integers(0, 2**63))
        planes.append(board_to_planes(black, white, bool(rng.integers(0, 2)), legal))
    return np.stack(planes)


def _trained_model():
    """A network with non-trivial weights and BatchNorm statistics."""
    rng = np.random.default_rng(1)
    model = create_model()
    for _ in range(5):
        n = 64
        policies = rng.random((n, 64), dtype=np.float32)
        policies /= policies.sum(axis=1, keepdims=True)
        outcomes = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), size=n)
        ones = np.ones(n, dtype=np.float32)
        model.train_step(
            _random_planes(rng, n), policies, outcomes, outcomes * 0.5, ones, ones
        )
    return model


def _infer(weights: Path, planes: np.ndarray, *extra: str) -> tuple[np.ndarray, ...]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.f32", Path(tmp) / "out.f32"
        np.ascontiguousarray(planes, dtype="<f4").tofile(src)
        _run(
            "infer",
            "--weights",
            str(weights),
            "--input",
            str(src),
            "--output",
            str(dst),
            *extra,
        )
        out = np.fromfile(dst, dtype="<f4").reshape(len(planes), 65)
    return out[:, :64], out[:, 64]


def test_infer_parity(tmp: Path) -> None:
    import reversi_mcts

    model = _trained_model()
    path = tmp / "parity.bin"
    weights_io.save_bin(model.weights(), path)
    planes = _random_planes(np.random.default_rng(2), 512)
    want_policy, want_value = model.infer(planes)
    cuda = _cuda()

    got_policy, got_value = _infer(path, planes)
    assert np.isfinite(got_policy).all() and np.isfinite(got_value).all(), "non-finite"
    assert np.allclose(got_policy.sum(axis=1), 1.0, atol=1e-3), "policy sums"
    policy_diff = float(np.abs(got_policy - want_policy).max())
    value_diff = float(np.abs(got_value - want_value).max())
    policy_tol, value_tol = (
        (HALF_POLICY_ATOL, HALF_VALUE_ATOL) if cuda else (FULL_ATOL, FULL_ATOL)
    )
    print(
        f"  default precision ({'cuda half' if cuda else 'cpu f32'}) vs "
        f"{backend_name()} model.infer: policy {policy_diff:.2e}, "
        f"value {value_diff:.2e}"
    )
    assert policy_diff <= policy_tol, f"policy diff {policy_diff}"
    assert value_diff <= value_tol, f"value diff {value_diff}"

    # Full precision against the Rust CPU reference (the browser bots' code).
    full_policy, full_value = _infer(path, planes, "--fp32")
    raw = weights_io.to_bytes(model.weights())
    ref = [reversi_mcts.nn_forward(raw, p.reshape(-1).tolist()) for p in planes[:64]]
    ref_policy = np.array([r[0] for r in ref], dtype=np.float32)
    ref_value = np.array([r[1] for r in ref], dtype=np.float32)
    full_policy_diff = float(np.abs(full_policy[:64] - ref_policy).max())
    full_value_diff = float(np.abs(full_value[:64] - ref_value).max())
    print(
        f"  --fp32 vs Rust CPU reference: policy {full_policy_diff:.2e}, "
        f"value {full_value_diff:.2e}"
    )
    assert full_policy_diff <= FULL_ATOL and full_value_diff <= FULL_ATOL
    print("✓ inference parity")


def _selfplay(run_dir: Path, *extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            str(BINARY),
            "selfplay",
            "--run-dir",
            str(run_dir),
            "--log-seconds",
            "2",
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _load_shards(spool: Path) -> dict[str, np.ndarray]:
    paths = sorted(spool.glob("*.npz"))
    assert paths, "no shards"
    assert not list(spool.glob("*.tmp")), "temporary shard left behind"
    parts: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as data:
            for key in data.files:
                assert data[key].dtype == np.float32, f"{key}: {data[key].dtype}"
                parts.setdefault(key, []).append(data[key])
        replay = ReplayBuffer(max_size=100_000)
        count = replay.ingest_shard(path, unlink=False)
        assert count == len(replay) > 0
        assert path.exists(), "ingest_shard(unlink=False) must keep the file"
    return {k: np.concatenate(v) for k, v in parts.items()}


def _check_samples(s: dict[str, np.ndarray]) -> None:
    n = len(s["planes"])
    assert s["planes"].shape == (n, 4, 8, 8)
    assert s["policies"].shape == (n, 64)
    for key in ("outcomes", "margins", "policy_weights", "value_weights"):
        assert s[key].shape == (n,), key
    planes = s["planes"]
    discs = planes[:, 0].sum(axis=(1, 2)) + planes[:, 1].sum(axis=(1, 2))
    # The phase plane is constant: discs on the board / 64.
    assert np.all(planes[:, 3] == planes[:, 3, :1, :1])
    assert np.allclose(planes[:, 3, 0, 0], discs / 64.0)
    assert not np.any(planes[:, 0] * planes[:, 1]), "a square with both colours"
    # Value-only positions carry the game result, whose sign the margin has;
    # searched ones are halfway between it and the search value (in [-1, 1]).
    result = np.sign(s["margins"])
    value_only = s["policy_weights"] == 0
    assert np.all(s["outcomes"][value_only] == result[value_only])
    assert np.all(np.abs(s["outcomes"] - 0.5 * result)[~value_only] <= 0.5 + 1e-6)
    searched = s["policy_weights"] > 0
    legal = planes[:, 2].reshape(n, 64)
    sums = (s["policies"] * legal).sum(axis=1)
    assert np.allclose(sums[searched], 1.0, atol=1e-4), (
        "policy mass off the legal moves"
    )
    assert np.allclose(s["policies"][searched].sum(axis=1), 1.0, atol=1e-4)
    # Terminal samples: full or blocked board, no legal moves, no policy.
    terminal = (s["policy_weights"] == 0) & (legal.sum(axis=1) == 0)
    assert terminal.sum() >= 2, "no terminal samples"
    assert np.all(s["policies"][terminal] == 0)
    assert np.all(s["value_weights"][terminal] == 2.0)
    print(
        f"  {n} samples, {int(searched.sum())} searched, "
        f"{int(terminal.sum())} terminal value-only"
    )


def test_shards_and_shutdown(tmp: Path, bin_path: Path) -> None:
    run = tmp / "run"
    run.mkdir()
    (run / "selfplay_latest.bin").write_bytes(bin_path.read_bytes())
    proc = _selfplay(
        run,
        "--simulations", "16",
        "--workers", "2",
        "--pools-per-worker", "2",
        "--games-per-pool", "32",
        "--batch-target", "256",
        "--positions-per-shard", "1000",
        "--parent-pid", str(os.getpid()),
    )  # fmt: skip
    # Wait for at least one full shard, then stop it mid-way.
    deadline = time.monotonic() + 120
    spool = run / "replay_spool"
    while time.monotonic() < deadline and not list(spool.glob("*.npz")):
        time.sleep(0.5)
    assert (run / "selfplay-rs.pid").read_text().strip() == str(proc.pid)
    time.sleep(3)
    stopped = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    output, _ = proc.communicate(timeout=10)
    seconds = time.monotonic() - stopped
    assert proc.returncode == 0, output
    assert seconds < 5, f"took {seconds:.1f}s to stop"
    assert "[selfplay] signal" in output and "progress process=0 games=" in output
    assert not (run / "selfplay-rs.pid").exists(), "pid file left behind"
    # Every position it reports is in a shard: the partial one was flushed.
    exit_line = next(line for line in output.splitlines() if "[selfplay] exit" in line)
    reported = int(exit_line.split("positions=")[1].split()[0])
    samples = _load_shards(spool)
    assert len(samples["planes"]) == reported, (len(samples["planes"]), reported)
    print(
        f"  SIGTERM: exited in {seconds:.2f}s, all {reported} positions in "
        f"{len(list(spool.glob('*.npz')))} shards"
    )
    _check_samples(samples)
    print("✓ shards round-trip and shutdown")


def _stats(s: dict[str, np.ndarray]) -> dict[str, float]:
    searched = s["policy_weights"] == 1
    legal = s["planes"][:, 2].reshape(len(s["planes"]), 64)
    terminal = (s["policy_weights"] == 0) & (legal.sum(axis=1) == 0)
    # Terminal samples come in pairs, black's view first.
    first = np.flatnonzero(terminal)[::2]
    games = len(first)
    p = s["policies"][searched]
    entropy = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)
    return {
        "games": games,
        "moves_per_game": float(searched.sum() / games),
        "black_win_rate": float((s["outcomes"][first] > 0).mean()),
        "policy_entropy": float(entropy.mean()),
    }


def _python_selfplay(bin_path: Path, games: int, simulations: int) -> dict:
    """The Python self-play loop of train/loop.py, without the shards."""
    import reversi_mcts

    model = create_model()
    model.load_weights(weights_io.load(bin_path))
    pool = reversi_mcts.BatchedSelfPlay(256, simulations, stagger=False)
    parts: list[tuple[np.ndarray, ...]] = []
    done = 0
    while done < games:
        planes = np.frombuffer(pool.collect(), dtype=np.float32).reshape(-1, 4, 8, 8)
        policies, values = model.infer(planes)
        pool.apply(
            np.ascontiguousarray(policies, dtype=np.float32).tobytes(),
            np.ascontiguousarray(values, dtype=np.float32).tobytes(),
        )
        for record in pool.take_finished():
            parts.append(materialize_record(record))
            done += 1
    keys = (
        "planes",
        "policies",
        "outcomes",
        "margins",
        "policy_weights",
        "value_weights",
    )
    return {k: np.concatenate([p[i] for p in parts]) for i, k in enumerate(keys)}


def test_statistics(
    tmp: Path, bin_path: Path, games: int = 512, sims: int = 32
) -> None:
    run = tmp / "stats"
    run.mkdir()
    (run / "selfplay_latest.bin").write_bytes(bin_path.read_bytes())
    proc = _selfplay(
        run,
        "--simulations", str(sims),
        "--workers", "2",
        "--pools-per-worker", "1",
        "--games-per-pool", "256",
        "--batch-target", "512",
        "--no-stagger",
        "--max-games", str(games),
    )  # fmt: skip
    output, _ = proc.communicate(timeout=600)
    assert proc.returncode == 0, output
    rust = _stats(_load_shards(run / "replay_spool"))
    python = _stats(_python_selfplay(bin_path, games, sims))
    for key in ("games", "moves_per_game", "black_win_rate", "policy_entropy"):
        print(f"  {key:15s} rust {rust[key]:8.3f}   python {python[key]:8.3f}")
    assert abs(rust["moves_per_game"] - python["moves_per_game"]) < 1.5
    assert abs(rust["black_win_rate"] - python["black_win_rate"]) < 0.1
    assert abs(rust["policy_entropy"] - python["policy_entropy"]) < 0.1
    print("✓ game statistics match Python self-play")


def main() -> None:
    if not BINARY.exists():
        sys.exit(
            f"{BINARY} not found: run `just build-selfplay` or set REVERSI_SELFPLAY_BIN"
        )
    print(f"backend: {backend_name()}, binary: {BINARY}")
    weights = os.environ.get("REVERSI_TEST_WEIGHTS")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        test_infer_parity(tmp)
        # Real weights give realistic games; trained-on-noise ones otherwise.
        bin_path = Path(weights) if weights else tmp / "game.bin"
        if not weights:
            weights_io.save_bin(_trained_model().weights(), bin_path)
        test_shards_and_shutdown(tmp, bin_path)
        test_statistics(tmp, bin_path)
    print("All Rust self-play tests passed!")


if __name__ == "__main__":
    main()
