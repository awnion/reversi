"""
Tournament to select the true champion from saved checkpoints.

After the tournament:
  - exports the winner to weights/champion.bin
  - rebuilds the WASM bot
  - saves the champion weights as the starting point for next training

The current champion (champion.bin) is always included as a reference
baseline. If it wins, no export/rebuild is done (nothing changed).

Usage:
    uv run python -m train.tournament [--top N] [--games G] [--sims S]
                                      [--no-champion]
"""

import argparse
import os
import struct
import subprocess
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import mlx.core as mx
import numpy as np

from .eval_server import board_to_planes
from .export import MAGIC, VERSION, export_model
from .loop import _load_compatible_weights, _record_champion
from .model import AlphaZeroNet

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
REPO_ROOT = WEIGHTS_DIR.parent
CHAMPION_BIN = WEIGHTS_DIR / "champion.bin"


def load_model(checkpoint: Path) -> AlphaZeroNet:
    model = AlphaZeroNet()
    data = np.load(str(checkpoint))
    weights = [(k, mx.array(data[k])) for k in data.files]
    _load_compatible_weights(model, weights)
    mx.eval(model.parameters())
    model.eval()
    return model


def load_model_from_bin(bin_path: Path) -> AlphaZeroNet:
    """Load model weights from the custom .bin export format."""
    weights = []
    with open(bin_path, "rb") as f:
        magic, version = struct.unpack("<IH", f.read(6))
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"Unrecognised .bin format in {bin_path}")
        (count,) = struct.unpack("<H", f.read(2))
        for _ in range(count):
            (name_len,) = struct.unpack("<B", f.read(1))
            name = f.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<B", f.read(1))
            shape = struct.unpack(f"<{ndim}I", f.read(ndim * 4))
            n_elems = 1
            for d in shape:
                n_elems *= d
            arr = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).reshape(shape)
            weights.append((name, mx.array(arr)))
    model = AlphaZeroNet()
    _load_compatible_weights(model, weights)
    mx.eval(model.parameters())
    model.eval()
    return model


def make_eval_fn(model: AlphaZeroNet):
    def evaluate(board_black: int, board_white: int, is_black: bool, legal: int):
        planes = board_to_planes(board_black, board_white, is_black, legal)
        x = mx.array(planes[None])
        with mx.stream(mx.cpu):
            policy_logits, value = model(x)
            mx.eval(policy_logits, value)
        return policy_logits[0].tolist(), float(value[0])

    return evaluate


# Module-level state for multiprocessing workers (populated by initializer)
_worker_eval_fns: dict = {}


def _worker_init(checkpoint_paths: list):
    for cp in checkpoint_paths:
        cp = Path(cp)
        m = load_model_from_bin(cp) if cp.suffix == ".bin" else load_model(cp)
        _worker_eval_fns[str(cp)] = make_eval_fn(m)


def _play_game(args):
    import reversi_mcts

    cp_a, cp_b, black_is_a, sims = args
    black, white = (cp_a, cp_b) if black_is_a else (cp_b, cp_a)
    outcome = reversi_mcts.run_match(
        _worker_eval_fns[black], _worker_eval_fns[white], sims
    )
    return cp_a, cp_b, black_is_a, outcome


def run_tournament(
    checkpoints: list[Path],
    games_per_pair: int,
    sims: int,
    workers: int | None = None,
) -> tuple[list[tuple[Path, float]], dict[Path, AlphaZeroNet]]:
    """Returns (ranking, models_dict) — models kept in memory to allow safe export."""
    n_workers = workers or max(1, (os.cpu_count() or 4) // 2)
    cp_strs = [str(cp) for cp in checkpoints]
    pairs = list(combinations(cp_strs, 2))
    tasks = [
        (cp_a, cp_b, i % 2 == 0, sims)
        for cp_a, cp_b in pairs
        for i in range(games_per_pair)
    ]
    total = len(tasks)

    print(f"Loading {len(checkpoints)} models... workers={n_workers}")
    scores: dict[str, float] = {cp: 0.0 for cp in cp_strs}

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(cp_strs,),
    ) as pool:
        for done, (cp_a, cp_b, black_is_a, outcome) in enumerate(
            pool.imap_unordered(_play_game, tasks), 1
        ):
            black, white = (cp_a, cp_b) if black_is_a else (cp_b, cp_a)
            if outcome > 0:
                scores[black] += 1.0
            elif outcome < 0:
                scores[white] += 1.0
            else:
                scores[black] += 0.5
                scores[white] += 0.5
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  [{done}/{total}]", end="\r", flush=True)

    print()

    # Load models in main process only for the winner export
    def _load(cp: Path) -> AlphaZeroNet:
        return load_model_from_bin(cp) if cp.suffix == ".bin" else load_model(cp)

    ranking_paths = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Only load the winner model (needed for potential export)
    winner_cp = Path(ranking_paths[0][0])
    models = {winner_cp: _load(winner_cp)}

    return [(Path(cp), s) for cp, s in ranking_paths], models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="Top N checkpoints by loss")
    parser.add_argument(
        "--games", type=int, default=2, help="Games per pair (use even number)"
    )
    parser.add_argument(
        "--sims", type=int, default=50, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Parallel game workers"
    )
    parser.add_argument(
        "--no-champion",
        action="store_true",
        help="Exclude the current champion.bin from the tournament",
    )
    args = parser.parse_args()

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    checkpoints = sorted(WEIGHTS_DIR.glob("iter_*.npz"), key=_loss)[: args.top]
    if not checkpoints:
        print("No checkpoints found in", WEIGHTS_DIR)
        return

    # Optionally include the current deployed champion as a reference baseline
    include_champion = not args.no_champion and CHAMPION_BIN.exists()
    if include_champion:
        checkpoints = [CHAMPION_BIN] + checkpoints

    print(
        f"Tournament: {len(checkpoints)} models"
        f" · {args.games} games/pair · {args.sims} sims/move"
    )
    for cp in checkpoints:
        tag = " [current champion]" if cp == CHAMPION_BIN else ""
        print(f"  {cp.name}{tag}")
    print()

    ranking, models = run_tournament(checkpoints, args.games, args.sims, args.workers)

    max_pts = (len(checkpoints) - 1) * args.games
    print("\n=== Results ===")
    for rank, (cp, score) in enumerate(ranking, 1):
        tag = " [current champion]" if cp == CHAMPION_BIN else ""
        print(f"  {rank:2d}. {score:5.1f}/{max_pts}  {cp.name}{tag}")

    winner_path = ranking[0][0]
    print(f"\nWinner: {winner_path.name}")

    if winner_path == CHAMPION_BIN:
        print("Current champion is still the best — no export needed.")
        return

    # Export new champion directly from the in-memory model
    # (the original .npz may have been pruned by a concurrent training run)
    output = WEIGHTS_DIR / "champion.bin"
    export_model(models[winner_path], output)

    # Record in champion history
    def _loss_from_path(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    winner_loss = _loss_from_path(winner_path)
    max_pts = (len(ranking) - 1) * args.games
    winner_score = ranking[0][1]
    win_rate = winner_score / max_pts if max_pts > 0 else 0.0
    _record_champion(models[winner_path], win_rate, winner_loss)

    # Rebuild WASM
    print("Rebuilding WASM...")
    subprocess.run(
        ["bun", "--filter", "@reversi/bot-alphazero", "build:wasm"],
        cwd=REPO_ROOT,
        check=True,
    )
    print("WASM rebuilt.")

    # Next training run will pick up winner via _load_best_checkpoint
    print(f"Next training will resume from {winner_path.name}")


if __name__ == "__main__":
    main()
