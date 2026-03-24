"""
Tournament to select the true champion from saved checkpoints.

After the tournament:
  - exports the winner to weights/champion.bin
  - rebuilds the WASM bot
  - saves the champion weights as the starting point for next training

Usage:
    uv run python -m train.tournament [--top N] [--games G] [--sims S]
"""

import argparse
import subprocess
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import numpy as np

from .eval_server import board_to_planes
from .export import export
from .model import AlphaZeroNet

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
REPO_ROOT = WEIGHTS_DIR.parent


def load_model(checkpoint: Path) -> AlphaZeroNet:
    model = AlphaZeroNet()
    data = np.load(str(checkpoint))
    weights = [(k, mx.array(data[k])) for k in data.files]
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model


def make_eval_fn(model: AlphaZeroNet):
    def evaluate(board_black: int, board_white: int, is_black: bool, legal: int):
        planes = board_to_planes(board_black, board_white, is_black, legal)
        x = mx.array(planes[None])
        policy_logits, value = model(x)
        mx.eval(policy_logits, value)
        return policy_logits[0].tolist(), float(value[0])

    return evaluate


def run_tournament(
    checkpoints: list[Path],
    games_per_pair: int,
    sims: int,
) -> list[tuple[Path, float]]:
    import reversi_mcts

    print(f"Loading {len(checkpoints)} models...")
    eval_fns = {cp: make_eval_fn(load_model(cp)) for cp in checkpoints}

    scores: dict[Path, float] = {cp: 0.0 for cp in checkpoints}
    pairs = list(combinations(checkpoints, 2))
    total = len(pairs) * games_per_pair
    done = 0

    for cp_a, cp_b in pairs:
        for i in range(games_per_pair):
            black, white = (cp_a, cp_b) if i % 2 == 0 else (cp_b, cp_a)
            outcome = reversi_mcts.run_match(eval_fns[black], eval_fns[white], sims)
            if outcome > 0:
                scores[black] += 1.0
            elif outcome < 0:
                scores[white] += 1.0
            else:
                scores[black] += 0.5
                scores[white] += 0.5
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  [{done}/{total}]", end="\r", flush=True)

    print()
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="Top N checkpoints by loss")
    parser.add_argument(
        "--games", type=int, default=4, help="Games per pair (use even number)"
    )
    parser.add_argument(
        "--sims", type=int, default=50, help="MCTS simulations per move"
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

    print(
        f"Tournament: {len(checkpoints)} models"
        f" · {args.games} games/pair · {args.sims} sims/move"
    )
    for cp in checkpoints:
        print(f"  {cp.name}")
    print()

    ranking = run_tournament(checkpoints, args.games, args.sims)

    max_pts = (len(checkpoints) - 1) * args.games
    print("\n=== Results ===")
    for rank, (cp, score) in enumerate(ranking, 1):
        print(f"  {rank:2d}. {score:5.1f}/{max_pts}  {cp.name}")

    champion_path = ranking[0][0]
    print(f"\nChampion: {champion_path.name}")

    # Export to champion.bin
    output = WEIGHTS_DIR / "champion.bin"
    export(champion_path, output)

    # Rebuild WASM
    print("Rebuilding WASM...")
    subprocess.run(
        ["bun", "--filter", "@reversi/bot-alphazero", "build:wasm"],
        cwd=REPO_ROOT,
        check=True,
    )
    print("WASM rebuilt.")

    # Copy champion as the resume point for next training run
    # (loop.py will pick it up via _load_best_checkpoint)
    print(f"Next training will resume from {champion_path.name}")


if __name__ == "__main__":
    main()
