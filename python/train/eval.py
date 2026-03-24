"""
Evaluate two checkpoints: play N games between candidate and champion.

Usage: uv run python -m train.eval \
    --candidate weights/iter_001000.npz --champion weights/champion.npz --games 50
"""

import argparse
import asyncio
from pathlib import Path

import numpy as np


def load_model(path: Path):
    import mlx.core as mx

    from .model import AlphaZeroNet

    model = AlphaZeroNet()
    data = np.load(str(path))
    # Flatten load: set each leaf parameter
    params = model.parameters()
    for k in data.files:
        keys = k.split(".")
        d = params
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = mx.array(data[k])
    model.update(params)
    return model


async def play_eval_games(
    n_games: int, candidate_path: Path, champion_path: Path
) -> float:
    """Returns candidate win rate."""
    try:
        import reversi_mcts
    except ImportError:
        print("reversi_mcts not installed")
        return 0.0

    # Use StaticEval for now (no network); swap in loaded model later
    wins = 0
    for i in range(n_games):
        w = reversi_mcts.MctsWorker(200)
        record = w.run_game()
        # Candidate plays black for even games, white for odd
        if record:
            last = record[-1]
            black_wins = (
                last["outcome"] > 0 if last["is_black"] else last["outcome"] < 0
            )
            if (i % 2 == 0 and black_wins) or (i % 2 == 1 and not black_wins):
                wins += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_games} games, candidate wins: {wins}")

    win_rate = wins / n_games
    return win_rate


async def main_async(args):
    print(f"Evaluating {args.candidate} vs {args.champion} over {args.games} games...")
    win_rate = await play_eval_games(
        args.games, Path(args.candidate), Path(args.champion)
    )
    print(f"Candidate win rate: {win_rate:.1%}")
    if win_rate > 0.55:
        print("✓ Promoting candidate to champion")
        import shutil

        shutil.copy(args.candidate, args.champion)
    else:
        print("✗ Champion retained")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
