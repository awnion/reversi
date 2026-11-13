"""
Tournament to select the true champion from saved checkpoints.

After the tournament:
  - exports the winner to weights/<run>/champion.bin
  - rebuilds the WASM bots
  - training is not affected: it resumes from the newest checkpoint
    (iter_*.npz), and from champion.bin only when that is newer

The current champion (champion.bin) is always included as a reference
baseline. If it wins, no export/rebuild is done (nothing changed).

The games are played in Rust on CPU threads with the same inference code as
the browser bots, so this needs no GPU and no ML framework.

Usage:
    uv run python -m train.tournament --run-name NAME [--top N] [--games G]
                                      [--sims S] [--no-champion]
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import reversi_mcts

from . import loop
from . import weights as weights_io

_WEIGHTS_BASE = Path(__file__).parent.parent.parent / "weights"
REPO_ROOT = _WEIGHTS_BASE.parent


def run_tournament(
    checkpoints: list[Path],
    games_per_pair: int,
    sims: int,
    workers: int | None = None,
    opening_plies: int = 4,
) -> list[tuple[Path, float]]:
    """Round-robin between `checkpoints`; returns them ranked with their points."""
    threads = workers or max(1, (os.cpu_count() or 4) // 2)
    blobs = [weights_io.to_bytes(weights_io.load(cp)) for cp in checkpoints]
    scores = reversi_mcts.run_tournament(
        blobs,
        games_per_pair,
        sims,
        opening_plies=opening_plies,
        seed=time.time_ns() & 0xFFFF_FFFF,
        threads=threads,
    )
    ranked = zip(checkpoints, scores, strict=True)
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Named run; reads weights from weights/<run-name>/",
    )
    parser.add_argument("--top", type=int, default=10, help="Top N checkpoints by loss")
    parser.add_argument(
        "--games", type=int, default=6, help="Games per pair (use even number)"
    )
    parser.add_argument(
        "--sims", type=int, default=200, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Parallel game threads"
    )
    parser.add_argument(
        "--no-champion",
        action="store_true",
        help="Exclude the current champion.bin from the tournament",
    )
    args = parser.parse_args()

    loop._apply_run_name(args.run_name)
    weights_dir = _WEIGHTS_BASE / args.run_name
    champion_bin = weights_dir / "champion.bin"

    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss")[-1])
        except ValueError:
            return float("inf")

    checkpoints = sorted(weights_dir.glob("iter_*.npz"), key=_loss)[: args.top]
    if not checkpoints:
        print("No checkpoints found in", weights_dir)
        return

    # Optionally include the current deployed champion as a reference baseline
    if not args.no_champion and champion_bin.exists():
        checkpoints = [champion_bin] + checkpoints

    print(
        f"Tournament: {len(checkpoints)} models"
        f" · {args.games} games/pair · {args.sims} sims/move"
    )
    for cp in checkpoints:
        tag = " [current champion]" if cp == champion_bin else ""
        print(f"  {cp.name}{tag}")
    print()

    started = time.monotonic()
    ranking = run_tournament(checkpoints, args.games, args.sims, args.workers)

    max_pts = (len(checkpoints) - 1) * args.games
    print(f"=== Results ({time.monotonic() - started:.0f}s) ===")
    for rank, (cp, score) in enumerate(ranking, 1):
        tag = " [current champion]" if cp == champion_bin else ""
        print(f"  {rank:2d}. {score:5.1f}/{max_pts}  {cp.name}{tag}")

    winner_path, winner_score = ranking[0]
    print(f"\nWinner: {winner_path.name}")

    if winner_path == champion_bin:
        print("Current champion is still the best — no export needed.")
        return

    winner = weights_io.load(winner_path)
    weights_io.save_bin(winner, champion_bin)
    win_rate = winner_score / max_pts if max_pts > 0 else 0.0
    try:
        step = int(winner_path.stem.split("_")[1])
    except (IndexError, ValueError):
        step = None
    loop._record_champion(winner, win_rate, _loss(winner_path), step=step)

    # Rebuild WASM
    print("Rebuilding WASM...")
    subprocess.run(
        ["bun", "run", "build:wasm"],
        cwd=REPO_ROOT,
        check=True,
    )
    print("WASM rebuilt.")

    print(f"Exported {winner_path.name} as champion.bin; training is not affected.")


if __name__ == "__main__":
    main()
