# Reversi

A browser-based [Reversi](https://en.wikipedia.org/wiki/Reversi) (Othello) game built with TypeScript.

## Structure

Bun workspace monorepo:

- `packages/core` — game rules and logic
- `packages/board-ui` — DOM-based board renderer
- `apps/web` — playable web app

## Quick start

```sh
bun install
bun dev
```

Open `http://localhost:5173` in your browser.

## Build

```sh
bun run build
```

## Test

```sh
# unit tests (core logic)
bun run test:unit

# browser integration tests (vitest + playwright)
bun run test:browser

# e2e tests (requires browsers: bun run install:browsers)
bun run test:e2e
```

## AlphaZero training

The AlphaZero bot is trained via self-play using Rust MCTS + Python/MLX neural network.
All training commands run from `python/`:

```sh
cd python
```

### Prerequisites

```sh
# Install Python deps + build Rust extension
uv sync
uv run maturin develop --features python
```

### Run training

```sh
PYTHONUNBUFFERED=1 uv run python -m train.loop > /tmp/reversi_train.log 2>&1 &
tail -f /tmp/reversi_train.log
```

Training auto-resumes from the best checkpoint in `weights/` on restart.
Checkpoints are saved to `weights/iter_<step>_loss<value>.npz`.

### Run tournament

After accumulating checkpoints, run a round-robin tournament to find the true champion
(loss alone doesn't reflect actual playing strength):

```sh
# Top 10 checkpoints by loss, 4 games per pair, 50 MCTS sims per move
uv run python -m train.tournament --top 10 --games 4 --sims 50
```

The tournament automatically:
1. Plays all pairs against each other (alternating colors)
2. Ranks by win points
3. Exports the winner to `weights/champion.bin`
4. Rebuilds the WASM bot

### Export champion manually

```sh
uv run python -m train.export \
  --checkpoint weights/iter_<step>_loss<value>.npz \
  --output weights/champion.bin
```

### Rebuild WASM bot

After exporting a new `champion.bin`, rebuild the browser bot:

```sh
cd ..  # repo root
bun run build:wasm
```

The WASM bot embeds `weights/champion.bin` at compile time — the file must exist before building.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
