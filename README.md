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
# Start in background, append to log
nohup uv run python -u -m train.loop >> /tmp/reversi_train.log 2>&1 &

# Watch the log
tail -f /tmp/reversi_train.log
```

Training runs continuously and:
- Waits until 30 000 positions are collected before starting gradient updates
- Uses 80 MCTS simulations per move during self-play
- Rate-limits training to 1 gradient step per 10 new positions (prevents overfitting)
- Uses AdamW (`lr=3e-4`, `weight_decay=1e-4`) with step-decay ×0.1 every 5 000 steps
- Policy targets use label smoothing (ε=0.1) to prevent memorisation of peaked MCTS distributions;
  loss floor with smoothing is ~0.69, healthy training range is 0.70–0.80
- Auto-resumes from `weights/champion.bin` on restart; falls back to the latest checkpoint in `weights/`
- Stores the legal-move bitmask in replay so training sees the same spatial planes as self-play/eval
- Adds a phase feature (`occupied_count / 64`) so the net can distinguish opening from endgame
- Adds value-only endgame samples for forced last moves and terminal boards
- Self-play uses stochastic openings (root noise + sampling from visit counts in the early game)
- Saves checkpoints to `weights/iter_<step>_loss<value>.npz`
- Runs a **mini-tournament** every 3 000 steps: `champion.bin`, the current model,
  and 2 recent checkpoints play a small round-robin (8 games per pairing)
- If the winner is not the current champion and scores at least 55 % of the maximum points,
  it becomes the new champion automatically
- Keeps the last 5 champions with date and win-rate in `weights/champions/history.json`

### Run tournament (manual)

Run a round-robin tournament over the top N checkpoints to find the true champion
(loss alone doesn't reflect playing strength):

```sh
# Top 10 checkpoints by loss + current champion as baseline, 4 games/pair, 100 sims/move
uv run python -m train.tournament --top 10 --games 4 --sims 100

# Exclude the current champion from the bracket
uv run python -m train.tournament --top 10 --games 4 --sims 100 --no-champion
```

The tournament automatically:
1. Loads all candidates (including `champion.bin` as reference baseline)
2. Plays every pair against each other, alternating colours
3. Ranks by win points
4. If the winner is not the current champion: exports it to `weights/champion.bin`,
   rebuilds the WASM bot, and records it in `weights/champions/history.json`

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
bun --filter '@reversi/bot-alphazero' build:wasm
```

The WASM bot embeds `weights/champion.bin` at compile time — the file must exist before building.

### Champion history

Past champions are stored in `weights/champions/` (last 5 kept):

```
weights/champions/
  history.json                              # date, loss, win_rate for each
  champion_20260324_142030_loss0.0938.npz   # model weights snapshot
  ...
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
