# Reversi

A browser-based [Reversi](https://en.wikipedia.org/wiki/Reversi) (Othello) game built with TypeScript.

## Structure

Rust is built by Bazel, TypeScript by Bun/Vite, and the Python training
pipeline by uv:

- `rust/minimax`, `rust/mcts`, `rust/nn` — engine, search and inference crates
- `rust/alphazero`, `rust/zerorev` — WASM bots with champion weights embedded
- `packages/core` — game rules and logic
- `packages/board-ui` — DOM-based board renderer
- `packages/bot`, `packages/bot-*` — bot players (the `bot-*` ones wrap the WASM crates)
- `apps/web` — playable web app (Vite)
- `python/train` — self-play training (MLX on Apple Silicon, PyTorch on Linux/CUDA)
- `weights/<bot>/` — checkpoints and `champion.bin` per bot

## Prerequisites

- [Bazelisk](https://github.com/bazelbuild/bazelisk) (installed as `bazel`) —
  compiles Rust to WASM; it downloads the Rust toolchain and wasm-bindgen itself
- [Bun](https://bun.sh) — package manager, test runner and script runner
- Rust nightly with `rust-src` and the `wasm32-unknown-unknown` target — for
  the multi-threaded bots:
  `rustup toolchain install nightly --component rust-src --target wasm32-unknown-unknown`
- optional: [`just`](https://github.com/casey/just) wraps the common commands

## Quick start

```sh
bun install
bun dev
```

`bun dev` first runs `bun run build:wasm` (Bazel builds the bots and the output
is copied to `packages/bot-*/wasm/`), then starts Vite. Open
`http://localhost:5173` in your browser.

## Build

```sh
bun run build
```

Minimax, AlphaZero and ZeroRev search on several threads when the page is cross-origin
isolated. The dev and preview servers send the required headers; a production
host has to send them too, otherwise the bots fall back to one thread:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Test

```sh
# Rust tests
bazel test //...

# unit tests (bun test; builds the WASM bots first)
bun run test:unit

# e2e tests (requires browsers: bun run install:browsers)
bun run test:e2e
```

## Lint and format

```sh
just lint
just fmt
```

Rust dependencies: edit `Cargo.toml`, then `cargo update` — Bazel reads
`Cargo.lock`. `wasm-bindgen` is pinned to the version of the CLI bundled with
`rules_rust_wasm_bindgen`; bump both together.

## AlphaZero training

The AlphaZero bot is trained via self-play: Rust MCTS plus a neural network in Python (MLX on
Apple Silicon, PyTorch on Linux/CUDA; checkpoints are interchangeable).
All training commands run from `python/`:

```sh
cd python
```

### Prerequisites

```sh
# Install Python deps + build Rust extension
uv sync
uv run maturin develop --release --features python
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
- Uses 400 MCTS simulations per move during self-play (reusing the subtree of the move played), always with the newest weights
- Does 1 gradient step (batch 512) per 100 new positions, so each position is
  trained on about 5 times
- Keeps the last 2 000 000 positions in the replay buffer
- Solves the last 12 empty squares exactly instead of searching them
- Turns every training position to a random one of the 8 symmetries of the board
- Learns the move policy, the game result and the final disc margin
  (an auxiliary head, not used by the bots)
- Uses AdamW with a learning-rate schedule of a short warmup and three long
  constant stages (`LR_STAGES` in `train/loop.py`); the schedule's start is
  kept in `weights/<run>/lr_schedule.json`, so restarts continue it
- Logs the loss and its parts: `policy_kl` (the part of the policy loss that
  can still go down), `value` and `margin`
- Self-play uses stochastic openings (root noise + sampling from visit counts in the early game)
- Every 30 minutes (and at least 5 000 steps) plays the current network against `champion.bin` (200 games);
  it becomes the new champion with at least 55 % of the points
- Keeps the last 5 champions with date and win-rate in `weights/<run>/champions/history.json`

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

After exporting a new `champion.bin`, rebuild the browser bots:

```sh
bun run build:wasm
```

The WASM bots embed `weights/<bot>/champion.bin` at compile time — the files
must exist before building.

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
