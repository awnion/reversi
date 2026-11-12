# AGENTS.md

Browser Reversi (Othello) with bots: TypeScript UI, Rust engines compiled to
WASM, and a Python self-play training pipeline (MLX or PyTorch).

## Layout

- `rust/minimax`, `rust/mcts`, `rust/nn` — bitboard engine, MCTS, NN inference
- `rust/alphazero`, `rust/zerorev` — WASM bot crates; each embeds
  `weights/<bot>/champion.bin` via `include_bytes!`
- `packages/core` — game rules; `packages/board-ui` — DOM board;
  `packages/bot` — greedy bot; `packages/bot-{minimax,alphazero,zerorev}` — TS
  wrappers around the WASM crates
- `apps/web` — Vite app; bots run in a Web Worker (`src/botWorker.ts`)
- `e2e` — Playwright specs against the production bundle
- `python/train` — training loop, backends, tournament, export (uv project,
  not in Bazel)
- `tools/` — `build_wasm.sh` (Bazel → `packages/bot-*/wasm`) and the workspace
  status script

## Skills

Agent skills live in `.agents/skills/` (`.claude/skills` links there).
`frontend-design` is Anthropic's design skill: read it before changing the UI
in `apps/web` or `packages/board-ui`. The interface is deliberately minimal —
the board, the two player columns, one status line.

## Build system

Two layers with a hard boundary:

- **Bazel** (bzlmod, version in `.bazelversion`) compiles Rust and produces the
  WASM bots — nothing else. `apps/`, `packages/`, `e2e/` and `python/` are in
  `.bazelignore`; do not add `BUILD.bazel` files or JS rules there.
- **Bun** owns everything JS/TS: dependencies (`bun.lock`), scripts, unit tests
  (`bun test`), Vite for dev/build.

The bridge is `tools/build_wasm.sh` (`bun run build:wasm`): it runs
`bazel build //rust/<bot>:wasm` and copies the wasm-bindgen output into
`packages/bot-<bot>/wasm/` (gitignored), which the TS wrappers import as
`../wasm/reversi_<bot>.js`.

One deliberate exception to "Bazel builds Rust": the WASM bots also get a
multi-threaded build in `packages/bot-<bot>/wasm-threads/` (feature `threads`:
the network bots share one search tree between Web Workers, Minimax splits the
root moves between them). It needs the standard library
rebuilt with atomics, which rules_rust cannot do, so the same script builds it
with `cargo +nightly -Z build-std` and the wasm-bindgen CLI that Bazel built.
Nightly Rust is fine in this project. The page loads the threaded build only
when it is cross-origin isolated (COOP/COEP headers, set in
`apps/web/vite.config.ts`); otherwise the bots search on one thread. Any host
serving the bundle must send the same two headers to get threads.

```sh
bun install
bun dev                                # build wasm + Vite dev server
bun run build                          # build wasm + production bundle
bazel test //...                       # Rust tests
bun run test:unit                      # build wasm + bun test
bun run test:e2e                       # Playwright against the bundle
bazel build --config=clippy //rust/... # clippy
just lint / just fmt                   # biome, tsc, ruff, rustfmt, taplo
```

- e2e tests need `bun run install:browsers` once.
- Bazel defaults to `-c opt` (the bots are compute-bound); `--config=dbg` for
  debug builds.
- `BUILD_DATE` is stamped into the WASM bots from `tools/workspace_status.sh`.
- After changing Rust code, rerun `bun run build:wasm`; Vite and `bun test`
  only see the copied output.
- `bun test` and `just vs-minimax` run outside a browser and always use the
  single-threaded build; threads are exercised by the e2e tests.

## Conventions

- TS packages export sources directly (`"main": "./src/index.ts"`); `tsc -b`
  is used for typechecking only.
- Unit tests live in `<pkg>/test/*.test.ts` and import from `bun:test` — the
  only test runner; do not add vitest/jest. DOM tests register happy-dom
  inside the file (see `packages/board-ui/test`), not globally: the WASM bot
  tests need Bun's native `fetch`. Real browsers are covered by `e2e/`.
- Rust deps: `Cargo.toml` + `Cargo.lock` are the source of truth
  (crate_universe). `wasm-bindgen` must stay at the exact version bundled with
  `rules_rust_wasm_bindgen`.
- Adding a bot: crate in `rust/<bot>` (add to the Cargo workspace, to
  `crate.from_cargo` manifests in `MODULE.bazel` and to `tools/build_wasm.sh`),
  wrapper package in `packages/bot-<bot>`, weights in `weights/<bot>/`, and
  register it in `apps/web/src/botWorker.ts`.
- Lint: biome (TS/JSON), clippy + rustfmt (nightly features, see
  `rustfmt.toml`), ruff (Python), taplo (TOML). Non-null assertions are
  rejected by biome.

## Training (Python, outside Bazel)

```sh
cd python && uv sync && uv run maturin develop --release --features python
just train <bot> / just logs <bot> / just stop <bot> / just tournament <bot>
```

- **Two ML backends, one loop.** `python/train/backend/` implements the same
  small interface twice: MLX on Apple Silicon, PyTorch on Linux/CUDA (chosen by
  platform; override with `REVERSI_BACKEND=mlx|torch`). Everything else is
  shared: batched self-play and tournaments run in Rust (`rust/mcts`), and
  `train/loop.py` must stay free of framework imports.
- **Checkpoints are framework-neutral.** `.npz` and `.bin` hold named float32
  arrays in the canonical (MLX) layout described in `train/weights.py`; a run
  trained on one machine continues on the other. The Rust inference code
  (`rust/nn`, browser bots) reads the same `.bin`. `train.test_smoke` checks
  the backend against the Rust inference; run it on each machine after
  touching a backend or the format.
- **Remote machine:** `just setup-remote <host>`, `just push-run <host> <bot>`,
  `just train-remote <host> <bot>`, `just pull-run <host> <bot>`.

Training writes to `weights/<bot>/`. Do not delete or rewrite files there
(checkpoints, replay buffers, `champion.bin`) unless asked; a training run may
be in progress (`weights/<bot>/train.pid`).
