# AGENTS.md

Browser Reversi (Othello) with bots: TypeScript UI, Rust engines compiled to
WASM, and a Python/MLX self-play training pipeline.

## Layout

- `rust/minimax`, `rust/mcts`, `rust/nn` — bitboard engine, MCTS, NN inference
- `rust/alphazero`, `rust/zerorev` — WASM bot crates; each embeds
  `weights/<bot>/champion.bin` via `include_bytes!`
- `packages/core` — game rules; `packages/board-ui` — DOM board;
  `packages/bot` — greedy bot; `packages/bot-{minimax,alphazero,zerorev}` — TS
  wrappers around the WASM crates
- `apps/web` — Vite app; bots run in a Web Worker (`src/botWorker.ts`)
- `e2e` — Playwright specs against the production bundle
- `python/train` — training loop, tournament, export (uv project, not in Bazel)
- `tools/` — `build_wasm.sh` (Bazel → `packages/bot-*/wasm`) and the workspace
  status script

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

## Training (Python, Apple Silicon only)

Runs through uv, outside Bazel:

```sh
cd python && uv sync && uv run maturin develop --features python
just train <bot> / just logs <bot> / just stop <bot> / just tournament <bot>
```

Training writes to `weights/<bot>/`. Do not delete or rewrite files there
(checkpoints, replay buffers, `champion.bin`) unless asked; a training run may
be in progress (`weights/<bot>/train.pid`).
