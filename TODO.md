# TODO — AlphaZero MCTS Bot

## Phase 1 — Rust MCTS crate (`rust/mcts`)

- [x] Add `rust/mcts` to Cargo workspace
- [x] `rust/mcts/Cargo.toml` with dep on `reversi_minimax`
- [x] `EvalFn` trait + `StaticEval` (uniform policy + normalized minimax score)
- [x] `Node` arena + `MctsSearch` struct
- [x] MCTS: `select` (PUCT), `expand_and_eval`, `backup` (negamax)
- [x] `best_move()` and `policy_target()` on `MctsSearch`
- [x] `play_game()` — full self-play game returning `GameRecord`
- [x] Tests: policy sums to 1, best move is legal, game completes

## Phase 2 — PyO3 bindings (`reversi_mcts` Python extension)

- [x] Add `pyo3` + `maturin` to `rust/mcts` dependencies
- [x] `#[pyclass] MctsWorker` with `run_game(eval_fn)` method
- [x] `GameRecord` → Python dict conversion
- [x] `uv add maturin --dev` + `maturin develop` build target
- [x] Python smoke test: `import reversi_mcts; reversi_mcts.MctsWorker(200).run_game(...)`

## Phase 3 — MLX model + LeafEvalServer

- [x] `python/train/model.py` — `AlphaZeroNet` (mlx.nn, 4 ResBlocks × 32ch)
- [x] `python/train/eval_server.py` — `LeafEvalServer` (asyncio batching, batch=64)
- [x] `python/train/replay.py` — `ReplayBuffer` (numpy ring buffer, 500K positions)
- [x] Smoke test: model forward pass, eval server handles concurrent requests

## Phase 4 — Training loop

- [x] `python/train/loss.py` — policy cross-entropy + value MSE
- [x] `python/train/loop.py` — asyncio main: workers + eval_server + train_loop
- [x] `python/train/export.py` — weights → binary format for Rust

## Phase 5 — Checkpoint eval & promotion

- [x] Every 1000 steps: save checkpoint `.npz`
- [x] 50 eval games: candidate vs previous champion via `MctsWorker`
- [x] Auto-promote if win_rate > 55%

## Phase 6 — Rust NN inference (`rust/nn`)

- [x] Binary weight format: `[magic u32][version u16][n_tensors u16][layer blobs]`
- [x] `Conv2d`, `BatchNorm2d`, `Linear`, `relu`, `tanh_inplace`, `softmax`
- [x] `AlphaZeroNet::load(bytes: &[u8])` + `forward(planes) → ([f32;64], f32)`
- [x] 5 unit tests, clippy clean

## Phase 7 — AlphaZero WASM bot

- [x] `rust/alphazero` crate: MCTS + `rust/nn` + `include_bytes!` champion weights
- [ ] wasm-pack build → `packages/bot-alphazero/wasm` (run: `bun run build:wasm`)
- [x] TypeScript wrapper implementing `BotPlayer`
- [x] Add to web app tsconfig + path aliases
