# AlphaZero-style MCTS Bot — Implementation Plan

## How It Works

Classic MCTS uses **random rollouts** to evaluate positions. AlphaZero replaces them with a
**neural network** that outputs two things simultaneously:

- **Policy** — probability distribution over all moves
- **Value** — position evaluation for the current player (scalar in [-1, 1])

The search tree is built using the **PUCT formula**:
```
score(s, a) = Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```
where `P(s,a)` is the policy prior from the network and `N` is visit counts. This gives
directed search instead of random exploration.

---

## Key Design Decisions

### Network Architecture

A small network is sufficient for 8×8 Reversi:

```
Input:  3 × 8 × 8
          ├── current player's discs
          ├── opponent's discs
          └── legal moves mask

Body:   4 ResBlocks × 32 channels (3×3 conv + BN + ReLU)

Policy head:  Conv 2×1 → Flatten → 64 logits (one per square)
Value head:   Conv 1×1 → Flatten → Linear → Tanh → scalar
```

~100K parameters → ~200KB fp32 → ~50KB int8 after quantization.

### NN Inference in WASM

| Approach | Pros | Cons |
|----------|------|------|
| `tract` crate (ONNX) | Ready infrastructure | +3–5 MB to WASM |
| `candle` (HuggingFace) | Modern API | Heavy |
| Hand-rolled forward pass | Minimal size, full control | Need to implement Conv2d, BN |

**Choice: hand-rolled forward pass.** For a 4-block ResNet this is ~300 lines of Rust.
The WASM stays small. Weights are loaded from `&[u8]` embedded via `include_bytes!`.

### Training

Python + PyTorch self-play loop, managed with **uv**:

```
1. Run tournament: current_model vs current_model via MCTS
2. Record (state, policy_target, value_target) for each position
3. Train network on batches of recorded positions
4. Evaluate: candidate vs champion (run N games via tournament tool)
5. If candidate win rate > 55% → promote to champion
6. Repeat
```

Reversi 8×8 converges in a few hours on a GPU (or ~1 day on CPU). Trained weights are
exported to a binary format and committed to the repo, then embedded into the WASM binary.

---

## Tournament Tool

A dedicated **Rust CLI binary** for running games between any two strategies.
It is the engine driving both self-play data generation and model evaluation.

### Architecture

```
tools/tournament/
  src/
    main.rs      ← CLI: arg parsing, stats output
    runner.rs    ← plays N games between two strategies, rotates colors
    strategy.rs  ← Strategy trait + implementations
    record.rs    ← serialization of game records for training
```

Strategy trait:
```rust
trait Strategy {
    fn choose_move(&mut self, board: Board, is_black: bool) -> Option<u64>;
}
// Implementations: Random, Greedy, Minimax { depth }, Mcts { sims, network }
```

### CLI Interface

```bash
# Evaluation: candidate vs champion, 200 games
tournament --black mcts:weights=weights/candidate.bin \
           --white mcts:weights=weights/champion.bin \
           --games 200 --eval-mode

# Self-play data generation, saves game records
tournament --black mcts:weights=weights/v3.bin \
           --white mcts:weights=weights/v3.bin \
           --games 500 --save-records records/v3_selfplay.bin

# Ad-hoc: minimax vs random
tournament --black minimax:depth=6 --white random --games 1000
```

Output:
```
Games: 200  Black: 112W / 43L / 45D (58.5%)
ELO delta: +47
Records saved: records/v3_selfplay.bin (500 games, 34 821 positions)
```

### Game Record Format

Binary (not JSON) — one game ≈ 5 KB, 10K games ≈ 50 MB:
```
GameRecord {
  positions: Vec<{
    board_black:  u64,
    board_white:  u64,
    is_black:     bool,
    mcts_policy:  [f32; 64],   // MCTS visit count distribution
    outcome:      f32,         // +1 / −1 / 0 from current player's POV
  }>
}
```

### Python Training Loop

```python
# python/train/loop.py  (run with: uv run python -m train.loop)

for iteration in range(NUM_ITERATIONS):
    # 1. Generate self-play data
    run_tournament(
        black=f"mcts:weights=weights/champion.bin",
        white=f"mcts:weights=weights/champion.bin",
        games=500,
        save_records=f"records/iter_{iteration}.bin",
    )

    # 2. Train on collected records
    dataset = load_records(f"records/iter_{iteration}.bin")
    train_epoch(model, dataset)          # policy loss + value loss
    save_weights(model, "weights/candidate.bin")

    # 3. Evaluate candidate vs champion
    result = run_tournament(
        black="mcts:weights=weights/candidate.bin",
        white="mcts:weights=weights/champion.bin",
        games=200, eval_mode=True,
    )

    # 4. Promote if better
    if result.win_rate > 0.55:
        promote_candidate_to_champion()
```

---

## Implementation Phases

### Phase 1 — Pure MCTS (no network, structural validation)

New crate `rust/mcts`:
- **Selection** — PUCT with uniform prior (no network yet)
- **Expansion** — add a node for each legal move
- **Simulation** — random rollout OR minimax static eval at leaf
- **Backup** — update `W(s,a)` and `N(s,a)` up the tree

This gives a **working bot immediately**; the network plugs in later as an upgrade.
Reuses `Board` and `evaluate` from `rust/minimax`.

### Phase 2 — Tournament Runner

New binary `tools/tournament`:
- `Strategy` trait + `Random`, `Greedy`, `Minimax`, `MctsRollout` implementations
- Parallel game execution (rayon)
- Game record serialization
- ELO tracking across runs

This enables immediate benchmarking of Phase 1 MCTS against minimax,
and becomes the data pipeline for training.

### Phase 3 — NN Inference Engine in Rust

New crate `rust/nn`:

```rust
struct Network { /* layer weights */ }

impl Network {
    fn load(weights: &[u8]) -> Self { ... }
    fn forward(&self, planes: &[f32; 3 * 64]) -> (PolicyOutput, ValueOutput) { ... }
}
```

Layers: `Conv2d`, `BatchNorm2d`, `Linear`, `ReLU`, `Tanh`.
Weight binary format: custom (magic bytes + version + raw f32/i8 blobs per layer).

### Phase 4 — Python Self-Play Pipeline

```
python/                      # managed with uv
  pyproject.toml
  train/
    model.py       ← PyTorch model (mirrors Rust architecture exactly)
    self_play.py   ← calls tournament binary, parses records
    loop.py        ← main training loop
    export.py      ← export weights to binary format for Rust

weights/
  champion.bin     ← current champion (committed to repo)
  history/         ← checkpoint archive
```

Run with: `uv run python -m train.loop`

### Phase 5 — Integration: MCTS + NN → BotPlayer

New crate `rust/alphazero` (or extend `rust/mcts`):
- Loads weights via `include_bytes!("../../weights/champion.bin")`
- MCTS leaf evaluation: instead of rollout → `network.forward(state)`
- Exported via wasm-pack as `packages/bot-alphazero`
- Implements the same `BotPlayer` interface as `bot-minimax`
- Tournament `Strategy` implementation added: `MctsNN { weights_path }`

### Phase 6 — Improvements

- Int8 weight quantization (−75% weight size)
- Temperature parameter for Analysis mode (expose policy distribution)
- Dirichlet noise at tree root (exploration during self-play)
- Progressive widening for very early tree nodes

---

## Repository Structure

```
rust/
  minimax/          ← exists
  mcts/             ← new: pure MCTS + NN-guided MCTS
  nn/               ← new: hand-rolled inference engine

tools/
  tournament/       ← new: Rust CLI binary for running strategy vs strategy

python/             ← managed with uv (pyproject.toml)
  train/
    model.py
    loop.py
    self_play.py
    export.py

weights/
  champion.bin      ← trained weights committed to the repo
  history/          ← version archive

packages/
  bot-minimax/      ← exists
  bot-alphazero/    ← new: WASM wrapper, same BotPlayer interface
```

---

## Tooling & Linting

| Layer | Tool | How to run |
|-------|------|------------|
| Rust | `cargo fmt` + `cargo clippy` | `cargo fmt && cargo clippy --workspace` |
| JS/TS | Biome | `bunx biome check` |
| Python | Ruff | `uvx ruff check python/` |
| TOML | Taplo | `uvx taplo fmt` |
| All (pre-commit) | pre-commit | `uvx pre-commit run --all-files` |

Install hook: `uvx pre-commit install`

---

## Complexity Estimate

| Component | Complexity | Notes |
|-----------|------------|-------|
| Phase 1: Pure MCTS | Medium | Pure Rust, well-understood algorithm |
| Phase 2: Tournament runner | Medium | Straightforward CLI, parallelism with rayon |
| Phase 3: NN inference in Rust | Medium | ~300 lines, needs care with Conv2d |
| Phase 4: Python self-play | High | Many moving parts, fragile to get right |
| Training to good quality | High | Hyperparameter tuning, stability |
| Phase 5: Integration + WASM | Low | Mirrors existing bot-minimax pattern |

The hardest part is the **Python self-play pipeline** and **training stability**.
Everything else is straightforward engineering.

---

## Implementation Order

| # | What | Why first |
|---|------|-----------|
| 1 | Pure MCTS in `rust/mcts` | Validate tree structure, get a working bot |
| 2 | Tournament runner `tools/tournament` | Enables benchmarking + data generation |
| 3 | NN inference engine `rust/nn` | Foundation for plugging in the network |
| 4 | Python model + export script | Defines the weight format both sides agree on |
| 5 | Python self-play + training loop | Actual learning, most iteration needed |
| 6 | Wire NN into MCTS, build WASM | Connect all pieces |
| 7 | Quantization + Analysis mode polish | Nice to have once it works |
