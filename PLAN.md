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
where `P(s,a)` is the policy prior from the network and `N` is visit counts.

---

## Hardware Target: MacBook Pro M2 Max

The M2 Max has **unified memory** — CPU and GPU share the same RAM with no copy overhead.
This fundamentally changes the architecture: Rust workers and the MLX model can exchange
board states as raw numpy arrays with zero serialization cost.

Key numbers on M2 Max:
- ~400 GB/s memory bandwidth
- 30–38 GPU cores (Metal)
- 12 performance CPU cores (rayon workers)
- NN inference batch=64: ~0.5 ms on Metal
- Expected throughput: ~15–25 games/sec → ~1 000–1 500 positions/sec
- 1 hour of training ≈ 3–5 M positions → sufficient for 8×8 Reversi convergence

---

## Key Design Decisions

### ML Framework: MLX (not PyTorch)

| | MLX | PyTorch MPS |
|-|-----|-------------|
| Backend | Native Metal | MPS (wrapper) |
| Memory copies | Zero (unified) | Sometimes |
| Lazy eval + JIT | Yes | No |
| Speed on M-series | ✅ Best | Slower |

```python
import mlx.core as mx
import mlx.nn as nn
# mx.default_device() → Metal GPU automatically on M2
```

### Network Architecture

```
Input:  3 × 8 × 8
          ├── current player's discs
          ├── opponent's discs
          └── legal moves mask

Body:   4 ResBlocks × 32 channels (3×3 conv + BN + ReLU)

Policy head:  Conv 2×1 → Flatten → 64 logits
Value head:   Conv 1×1 → Flatten → Linear → Tanh → scalar
```

~100K parameters → ~200 KB fp32 → ~50 KB int8 after quantization.

### Parallelism: Batched Leaf Evaluation

The key pattern: instead of evaluating each MCTS leaf node individually, accumulate
requests from N concurrent games and evaluate them as a single GPU batch.

```
game_1 MCTS → hit leaf → ─────────────────────────────┐
game_2 MCTS → hit leaf → ──────────────────────────┐   │
...                                                 ▼   ▼
game_N MCTS → hit leaf → ──► [LeafEvalServer] → batch=64 → Metal GPU
                                                        │
                              ◄─── (policy, value) ×N ─┘
```

No copies. All in unified memory. Workers suspend (asyncio) while waiting for the batch.

### Rust ↔ Python Bridge: PyO3

MCTS game loop runs in Rust (fast), calls back into Python for leaf evaluation:

```rust
// rust/mcts/src/lib.rs
#[pyclass]
pub struct MctsWorker { ... }

#[pymethods]
impl MctsWorker {
    // Called from Python asyncio; eval_fn is LeafEvalServer.evaluate
    pub fn run_game(&mut self, eval_fn: PyObject) -> PyResult<GameRecord> { ... }
}
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Single Python process  (uv run python -m train.loop)        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLX Model  (Metal GPU, unified memory)              │   │
│  │                                                      │   │
│  │  LeafEvalServer  batch_size=64, timeout=2ms          │   │
│  │    ← board states (np.array, zero-copy)              │   │
│  │    → (policy [64], value scalar)                     │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │  Rust MCTS Workers  (PyO3 + rayon, N = cpu_count)    │   │
│  │                                                      │   │
│  │  game_1: selection → expansion → [eval] → backup    │   │
│  │  game_2: selection → expansion → [eval] → backup    │   │
│  │  ...                                                 │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │  completed GameRecord            │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │  Replay Buffer  (numpy ring buffer, 500K positions)  │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │  sample batch=256                │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │  Learner  (asyncio task, runs between eval batches)  │   │
│  │  policy loss + value loss → MLX grad update ~1ms     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Concurrency model

Everything runs in **one Python process**, one asyncio event loop:
- `eval_server.run()` — drains the leaf eval queue, fires GPU batches
- `worker_i.run_game()` — suspends at each leaf eval request, resumes with result
- `train_loop()` — runs between GPU batches, does one gradient step each time
- No multiprocessing, no IPC, no locks on the hot path

### Game Record Format

```python
@dataclass
class Position:
    board_black: np.uint64
    board_white: np.uint64
    is_black: bool
    mcts_policy: np.ndarray   # shape [64], float32 — visit count distribution
    outcome: np.float32       # +1 / −1 / 0 from current player's POV

@dataclass
class GameRecord:
    positions: list[Position]
```

---

## Training Loop

```python
# python/train/loop.py

async def main():
    model = AlphaZeroNet()
    eval_server = LeafEvalServer(model, batch_size=64, timeout_ms=2.0)
    replay = ReplayBuffer(max_size=500_000)

    async def worker(i):
        w = reversi_mcts.MctsWorker(simulations=400)
        while True:
            record = await w.run_game(eval_server.evaluate)
            replay.add(record)

    async def train_loop():
        opt = optim.Adam(learning_rate=1e-3)
        loss_grad_fn = nn.value_and_grad(model, compute_loss)
        while True:
            if len(replay) < 10_000:
                await asyncio.sleep(0.05)
                continue
            boards, policies, values = replay.sample(256)
            loss, grads = loss_grad_fn(model, mx.array(boards),
                                       mx.array(policies), mx.array(values))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)  # flush lazy ops
            await asyncio.sleep(0)  # yield to eval_server

    await asyncio.gather(
        eval_server.run(),
        train_loop(),
        *[worker(i) for i in range(os.cpu_count() - 1)],
    )
```

### Self-play schedule (AlphaZero-style)

```
Every 1 000 gradient steps:
  1. Save checkpoint  weights/iter_{n}.safetensors
  2. Run 50 eval games: candidate vs previous champion
  3. If win_rate > 55% → promote candidate to champion
  4. Export champion weights to binary for WASM: python -m train.export
```

---

## Implementation Phases

### Phase 1 — Pure MCTS (no network)

New crate `rust/mcts`, new PyO3 extension `reversi_mcts`:
- **Selection** — PUCT with uniform prior
- **Expansion** — add node for each legal move
- **Simulation** — minimax static eval at leaf (reuse `rust/minimax`)
- **Backup** — update `W(s,a)` and `N(s,a)`
- PyO3 bindings: `MctsWorker.run_game(eval_fn)`
- Reuses `Board` and `evaluate` from `rust/minimax`

This gives a working bot immediately. NN plugs in as the eval function later.

### Phase 2 — MLX Network + LeafEvalServer

```
python/train/
  model.py        ← AlphaZeroNet (mlx.nn.Module)
  eval_server.py  ← LeafEvalServer (asyncio + batching)
  replay.py       ← ReplayBuffer (numpy ring buffer)
```

Validate that Rust workers can call Python eval_fn and get results back.

### Phase 3 — Training Loop

```
python/train/
  loop.py         ← asyncio main: workers + eval_server + train_loop
  loss.py         ← policy cross-entropy + value MSE
  export.py       ← save weights to binary for Rust WASM
```

Run: `uv run python -m train.loop`

### Phase 4 — NN Inference in Rust (for WASM)

New crate `rust/nn`:
- Hand-rolled forward pass: `Conv2d`, `BatchNorm2d`, `Linear`, `ReLU`, `Tanh`
- Loads weights from `&[u8]` via `include_bytes!`
- ~300 lines, zero dependencies

Custom weight format: `[magic: u32][version: u16][n_layers: u16][layer blobs...]`

### Phase 5 — AlphaZero WASM Bot

New crate `rust/alphazero`:
- MCTS + `rust/nn` inference
- `include_bytes!("../../weights/champion.bin")`
- Exported via wasm-pack as `packages/bot-alphazero`
- Same `BotPlayer` interface as `bot-minimax`

### Phase 6 — Improvements

- Int8 weight quantization (−75% weight size)
- Temperature schedule: high early in game (exploration), low late (exploitation)
- Dirichlet noise at tree root during self-play
- `mlx.optimizers.cosine_decay` learning rate schedule

---

## Repository Structure

```
rust/
  minimax/          ← exists: alpha-beta search
  mcts/             ← new: MCTS engine + PyO3 bindings
  nn/               ← new: hand-rolled inference for WASM

python/             ← uv project (pyproject.toml)
  train/
    model.py        ← AlphaZeroNet (mlx.nn)
    eval_server.py  ← batched leaf evaluation
    replay.py       ← replay buffer
    loop.py         ← main training loop
    loss.py         ← policy + value loss
    export.py       ← weights → binary for Rust

weights/
  champion.bin      ← latest champion (committed, embedded in WASM)
  history/          ← checkpoint archive (.safetensors)

packages/
  bot-minimax/      ← exists
  bot-alphazero/    ← new: WASM wrapper, same BotPlayer interface
```

---

## Tooling

| Layer | Tool | Command |
|-------|------|---------|
| Rust | cargo fmt + clippy | `cargo fmt && cargo clippy --workspace` |
| JS/TS | Biome | `bunx biome check` |
| Python | Ruff | `uvx ruff check python/` |
| TOML | Taplo | `uvx taplo fmt` |
| All | pre-commit | `uvx pre-commit run --all-files` |

Install hook: `uvx pre-commit install`

Python deps: `uv sync` in `python/`
Run training: `uv run python -m train.loop`

---

## Complexity Estimate

| Component | Complexity | Notes |
|-----------|------------|-------|
| Phase 1: Rust MCTS + PyO3 | Medium | PyO3 async bridge is non-trivial |
| Phase 2: MLX model + eval server | Low | MLX API is clean, asyncio batching is ~50 lines |
| Phase 3: Training loop | Medium | Many moving parts but each piece is simple |
| Training to good quality | Medium | M2 Max + MLX makes iteration fast |
| Phase 4: Rust NN inference | Medium | ~300 lines, tedious but mechanical |
| Phase 5: WASM integration | Low | Mirrors bot-minimax pattern |

The hardest piece is the **PyO3 async bridge** between Rust workers and the Python
asyncio eval server. Everything else is straightforward given the architecture above.

---

## Implementation Order

| # | What | Why first |
|---|------|-----------|
| 1 | Rust MCTS crate (`rust/mcts`) | Core algorithm, test in isolation |
| 2 | PyO3 bindings (`reversi_mcts` Python ext) | Connects Rust speed to Python orchestration |
| 3 | MLX model + LeafEvalServer | Enables NN-guided search |
| 4 | Training loop + replay buffer | Actual learning |
| 5 | Checkpoint eval + promotion | Quality gate for weights |
| 6 | `rust/nn` inference engine | Required for WASM deployment |
| 7 | `bot-alphazero` WASM package | Final browser integration |
