# Bot Players — Implementation Plan

## Architecture Overview

The existing codebase is well-suited for extension — `GameState` is immutable,
`applyMove` returns a new state, `getLegalMoves` is exported. The plan adds:

```
packages/
  core/         ← add bot types (BotPlayer, PositionAnalysis)
  bot/          ← new: JS bot interface + pure JS bots for testing
  bot-minimax/  ← new: WASM wrapper around the Rust bot
rust/
  minimax/      ← new: Rust crate → wasm-pack → dist/
apps/
  web/          ← GameController + player selection UI + analysis mode
```

---

## Phase 1: Types and Interfaces (`@reversi/core`)

Add to core exports:

```typescript
interface MoveEval {
  position: Position
  score: number   // from the perspective of the current player
  depth: number   // search depth reached
}

interface PositionAnalysis {
  bestMove: Position | null
  score: number
  depth: number
  moves: MoveEval[]  // all legal moves with scores
}

interface BotPlayer {
  readonly name: string
  chooseMove(state: GameState, timeLimitMs: number): Promise<Position>
  analyzePosition(state: GameState, timeLimitMs: number): Promise<PositionAnalysis>
  destroy?(): void
}
```

---

## Phase 2: Rust + WASM Bot (`rust/minimax`)

**Structure:**
```
rust/minimax/
  Cargo.toml
  src/
    lib.rs      ← wasm-bindgen exports
    engine.rs   ← negamax + alpha-beta pruning
    eval.rs     ← position evaluation function
    board.rs    ← bitboard representation (u64)
```

**Algorithm:**
- **Bitboard**: board as two `u64` values (black/white) — enables fast bitwise ops
- **Negamax with alpha-beta** pruning
- **Iterative deepening**: deepens until time runs out (depths 1, 2, 3, …)
- **Evaluation function:**
  - Positional weights (corners = very high bonus, C-squares = large penalty, edges = bonus)
  - Mobility (current player's move count vs opponent's)
  - Stable discs
  - Pure disc count in endgame

**WASM API (via wasm-bindgen):**
```typescript
class MinimaxBot {
  constructor()
  choose_move(board_black: bigint, board_white: bigint, is_black: boolean, time_ms: number): number
  analyze_position(board_black: bigint, board_white: bigint, is_black: boolean, time_ms: number): AnalysisResult
}
```

**Build:** `wasm-pack build --target web --out-dir ../../packages/bot-minimax/wasm`

---

## Phase 3: TypeScript Wrapper (`@reversi/bot-minimax`)

```typescript
// packages/bot-minimax/src/index.ts
import init, { MinimaxBot as WasmBot } from '../wasm/reversi_minimax.js'

export async function createMinimaxBot(name = 'Minimax'): Promise<BotPlayer> {
  await init()
  const wasm = new WasmBot()
  return {
    name,
    async chooseMove(state, timeLimitMs) {
      // convert GameState.board → bitboards
      // call wasm.choose_move(...)
      // convert index → Position
    },
    async analyzePosition(state, timeLimitMs) {
      // wasm.analyze_position(...)
    },
    destroy() { wasm.free() }
  }
}
```

Also add a pure **JS greedy bot** (no WASM) for testing and as a fallback:

```typescript
// packages/bot/src/greedyBot.ts
export const greedyBot: BotPlayer = {
  name: 'Greedy',
  async chooseMove(state) {
    // picks the move that flips the most discs
  }
}
```

---

## Phase 4: Game Controller (`apps/web`)

```typescript
// apps/web/src/gameController.ts
type PlayerConfig =
  | { type: 'human' }
  | { type: 'bot'; player: BotPlayer; thinkMs: number }

class GameController {
  constructor(
    private state: GameState,
    private players: { black: PlayerConfig; white: PlayerConfig },
    private onStateChange: (state: GameState, analysis?: PositionAnalysis) => void
  )

  async makeMove(pos: Position): Promise<void>   // for human player
  private async triggerBotMove(): Promise<void>  // auto-called after turn change
  async startAnalysis(timeLimitMs: number): Promise<PositionAnalysis>
  newGame(players: { black: PlayerConfig; white: PlayerConfig }): void
}
```

Controller behavior:
- After each move, checks if the next player is a bot
- If yes — schedules `bot.chooseMove()` via `setTimeout(0)` and applies the result
- Two bots can play each other with a configurable visual delay (~300 ms)

---

## Phase 5: Player Selection UI

Add to the sidebar:

```
[ Black: ▼ Human        ]  [ White: ▼ Minimax (2s) ]
               [ New Game ]
```

- Dropdowns: `Human`, `Greedy (instant)`, `Minimax (1s)`, `Minimax (5s)`
- `New Game` button starts a game with the selected players

---

## Phase 6: Analysis Mode

- `Analyze` button in the sidebar (active when game is over or paused)
- Calls `analyzePosition()` on the current state
- Overlays move scores on the board (numbers or color scale on hint squares)
- Sidebar shows a ranked move list with scores, best move highlighted
- Similar to Lichess: clicking a move in the list previews it on the board

---

## Implementation Order

| # | What | Why first |
|---|------|-----------|
| 1 | `BotPlayer`, `PositionAnalysis` types in core | Foundation for everything |
| 2 | JS greedy bot in `@reversi/bot` | Test bot flow without WASM |
| 3 | `GameController` + player selection UI in web app | Makes bots playable |
| 4 | Rust/WASM minimax bot | Requires toolchain setup (wasm-pack, Rust) |
| 5 | WASM wrapper + integration | Plugs the real bot in |
| 6 | Analysis mode UI | Requires fully working `analyzePosition` |
