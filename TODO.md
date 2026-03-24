# Reversi Bot Players TODO

## Phase 1: Types and Interfaces (`@reversi/core`)
- [x] Add `MoveEval` and `PositionAnalysis` interfaces to `@reversi/core`.
- [x] Add `BotPlayer` interface to `@reversi/core`.

## Phase 2: JS bot package (`@reversi/bot`)
- [x] Set up `@reversi/bot` package (`package.json`, `tsconfig.json`).
- [x] Implement `greedyBot.ts`.
- [x] Export the bot.

## Phase 3: Game Controller & UI (`apps/web`)
- [x] Create `GameController` logic (handle human vs bot, bot vs bot).
- [x] Update Game board to use `GameController`.
- [x] Add Player Selection UI to sidebar (Dropdowns for Black/White).
- [x] Implement `New Game` button logic.

## Phase 4: Rust + WASM Bot (`rust/minimax`)
- [x] Initialize `rust/minimax` crate.
- [x] Implement bitboard representation.
- [x] Implement Negamax with alpha-beta pruning.
- [x] Implement evaluation function.
- [x] Expose WASM API via `wasm-bindgen`.
- [x] Set up `wasm-pack` build script.

## Phase 5: TypeScript Wrapper (`@reversi/bot-minimax`)
- [x] Set up `@reversi/bot-minimax` package.
- [x] Implement `createMinimaxBot` wrapper around the WASM module.
- [x] Integrate Minimax bot into `apps/web` selection UI.

## Phase 6: Analysis Mode
- [x] Add `Analyze` button in the sidebar.
- [x] Implement UI overlay for move scores and hints.
- [x] Show ranked move list in the sidebar.
