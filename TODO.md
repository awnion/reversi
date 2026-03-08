# TODO

## Foundation

- [x] Define initial implementation plan for the monorepo
- [x] Create Bun workspace monorepo scaffold
- [x] Add root TypeScript and tooling configuration
- [x] Add root scripts for build, dev, test, and lint

## Core Game Logic

- [x] Create `packages/core`
- [x] Implement board and move types
- [x] Implement initial Reversi position
- [x] Implement legal move generation
- [x] Implement move application and disc flipping
- [x] Implement pass and game-over detection
- [x] Add unit tests for core rules

## Board GUI Engine

- [x] Create `packages/board-ui`
- [x] Implement board state model
- [x] Implement DOM renderer for 8x8 board
- [x] Implement fast redraw scheduling with `requestAnimationFrame`
- [x] Implement pointer interaction for legal moves
- [ ] Implement disc placement and flip animation hooks
- [x] Add browser integration tests for board interactions

## Web App

- [x] Create `apps/web`
- [x] Wire `core` and `board-ui` together
- [x] Render playable local 8x8 Reversi game
- [x] Show current player, score, legal moves, and game status
- [x] Add basic responsive layout

## Testing

- [x] Configure `bun test` for logic packages
- [x] Configure `vitest` browser mode with Playwright provider
- [x] Configure Playwright end-to-end tests
- [x] Install browser binaries through Playwright

## Verification

- [x] Run unit tests
- [x] Run browser integration tests
- [x] Run end-to-end smoke tests
- [x] Update this TODO after each completed milestone
