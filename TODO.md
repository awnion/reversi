# TODO

## Critical

- [x] Implement `analyzePosition()` in Rust — add a function that evaluates all legal moves
      and returns `MoveEval[]` with scores and depth; currently always returns an empty stub

## Medium

- [x] Add `@reversi/bot` and `@reversi/bot-minimax` to `paths` in `tsconfig.base.json`
      for consistency with `@reversi/core` and `@reversi/board-ui`

## Minor

- [x] Remove unnecessary type cast in `gameController.ts:52`:
      `(this._state.status as string) === 'finished'` → `this._state.status === 'finished'`

- [x] Implement iterative deepening in the Rust engine instead of fixed depth=5;
      `_time_ms` parameter is accepted but ignored

- [x] Fix confusing direction comments in `board.rs` — the inline note
      "Wait, standard representation..." contradicts the surrounding labels
