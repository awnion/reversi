import { describe, expect, test } from 'bun:test';
import {
  applyMove,
  countDiscs,
  createInitialState,
  type GameState,
  passTurn,
} from '@reversi/core';
import { createZeroRevBot } from '../src/index';

/** Play first legal moves until at most `empties` squares are free. */
function positionWithEmpties(empties: number): GameState {
  let state = createInitialState();
  while (state.status !== 'finished') {
    const { black, white } = countDiscs(state.board);
    if (64 - black - white <= empties && state.legalMoves.length > 1) break;
    const move = state.legalMoves[0];
    state = move ? applyMove(state, move) : passTurn(state);
  }
  return state;
}

describe('ZeroRevBot', () => {
  test('plays the last 12 squares at once, without using its time', async () => {
    const bot = await createZeroRevBot();
    const budgetMs = 3000;

    // The last 12 squares are solved exactly.
    const endgame = positionWithEmpties(12);
    expect(endgame.status).not.toBe('finished');
    const started = performance.now();
    const move = await bot.chooseMove(endgame, budgetMs);
    expect(performance.now() - started).toBeLessThan(budgetMs / 4);
    expect(
      endgame.legalMoves.some((m) => m.row === move.row && m.col === move.col),
    ).toBe(true);
  });
});
