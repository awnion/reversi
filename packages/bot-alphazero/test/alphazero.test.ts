import { describe, expect, test } from 'bun:test';
import {
  applyMove,
  countDiscs,
  createInitialState,
  passTurn,
} from '@reversi/core';
import { createAlphaZeroBot } from '../src/index';

describe('AlphaZeroBot', () => {
  test('returns a valid move from initial position and embeds build date', async () => {
    const state = createInitialState();
    const bot = await createAlphaZeroBot();

    expect(bot.name).toMatch(/^AlphaZero \(\d{4}-\d{2}-\d{2}\)$/);

    const move = await bot.chooseMove(state, 200);
    expect(move).toBeDefined();
    const isLegal = state.legalMoves.find(
      (m) => m.row === move.row && m.col === move.col,
    );
    expect(isLegal).toBeDefined();

    bot.destroy?.();
  });

  test('alphazero vs alphazero plays a complete game', async () => {
    const bot = await createAlphaZeroBot();
    let state = createInitialState();
    let turns = 0;
    const maxTurns = 130; // reversi can't exceed 64 moves + passes

    while (state.status !== 'finished' && turns < maxTurns) {
      if (state.legalMoves.length === 0) {
        state = passTurn(state);
      } else {
        const pos = await bot.chooseMove(state, 100);
        state = applyMove(state, pos);
      }
      turns++;
    }

    expect(state.status).toBe('finished');
    expect(['black', 'white', 'draw']).toContain(state.winner);

    const counts = countDiscs(state.board);
    expect(counts.black + counts.white).toBeGreaterThan(4);
    expect(counts.black + counts.white).toBeLessThanOrEqual(64);

    bot.destroy?.();
  }, 30_000);
});
