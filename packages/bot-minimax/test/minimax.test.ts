import { describe, expect, test } from 'bun:test';
import { createInitialState } from '@reversi/core';
import { createMinimaxBot } from '../src/index';

describe('MinimaxBot', () => {
  test('wrapper handles initialized bot and returns a valid starting move', async () => {
    const state = createInitialState();
    const bot = await createMinimaxBot('Test Minimax');
    expect(bot.name).toBe('Test Minimax');

    const move = await bot.chooseMove(state, 1000);
    expect(move).toBeDefined();
    const isLegal = state.legalMoves.find(
      (m) => m.row === move.row && m.col === move.col,
    );
    expect(isLegal).toBeDefined();

    bot.destroy?.();
  });
});
