import { describe, expect, test } from 'bun:test';
import { createInitialState } from '@reversi/core';
import { greedyBot } from '../src/greedyBot';

describe('greedyBot', () => {
  test('picks the move that gives the highest flip count', async () => {
    const state = createInitialState();
    // The initial board has 4 moves, all flip 1 disc.
    // Greedy should just pick the first best one without crashing.
    const move = await greedyBot.chooseMove(state, 100);
    expect(move).toBeDefined();
    expect(
      state.legalMoves.find((m) => m.row === move.row && m.col === move.col),
    ).toBeDefined();
  });

  test('analyze position returns list of ranked moves', async () => {
    const state = createInitialState();
    const analysis = await greedyBot.analyzePosition(state, 100);
    expect(analysis.moves.length).toBe(state.legalMoves.length);
    expect(analysis.bestMove).toBeDefined();
    // Since all initial moves flip 1 disc, max score should be 1
    expect(analysis.score).toBe(1);
  });
});
