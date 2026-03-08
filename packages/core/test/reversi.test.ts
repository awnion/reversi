import { describe, expect, test } from 'bun:test';

import { applyMove, countDiscs, createInitialState, getLegalMoves, passTurn } from '../src/reversi';

describe('reversi core', () => {
  test('creates initial state with four legal moves for black', () => {
    const state = createInitialState();
    expect(state.currentPlayer).toBe('black');
    expect(state.legalMoves).toHaveLength(4);
  });

  test('applies opening move and flips one disc', () => {
    const state = createInitialState();
    const next = applyMove(state, { row: 2, col: 3 });
    const counts = countDiscs(next.board);
    expect(next.currentPlayer).toBe('white');
    expect(counts.black).toBe(4);
    expect(counts.white).toBe(1);
  });

  test('pass is rejected when moves exist', () => {
    const state = createInitialState();
    expect(() => passTurn(state)).toThrow();
  });

  test('legal move generation returns no moves on filled board', () => {
    const board = Array(64).fill('black');
    expect(getLegalMoves(board, 'white')).toEqual([]);
  });
});
