import { describe, expect, test } from 'bun:test';

import {
  applyMove,
  type Cell,
  countDiscs,
  createInitialState,
  type Disc,
  type GameState,
  getLegalMoves,
  indexOf,
  passTurn,
} from '../src/reversi';

/** Build a board from 8 rows of space-separated chars: B=black, W=white, .=empty */
function makeBoard(rows: string[]): Cell[] {
  const board: Cell[] = Array(64).fill(null);
  for (let r = 0; r < 8; r++) {
    const cells = rows[r].trim().split(/\s+/);
    for (let c = 0; c < 8; c++) {
      if (cells[c] === 'B') board[indexOf(r, c)] = 'black';
      else if (cells[c] === 'W') board[indexOf(r, c)] = 'white';
    }
  }
  return board;
}

/** Build a minimal GameState from a board (status is approximate — use passTurn/applyMove to get real derived state). */
function makeState(
  board: Cell[],
  currentPlayer: Disc,
  consecutivePasses = 0,
): GameState {
  const legalMoves = getLegalMoves(board, currentPlayer);
  return {
    board,
    currentPlayer,
    legalMoves,
    status: legalMoves.length > 0 ? 'playing' : 'passed',
    winner: null,
    consecutivePasses,
  };
}

// ── Board where black has no moves but white does ─────────────────────
// White can play at (4,0) flipping (3,0) and at (4,1) flipping (3,1).
// Black has no legal moves anywhere.
const PASS_BOARD = [
  'W W W W W W W W',
  'W W W W W W W W',
  'W W W W W W W W',
  'B B W W W W W W',
  '. . W W W W W W',
  'W W W W W W W W',
  'W W W W W W W W',
  'W W W W W W W W',
];

// ── Board where neither player can move (isolated discs) ─────────────
const DEADLOCK_BOARD = [
  'B B . . . . . .',
  '. . . . . . . .',
  '. . . . . . . .',
  '. . . . . . . .',
  '. . . . . . . .',
  '. . . . . . . .',
  '. . . . . . . .',
  '. . . . . . . W',
];

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

  test('illegal move position throws', () => {
    const state = createInitialState();
    expect(() => applyMove(state, { row: 0, col: 0 })).toThrow('Illegal move');
  });
});

describe('pass turn', () => {
  test('black has no legal moves on PASS_BOARD', () => {
    const board = makeBoard(PASS_BOARD);
    expect(getLegalMoves(board, 'black')).toHaveLength(0);
  });

  test('white has legal moves on PASS_BOARD', () => {
    const board = makeBoard(PASS_BOARD);
    expect(getLegalMoves(board, 'white').length).toBeGreaterThan(0);
  });

  test('passTurn succeeds when current player has no moves', () => {
    const state = makeState(makeBoard(PASS_BOARD), 'black');
    expect(state.legalMoves).toHaveLength(0);

    const next = passTurn(state);
    expect(next.currentPlayer).toBe('white');
    expect(next.consecutivePasses).toBe(1);
  });

  test('after pass, game continues if opponent has moves', () => {
    const state = makeState(makeBoard(PASS_BOARD), 'black');
    const next = passTurn(state);

    expect(next.status).toBe('playing');
    expect(next.legalMoves.length).toBeGreaterThan(0);
  });

  test('applyMove after pass resets consecutivePasses', () => {
    const state = makeState(makeBoard(PASS_BOARD), 'black');
    const afterPass = passTurn(state);
    const afterMove = applyMove(afterPass, afterPass.legalMoves[0]);

    expect(afterMove.consecutivePasses).toBe(0);
  });

  test('pass then opponent plays — full sequence', () => {
    const state = makeState(makeBoard(PASS_BOARD), 'black');

    // black passes
    const afterPass = passTurn(state);
    expect(afterPass.currentPlayer).toBe('white');
    expect(afterPass.status).toBe('playing');

    // white plays at (4,0), flipping (3,0) from B→W
    const afterMove = applyMove(afterPass, { row: 4, col: 0 });
    expect(afterMove.currentPlayer).toBe('black');
    expect(afterMove.consecutivePasses).toBe(0);

    // (3,0) was black, should now be white
    expect(afterMove.board[indexOf(3, 0)]).toBe('white');
  });
});

describe('game over', () => {
  test('full board is finished', () => {
    // all black except one white — to have a valid Reversi-ish board
    const board: Cell[] = Array(64).fill('black');
    board[indexOf(7, 7)] = 'white';

    const state = makeState(board, 'black');
    // no empty cells → no moves for either side
    expect(state.legalMoves).toHaveLength(0);

    const next = passTurn(state);
    expect(next.status).toBe('finished');
    expect(next.winner).toBe('black');
  });

  test('game finishes when neither player can move on partial board', () => {
    const board = makeBoard(DEADLOCK_BOARD);
    expect(getLegalMoves(board, 'black')).toHaveLength(0);
    expect(getLegalMoves(board, 'white')).toHaveLength(0);

    const state = makeState(board, 'black');
    const next = passTurn(state);

    expect(next.status).toBe('finished');
  });

  test('draw when disc counts are equal', () => {
    const board = makeBoard([
      'B . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . W',
    ]);
    const state = makeState(board, 'black');
    const next = passTurn(state);

    expect(next.status).toBe('finished');
    expect(next.winner).toBe('draw');
  });

  test('black wins with more discs', () => {
    const board = makeBoard(DEADLOCK_BOARD); // B=2, W=1
    const state = makeState(board, 'black');
    const next = passTurn(state);

    expect(next.status).toBe('finished');
    expect(next.winner).toBe('black');
  });

  test('white wins with more discs', () => {
    const board = makeBoard([
      'W W . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . .',
      '. . . . . . . B',
    ]);
    const state = makeState(board, 'white');
    const next = passTurn(state);

    expect(next.status).toBe('finished');
    expect(next.winner).toBe('white');
  });

  test('completely filled board has no moves for either side', () => {
    const board: Cell[] = Array(64).fill('white');
    expect(getLegalMoves(board, 'black')).toHaveLength(0);
    expect(getLegalMoves(board, 'white')).toHaveLength(0);
  });
});
