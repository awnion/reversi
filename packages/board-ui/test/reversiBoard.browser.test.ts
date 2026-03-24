import {
  applyMove,
  type Cell,
  createInitialState,
  type Disc,
  type GameState,
  getLegalMoves,
  indexOf,
} from '@reversi/core';
import { describe, expect, test, vi } from 'vitest';
import { page } from 'vitest/browser';
import { createReversiBoard } from '../src/reversiBoard';

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

function setup(
  state: GameState,
  onMove?: (pos: { row: number; col: number }) => void,
) {
  document.body.innerHTML = '<div id="board"></div>';
  const root = document.getElementById('board') as HTMLElement;
  const board = createReversiBoard(root, { state, onMove });
  return { root, board };
}

describe('reversi board ui', () => {
  test('renders 64 squares and 4 starting discs', async () => {
    const { root } = setup(createInitialState());

    await expect.element(page.getByRole('button').nth(0)).toBeInTheDocument();
    expect(root.querySelectorAll('.rv-square')).toHaveLength(64);
    expect(root.querySelectorAll('.rv-disc')).toHaveLength(4);
  });

  test('marks legal move squares with data-legal and hint', async () => {
    const state = createInitialState();
    const { root } = setup(state);

    await expect.element(page.getByRole('button').nth(0)).toBeInTheDocument();

    const legalSquares = root.querySelectorAll('.rv-square[data-legal="true"]');
    expect(legalSquares).toHaveLength(state.legalMoves.length);

    const hints = root.querySelectorAll('.rv-hint');
    expect(hints).toHaveLength(state.legalMoves.length);
  });

  test('clicking a legal square triggers onMove', async () => {
    const state = createInitialState();
    const onMove = vi.fn();
    setup(state, onMove);

    const move = state.legalMoves[0];
    const button = page.getByRole('button', {
      name: `row ${move.row + 1} column ${move.col + 1}`,
    });
    await button.click();

    expect(onMove).toHaveBeenCalledWith({ row: move.row, col: move.col });
  });

  test('clicking a non-legal square does not trigger onMove', async () => {
    const onMove = vi.fn();
    setup(createInitialState(), onMove);

    // (0,0) is never a legal opening move
    const button = page.getByRole('button', { name: 'row 1 column 1' });
    await button.click();

    expect(onMove).not.toHaveBeenCalled();
  });

  test('setState updates discs on the board', async () => {
    const state = createInitialState();
    const { root, board } = setup(state);

    await expect.element(page.getByRole('button').nth(0)).toBeInTheDocument();
    expect(root.querySelectorAll('.rv-disc')).toHaveLength(4);

    const next = applyMove(state, state.legalMoves[0]);
    board.setState(next);

    // wait for rAF
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );

    // opening move: 3 discs of one color + 1 flipped = at least 5 total discs
    const discs = root.querySelectorAll('.rv-disc');
    expect(discs.length).toBe(5);
  });

  test('no hints shown when current player has no legal moves', async () => {
    const board = makeBoard([
      'W W W W W W W W',
      'W W W W W W W W',
      'W W W W W W W W',
      'B B W W W W W W',
      '. . W W W W W W',
      'W W W W W W W W',
      'W W W W W W W W',
      'W W W W W W W W',
    ]);
    const state = makeState(board, 'black');
    expect(state.legalMoves).toHaveLength(0);

    const { root } = setup(state);

    await expect.element(page.getByRole('button').nth(0)).toBeInTheDocument();
    expect(root.querySelectorAll('.rv-hint')).toHaveLength(0);
    expect(root.querySelectorAll('.rv-square[data-legal="true"]')).toHaveLength(
      0,
    );
  });

  test('destroy removes board content', async () => {
    const { root, board } = setup(createInitialState());

    await expect.element(page.getByRole('button').nth(0)).toBeInTheDocument();
    expect(root.querySelectorAll('.rv-square').length).toBe(64);

    board.destroy();
    expect(root.querySelectorAll('.rv-square').length).toBe(0);
    expect(root.classList.contains('rv-board')).toBe(false);
  });
});
