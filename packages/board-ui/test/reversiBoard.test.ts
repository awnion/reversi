import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import {
  applyMove,
  type Cell,
  createInitialState,
  type Disc,
  type GameState,
  getLegalMoves,
  indexOf,
} from '@reversi/core';
import { createReversiBoard } from '../src/reversiBoard';

// Simulated DOM, scoped to this file so the other suites keep Bun's native
// globals (the WASM bots load through the real fetch). Real browsers are
// covered by the Playwright e2e tests.
beforeAll(() => {
  GlobalRegistrator.register();
});
afterAll(async () => {
  await GlobalRegistrator.unregister();
});

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

function squareAt(root: HTMLElement, row: number, col: number): HTMLElement {
  const square = root.querySelector<HTMLElement>(
    `[aria-label="row ${row + 1} column ${col + 1}"]`,
  );
  if (!square) throw new Error(`no square at row ${row}, column ${col}`);
  return square;
}

describe('reversi board ui', () => {
  test('renders 64 squares and 4 starting discs', async () => {
    const { root } = setup(createInitialState());

    expect(root.querySelectorAll('.rv-square')).toHaveLength(64);
    expect(root.querySelectorAll('.rv-disc')).toHaveLength(4);
  });

  test('marks legal move squares with data-legal and hint', async () => {
    const state = createInitialState();
    const { root } = setup(state);

    const legalSquares = root.querySelectorAll('.rv-square[data-legal="true"]');
    expect(legalSquares).toHaveLength(state.legalMoves.length);

    const hints = root.querySelectorAll('.rv-hint');
    expect(hints).toHaveLength(state.legalMoves.length);
  });

  test('clicking a legal square triggers onMove', async () => {
    const state = createInitialState();
    const onMove = mock();
    const { root } = setup(state, onMove);

    const move = state.legalMoves[0];
    squareAt(root, move.row, move.col).click();

    expect(onMove).toHaveBeenCalledWith({ row: move.row, col: move.col });
  });

  test('clicking a non-legal square does not trigger onMove', async () => {
    const onMove = mock();
    const { root } = setup(createInitialState(), onMove);

    // (0,0) is never a legal opening move
    squareAt(root, 0, 0).click();

    expect(onMove).not.toHaveBeenCalled();
  });

  test('setState updates discs on the board', async () => {
    const state = createInitialState();
    const { root, board } = setup(state);

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

    expect(root.querySelectorAll('.rv-hint')).toHaveLength(0);
    expect(root.querySelectorAll('.rv-square[data-legal="true"]')).toHaveLength(
      0,
    );
  });

  test('destroy removes board content', async () => {
    const { root, board } = setup(createInitialState());

    expect(root.querySelectorAll('.rv-square').length).toBe(64);

    board.destroy();
    expect(root.querySelectorAll('.rv-square').length).toBe(0);
    expect(root.classList.contains('rv-board')).toBe(false);
  });
});

describe('last move marker', () => {
  const nextFrame = () =>
    new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );

  test('marks the disc placed by the most recent move', async () => {
    const state = createInitialState();
    const { root, board } = setup(state);
    expect(root.querySelectorAll('[data-last="true"]')).toHaveLength(0);

    const move = state.legalMoves[0];
    board.setState(applyMove(state, move));
    await nextFrame();

    const marked = root.querySelectorAll<HTMLElement>('[data-last="true"]');
    expect(marked).toHaveLength(1);
    expect(marked[0]).toBe(squareAt(root, move.row, move.col));
    expect(root.dataset.turn).toBe('white');
  });

  test('clears the marker when a new game starts', async () => {
    const state = createInitialState();
    const { root, board } = setup(state);
    board.setState(applyMove(state, state.legalMoves[0]));
    await nextFrame();
    board.setState(createInitialState());
    await nextFrame();
    expect(root.querySelectorAll('[data-last="true"]')).toHaveLength(0);
  });
});
