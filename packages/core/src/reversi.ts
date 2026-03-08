export type Disc = 'black' | 'white';
export type Cell = Disc | null;
export type GameStatus = 'playing' | 'passed' | 'finished';

export interface Position {
  row: number;
  col: number;
}

export interface Move extends Position {
  flips: Position[];
}

export interface GameState {
  board: Cell[];
  currentPlayer: Disc;
  status: GameStatus;
  winner: Disc | 'draw' | null;
  legalMoves: Move[];
  consecutivePasses: number;
}

const BOARD_SIZE = 8;
const BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 0],
  [1, 1]
];

export function createInitialState(): GameState {
  const board = Array<Cell>(BOARD_CELLS).fill(null);
  setCell(board, 3, 3, 'white');
  setCell(board, 3, 4, 'black');
  setCell(board, 4, 3, 'black');
  setCell(board, 4, 4, 'white');
  return withDerivedState({
    board,
    currentPlayer: 'black',
    status: 'playing',
    winner: null,
    consecutivePasses: 0
  });
}

export function getLegalMoves(board: ReadonlyArray<Cell>, player: Disc): Move[] {
  const moves: Move[] = [];
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      if (getCell(board, row, col) !== null) continue;
      const flips = collectFlips(board, player, row, col);
      if (flips.length > 0) moves.push({ row, col, flips });
    }
  }
  return moves;
}

export function applyMove(state: GameState, position: Position): GameState {
  const move = state.legalMoves.find(candidate => candidate.row === position.row && candidate.col === position.col);
  if (!move) {
    throw new Error(`Illegal move at (${position.row}, ${position.col})`);
  }

  const board = state.board.slice();
  setCell(board, move.row, move.col, state.currentPlayer);
  for (const flip of move.flips) setCell(board, flip.row, flip.col, state.currentPlayer);

  return withDerivedState({
    board,
    currentPlayer: opposite(state.currentPlayer),
    status: 'playing',
    winner: null,
    consecutivePasses: 0
  });
}

export function passTurn(state: GameState): GameState {
  if (state.legalMoves.length > 0) throw new Error('Cannot pass when legal moves are available');
  return withDerivedState({
    board: state.board.slice(),
    currentPlayer: opposite(state.currentPlayer),
    status: 'passed',
    winner: null,
    consecutivePasses: state.consecutivePasses + 1
  });
}

export function countDiscs(board: ReadonlyArray<Cell>): Record<Disc, number> {
  let black = 0;
  let white = 0;
  for (const cell of board) {
    if (cell === 'black') black += 1;
    else if (cell === 'white') white += 1;
  }
  return { black, white };
}

export function indexOf(row: number, col: number): number {
  return row * BOARD_SIZE + col;
}

export function isInsideBoard(row: number, col: number): boolean {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

function withDerivedState(base: Omit<GameState, 'legalMoves' | 'status' | 'winner'> & Partial<Pick<GameState, 'status' | 'winner'>>): GameState {
  const legalMoves = getLegalMoves(base.board, base.currentPlayer);
  const next: GameState = {
    board: base.board,
    currentPlayer: base.currentPlayer,
    legalMoves,
    status: base.status ?? 'playing',
    winner: base.winner ?? null,
    consecutivePasses: base.consecutivePasses
  };

  if (legalMoves.length > 0) {
    next.status = 'playing';
    next.winner = null;
    return next;
  }

  const otherMoves = getLegalMoves(base.board, opposite(base.currentPlayer));
  if (otherMoves.length > 0 && next.consecutivePasses < 2) {
    next.status = 'passed';
    next.winner = null;
    return next;
  }

  const counts = countDiscs(base.board);
  next.status = 'finished';
  next.winner = counts.black === counts.white ? 'draw' : counts.black > counts.white ? 'black' : 'white';
  return next;
}

function collectFlips(board: ReadonlyArray<Cell>, player: Disc, row: number, col: number): Position[] {
  const flips: Position[] = [];
  for (const [deltaRow, deltaCol] of DIRECTIONS) {
    const line = collectDirectionalFlips(board, player, row, col, deltaRow, deltaCol);
    if (line.length > 0) flips.push(...line);
  }
  return flips;
}

function collectDirectionalFlips(
  board: ReadonlyArray<Cell>,
  player: Disc,
  startRow: number,
  startCol: number,
  deltaRow: number,
  deltaCol: number
): Position[] {
  const seen: Position[] = [];
  let row = startRow + deltaRow;
  let col = startCol + deltaCol;

  while (isInsideBoard(row, col)) {
    const cell = getCell(board, row, col);
    if (cell === null) return [];
    if (cell === player) return seen.length > 0 ? seen : [];
    seen.push({ row, col });
    row += deltaRow;
    col += deltaCol;
  }

  return [];
}

function getCell(board: ReadonlyArray<Cell>, row: number, col: number): Cell {
  return board[indexOf(row, col)] ?? null;
}

function setCell(board: Cell[], row: number, col: number, value: Cell): void {
  board[indexOf(row, col)] = value;
}

function opposite(player: Disc): Disc {
  return player === 'black' ? 'white' : 'black';
}
