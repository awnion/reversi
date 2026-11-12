import type { Disc, GameState, Position } from '@reversi/core';

export interface BoardConfig {
  state: GameState;
  onMove?: (position: Position) => void;
}

export interface BoardApi {
  setState(state: GameState): void;
  destroy(): void;
}

interface BoardInternalState {
  state: GameState;
  selected?: Position;
  /** Index of the disc placed by the most recent move, if any. */
  lastMove?: number | undefined;
}

/** The square that was empty in `previous` and holds a disc in `next`. */
function placedIndex(previous: GameState, next: GameState): number | undefined {
  let placed: number | undefined;
  for (let i = 0; i < next.board.length; i += 1) {
    if (previous.board[i] === null && next.board[i] !== null) {
      // More than one new disc is not a single move (e.g. a loaded position).
      if (placed !== undefined) return undefined;
      placed = i;
    } else if (previous.board[i] !== null && next.board[i] === null) {
      return undefined; // discs disappeared: a new game
    }
  }
  return placed;
}

export function createReversiBoard(
  element: HTMLElement,
  config: BoardConfig,
): BoardApi {
  const internal: BoardInternalState = {
    state: config.state,
  };

  element.classList.add('rv-board');
  let frame = 0;

  const redraw = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      render(element, internal, config);
    });
  };

  render(element, internal, config);

  return {
    setState(state) {
      // A pass keeps the board as is; keep the marker on the last disc placed.
      const changed = state.board.some(
        (cell, i) => cell !== internal.state.board[i],
      );
      if (changed) internal.lastMove = placedIndex(internal.state, state);
      internal.state = state;
      redraw();
    },
    destroy() {
      if (frame) cancelAnimationFrame(frame);
      element.replaceChildren();
      element.classList.remove('rv-board');
      delete element.dataset.turn;
    },
  };
}

function render(
  element: HTMLElement,
  internal: BoardInternalState,
  config: BoardConfig,
): void {
  element.replaceChildren();
  element.dataset.turn = internal.state.currentPlayer;

  const fragment = document.createDocumentFragment();
  const legalLookup = new Set(
    internal.state.legalMoves.map((move: Position) =>
      keyOf(move.row, move.col),
    ),
  );
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const square = document.createElement('button');
      square.type = 'button';
      square.className = 'rv-square';
      if ((row + col) % 2 === 0) square.dataset.tone = 'light';
      else square.dataset.tone = 'dark';
      square.dataset.row = String(row);
      square.dataset.col = String(col);
      square.setAttribute('aria-label', `row ${row + 1} column ${col + 1}`);
      if (legalLookup.has(keyOf(row, col))) square.dataset.legal = 'true';
      if (internal.lastMove === row * 8 + col) square.dataset.last = 'true';
      square.addEventListener('click', () => {
        if (legalLookup.has(keyOf(row, col))) config.onMove?.({ row, col });
      });

      const disc = internal.state.board[row * 8 + col];
      if (disc) square.appendChild(renderDisc(disc));
      else if (legalLookup.has(keyOf(row, col)))
        square.appendChild(renderHint());

      fragment.appendChild(square);
    }
  }

  element.appendChild(fragment);
}

function renderDisc(disc: Disc): HTMLElement {
  const token = document.createElement('span');
  token.className = 'rv-disc';
  token.dataset.disc = disc;
  return token;
}

function renderHint(): HTMLElement {
  const hint = document.createElement('span');
  hint.className = 'rv-hint';
  return hint;
}

function keyOf(row: number, col: number): string {
  return `${row}:${col}`;
}
