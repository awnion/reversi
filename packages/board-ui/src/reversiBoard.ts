import type { Disc, GameState, Position } from '@reversi/core';

export interface BoardConfig {
  state: GameState;
  onMove?: (position: Position) => void;
  /** Show the board mirrored left to right. */
  mirrored?: boolean;
}

/** A short label drawn on a square, e.g. an engine's opinion of a move. */
export interface SquareNote {
  position: Position;
  label: string;
  /** The move the note's author likes best. */
  best?: boolean;
}

export interface BoardApi {
  setState(state: GameState): void;
  /** Replace the notes on the board; they are dropped when the position changes. */
  setNotes(notes: SquareNote[]): void;
  /** Show the board mirrored left to right (column h on the left), or not. */
  setMirrored(mirrored: boolean): void;
  destroy(): void;
}

interface BoardInternalState {
  state: GameState;
  selected?: Position;
  /** Index of the disc placed by the most recent move, if any. */
  lastMove?: number | undefined;
  /** Indices of the discs that move turned over. */
  flipped: number[];
  /** Notes by square index. */
  notes: Map<number, SquareNote>;
  mirrored: boolean;
}

/** Discs that changed colour between two positions. */
function flippedIndices(previous: GameState, next: GameState): number[] {
  const flipped: number[] = [];
  for (let i = 0; i < next.board.length; i += 1) {
    const before = previous.board[i];
    if (before && next.board[i] && before !== next.board[i]) flipped.push(i);
  }
  return flipped;
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
    notes: new Map(),
    flipped: [],
    mirrored: config.mirrored ?? false,
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
      if (changed) {
        internal.lastMove = placedIndex(internal.state, state);
        internal.flipped =
          internal.lastMove === undefined
            ? []
            : flippedIndices(internal.state, state);
      }
      // Notes describe one position; a new one makes them stale.
      if (state !== internal.state) internal.notes.clear();
      internal.state = state;
      redraw();
    },
    setNotes(notes) {
      internal.notes = new Map(
        notes.map((note) => [note.position.row * 8 + note.position.col, note]),
      );
      redraw();
    },
    setMirrored(mirrored) {
      if (mirrored === internal.mirrored) return;
      internal.mirrored = mirrored;
      redraw();
    },
    destroy() {
      if (frame) cancelAnimationFrame(frame);
      element.replaceChildren();
      element.classList.remove('rv-board');
      delete element.dataset.turn;
      delete element.dataset.mirrored;
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
  element.dataset.mirrored = String(internal.mirrored);

  const fragment = document.createDocumentFragment();
  const legalLookup = new Set(
    internal.state.legalMoves.map((move: Position) =>
      keyOf(move.row, move.col),
    ),
  );
  for (let row = 0; row < 8; row += 1) {
    // Squares go in the order they are seen, so focus moves the way it looks.
    for (let shown = 0; shown < 8; shown += 1) {
      const col = internal.mirrored ? 7 - shown : shown;
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
      const note = internal.notes.get(row * 8 + col);
      if (disc) square.appendChild(renderDisc(disc));
      else if (note) square.appendChild(renderNote(note));
      else if (legalLookup.has(keyOf(row, col)))
        square.appendChild(renderHint());

      fragment.appendChild(square);
    }
  }

  element.appendChild(fragment);
  const mover =
    internal.lastMove === undefined
      ? null
      : internal.state.board[internal.lastMove];
  if (mover && internal.lastMove !== undefined && internal.flipped.length > 0)
    element.appendChild(
      renderFlips(
        internal.lastMove,
        internal.flipped,
        mover,
        internal.mirrored,
      ),
    );
}

const SVG = 'http://www.w3.org/2000/svg';

/**
 * The last move's captures: a line from the placed disc through each captured
 * run and a dot on every disc it turned over, both in the colour those discs
 * had before (`mover` is the side that captured them).
 */
function renderFlips(
  placed: number,
  flipped: number[],
  mover: Disc,
  mirrored: boolean,
): SVGSVGElement {
  // Centre of a square, as drawn.
  const x = (index: number) => (mirrored ? 7 - (index % 8) : index % 8) + 0.5;
  const y = (index: number) => Math.floor(index / 8) + 0.5;
  const svg = document.createElementNS(SVG, 'svg');
  svg.classList.add('rv-flips');
  svg.dataset.by = mover;
  svg.setAttribute('viewBox', '0 0 8 8');
  svg.setAttribute('aria-hidden', 'true');

  const row0 = Math.floor(placed / 8);
  const col0 = placed % 8;
  // The far end of each captured run, by direction from the placed disc.
  const ends = new Map<string, { index: number; distance: number }>();
  for (const index of flipped) {
    const dr = Math.floor(index / 8) - row0;
    const dc = (index % 8) - col0;
    const key = `${Math.sign(dr)}:${Math.sign(dc)}`;
    const distance = Math.max(Math.abs(dr), Math.abs(dc));
    if (distance > (ends.get(key)?.distance ?? 0))
      ends.set(key, { index, distance });
  }
  for (const { index: end } of ends.values()) {
    const line = document.createElementNS(SVG, 'line');
    line.classList.add('rv-flip-line');
    line.setAttribute('x1', String(x(placed)));
    line.setAttribute('y1', String(y(placed)));
    line.setAttribute('x2', String(x(end)));
    line.setAttribute('y2', String(y(end)));
    svg.appendChild(line);
  }
  for (const index of flipped) {
    const ring = document.createElementNS(SVG, 'circle');
    ring.classList.add('rv-flip');
    ring.setAttribute('cx', String(x(index)));
    ring.setAttribute('cy', String(y(index)));
    ring.setAttribute('r', '0.075');
    svg.appendChild(ring);
  }
  return svg;
}

function renderDisc(disc: Disc): HTMLElement {
  const token = document.createElement('span');
  token.className = 'rv-disc';
  token.dataset.disc = disc;
  return token;
}

function renderNote(note: SquareNote): HTMLElement {
  const label = document.createElement('span');
  label.className = 'rv-note';
  label.textContent = note.label;
  if (note.best) label.dataset.best = 'true';
  return label;
}

function renderHint(): HTMLElement {
  const hint = document.createElement('span');
  hint.className = 'rv-hint';
  return hint;
}

function keyOf(row: number, col: number): string {
  return `${row}:${col}`;
}
