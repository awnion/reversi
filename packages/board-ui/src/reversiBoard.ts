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
      internal.state = state;
      redraw();
    },
    destroy() {
      if (frame) cancelAnimationFrame(frame);
      element.replaceChildren();
      element.classList.remove('rv-board');
    },
  };
}

function render(
  element: HTMLElement,
  internal: BoardInternalState,
  config: BoardConfig,
): void {
  element.replaceChildren();

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
