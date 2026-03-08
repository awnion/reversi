import { createReversiBoard } from '@reversi/board-ui';
import {
  applyMove,
  countDiscs,
  createInitialState,
  type GameState,
  type Position,
  passTurn,
} from '@reversi/core';

import './styles.css';

let state = createInitialState();

const app = document.getElementById('app');
if (!app) throw new Error('Missing app root');

app.innerHTML = `
  <main class="app-shell">
    <section class="board-panel">
      <div id="board-root" class="board-root"></div>
    </section>
    <aside class="sidebar">
      <h1>Reversi</h1>
      <p id="status"></p>
      <p id="score"></p>
      <button id="pass-button" type="button">Pass turn</button>
    </aside>
  </main>
`;

const boardRoot = requireElement<HTMLDivElement>('board-root');
const statusNode = requireElement<HTMLParagraphElement>('status');
const scoreNode = requireElement<HTMLParagraphElement>('score');
const passButton = requireElement<HTMLButtonElement>('pass-button');

const board = createReversiBoard(boardRoot, {
  state,
  onMove(position: Position) {
    state = applyMove(state, position);
    sync(state);
  },
});

passButton.addEventListener('click', () => {
  if (state.legalMoves.length === 0 && state.status !== 'finished') {
    state = passTurn(state);
    sync(state);
  }
});

sync(state);

function sync(nextState: GameState): void {
  board.setState(nextState);
  const score = countDiscs(nextState.board);
  const turn =
    nextState.status === 'finished'
      ? 'Game over'
      : `Turn: ${nextState.currentPlayer}`;
  const extra =
    nextState.status === 'finished'
      ? nextState.winner === 'draw'
        ? 'Winner: draw'
        : `Winner: ${nextState.winner}`
      : nextState.legalMoves.length === 0
        ? 'No legal moves available'
        : `${nextState.legalMoves.length} legal moves`;
  statusNode.textContent = `${turn}. ${extra}.`;
  scoreNode.textContent = `Black ${score.black} - ${score.white} White`;
  passButton.disabled =
    nextState.legalMoves.length > 0 || nextState.status === 'finished';
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement))
    throw new Error(`Missing element #${id}`);
  return element as T;
}
