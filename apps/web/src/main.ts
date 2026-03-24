import { createReversiBoard } from '@reversi/board-ui';
import {
  countDiscs,
  createInitialState,
  type GameState,
} from '@reversi/core';
import { greedyBot } from '@reversi/bot';
import { createMinimaxBot } from '@reversi/bot-minimax';
import { GameController, type PlayerConfig } from './gameController';

import './styles.css';

const minimax1s = await createMinimaxBot('Minimax 1s');
const minimax5s = await createMinimaxBot('Minimax 5s');

const app = document.getElementById('app');
if (!app) throw new Error('Missing app root');

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement))
    throw new Error(`Missing element #${id}`);
  return element as T;
}


app.innerHTML = `
  <main class="app-shell">
    <section class="board-panel">
      <div id="board-root" class="board-root"></div>
    </section>
    <aside class="sidebar">
      <h1>Reversi</h1>
      <div class="player-selection" style="margin-bottom: 20px;">
        <label>
          Black:
          <select id="player-black">
            <option value="human">Human</option>
            <option value="bot-greedy">Greedy (instant)</option>
            <option value="bot-minimax-1s">Minimax (1s)</option>
            <option value="bot-minimax-5s">Minimax (5s)</option>
          </select>
        </label>
        <br />
        <label>
          White:
          <select id="player-white">
            <option value="human">Human</option>
            <option value="bot-greedy">Greedy (instant)</option>
            <option value="bot-minimax-1s">Minimax (1s)</option>
            <option value="bot-minimax-5s">Minimax (5s)</option>
          </select>
        </label>
        <br /><br />
        <button id="new-game-button" type="button">New Game</button>
      </div>
      <hr style="margin-bottom: 20px;" />
      <p id="status"></p>
      <p id="score"></p>
      <button id="pass-button" type="button">Pass turn</button>
      <hr style="margin-bottom: 20px; margin-top: 20px;" />
      <button id="analyze-button" type="button">Analyze Position</button>
      <div id="analysis-results" class="analysis-results" style="margin-top: 10px; font-size: 14px; max-height: 200px; overflow-y: auto;"></div>
    </aside>
  </main>
`;

const boardRoot = requireElement<HTMLDivElement>('board-root');
const statusNode = requireElement<HTMLParagraphElement>('status');
const scoreNode = requireElement<HTMLParagraphElement>('score');
const passButton = requireElement<HTMLButtonElement>('pass-button');
const newGameButton = requireElement<HTMLButtonElement>('new-game-button');
const playerBlackSelect = requireElement<HTMLSelectElement>('player-black');
const playerWhiteSelect = requireElement<HTMLSelectElement>('player-white');

function getPlayerConfig(select: HTMLSelectElement): PlayerConfig {
  if (select.value === 'bot-greedy') {
    return { type: 'bot', player: greedyBot, thinkMs: 300 };
  }
  if (select.value === 'bot-minimax-1s') {
    return { type: 'bot', player: minimax1s, thinkMs: 1000 };
  }
  if (select.value === 'bot-minimax-5s') {
    return { type: 'bot', player: minimax5s, thinkMs: 5000 };
  }
  return { type: 'human' };
}


let controller: GameController | null = null;

const board = createReversiBoard(boardRoot, {
  state: createInitialState(),
  onMove(position) {
    if (controller) void controller.makeMove(position);
  },
});

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

controller = new GameController(
  createInitialState(),
  { black: getPlayerConfig(playerBlackSelect), white: getPlayerConfig(playerWhiteSelect) },
  sync
);

passButton.addEventListener('click', () => {
  if (controller) void controller.pass();
});

const analyzeButton = requireElement<HTMLButtonElement>('analyze-button');
const analysisResultsNode = requireElement<HTMLDivElement>('analysis-results');

newGameButton.addEventListener('click', () => {
  if (controller) {
    analysisResultsNode.innerHTML = '';
    controller.newGame(
      { black: getPlayerConfig(playerBlackSelect), white: getPlayerConfig(playerWhiteSelect) },
      createInitialState()
    );
  }
});

analyzeButton.addEventListener('click', async () => {
  if (!controller) return;
  analyzeButton.disabled = true;
  analysisResultsNode.innerHTML = 'Analyzing...';
  try {
    const analysis = await minimax5s.analyzePosition(controller.state, 1000);
    analysisResultsNode.innerHTML = `
      <h4>Ranked Moves</h4>
      <ul style="padding-left: 20px;">
        ${analysis.moves.length === 0 ? '<li>No valid moves</li>' : ''}
        ${analysis.moves.map(m => `<li>R${m.position.row} C${m.position.col}: Score ${m.score}</li>`).join('')}
      </ul>
    `;
  } catch (e) {
    analysisResultsNode.innerHTML = 'Analysis unavailable.';
  } finally {
    analyzeButton.disabled = false;
  }
});

void controller.start();

