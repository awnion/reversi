import { createReversiBoard } from '@reversi/board-ui';
import { greedyBot } from '@reversi/bot';
import { countDiscs, createInitialState, type GameState } from '@reversi/core';
import { GameController, type PlayerConfig } from './gameController';
import { BotWorkerPool, createWorkerBot } from './workerBot';

import './styles.css';

const pool = new BotWorkerPool();
const minimaxBot = createWorkerBot(pool, 'minimax', 'Minimax');
const alphazeroBot = createWorkerBot(pool, 'alphazero', 'AlphaZero');

const app = document.getElementById('app');
if (!app) throw new Error('Missing app root');

function requireElement<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLElement)) throw new Error(`Missing element #${id}`);
  return el as T;
}

app.innerHTML = `
  <main class="app-shell">
    <section class="board-panel">
      <div id="board-root" class="board-root"></div>
    </section>
    <aside class="sidebar">
      <h1>Reversi</h1>
      <div class="player-selection" style="margin-bottom: 20px;">
        <div class="player-row">
          <span class="player-label">Black:</span>
          <select id="strategy-black">
            <option value="human">Human</option>
            <option value="greedy">Greedy</option>
            <option value="minimax">Minimax</option>
            <option value="alphazero">AlphaZero</option>
          </select>
          <select id="time-black" class="time-select" disabled>
            ${timeOptions()}
          </select>
        </div>
        <div class="player-row" style="margin-top: 8px;">
          <span class="player-label">White:</span>
          <select id="strategy-white">
            <option value="human">Human</option>
            <option value="greedy">Greedy</option>
            <option value="minimax">Minimax</option>
            <option value="alphazero">AlphaZero</option>
          </select>
          <select id="time-white" class="time-select" disabled>
            ${timeOptions()}
          </select>
        </div>
        <br />
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

function timeOptions(): string {
  return Array.from({ length: 10 }, (_, i) => i + 1)
    .map(
      (s) =>
        `<option value="${s * 1000}"${s === 1 ? ' selected' : ''}>${s}s</option>`,
    )
    .join('');
}

const strategyBlack = requireElement<HTMLSelectElement>('strategy-black');
const strategyWhite = requireElement<HTMLSelectElement>('strategy-white');
const timeBlack = requireElement<HTMLSelectElement>('time-black');
const timeWhite = requireElement<HTMLSelectElement>('time-white');
const boardRoot = requireElement<HTMLDivElement>('board-root');
const statusNode = requireElement<HTMLParagraphElement>('status');
const scoreNode = requireElement<HTMLParagraphElement>('score');
const passButton = requireElement<HTMLButtonElement>('pass-button');
const newGameButton = requireElement<HTMLButtonElement>('new-game-button');

function syncTimeSelect(
  strategy: HTMLSelectElement,
  time: HTMLSelectElement,
): void {
  const needsTime =
    strategy.value === 'minimax' || strategy.value === 'alphazero';
  time.disabled = !needsTime;
}

strategyBlack.addEventListener('change', () =>
  syncTimeSelect(strategyBlack, timeBlack),
);
strategyWhite.addEventListener('change', () =>
  syncTimeSelect(strategyWhite, timeWhite),
);

function getPlayerConfig(
  strategy: HTMLSelectElement,
  time: HTMLSelectElement,
): PlayerConfig {
  const thinkMs = parseInt(time.value, 10);
  if (strategy.value === 'greedy') {
    return { type: 'bot', player: greedyBot, thinkMs: 300 };
  }
  if (strategy.value === 'minimax') {
    return { type: 'bot', player: minimaxBot, thinkMs };
  }
  if (strategy.value === 'alphazero') {
    return { type: 'bot', player: alphazeroBot, thinkMs };
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
  {
    black: getPlayerConfig(strategyBlack, timeBlack),
    white: getPlayerConfig(strategyWhite, timeWhite),
  },
  sync,
);

passButton.addEventListener('click', () => {
  if (controller) void controller.pass();
});

newGameButton.addEventListener('click', () => {
  if (controller) {
    requireElement<HTMLDivElement>('analysis-results').innerHTML = '';
    controller.newGame(
      {
        black: getPlayerConfig(strategyBlack, timeBlack),
        white: getPlayerConfig(strategyWhite, timeWhite),
      },
      createInitialState(),
    );
  }
});

const analyzeButton = requireElement<HTMLButtonElement>('analyze-button');
const analysisResultsNode = requireElement<HTMLDivElement>('analysis-results');

analyzeButton.addEventListener('click', async () => {
  if (!controller) return;
  analyzeButton.disabled = true;
  analysisResultsNode.innerHTML = 'Analyzing...';
  try {
    const analysis = await minimaxBot.analyzePosition(controller.state, 5000);
    analysisResultsNode.innerHTML = `
      <h4>Ranked Moves</h4>
      <ul style="padding-left: 20px;">
        ${analysis.moves.length === 0 ? '<li>No valid moves</li>' : ''}
        ${analysis.moves.map((m) => `<li>R${m.position.row} C${m.position.col}: Score ${m.score}</li>`).join('')}
      </ul>
    `;
  } catch {
    analysisResultsNode.innerHTML = 'Analysis unavailable.';
  } finally {
    analyzeButton.disabled = false;
  }
});

void controller.start();
