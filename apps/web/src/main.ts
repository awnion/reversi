import { createReversiBoard } from '@reversi/board-ui';
import { greedyBot } from '@reversi/bot';
import {
  type BotPlayer,
  countDiscs,
  createInitialState,
  type Disc,
  type GameState,
} from '@reversi/core';
import { GameController, type PlayerConfig } from './gameController';
import { BotWorkerPool, createWorkerBot } from './workerBot';

import './styles.css';

const pool = new BotWorkerPool();

type BotId = 'greedy' | 'minimax' | 'alphazero' | 'zerorev';

/**
 * The bots a side can be handed to. A side without a bot is played by hand.
 *
 * `timed`: the bot searches for as long as it is allowed to.
 * `detail`: one line under the name; the network bots replace theirs with
 * facts about their weights once the WASM is loaded.
 */
const BOTS: {
  id: BotId;
  name: string;
  detail: string;
  timed: boolean;
  player: BotPlayer;
}[] = [
  {
    id: 'greedy',
    name: 'Greedy',
    detail: 'Flips the most discs',
    timed: false,
    player: greedyBot,
  },
  {
    id: 'minimax',
    name: 'Minimax',
    detail: 'Searches moves ahead',
    timed: true,
    player: createWorkerBot(pool, 'minimax', 'Minimax'),
  },
  {
    id: 'alphazero',
    name: 'AlphaZero',
    detail: 'Neural network',
    timed: true,
    player: createWorkerBot(pool, 'alphazero', 'AlphaZero'),
  },
  {
    id: 'zerorev',
    name: 'ZeroRev',
    detail: 'Neural network',
    timed: true,
    player: createWorkerBot(pool, 'zerorev', 'ZeroRev'),
  },
];

const SIDES: Disc[] = ['black', 'white'];
/** Time per move a bot can be given with one click, in milliseconds. */
const THINK_PRESETS_MS = [500, 1000, 2000, 5000];
const DEFAULT_THINK_MS = 1000;
const MIN_THINK_MS = 10;
const MAX_THINK_MS = 60_000;
/** Pause between moves of a bot that does not search. */
const UNTIMED_BOT_MS = 300;
/** Search budget for the Analyze button. */
const ANALYSIS_MS = 1000;

const app = document.getElementById('app');
if (!app) throw new Error('Missing app root');

function requireElement<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLElement)) throw new Error(`Missing element #${id}`);
  return el as T;
}

function capitalize(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function playerColumn(side: Disc): string {
  const options = BOTS.map(
    (bot) => `
      <label class="option">
        <input type="checkbox" name="bot-${side}" value="${bot.id}" />
        <span class="option-body">
          <span class="option-name">${bot.name}</span>
          <span class="option-detail" data-bot="${bot.id}">${bot.detail}</span>
        </span>
      </label>`,
  ).join('');

  return `
    <section class="player" id="player-${side}" data-side="${side}">
      <header class="player-head">
        <span class="player-disc" data-disc="${side}" aria-hidden="true"></span>
        <h2 class="player-name">${capitalize(side)}</h2>
        <span class="player-state" id="state-${side}" role="status"></span>
        <output class="player-count" id="score-${side}" aria-label="${capitalize(side)} discs">2</output>
        <span class="player-clock" id="clock-${side}"></span>
      </header>
      <div class="options" role="group" aria-label="Bot for ${side}">${options}</div>
      <fieldset class="think" id="think-${side}">
        <legend class="think-label">Seconds per move</legend>
        <div class="think-cells">
          ${THINK_PRESETS_MS.map(
            (ms) =>
              `<button type="button" class="think-cell" data-ms="${ms}">${ms / 1000}</button>`,
          ).join('')}
          <label class="think-cell think-custom">
            <input type="number" id="think-custom-${side}" inputmode="numeric"
              min="${MIN_THINK_MS}" max="${MAX_THINK_MS}" step="10"
              placeholder="…" aria-label="Custom time per move in milliseconds" />
            <span>ms</span>
          </label>
        </div>
      </fieldset>
    </section>`;
}

app.innerHTML = `
  <main class="table">
    <section class="board-area">
      <div class="board-frame">
        <div id="board-root" class="board-root"></div>
      </div>
    </section>
    <aside class="panel">
      <h1 class="visually-hidden">Reversi</h1>
      <div class="players">${SIDES.map(playerColumn).join('')}</div>
      <p id="status" class="status" role="status"></p>
      <div class="actions">
        <button id="play-button" type="button" class="action">Play</button>
        <button id="pass-button" type="button" class="action" hidden>Pass turn</button>
        <button id="new-game-button" type="button" class="action action-quiet">New game</button>
        <button id="analyze-button" type="button" class="action action-quiet">Analyze</button>
      </div>
      <div id="analysis-results" class="analysis"></div>
    </aside>
  </main>
`;

const boardRoot = requireElement<HTMLDivElement>('board-root');
const statusNode = requireElement<HTMLParagraphElement>('status');
const playButton = requireElement<HTMLButtonElement>('play-button');
const passButton = requireElement<HTMLButtonElement>('pass-button');
const newGameButton = requireElement<HTMLButtonElement>('new-game-button');
const analyzeButton = requireElement<HTMLButtonElement>('analyze-button');
const analysisResultsNode = requireElement<HTMLDivElement>('analysis-results');

const thinkMs: Record<Disc, number> = {
  black: DEFAULT_THINK_MS,
  white: DEFAULT_THINK_MS,
};

function botInputs(side: Disc): HTMLInputElement[] {
  return [
    ...document.querySelectorAll<HTMLInputElement>(`input[name="bot-${side}"]`),
  ];
}

/** The bot chosen for a side, if any. */
function chosenBot(side: Disc): (typeof BOTS)[number] | undefined {
  const id = botInputs(side).find((input) => input.checked)?.value;
  return BOTS.find((bot) => bot.id === id);
}

function playerConfig(side: Disc): PlayerConfig {
  const bot = chosenBot(side);
  if (!bot) return { type: 'human' };
  return {
    type: 'bot',
    player: bot.player,
    thinkMs: bot.timed ? thinkMs[side] : UNTIMED_BOT_MS,
  };
}

function players(): { black: PlayerConfig; white: PlayerConfig } {
  return { black: playerConfig('black'), white: playerConfig('white') };
}

/** The time is always shown, but only bots that search can change it. */
function syncThinkControl(side: Disc): void {
  const row = requireElement<HTMLFieldSetElement>(`think-${side}`);
  const custom = requireElement<HTMLInputElement>(`think-custom-${side}`);
  row.disabled = !(chosenBot(side)?.timed ?? false);

  const isPreset = THINK_PRESETS_MS.includes(thinkMs[side]);
  for (const cell of row.querySelectorAll<HTMLButtonElement>(
    'button.think-cell',
  )) {
    cell.setAttribute(
      'aria-pressed',
      String(Number(cell.dataset.ms) === thinkMs[side]),
    );
  }
  custom
    .closest('.think-custom')
    ?.setAttribute('data-active', String(!isPreset));
  // Leave the field alone while it is being typed in.
  if (isPreset) custom.value = '';
  else if (document.activeElement !== custom)
    custom.value = String(thinkMs[side]);
}

const board = createReversiBoard(boardRoot, {
  state: createInitialState(),
  onMove(position) {
    void controller.makeMove(position);
  },
});

/** What the scoreboard says next to a side's name. */
function sideState(state: GameState, side: Disc, botOnMove: boolean): string {
  if (state.status === 'finished') {
    if (state.winner === 'draw' || state.winner === null) return 'draw';
    return state.winner === side ? 'wins' : '';
  }
  if (state.currentPlayer !== side) return '';
  return botOnMove ? 'thinking' : 'to move';
}

/**
 * Milliseconds with the thousands set apart from the rest: `12 345 ms`.
 * The two groups are separate elements so that they can be coloured apart.
 */
function millisecondsHtml(ms: number): string {
  const total = Math.max(0, Math.round(ms));
  const thousands = Math.floor(total / 1000);
  const rest = total % 1000;
  const lower = thousands > 0 ? String(rest).padStart(3, '0') : String(rest);
  return `${thousands > 0 ? `<span class="ms-thousands">${thousands}</span>` : ''}<span class="ms-rest">${lower}</span><span class="ms-unit">ms</span>`;
}

/**
 * The clock of a side: for the side on the move, the time a searching bot has
 * left (or, for anyone else, how long the move is taking); for the other
 * side, how long its last move took.
 */
function clockHtml(
  state: GameState,
  side: Disc,
): { html: string; title: string } {
  const last = controller.lastMoveMs[side];
  if (state.status === 'finished' || state.currentPlayer !== side) {
    return last === null
      ? { html: '', title: '' }
      : { html: millisecondsHtml(last), title: 'Time taken for the last move' };
  }

  const now = performance.now();
  const bot = chosenBot(side);
  if (!controller.paused && bot?.timed) {
    const budget = thinkMs[side];
    const spent =
      controller.thinkingSince === null ? 0 : now - controller.thinkingSince;
    return {
      html: millisecondsHtml(budget - spent),
      title: 'Time left for this move',
    };
  }
  // Before the first move nobody is on the clock.
  if (!controller.started) return { html: '', title: '' };
  return {
    html: millisecondsHtml(now - controller.turnStartedAt),
    title: 'Time spent on this move so far',
  };
}

function syncClocks(): void {
  const state = controller.state;
  for (const side of SIDES) {
    const clock = requireElement(`clock-${side}`);
    const { html, title } = clockHtml(state, side);
    if (clock.innerHTML !== html) clock.innerHTML = html;
    clock.title = title;
  }
}

/**
 * Bots move only while the game is playing. Paused, every side is moved by
 * hand, so a person can always take over by pressing Pause.
 */
function sync(state: GameState): void {
  board.setState(state);
  const score = countDiscs(state.board);
  const finished = state.status === 'finished';
  const botOnMove =
    !finished &&
    !controller.paused &&
    chosenBot(state.currentPlayer) !== undefined;

  // The scoreboard carries the state of the game: whose move, or who won.
  for (const side of SIDES) {
    requireElement(`score-${side}`).textContent = String(score[side]);
    requireElement(`state-${side}`).textContent = sideState(
      state,
      side,
      botOnMove,
    );
    const column = requireElement(`player-${side}`);
    const onMove = !finished && state.currentPlayer === side;
    column.dataset.turn = String(onMove);
    column.dataset.thinking = String(onMove && botOnMove);
    column.dataset.result = finished
      ? state.winner === side
        ? 'won'
        : state.winner === 'draw' || state.winner === null
          ? 'draw'
          : 'lost'
      : '';
  }

  // While a bot is on the move the board only shows where it can go.
  boardRoot.dataset.locked = String(finished || botOnMove);
  const mustPass = !finished && !botOnMove && state.legalMoves.length === 0;
  // The line under the players is only for what the scoreboard cannot say.
  if (!botsFailed) {
    statusNode.textContent = mustPass
      ? `${capitalize(state.currentPlayer)} has no moves and must pass.`
      : '';
  }
  passButton.hidden = !mustPass;

  const anyBot = SIDES.some((side) => chosenBot(side) !== undefined);
  playButton.textContent = controller.paused ? 'Play' : 'Pause';
  playButton.disabled = finished || (controller.paused && !anyBot);
  playButton.title =
    controller.paused && !anyBot ? 'Choose a bot for black or white first' : '';
  syncClocks();
}

/** Set when the WASM bots could not be loaded; the status line says so. */
let botsFailed = false;

const controller = new GameController(createInitialState(), players(), sync);
// Nothing plays by itself: the bots wait for the Play button.
controller.pause();

/** Choosing a bot never starts it: the game pauses and waits for Play. */
function applyPlayers(): void {
  controller.pause();
  controller.setPlayers(players());
  sync(controller.state);
}

for (const side of SIDES) {
  for (const input of botInputs(side)) {
    input.addEventListener('change', () => {
      // One bot per side; clicking the chosen one again hands the side back.
      if (input.checked) {
        for (const other of botInputs(side)) {
          if (other !== input) other.checked = false;
        }
      }
      syncThinkControl(side);
      applyPlayers();
    });
  }

  // A new time applies to the bot's next move; it never starts a paused game.
  const setThinkMs = (ms: number) => {
    thinkMs[side] = Math.min(
      MAX_THINK_MS,
      Math.max(MIN_THINK_MS, Math.round(ms)),
    );
    syncThinkControl(side);
    controller.setPlayers(players());
  };

  for (const cell of requireElement(
    `think-${side}`,
  ).querySelectorAll<HTMLButtonElement>('button.think-cell')) {
    cell.addEventListener('click', () => setThinkMs(Number(cell.dataset.ms)));
  }

  const custom = requireElement<HTMLInputElement>(`think-custom-${side}`);
  custom.addEventListener('input', () => {
    const ms = Number(custom.value);
    if (custom.value !== '' && Number.isFinite(ms) && ms > 0) setThinkMs(ms);
  });
  // Show the value actually used (clamped) once editing is done.
  custom.addEventListener('blur', () => syncThinkControl(side));

  syncThinkControl(side);
}

playButton.addEventListener('click', () => {
  if (controller.paused) controller.play();
  else controller.pause();
  sync(controller.state);
});

passButton.addEventListener('click', () => {
  void controller.pass();
});

newGameButton.addEventListener('click', () => {
  analysisResultsNode.replaceChildren();
  controller.pause();
  controller.newGame(players(), createInitialState());
});

/** A score with an explicit sign, using a real minus. */
function signed(score: number): string {
  if (score > 0) return `+${score}`;
  return score < 0 ? `−${-score}` : '0';
}

/** Board coordinates as on a real board: columns a–h, rows 1–8. */
function squareName(row: number, col: number): string {
  return `${'abcdefgh'[col]}${row + 1}`;
}

const analysisBot = BOTS.find((bot) => bot.id === 'minimax');

analyzeButton.addEventListener('click', async () => {
  if (!analysisBot) return;
  analyzeButton.disabled = true;
  analysisResultsNode.textContent = 'Analyzing…';
  try {
    const analysis = await analysisBot.player.analyzePosition(
      controller.state,
      ANALYSIS_MS,
    );
    if (analysis.moves.length === 0) {
      analysisResultsNode.textContent = 'No legal moves here.';
      return;
    }
    const rows = analysis.moves
      .map(
        (m) =>
          `<li><span>${squareName(m.position.row, m.position.col)}</span><span>${signed(m.score)}</span></li>`,
      )
      .join('');
    analysisResultsNode.innerHTML = `<p class="analysis-title">Minimax ranks the moves</p><ol class="analysis-list">${rows}</ol>`;
  } catch {
    analysisResultsNode.textContent = 'Analysis is unavailable right now.';
  } finally {
    analyzeButton.disabled = false;
  }
});

// Show the starting position; a person can move right away.
sync(controller.state);

// The running clock of the side on the move.
(function tick(): void {
  if (controller.state.status !== 'finished') syncClocks();
  requestAnimationFrame(tick);
})();

function shortDate(isoDay: string): string {
  const date = new Date(`${isoDay}T00:00:00`);
  if (Number.isNaN(date.getTime())) return isoDay;
  return date.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

// The network bots describe their weights once the WASM is loaded.
pool.ready
  .then((metas) => {
    for (const [bot, meta] of Object.entries(metas)) {
      const loss = meta.loss === null ? '' : `, loss ${meta.loss.toFixed(2)}`;
      const winRate =
        meta.winRate === null
          ? ''
          : `, won ${Math.round(meta.winRate * 100)}% of its tournament`;
      for (const detail of document.querySelectorAll<HTMLElement>(
        `.option-detail[data-bot="${bot}"]`,
      )) {
        detail.textContent = `${shortDate(meta.trainedOn)}${loss}`;
        detail.title = `Run ${meta.run}${winRate}. Built ${meta.buildDate}.`;
      }
    }
  })
  .catch((err) => {
    console.error('Bots failed to load', err);
    botsFailed = true;
    statusNode.textContent = 'The bots could not be loaded. Reload the page.';
  });
