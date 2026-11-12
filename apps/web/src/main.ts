import { createReversiBoard } from '@reversi/board-ui';
import { greedyBot } from '@reversi/bot';
import {
  type BotPlayer,
  countDiscs,
  createInitialState,
  type Disc,
  type GameState,
  parseState,
  serializeState,
} from '@reversi/core';
import { GameController, type PlayerConfig } from './gameController';
import { BotWorkerPool, createWorkerBot } from './workerBot';

import './styles.css';

/** Logical cores of this machine, as far as the browser tells. */
const CPU_CORES = Math.min(Math.max(navigator.hardwareConcurrency || 1, 1), 16);
/**
 * Threads for a bot's search. Sharing memory between threads needs a
 * cross-origin isolated page; elsewhere every search runs on one thread.
 * Two cores are left to the page and the system.
 */
const BOT_THREADS = globalThis.crossOriginIsolated
  ? Math.max(1, Math.min(CPU_CORES - 2, 8))
  : 1;
/** Analysis runs next to the bots, so it gets half as many. */
const ANALYSIS_THREADS = Math.max(1, Math.floor(BOT_THREADS / 2));
/** Bots whose search uses every thread it is given. */
const THREADED_BOTS: ReadonlySet<string> = new Set([
  'minimax',
  'alphazero',
  'zerorev',
]);

const pool = new BotWorkerPool(BOT_THREADS);

type BotId = 'greedy' | 'minimax' | 'alphazero' | 'zerorev';

/**
 * The bots a side can be handed to. A side without a bot is played by hand.
 *
 * `timed`: the bot searches for as long as it is allowed to.
 * `detail`: one line under the name. The network bots have none until the
 * WASM is loaded; then they tell when their weights were trained.
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
    detail: '',
    timed: true,
    player: createWorkerBot(pool, 'alphazero', 'AlphaZero'),
  },
  {
    id: 'zerorev',
    name: 'ZeroRev',
    detail: '',
    timed: true,
    player: createWorkerBot(pool, 'zerorev', 'ZeroRev'),
  },
];

const SIDES: Disc[] = ['black', 'white'];
/** Bots that can annotate a position, in the order they are offered. */
const ANALYSIS_ENGINES: BotId[] = ['zerorev', 'minimax'];
/** Time per move a bot can be given with one click, in milliseconds. */
const THINK_PRESETS_MS = [500, 1000, 2000, 5000];
const DEFAULT_THINK_MS = 1000;
const MIN_THINK_MS = 10;
const MAX_THINK_MS = 60_000;
/** Pause between moves of a bot that does not search. */
const UNTIMED_BOT_MS = 300;
/**
 * Analysis keeps going while the switch is on, in slices of this length: each
 * call carries the engine's search of the position further, and the board
 * shows where it has got to.
 */
const ANALYSIS_SLICE_MS = 500;

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
      <label class="option" title="${bot.detail || 'Neural network'}" data-bot="${bot.id}">
        <input type="checkbox" name="bot-${side}" value="${bot.id}" />
        <span class="option-body">
          <span class="option-name">${bot.name}</span>
          <span class="option-detail" data-bot="${bot.id}"${bot.detail ? '' : ' data-pending'}>${bot.detail}</span>
        </span>
      </label>`,
  ).join('');

  return `
    <section class="player" id="player-${side}" data-side="${side}">
      <header class="player-head">
        <h2 class="player-name visually-hidden">${capitalize(side)}</h2>
        <div class="player-top">
          <span class="player-clock" id="clock-${side}" data-idle="true">00:00<span class="clock-tenth">.0</span></span>
          <span class="player-state" id="state-${side}" role="status"></span>
        </div>
        <div class="player-score">
          <span class="player-disc" data-disc="${side}" aria-hidden="true"></span>
          <output class="player-count" id="score-${side}" aria-label="${capitalize(side)} discs">2</output>
        </div>
      </header>
      <div class="options" role="group" aria-label="Bot for ${side}">${options}</div>
      <fieldset class="think" id="think-${side}" aria-label="Time per move for ${side}">
        <div class="think-cells">
          ${THINK_PRESETS_MS.map(
            (ms) =>
              `<button type="button" class="think-cell" data-ms="${ms}">${ms / 1000}s</button>`,
          ).join('')}
          <label class="think-cell think-custom">
            <input type="text" id="think-custom-${side}" inputmode="decimal"
              autocomplete="off" placeholder="0.2"
              aria-label="Custom time per move in seconds" />
            <span>s</span>
          </label>
        </div>
      </fieldset>
    </section>`;
}

app.innerHTML = `
  <main class="table">
    <section class="board-area">
      <div class="eval-bar" id="eval-bar" role="img" hidden>
        <span class="eval-white"></span>
        <span class="eval-label" id="eval-label"></span>
      </div>
      <div class="board-frame">
        <div id="board-root" class="board-root"></div>
      </div>
      <button type="button" class="mirror" id="mirror-board" aria-pressed="false"
        aria-label="Mirror the board" title="Mirror the board">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path class="mirror-axis" d="M12 3v18" />
          <path class="mirror-shape" d="M9 6 3 18h6z" />
          <path class="mirror-shape mirror-image" d="M15 6l6 12h-6z" />
        </svg>
      </button>
      <div class="pass-prompt" id="pass-prompt" role="status" hidden>
        <span id="pass-text"></span>
        <button id="pass-button" type="button" class="action">Pass</button>
      </div>
    </section>
    <aside class="panel">
      <h1 class="visually-hidden">Reversi</h1>
      <div class="players">${SIDES.map(playerColumn).join('')}</div>
      <p id="status" class="status" role="status"></p>
      <div class="actions">
        <button id="play-button" type="button" class="action">Play</button>
        <button id="new-game-button" type="button" class="action action-quiet">New game</button>
        <div class="cpu" id="cpu-meter" role="img">
          <span class="cpu-label" aria-hidden="true">CPU</span>
          <span class="cpu-cores" aria-hidden="true">${Array.from(
            { length: CPU_CORES },
            (_, core) =>
              `<i style="--core: ${core}">${'<b></b>'.repeat(4)}</i>`,
          ).join('')}</span>
        </div>
      </div>
      <div class="analysis">
        <div class="analysis-head">
          <button type="button" id="analysis-switch" class="switch" role="switch" aria-checked="false">
            <span class="switch-track" aria-hidden="true"><span class="switch-knob"></span></span>
            <span class="switch-label">Analysis</span>
          </button>
          <div class="analysis-engines" role="group" aria-label="Engine used for analysis">
            ${ANALYSIS_ENGINES.map(
              (id, index) =>
                `<button type="button" class="analysis-engine" data-engine="${id}" aria-pressed="${index === 0}">${
                  BOTS.find((bot) => bot.id === id)?.name ?? id
                }</button>`,
            ).join('')}
          </div>
        </div>
        <div id="analysis-results" class="analysis-summary"></div>
      </div>
    </aside>
  </main>
`;

const boardRoot = requireElement<HTMLDivElement>('board-root');
const statusNode = requireElement<HTMLParagraphElement>('status');
const playButton = requireElement<HTMLButtonElement>('play-button');
const passButton = requireElement<HTMLButtonElement>('pass-button');
const passPrompt = requireElement('pass-prompt');
const passText = requireElement('pass-text');
const newGameButton = requireElement<HTMLButtonElement>('new-game-button');
const analysisResultsNode = requireElement<HTMLDivElement>('analysis-results');
const analysisSwitch = requireElement<HTMLButtonElement>('analysis-switch');
const analysisButtons = [
  ...document.querySelectorAll<HTMLButtonElement>('.analysis-engine'),
];

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
    custom.value = String(thinkMs[side] / 1000);
}

/** Remembered between visits; the page works without it. */
const MIRROR_KEY = 'reversi.mirrored';
function storedMirror(): boolean {
  try {
    return localStorage.getItem(MIRROR_KEY) === 'true';
  } catch {
    return false;
  }
}

const board = createReversiBoard(boardRoot, {
  state: createInitialState(),
  mirrored: storedMirror(),
  onMove(position) {
    void controller.makeMove(position);
  },
});

// Some players are used to the board the other way round.
const mirrorButton = requireElement<HTMLButtonElement>('mirror-board');
mirrorButton.setAttribute('aria-pressed', String(storedMirror()));
mirrorButton.addEventListener('click', () => {
  const mirrored = mirrorButton.getAttribute('aria-pressed') !== 'true';
  mirrorButton.setAttribute('aria-pressed', String(mirrored));
  board.setMirrored(mirrored);
  try {
    localStorage.setItem(MIRROR_KEY, String(mirrored));
  } catch {
    // Not remembered; it still applies to this visit.
  }
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
 * A stopwatch reading, minutes and seconds to a tenth: `00:11.8`. Every digit
 * is always there, so in a monospace face nothing shifts as it runs. A
 * countdown rounds up, so it reads 00:00.0 only once the time is gone.
 */
function clockText(ms: number, countdown = false): string {
  const round = countdown ? Math.ceil : Math.floor;
  const tenths = Math.max(0, round(ms / 100));
  const seconds = Math.floor(tenths / 10);
  const mm = String(Math.min(99, Math.floor(seconds / 60))).padStart(2, '0');
  const ss = String(seconds % 60).padStart(2, '0');
  return `${mm}:${ss}<span class="clock-tenth">.${tenths % 10}</span>`;
}

/**
 * The clock of a side: for the side on the move, the time a searching bot has
 * left (or, for anyone else, how long the move is taking); for the other
 * side, how long its last move took.
 */
function clockHtml(
  state: GameState,
  side: Disc,
): { html: string | null; title: string } {
  const last = controller.lastMoveMs[side];
  if (state.status === 'finished' || state.currentPlayer !== side) {
    return last === null
      ? { html: null, title: '' }
      : { html: clockText(last), title: 'Time taken for the last move' };
  }

  const now = performance.now();
  const bot = chosenBot(side);
  if (!controller.paused && bot?.timed) {
    const budget = thinkMs[side];
    const spent =
      controller.thinkingSince === null ? 0 : now - controller.thinkingSince;
    return {
      html: clockText(budget - spent, true),
      title: 'Time left for this move',
    };
  }
  // Before the first move nobody is on the clock.
  if (!controller.started) return { html: null, title: '' };
  return {
    html: clockText(now - controller.turnStartedAt),
    title: 'Time spent on this move so far',
  };
}

/** True while an analysis search is running (on its own worker). */
let analyzing = false;

/**
 * A row of dots, one per core, lit for the cores a search is using right
 * now. The browser cannot report real CPU load, so this shows what the bots
 * themselves are doing.
 */
function syncCpuMeter(): void {
  const thinking = controller.thinkingSince !== null;
  const mover = chosenBot(controller.state.currentPlayer);
  // Threads kept busy by the bot on the move and by the analysis.
  const botThreads =
    thinking && mover?.timed
      ? THREADED_BOTS.has(mover.id)
        ? BOT_THREADS
        : 1
      : 0;
  const analysisThreads = analyzing
    ? THREADED_BOTS.has(analysisEngine)
      ? ANALYSIS_THREADS
      : 1
    : 0;
  const searching = botThreads + analysisThreads > 0;
  const busy = Math.min(botThreads + analysisThreads, CPU_CORES);

  const meter = requireElement('cpu-meter');
  if (meter.dataset.busy === String(busy)) return;
  meter.dataset.busy = String(busy);
  meter.title = searching
    ? `Bots are searching on ${busy} of ${CPU_CORES} CPU threads`
    : `CPU idle. A thinking bot uses up to ${BOT_THREADS} of ${CPU_CORES} threads`;
  meter.setAttribute('aria-label', meter.title);
  meter.querySelectorAll('.cpu-cores i').forEach((core, index) => {
    core.classList.toggle('busy', index < busy);
  });
}

function syncClocks(): void {
  const state = controller.state;
  for (const side of SIDES) {
    const clock = requireElement(`clock-${side}`);
    const { html, title } = clockHtml(state, side);
    // An idle clock keeps its place, showing zeros, dimmed.
    const shown = html ?? clockText(0);
    if (clock.innerHTML !== shown) clock.innerHTML = shown;
    clock.dataset.idle = String(html === null);
    clock.title = title;
  }
}

/**
 * Bots move only while the game is playing. Paused, every side is moved by
 * hand, so a person can always take over by pressing Pause.
 */
function sync(state: GameState): void {
  board.setState(state);
  storePosition(state);
  // An analysis belongs to one position: look at the new one.
  if (state !== analyzedState) restartAnalysis();
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
  // No legal move: say so where the player is looking, on the board.
  passText.textContent = `${capitalize(state.currentPlayer)} has no moves`;
  passPrompt.hidden = !mustPass;

  const anyBot = SIDES.some((side) => chosenBot(side) !== undefined);
  playButton.textContent = controller.paused ? 'Play' : 'Pause';
  playButton.disabled = finished || (controller.paused && !anyBot);
  playButton.title =
    controller.paused && !anyBot ? 'Choose a bot for black or white first' : '';
  syncClocks();
}

/**
 * The position on the board is kept across page loads; a reload continues
 * the game (paused, like any game the page opens with).
 */
const POSITION_KEY = 'reversi.position';

function storedPosition(): GameState | null {
  try {
    const text = localStorage.getItem(POSITION_KEY);
    return text === null ? null : parseState(text);
  } catch {
    return null;
  }
}

function storePosition(state: GameState): void {
  try {
    localStorage.setItem(POSITION_KEY, serializeState(state));
  } catch {
    // Storage may be unavailable (private mode, quota); the game goes on.
  }
}

const controller = new GameController(
  storedPosition() ?? createInitialState(),
  players(),
  sync,
);
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
  // Seconds, fractions welcome: "0.7", "0,2", ".5".
  custom.addEventListener('input', () => {
    const text = custom.value.trim().replace(',', '.');
    const seconds = Number(text);
    if (text !== '' && Number.isFinite(seconds) && seconds > 0) {
      setThinkMs(seconds * 1000);
    }
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
  controller.pause();
  controller.newGame(players(), createInitialState());
});

/** A score with an explicit sign, using a real minus. */
function signed(score: number): string {
  if (score > 0) return `+${score}`;
  return score < 0 ? `\u2212${-score}` : '0';
}

/** Board coordinates as on a real board: columns a–h, rows 1–8. */
function squareName(row: number, col: number): string {
  return `${'abcdefgh'[col]}${row + 1}`;
}

// ── Analysis ────────────────────────────────────────────────────────────────
//
// While the switch is on, the chosen engine keeps looking at the current
// position, deeper with each step, and writes its opinion of every legal move
// on the board. It runs on a worker of its own, so it never delays the bots.

let analysisOn = false;
let analysisEngine: BotId = ANALYSIS_ENGINES[0] ?? 'zerorev';
/** The position the analysis on the board refers to. */
let analyzedState: GameState | null = null;
/** Bumped to abandon the running analysis. */
let analysisRun = 0;
let analysisPool: BotWorkerPool | null = null;

/** The engine whose analysis call is running, if any. */
let analyzingEngine: BotId | null = null;
/** Bumped for every analysis call, so a stale call cannot clear the flags. */
let analysisCall = 0;

type Analysis = Awaited<ReturnType<BotPlayer['analyzePosition']>>;
/**
 * What each engine has found out about the current position so far. Kept
 * while the switch is off or another engine is shown, so that coming back
 * shows the result at once. The engine itself keeps its search too and
 * carries on from there.
 */
const analysisFound = new Map<BotId, { state: GameState; result: Analysis }>();

/** How far an analysis has got: simulations, or depth for Minimax. */
function analysisEffort(result: Analysis): number {
  return result.simulations ?? result.depth;
}

/** Minimax's score of a finished game: `±(WIN + disc margin)`. */
const MINIMAX_WIN = 10_000;
/** Minimax points at which the eval bar is about three quarters full. */
const MINIMAX_SCALE = 100;

const evalBar = requireElement('eval-bar');
const evalLabel = requireElement('eval-label');

/**
 * The eval bar beside the board, as on Lichess: Black's share fills it from
 * the top, White's from the bottom, and the number sits at the leading end.
 */
function showEval(
  blackShare: number,
  label: string,
  description: string,
): void {
  evalBar.hidden = false;
  evalBar.style.setProperty('--white-share', String(1 - blackShare));
  evalBar.dataset.lead = blackShare >= 0.5 ? 'black' : 'white';
  evalLabel.textContent = label;
  evalBar.setAttribute('aria-label', description);
}

function hideEval(): void {
  evalBar.hidden = true;
}

/** The final score as an eval: the winner fills the bar. */
function showFinalEval(state: GameState): void {
  const { black, white } = countDiscs(state.board);
  const margin = Math.abs(black - white);
  if (black === white) showEval(0.5, '0', 'Draw');
  else
    showEval(
      black > white ? 1 : 0,
      String(margin),
      `${black > white ? 'Black' : 'White'} won by ${margin} discs`,
    );
}

/** Minimax scores a finished game beyond any heuristic score. */
function solvedMargin(score: number): number | null {
  if (score >= MINIMAX_WIN) return score - MINIMAX_WIN;
  if (score <= -MINIMAX_WIN) return score + MINIMAX_WIN;
  return null;
}

/** An engine's verdict on the position, for the eval bar and the readout. */
interface Verdict {
  /** Black's share of the eval bar, 0 to 1. */
  black: number;
  /** The side ahead, if either is. */
  lead: Disc | null;
  value: string;
  unit: string;
  description: string;
}

function engineVerdict(engine: BotId, result: Analysis, mover: Disc): Verdict {
  const forBlack = (share: number) => (mover === 'black' ? share : 1 - share);
  const leadOf = (black: number): Disc | null =>
    black === 0.5 ? null : black > 0.5 ? 'black' : 'white';
  if (engine !== 'minimax') {
    // The network's value: −1 a loss, +1 a win for the side to move.
    const black = forBlack(((result.value ?? 0) + 1) / 2);
    const percent = Math.round(Math.max(black, 1 - black) * 100);
    const lead = percent === 50 ? null : leadOf(black);
    return {
      black,
      lead,
      value: `${percent}%`,
      unit: lead ? 'to win' : 'even',
      description: lead
        ? `${capitalize(lead)} wins ${percent}% of the time`
        : 'Even chances',
    };
  }
  const solved = solvedMargin(result.score);
  if (solved !== null) {
    const black = forBlack(solved > 0 ? 1 : 0);
    const margin = Math.abs(solved);
    const lead = leadOf(black);
    return {
      black,
      lead,
      value: String(margin),
      unit: 'discs, solved',
      description: `${capitalize(lead ?? 'black')} wins by ${margin} discs with best play`,
    };
  }
  // Heuristic points (mobility, corners, discs), squashed onto the bar.
  const black = forBlack(1 / (1 + Math.exp(-result.score / MINIMAX_SCALE)));
  const points = Math.abs(result.score);
  const lead = points === 0 ? null : leadOf(black);
  return {
    black,
    lead,
    value: String(points),
    unit: lead ? 'points ahead' : 'even',
    description: lead
      ? `${capitalize(lead)} ahead by ${points} points`
      : 'Even',
  };
}

const compactNumber = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
});

/** One figure of the analysis readout: a value over a quiet unit. */
function figureHtml(value: string, unit: string, disc?: Disc | null): string {
  const mark = disc
    ? `<i class="figure-disc" data-disc="${disc}" aria-label="${capitalize(disc)}"></i>`
    : '';
  return `<span class="figure"><span class="figure-value">${mark}${value}</span><span class="figure-unit">${unit}</span></span>`;
}

/** A Minimax score on its square: points, or the result once it is solved. */
function minimaxNote(score: number): string {
  const solved = solvedMargin(score);
  if (solved === null) return signed(score);
  return solved > 0 ? `W${solved}` : `L${-solved}`;
}

function showAnalysis(engine: BotId, state: GameState, result: Analysis): void {
  const name = BOTS.find((bot) => bot.id === engine)?.name ?? engine;
  const isSearchShare = engine !== 'minimax';
  board.setNotes(
    result.moves.map((move, index) => ({
      position: move.position,
      label: isSearchShare
        ? `${Math.round(move.score * 100)}%`
        : minimaxNote(move.score),
      best: index === 0,
    })),
  );

  const verdict = engineVerdict(engine, result, state.currentPlayer);
  showEval(verdict.black, verdict.value, verdict.description);

  // The readout: the verdict, the move the engine likes, and how far it has
  // looked. The numbers on the board are explained on hover.
  const best = result.bestMove;
  const final = result.complete ? ', final' : '';
  const effort = isSearchShare
    ? figureHtml(
        compactNumber.format(result.simulations ?? 0),
        `simulations${final}`,
      )
    : figureHtml(String(result.depth), `moves deep${final}`);
  analysisResultsNode.dataset.kind = 'readout';
  delete analysisResultsNode.dataset.pending;
  analysisResultsNode.title = isSearchShare
    ? `On the board: the share of ${name}'s search each move got.`
    : `On the board: ${name}'s score for each move.`;
  analysisResultsNode.innerHTML =
    figureHtml(verdict.value, verdict.unit, verdict.lead) +
    figureHtml(best ? squareName(best.row, best.col) : '–', 'best move') +
    effort;
}

/** The readout before the first result: the figures, still blank. */
function showAnalysisPending(engine: BotId): void {
  const isSearchShare = engine !== 'minimax';
  analysisResultsNode.dataset.kind = 'readout';
  analysisResultsNode.dataset.pending = 'true';
  analysisResultsNode.title = '';
  analysisResultsNode.innerHTML =
    figureHtml('–', isSearchShare ? 'to win' : 'points ahead') +
    figureHtml('–', 'best move') +
    figureHtml('–', isSearchShare ? 'simulations' : 'moves deep');
}

/** A short line in place of the readout. */
function showAnalysisNote(text: string): void {
  delete analysisResultsNode.dataset.pending;
  analysisResultsNode.dataset.kind = 'note';
  analysisResultsNode.title = '';
  analysisResultsNode.textContent = text;
}

/**
 * Drop whatever analysis is shown and, if the switch is on, analyze the
 * current position: continuously, until the engine has gone as far as it
 * goes or the position changes.
 */
function restartAnalysis(): void {
  const run = ++analysisRun;
  const state = controller.state;
  // Findings about another position are of no use any more.
  if (state !== analyzedState) analysisFound.clear();
  analyzedState = state;
  board.setNotes([]);
  showAnalysisNote('');

  // A Minimax call finishes a whole depth and may take many seconds; it
  // cannot be interrupted, so it is thrown away together with its worker.
  // A ZeroRev call ends within a slice, and the next one waits behind it.
  if (analyzingEngine === 'minimax') {
    analysisPool?.terminate();
    analysisPool = null;
    analyzing = false;
    analyzingEngine = null;
  }
  // Off: the worker stays, idle, with the engine's search of this position.
  if (!analysisOn) {
    hideEval();
    return;
  }
  if (state.status === 'finished') {
    showFinalEval(state);
    return;
  }
  if (state.legalMoves.length === 0) {
    // The pass prompt on the board says it all.
    showAnalysisNote('');
    return;
  }

  const engine = analysisEngine;
  const name = BOTS.find((bot) => bot.id === engine)?.name ?? engine;
  analysisPool ??= new BotWorkerPool(ANALYSIS_THREADS);
  const player = createWorkerBot(
    analysisPool,
    engine as 'minimax' | 'alphazero' | 'zerorev',
    name,
  );
  // Show at once what this engine has found here before.
  const found = analysisFound.get(engine);
  if (found?.state === state) {
    showAnalysis(engine, state, found.result);
    if (found.result.complete) return;
  } else {
    showAnalysisPending(engine);
  }

  void (async () => {
    while (run === analysisRun) {
      const call = ++analysisCall;
      analyzing = true;
      analyzingEngine = engine;
      let result: Analysis;
      try {
        result = await player.analyzePosition(state, ANALYSIS_SLICE_MS);
      } catch {
        if (run === analysisRun) {
          showAnalysisNote('Analysis is unavailable right now.');
        }
        return;
      } finally {
        if (call === analysisCall) {
          analyzing = false;
          analyzingEngine = null;
        }
      }
      if (run !== analysisRun) return;
      // An engine whose worker was restarted starts over; the deeper result
      // stays on the board until the new search catches up with it.
      const shown = analysisFound.get(engine);
      if (
        shown?.state !== state ||
        analysisEffort(result) >= analysisEffort(shown.result)
      ) {
        analysisFound.set(engine, { state, result });
        showAnalysis(engine, state, result);
      }
      if (result.complete) return;
    }
  })();
}

analysisSwitch.addEventListener('click', () => {
  analysisOn = !analysisOn;
  analysisSwitch.setAttribute('aria-checked', String(analysisOn));
  restartAnalysis();
});

for (const button of analysisButtons) {
  button.addEventListener('click', () => {
    analysisEngine = button.dataset.engine as BotId;
    for (const other of analysisButtons) {
      other.setAttribute('aria-pressed', String(other === button));
    }
    if (analysisOn) restartAnalysis();
  });
}

// Show the starting position; a person can move right away.
sync(controller.state);

// The running clock of the side on the move.
(function tick(): void {
  if (controller.state.status !== 'finished') syncClocks();
  syncCpuMeter();
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
        // Next frame, so the transition starts from the hidden state.
        requestAnimationFrame(() => detail.removeAttribute('data-pending'));
      }
      // The whole story on hover.
      for (const option of document.querySelectorAll<HTMLElement>(
        `.option[data-bot="${bot}"]`,
      )) {
        option.title = `${shortDate(meta.trainedOn)}${loss}. Run ${meta.run}${winRate}. Built ${meta.buildDate}.`;
      }
    }
  })
  .catch((err) => {
    console.error('Bots failed to load', err);
    for (const detail of document.querySelectorAll<HTMLElement>(
      '.option-detail[data-pending]',
    )) {
      detail.textContent = 'Neural network';
      detail.removeAttribute('data-pending');
    }
    statusNode.textContent = 'The bots could not be loaded. Reload the page.';
  });
