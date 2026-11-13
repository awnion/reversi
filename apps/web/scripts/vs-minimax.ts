/**
 * Play the ZeroRev bot against Minimax with alternating colours, using the
 * same WASM engines as the browser:
 *
 *   bun apps/web/scripts/vs-minimax.ts [ms per move = 300] [games = 2] [jitter = 0]
 *
 * Jitter varies each move's time by up to ±jitter (0.2 = ±20 %), so that
 * repeated games between the two deterministic engines differ.
 */
import { createMinimaxBot } from '@reversi/bot-minimax';
import { createZeroRevBot } from '@reversi/bot-zerorev';
import {
  applyMove,
  type BotPlayer,
  countDiscs,
  createInitialState,
  passTurn,
} from '@reversi/core';

const thinkMs = Number(process.argv[2] ?? 300);
const games = Number(process.argv[3] ?? 2);
const jitter = Number(process.argv[4] ?? 0);
const minimax = await createMinimaxBot('Minimax');
const zerorev = await createZeroRevBot();

async function play(black: BotPlayer, white: BotPlayer) {
  let state = createInitialState();
  while (state.status !== 'finished') {
    if (state.legalMoves.length === 0) {
      state = passTurn(state);
      continue;
    }
    const bot = state.currentPlayer === 'black' ? black : white;
    const ms = Math.round(thinkMs * (1 + (Math.random() * 2 - 1) * jitter));
    state = applyMove(state, await bot.chooseMove(state, ms));
  }
  return countDiscs(state.board);
}

function verdict(mine: number, theirs: number): string {
  if (mine === theirs) return 'draw';
  return mine > theirs ? 'ZeroRev wins' : 'Minimax wins';
}

const { meta } = zerorev;
console.log(
  `ZeroRev: run ${meta.run}, ${meta.trainedOn}, loss ${meta.loss ?? 'n/a'} — ${thinkMs} ms per move` +
    (jitter ? ` ±${Math.round(jitter * 100)} %` : ''),
);
const score = { zerorev: 0, minimax: 0, draws: 0 };
for (let game = 0; game < games; game++) {
  const zerorevBlack = game % 2 === 0;
  const discs = zerorevBlack
    ? await play(zerorev, minimax)
    : await play(minimax, zerorev);
  const [mine, theirs] = zerorevBlack
    ? [discs.black, discs.white]
    : [discs.white, discs.black];
  if (mine > theirs) score.zerorev++;
  else if (mine < theirs) score.minimax++;
  else score.draws++;
  console.log(
    `  as ${zerorevBlack ? 'black' : 'white'}: ${mine}:${theirs}  ${verdict(mine, theirs)}`,
  );
}
if (games > 2) {
  console.log(
    `ZeroRev ${score.zerorev} – Minimax ${score.minimax}` +
      (score.draws ? `, ${score.draws} drawn` : ''),
  );
}
