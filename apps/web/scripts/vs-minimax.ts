/**
 * Play the ZeroRev bot against Minimax, once with each colour, using the same
 * WASM engines as the browser:
 *
 *   bun apps/web/scripts/vs-minimax.ts [ms per move, default 300]
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
    state = applyMove(state, await bot.chooseMove(state, thinkMs));
  }
  return countDiscs(state.board);
}

function verdict(mine: number, theirs: number): string {
  if (mine === theirs) return 'draw';
  return mine > theirs ? 'ZeroRev wins' : 'Minimax wins';
}

const { meta } = zerorev;
console.log(
  `ZeroRev: run ${meta.run}, ${meta.trainedOn}, loss ${meta.loss ?? 'n/a'} — ${thinkMs} ms per move`,
);
const asBlack = await play(zerorev, minimax);
console.log(
  `  as black: ${asBlack.black}:${asBlack.white}  ${verdict(asBlack.black, asBlack.white)}`,
);
const asWhite = await play(minimax, zerorev);
console.log(
  `  as white: ${asWhite.white}:${asWhite.black}  ${verdict(asWhite.white, asWhite.black)}`,
);
