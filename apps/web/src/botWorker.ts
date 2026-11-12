import { createAlphaZeroBot } from '@reversi/bot-alphazero';
import { createMinimaxBot } from '@reversi/bot-minimax';
import { createZeroRevBot } from '@reversi/bot-zerorev';
import type {
  BotPlayer,
  GameState,
  NetworkBotMeta,
  PositionAnalysis,
} from '@reversi/core';
import type { WorkerBotId } from './workerBot';

let minimax: BotPlayer;
let alphazero: BotPlayer & { meta: NetworkBotMeta };
let gladiator0: BotPlayer & { meta: NetworkBotMeta };
let zerorev: BotPlayer & { meta: NetworkBotMeta };

// The page says how many threads this worker's searches may use.
const threads = Number(self.name) || 1;

void (async () => {
  try {
    [minimax, alphazero, gladiator0, zerorev] = await Promise.all([
      createMinimaxBot('Minimax', { threads }),
      createAlphaZeroBot(undefined, { threads }),
      // Same engine as ZeroRev, other weights; the two share its threads.
      createZeroRevBot(undefined, { threads, network: 'gladiator0' }),
      createZeroRevBot(undefined, { threads }),
    ]);
  } catch (err) {
    // Without this the page would wait for the bots forever.
    self.postMessage({ type: 'failed', error: String(err) });
    return;
  }
  self.postMessage({
    type: 'ready',
    alphazero: alphazero.meta,
    gladiator0: gladiator0.meta,
    zerorev: zerorev.meta,
  });
})();

export type WorkerRequest = {
  id: number;
  bot: WorkerBotId;
  action: 'chooseMove' | 'analyzePosition';
  state: GameState;
  thinkMs: number;
};

self.addEventListener('message', async (e: MessageEvent<WorkerRequest>) => {
  const { id, bot, action, state, thinkMs } = e.data;
  const player = { minimax, alphazero, gladiator0, zerorev }[bot];
  try {
    if (action === 'chooseMove') {
      const position = await player.chooseMove(state, thinkMs);
      self.postMessage({ id, position });
    } else {
      const analysis: PositionAnalysis = await player.analyzePosition(
        state,
        thinkMs,
      );
      self.postMessage({ id, analysis });
    }
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
});
