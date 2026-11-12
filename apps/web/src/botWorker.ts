import { createAlphaZeroBot } from '@reversi/bot-alphazero';
import { createMinimaxBot } from '@reversi/bot-minimax';
import { createZeroRevBot } from '@reversi/bot-zerorev';
import type {
  BotPlayer,
  GameState,
  NetworkBotMeta,
  PositionAnalysis,
} from '@reversi/core';

let minimax: BotPlayer;
let alphazero: BotPlayer & { meta: NetworkBotMeta };
let zerorev: BotPlayer & { meta: NetworkBotMeta };

void (async () => {
  try {
    [minimax, alphazero, zerorev] = await Promise.all([
      createMinimaxBot('Minimax'),
      createAlphaZeroBot(),
      createZeroRevBot(),
    ]);
  } catch (err) {
    // Without this the page would wait for the bots forever.
    self.postMessage({ type: 'failed', error: String(err) });
    return;
  }
  self.postMessage({
    type: 'ready',
    alphazero: alphazero.meta,
    zerorev: zerorev.meta,
  });
})();

export type WorkerRequest = {
  id: number;
  bot: 'minimax' | 'alphazero' | 'zerorev';
  action: 'chooseMove' | 'analyzePosition';
  state: GameState;
  thinkMs: number;
};

self.addEventListener('message', async (e: MessageEvent<WorkerRequest>) => {
  const { id, bot, action, state, thinkMs } = e.data;
  const player =
    bot === 'minimax' ? minimax : bot === 'alphazero' ? alphazero : zerorev;
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
