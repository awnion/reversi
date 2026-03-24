import { createAlphaZeroBot } from '@reversi/bot-alphazero';
import { createMinimaxBot } from '@reversi/bot-minimax';
import type { BotPlayer, GameState, PositionAnalysis } from '@reversi/core';

let minimax: BotPlayer;
let alphazero: BotPlayer;

(async () => {
  [minimax, alphazero] = await Promise.all([
    createMinimaxBot('Minimax'),
    createAlphaZeroBot(),
  ]);
  self.postMessage({ type: 'ready', alphazeroName: alphazero.name });
})();

export type WorkerRequest = {
  id: number;
  bot: 'minimax' | 'alphazero';
  action: 'chooseMove' | 'analyzePosition';
  state: GameState;
  thinkMs: number;
};

self.addEventListener('message', async (e: MessageEvent<WorkerRequest>) => {
  const { id, bot, action, state, thinkMs } = e.data;
  const player = bot === 'minimax' ? minimax : alphazero;
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
