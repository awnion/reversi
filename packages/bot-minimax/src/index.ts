import type { BotPlayer, GameState, Position, PositionAnalysis } from '@reversi/core';
import init, { MinimaxBot as WasmBot } from '../wasm/reversi_minimax.js';

export async function createMinimaxBot(name: string = 'Minimax'): Promise<BotPlayer> {
  await init();
  const wasm = new WasmBot();

  return {
    name,
    async chooseMove(state: GameState, timeLimitMs: number): Promise<Position> {
      let blackLow = 0, blackHigh = 0;
      let whiteLow = 0, whiteHigh = 0;

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const idx = r * 8 + c;
          const cell = state.board[idx];
          if (cell === 'black') {
            if (idx < 32) blackLow |= (1 << idx);
            else blackHigh |= (1 << (idx - 32));
          } else if (cell === 'white') {
            if (idx < 32) whiteLow |= (1 << idx);
            else whiteHigh |= (1 << (idx - 32));
          }
        }
      }

      // We actually want unsigned shift because JS bitwise ops convert to signed 32-bit.
      // So use `>>> 0` if needed, but the WASM takes unsigned u32 anyway.
      const moveIdx = wasm.choose_move(
        blackLow >>> 0, blackHigh >>> 0,
        whiteLow >>> 0, whiteHigh >>> 0,
        state.currentPlayer === 'black',
        timeLimitMs
      );

      if (moveIdx === -1) {
        throw new Error('Minimax bot could not find a valid move');
      }

      const row = Math.floor(moveIdx / 8);
      const col = moveIdx % 8;

      return { row, col };
    },

    async analyzePosition(state: GameState, timeLimitMs: number): Promise<PositionAnalysis> {
      // Stub to satisfy interface
      return {
         bestMove: null,
         score: 0,
         depth: 0,
         moves: []
      };
    },

    destroy() {
      wasm.free();
    }
  };
}
