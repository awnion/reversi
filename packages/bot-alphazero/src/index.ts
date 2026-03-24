import type {
  BotPlayer,
  GameState,
  MoveEval,
  Position,
  PositionAnalysis,
} from '@reversi/core';
import init, { AlphaZeroBot as WasmBot } from '../wasm/reversi_alphazero.js';

function toBitboards(state: GameState) {
  let blackLow = 0,
    blackHigh = 0;
  let whiteLow = 0,
    whiteHigh = 0;
  for (let idx = 0; idx < 64; idx++) {
    const cell = state.board[idx];
    if (cell === 'black') {
      if (idx < 32) blackLow |= 1 << idx;
      else blackHigh |= 1 << (idx - 32);
    } else if (cell === 'white') {
      if (idx < 32) whiteLow |= 1 << idx;
      else whiteHigh |= 1 << (idx - 32);
    }
  }
  return {
    blackLow: blackLow >>> 0,
    blackHigh: blackHigh >>> 0,
    whiteLow: whiteLow >>> 0,
    whiteHigh: whiteHigh >>> 0,
  };
}

export async function createAlphaZeroBot(
  name: string = 'AlphaZero',
): Promise<BotPlayer> {
  await init();
  const wasm = new WasmBot();

  return {
    name,

    async chooseMove(state: GameState, timeLimitMs: number): Promise<Position> {
      const { blackLow, blackHigh, whiteLow, whiteHigh } = toBitboards(state);
      const moveIdx = wasm.choose_move(
        blackLow,
        blackHigh,
        whiteLow,
        whiteHigh,
        state.currentPlayer === 'black',
        timeLimitMs,
      );
      if (moveIdx === -1) {
        throw new Error('AlphaZero bot could not find a valid move');
      }
      return { row: Math.floor(moveIdx / 8), col: moveIdx % 8 };
    },

    async analyzePosition(
      state: GameState,
      timeLimitMs: number,
    ): Promise<PositionAnalysis> {
      const { blackLow, blackHigh, whiteLow, whiteHigh } = toBitboards(state);
      const policyArr = wasm.analyze_position(
        blackLow,
        blackHigh,
        whiteLow,
        whiteHigh,
        state.currentPlayer === 'black',
        timeLimitMs,
      );

      const moves: MoveEval[] = [];
      for (let i = 0; i < 64; i++) {
        const score = policyArr[i];
        if (score !== undefined && score > 0) {
          moves.push({
            position: { row: Math.floor(i / 8), col: i % 8 },
            score,
            depth: 0,
          });
        }
      }
      moves.sort((a, b) => b.score - a.score);

      const best = moves[0];
      return {
        bestMove: best?.position ?? null,
        score: best?.score ?? 0,
        depth: 0,
        moves,
      };
    },

    destroy() {
      wasm.free();
    },
  };
}
