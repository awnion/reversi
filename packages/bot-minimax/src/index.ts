import type { BotPlayer, GameState, MoveEval, Position, PositionAnalysis } from '@reversi/core';
import init, { MinimaxBot as WasmBot } from '../wasm/reversi_minimax.js';

function toBitboards(state: GameState) {
  let blackLow = 0, blackHigh = 0;
  let whiteLow = 0, whiteHigh = 0;
  for (let idx = 0; idx < 64; idx++) {
    const cell = state.board[idx];
    if (cell === 'black') {
      if (idx < 32) blackLow |= (1 << idx);
      else blackHigh |= (1 << (idx - 32));
    } else if (cell === 'white') {
      if (idx < 32) whiteLow |= (1 << idx);
      else whiteHigh |= (1 << (idx - 32));
    }
  }
  return {
    blackLow: blackLow >>> 0,
    blackHigh: blackHigh >>> 0,
    whiteLow: whiteLow >>> 0,
    whiteHigh: whiteHigh >>> 0,
  };
}

export async function createMinimaxBot(name: string = 'Minimax'): Promise<BotPlayer> {
  await init();
  const wasm = new WasmBot();

  return {
    name,

    async chooseMove(state: GameState, timeLimitMs: number): Promise<Position> {
      const { blackLow, blackHigh, whiteLow, whiteHigh } = toBitboards(state);
      const moveIdx = wasm.choose_move(
        blackLow, blackHigh,
        whiteLow, whiteHigh,
        state.currentPlayer === 'black',
        timeLimitMs,
      );
      if (moveIdx === -1) {
        throw new Error('Minimax bot could not find a valid move');
      }
      return { row: Math.floor(moveIdx / 8), col: moveIdx % 8 };
    },

    async analyzePosition(state: GameState, timeLimitMs: number): Promise<PositionAnalysis> {
      const { blackLow, blackHigh, whiteLow, whiteHigh } = toBitboards(state);
      const raw = wasm.analyze_position(
        blackLow, blackHigh,
        whiteLow, whiteHigh,
        state.currentPlayer === 'black',
        timeLimitMs,
      );

      // No legal moves
      if (raw.length === 0) {
        return { bestMove: null, score: 0, depth: 0, moves: [] };
      }

      // Layout: [depth, idx0, score0, idx1, score1, ...]
      const depth = raw[0] as number;
      const moves: MoveEval[] = [];
      for (let i = 1; i < raw.length; i += 2) {
        const idx = raw[i] as number;
        const score = raw[i + 1] as number;
        moves.push({ position: { row: Math.floor(idx / 8), col: idx % 8 }, score, depth });
      }
      moves.sort((a, b) => b.score - a.score);

      const best = moves[0];
      return {
        bestMove: best?.position ?? null,
        score: best?.score ?? 0,
        depth,
        moves,
      };
    },

    destroy() {
      wasm.free();
    },
  };
}
