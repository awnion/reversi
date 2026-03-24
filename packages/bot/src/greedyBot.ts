import type { BotPlayer, GameState, Position } from '@reversi/core';

export const greedyBot: BotPlayer = {
  name: 'Greedy',
  async chooseMove(state: GameState, _timeLimitMs: number): Promise<Position> {
    let bestMove: Position | null = null;
    let maxFlips = -1;

    for (const move of state.legalMoves) {
      if (move.flips.length > maxFlips) {
        maxFlips = move.flips.length;
        bestMove = { row: move.row, col: move.col };
      }
    }

    if (!bestMove) {
      throw new Error('No legal moves available');
    }

    // Optional delay for better presentation
    await new Promise((res) => setTimeout(res, 100));

    return bestMove;
  },
  async analyzePosition(state: GameState, _timeLimitMs: number) {
    const moves = state.legalMoves.map((move) => ({
      position: { row: move.row, col: move.col },
      score: move.flips.length,
      depth: 1,
    }));

    moves.sort((a, b) => b.score - a.score);

    return {
      bestMove: moves[0]?.position ?? null,
      score: moves[0]?.score ?? 0,
      depth: 1,
      moves,
    };
  },
};
