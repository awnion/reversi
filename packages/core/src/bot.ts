import type { GameState, Position } from './reversi';

export interface MoveEval {
  position: Position;
  score: number;
  depth: number;
}

export interface PositionAnalysis {
  bestMove: Position | null;
  score: number;
  depth: number;
  moves: MoveEval[];
}

export interface BotPlayer {
  readonly name: string;
  chooseMove(state: GameState, timeLimitMs: number): Promise<Position>;
  analyzePosition(
    state: GameState,
    timeLimitMs: number,
  ): Promise<PositionAnalysis>;
  destroy?(): void;
}
