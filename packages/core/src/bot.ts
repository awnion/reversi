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
  /** Simulations behind the analysis, for bots that search by simulation. */
  simulations?: number;
  /**
   * The expected result for the side to move, from −1 (a loss) to +1 (a
   * win), for bots that estimate one (the network bots).
   */
  value?: number;
  /**
   * For bots whose analysis carries on from call to call: it has gone as far
   * as it goes, and asking again returns the same result.
   */
  complete?: boolean;
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

/** What a neural-network bot knows about the weights it was built with. */
export interface NetworkBotMeta {
  /** Day the WASM bot was built, `YYYY-MM-DD` (`dev` for unstamped builds). */
  buildDate: string;
  /** Training run the champion came from. */
  run: string;
  /** Day the champion won its tournament, `YYYY-MM-DD`. */
  trainedOn: string;
  /** Training loss when it was promoted. */
  loss: number | null;
  /** Share of tournament points it scored, 0..1. */
  winRate: number | null;
  /** CPU threads its search runs on (1 unless the page is cross-origin isolated). */
  threads: number;
}
