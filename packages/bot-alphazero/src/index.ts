import type {
  BotPlayer,
  GameState,
  MoveEval,
  NetworkBotMeta,
  Position,
  PositionAnalysis,
} from '@reversi/core';

/** Raw shape of the JSON returned by the WASM `meta()`. */
interface WasmMeta {
  build_date: string;
  champion: {
    run: string;
    date: string;
    loss: number | null;
    win_rate: number | null;
    backend?: string;
  };
}

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

export interface AlphaZeroOptions {
  /** Threads to search on, where the page allows it. Default: one. */
  threads?: number;
}

/**
 * Load the WASM engine. A cross-origin isolated page may share memory between
 * workers, so there the search runs on a pool of threads working on one tree;
 * everywhere else the single-threaded build is used.
 */
async function loadEngine(threads: number) {
  if (threads > 1 && globalThis.crossOriginIsolated) {
    const engine = await import('../wasm-threads/reversi_alphazero.js');
    await engine.default();
    await engine.initThreadPool(threads);
    return engine;
  }
  const engine = await import('../wasm/reversi_alphazero.js');
  await engine.default();
  return engine;
}

export async function createAlphaZeroBot(
  name?: string,
  options: AlphaZeroOptions = {},
): Promise<BotPlayer & { meta: NetworkBotMeta }> {
  const engine = await loadEngine(Math.max(1, options.threads ?? 1));
  const wasm = new engine.AlphaZeroBot();
  const m: WasmMeta = JSON.parse(engine.meta());
  const displayName = name ?? `AlphaZero (${m.build_date})`;

  return {
    name: displayName,
    meta: {
      buildDate: m.build_date,
      run: m.champion.run,
      trainedOn: m.champion.date,
      loss: m.champion.loss,
      winRate: m.champion.win_rate,
      threads: engine.search_threads(),
    },

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
        simulations: wasm.last_simulations(),
      };
    },

    destroy() {
      wasm.free();
    },
  };
}
