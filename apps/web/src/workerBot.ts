import type {
  BotPlayer,
  GameState,
  NetworkBotMeta,
  Position,
  PositionAnalysis,
} from '@reversi/core';

/** Sent by the worker once the WASM bots are loaded. */
export type BotsReady = { alphazero: NetworkBotMeta; zerorev: NetworkBotMeta };

type PendingEntry = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export class BotWorkerPool {
  private worker: Worker;
  private pending = new Map<number, PendingEntry>();
  private idCounter = 0;
  readonly ready: Promise<BotsReady>;

  constructor() {
    this.worker = new Worker(new URL('./botWorker.ts', import.meta.url), {
      type: 'module',
    });
    let resolveReady!: (value: BotsReady) => void;
    let rejectReady!: (reason: Error) => void;
    this.ready = new Promise((res, rej) => {
      resolveReady = res;
      rejectReady = rej;
    });
    this.worker.addEventListener('message', (e: MessageEvent) => {
      if (e.data.type === 'ready') {
        resolveReady({
          alphazero: e.data.alphazero as NetworkBotMeta,
          zerorev: e.data.zerorev as NetworkBotMeta,
        });
        return;
      }
      if (e.data.type === 'failed') {
        rejectReady(new Error(e.data.error as string));
        return;
      }
      const entry = this.pending.get(e.data.id as number);
      if (!entry) return;
      this.pending.delete(e.data.id as number);
      if (e.data.error !== undefined) {
        entry.reject(new Error(e.data.error as string));
      } else {
        entry.resolve(e.data.position ?? e.data.analysis);
      }
    });
  }

  request(
    bot: 'minimax' | 'alphazero' | 'zerorev',
    action: 'chooseMove' | 'analyzePosition',
    state: GameState,
    thinkMs: number,
  ): Promise<unknown> {
    return this.ready.then(
      (_) =>
        new Promise((resolve, reject) => {
          const id = ++this.idCounter;
          this.pending.set(id, { resolve, reject });
          this.worker.postMessage({ id, bot, action, state, thinkMs });
        }),
    );
  }

  terminate() {
    this.worker.terminate();
  }
}

export function createWorkerBot(
  pool: BotWorkerPool,
  botType: 'minimax' | 'alphazero' | 'zerorev',
  name: string,
): BotPlayer {
  return {
    name,
    chooseMove(state: GameState, timeLimitMs: number): Promise<Position> {
      return pool.request(
        botType,
        'chooseMove',
        state,
        timeLimitMs,
      ) as Promise<Position>;
    },
    analyzePosition(
      state: GameState,
      timeLimitMs: number,
    ): Promise<PositionAnalysis> {
      return pool.request(
        botType,
        'analyzePosition',
        state,
        timeLimitMs,
      ) as Promise<PositionAnalysis>;
    },
  };
}
