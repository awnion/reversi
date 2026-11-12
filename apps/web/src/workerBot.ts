import type {
  BotPlayer,
  GameState,
  NetworkBotMeta,
  Position,
  PositionAnalysis,
} from '@reversi/core';
import botWorkerUrl from './botWorker.ts?worker&url';

/** Sent by the worker once the WASM bots are loaded. */
export type BotsReady = {
  alphazero: NetworkBotMeta;
  gladiator0: NetworkBotMeta;
  zerorev: NetworkBotMeta;
};

/** The bots the worker runs. */
export type WorkerBotId = 'minimax' | 'alphazero' | 'gladiator0' | 'zerorev';

type PendingEntry = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

let poolCount = 0;

export class BotWorkerPool {
  private worker: Worker;
  private pending = new Map<number, PendingEntry>();
  private idCounter = 0;
  readonly ready: Promise<BotsReady>;

  /** `threads`: CPU threads the worker's searches may use, where possible. */
  constructor(threads = 1) {
    // Each pool gets its own script URL: under cross-origin isolation Safari
    // refuses a second worker whose script it serves from its cache.
    const url = new URL(botWorkerUrl, import.meta.url);
    url.searchParams.set('pool', String(++poolCount));
    this.worker = new Worker(url, { type: 'module', name: String(threads) });
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
          gladiator0: e.data.gladiator0 as NetworkBotMeta,
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
    bot: WorkerBotId,
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
  botType: WorkerBotId,
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
