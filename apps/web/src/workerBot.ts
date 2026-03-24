import type {
  BotPlayer,
  GameState,
  Position,
  PositionAnalysis,
} from '@reversi/core';

type PendingEntry = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export class BotWorkerPool {
  private worker: Worker;
  private pending = new Map<number, PendingEntry>();
  private idCounter = 0;
  readonly ready: Promise<void>;

  constructor() {
    this.worker = new Worker(new URL('./botWorker.ts', import.meta.url), {
      type: 'module',
    });
    let resolveReady!: () => void;
    this.ready = new Promise((res) => {
      resolveReady = res;
    });
    this.worker.addEventListener('message', (e: MessageEvent) => {
      if (e.data.type === 'ready') {
        resolveReady();
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
    bot: 'minimax' | 'alphazero',
    action: 'chooseMove' | 'analyzePosition',
    state: GameState,
    thinkMs: number,
  ): Promise<unknown> {
    return this.ready.then(
      () =>
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
  botType: 'minimax' | 'alphazero',
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
