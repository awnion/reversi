import { describe, expect, test } from 'bun:test';
import { createInitialState } from '@reversi/core';
import { GameController } from '../src/gameController';

describe('GameController', () => {
  test('initializes and manages state changes correctly for human players', async () => {
    const initial = createInitialState();
    let lastState = initial;
    const controller = new GameController(
      initial,
      { black: { type: 'human' }, white: { type: 'human' } },
      (state) => {
        lastState = state;
      },
    );

    await controller.start();
    expect(lastState.currentPlayer).toBe('black');

    const move = lastState.legalMoves[0];
    await controller.makeMove({ row: move.row, col: move.col });

    expect(lastState.currentPlayer).toBe('white');
    expect(controller.state.currentPlayer).toBe('white');
  });

  test('does not allow human to make move when it is bot turn', async () => {
    const initial = createInitialState();
    const controller = new GameController(
      initial,
      {
        black: {
          type: 'bot',
          player: {
            name: 'b',
            chooseMove: async () => ({ row: 0, col: 0 }),
            analyzePosition: async () => ({
              bestMove: null,
              score: 0,
              depth: 0,
              moves: [],
            }),
          },
          thinkMs: 100,
        },
        white: { type: 'human' },
      },
      () => {},
    );
    await controller.makeMove({ row: 0, col: 0 });
    expect(controller.state.currentPlayer).toBe('black');
  });
});

describe('GameController restarts', () => {
  const never = new Promise<never>(() => {});

  test('a bot move that arrives after a new game is dropped', async () => {
    const initial = createInitialState();
    let lastState = initial;
    let release!: () => void;
    const thinking = new Promise<void>((resolve) => {
      release = resolve;
    });
    const slowBot = {
      name: 'slow',
      chooseMove: async (state: typeof initial) => {
        await thinking;
        return state.legalMoves[0];
      },
      analyzePosition: async () => never,
    };
    const controller = new GameController(
      initial,
      {
        black: { type: 'bot', player: slowBot, thinkMs: 1 },
        white: { type: 'human' },
      },
      (state) => {
        lastState = state;
      },
    );
    await controller.start();

    // Restart with two humans while the bot is still thinking.
    controller.newGame(
      { black: { type: 'human' }, white: { type: 'human' } },
      createInitialState(),
    );
    release();
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(lastState.currentPlayer).toBe('black');
    expect(lastState.board.filter((cell) => cell !== null)).toHaveLength(4);
  });
});

describe('GameController pause and play', () => {
  const never = new Promise<never>(() => {});
  const wait = (ms: number) =>
    new Promise((resolve) => setTimeout(resolve, ms));
  const firstMoveBot = {
    name: 'first-move',
    chooseMove: async (state: ReturnType<typeof createInitialState>) =>
      state.legalMoves[0],
    analyzePosition: async () => never,
  };
  const discs = (state: ReturnType<typeof createInitialState>) =>
    state.board.filter((cell) => cell !== null).length;

  test('a paused game lets a bot side be played by hand, and bots stay still', async () => {
    const initial = createInitialState();
    let lastState = initial;
    const controller = new GameController(
      initial,
      {
        black: { type: 'bot', player: firstMoveBot, thinkMs: 1 },
        white: { type: 'bot', player: firstMoveBot, thinkMs: 1 },
      },
      (state) => {
        lastState = state;
      },
    );
    controller.pause();
    await controller.start();
    await wait(10);
    expect(discs(lastState)).toBe(4);

    await controller.makeMove(initial.legalMoves[0]);
    await wait(10);
    // The hand move stands and white's bot does not answer while paused.
    expect(discs(lastState)).toBe(5);
    expect(lastState.currentPlayer).toBe('white');
  });

  test('play lets the bots move; a running bot side refuses hand moves', async () => {
    const initial = createInitialState();
    let lastState = initial;
    const controller = new GameController(
      initial,
      {
        black: { type: 'human' },
        white: { type: 'bot', player: firstMoveBot, thinkMs: 1 },
      },
      (state) => {
        lastState = state;
      },
    );
    controller.pause();
    await controller.makeMove(initial.legalMoves[0]);
    expect(lastState.currentPlayer).toBe('white');

    controller.play();
    await wait(10);
    expect(discs(lastState)).toBe(6);
    expect(lastState.currentPlayer).toBe('black');
  });

  test('pausing discards the move a bot is still thinking about', async () => {
    const initial = createInitialState();
    let lastState = initial;
    let release!: () => void;
    const thinking = new Promise<void>((resolve) => {
      release = resolve;
    });
    const slowBot = {
      name: 'slow',
      chooseMove: async (state: typeof initial) => {
        await thinking;
        return state.legalMoves[0];
      },
      analyzePosition: async () => never,
    };
    const controller = new GameController(
      initial,
      {
        black: { type: 'bot', player: slowBot, thinkMs: 1 },
        white: { type: 'human' },
      },
      (state) => {
        lastState = state;
      },
    );
    await controller.start();
    controller.pause();
    release();
    await wait(10);
    expect(discs(lastState)).toBe(4);
    expect(controller.paused).toBe(true);
  });
});

describe('GameController clocks', () => {
  const never = new Promise<never>(() => {});
  const wait = (ms: number) =>
    new Promise((resolve) => setTimeout(resolve, ms));

  test('records how long each side took over its last move', async () => {
    const initial = createInitialState();
    const slowBot = {
      name: 'slow',
      chooseMove: async (state: typeof initial) => {
        await wait(40);
        return state.legalMoves[0];
      },
      analyzePosition: async () => never,
    };
    const controller = new GameController(
      initial,
      {
        black: { type: 'human' },
        white: { type: 'bot', player: slowBot, thinkMs: 40 },
      },
      () => {},
    );
    expect(controller.lastMoveMs).toEqual({ black: null, white: null });

    // The wait before the opening move is not on the clock.
    expect(controller.started).toBe(false);
    await wait(20);
    await controller.makeMove(initial.legalMoves[0]);
    expect(controller.started).toBe(true);
    expect(controller.lastMoveMs.black).toBeNull();
    expect(controller.thinkingSince).not.toBeNull();

    await wait(70);
    expect(controller.lastMoveMs.white).toBeGreaterThanOrEqual(35);
    expect(controller.lastMoveMs.white).toBeLessThan(200);
    expect(controller.thinkingSince).toBeNull();

    // From now on the person's moves are timed.
    await wait(20);
    await controller.makeMove(controller.state.legalMoves[0]);
    expect(controller.lastMoveMs.black).toBeGreaterThanOrEqual(15);

    controller.newGame(
      { black: { type: 'human' }, white: { type: 'human' } },
      createInitialState(),
    );
    expect(controller.lastMoveMs).toEqual({ black: null, white: null });
    expect(controller.started).toBe(false);
  });
});
