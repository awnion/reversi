import {
  applyMove,
  type BotPlayer,
  type Disc,
  type GameState,
  type Position,
  type PositionAnalysis,
  passTurn,
} from '@reversi/core';

export type PlayerConfig =
  | { type: 'human' }
  | { type: 'bot'; player: BotPlayer; thinkMs: number };

export class GameController {
  private _state: GameState;
  private botTimeout?: ReturnType<typeof setTimeout>;
  /**
   * Bumped whenever the game, the players or the paused state change; a bot
   * move that was requested before the change is dropped.
   */
  private epoch = 0;
  /** While paused, bots do not move and any side can be played by hand. */
  private _paused = false;

  // ── clocks (milliseconds, `performance.now()` time base) ──────────────────
  /** When the current position appeared. */
  private _turnStartedAt = performance.now();
  /** When the bot on the move started thinking; `null` when none is. */
  private _thinkingSince: number | null = null;
  /** Moves and passes made in this game; the clocks start with the first. */
  private _moves = 0;
  /** How long each side took over its most recent move. */
  private _lastMoveMs: Record<Disc, number | null> = {
    black: null,
    white: null,
  };

  private players: { black: PlayerConfig; white: PlayerConfig };
  private onStateChange: (
    state: GameState,
    analysis?: PositionAnalysis,
  ) => void;

  constructor(
    initialState: GameState,
    players: { black: PlayerConfig; white: PlayerConfig },
    onStateChange: (state: GameState, analysis?: PositionAnalysis) => void,
  ) {
    this._state = initialState;
    this.players = players;
    this.onStateChange = onStateChange;
  }

  get state() {
    return this._state;
  }

  get paused() {
    return this._paused;
  }

  get turnStartedAt() {
    return this._turnStartedAt;
  }

  get thinkingSince() {
    return this._thinkingSince;
  }

  /** True once the first move of the game has been made. */
  get started() {
    return this._moves > 0;
  }

  get lastMoveMs(): Readonly<Record<Disc, number | null>> {
    return this._lastMoveMs;
  }

  /** Install a new position, recording how long the mover took over it. */
  private advance(next: GameState, tookMs: number | null): void {
    if (tookMs !== null) this._lastMoveMs[this._state.currentPlayer] = tookMs;
    this._moves += 1;
    this._state = next;
    this._turnStartedAt = performance.now();
    this._thinkingSince = null;
    this.onStateChange(this._state);
  }

  /** Stop the bots. A move a bot is still thinking about is discarded. */
  pause(): void {
    if (this.botTimeout) clearTimeout(this.botTimeout);
    this.epoch += 1;
    this._paused = true;
    this._thinkingSince = null;
  }

  /** Let the bots move again, starting with the side to move. */
  play(): void {
    if (!this._paused) return;
    this._paused = false;
    void this.triggerBotMove();
  }

  /** Change who plays each side without touching the position. */
  setPlayers(players: { black: PlayerConfig; white: PlayerConfig }): void {
    if (this.botTimeout) clearTimeout(this.botTimeout);
    this.epoch += 1;
    this._thinkingSince = null;
    this.players = players;
    void this.triggerBotMove();
  }

  /** A move by hand is allowed for a human side, or for any side while paused. */
  private manualMoveAllowed(): boolean {
    if (this._paused) return true;
    const mover =
      this._state.currentPlayer === 'black'
        ? this.players.black
        : this.players.white;
    return mover.type === 'human';
  }

  async makeMove(pos: Position): Promise<void> {
    if (!this.manualMoveAllowed()) return;

    // Nobody is on the clock before the game has begun: the wait before a
    // person's opening move is not thinking time.
    const tookMs = this.started
      ? performance.now() - this._turnStartedAt
      : null;
    this.advance(applyMove(this._state, pos), tookMs);
    void this.triggerBotMove();
  }

  async pass(): Promise<void> {
    if (!this.manualMoveAllowed()) return;

    this.advance(passTurn(this._state), null);
    void this.triggerBotMove();
  }

  async start(): Promise<void> {
    this.onStateChange(this._state);
    void this.triggerBotMove();
  }

  private async triggerBotMove(): Promise<void> {
    if (this._paused || this._state.status === 'finished') return;

    const currentPlayerConfig =
      this._state.currentPlayer === 'black'
        ? this.players.black
        : this.players.white;
    if (currentPlayerConfig.type !== 'bot') {
      return;
    }

    if (this.botTimeout) clearTimeout(this.botTimeout);

    const bot = currentPlayerConfig.player;
    if (this._state.legalMoves.length === 0) {
      this.advance(passTurn(this._state), null);
      this.botTimeout = setTimeout(() => {
        void this.triggerBotMove();
      }, currentPlayerConfig.thinkMs || 300);
      return;
    }

    const epoch = this.epoch;
    const thinkingSince = performance.now();
    this._thinkingSince = thinkingSince;
    try {
      const pos = await bot.chooseMove(
        this._state,
        currentPlayerConfig.thinkMs,
      );
      // The game moved on (restart, pause, new players) while the bot was thinking.
      if (epoch !== this.epoch) return;
      this.advance(
        applyMove(this._state, pos),
        performance.now() - thinkingSince,
      );
      this.botTimeout = setTimeout(() => {
        void this.triggerBotMove();
      }, 300);
    } catch (e) {
      console.error('Bot failed to move', e);
    }
  }

  newGame(
    players: { black: PlayerConfig; white: PlayerConfig },
    initialState: GameState,
  ) {
    if (this.botTimeout) clearTimeout(this.botTimeout);
    this.epoch += 1;
    this.players = players;
    this._state = initialState;
    this._lastMoveMs = { black: null, white: null };
    this._moves = 0;
    this._turnStartedAt = performance.now();
    this._thinkingSince = null;
    void this.start();
  }
}
