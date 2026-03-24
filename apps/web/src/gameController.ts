import { type GameState, type Position, type PositionAnalysis, type BotPlayer, passTurn, applyMove } from '@reversi/core';

export type PlayerConfig =
  | { type: 'human' }
  | { type: 'bot'; player: BotPlayer; thinkMs: number };

export class GameController {
  private _state: GameState;
  private botTimeout?: ReturnType<typeof setTimeout>;

  private players: { black: PlayerConfig; white: PlayerConfig };
  private onStateChange: (state: GameState, analysis?: PositionAnalysis) => void;

  constructor(
    initialState: GameState,
    players: { black: PlayerConfig; white: PlayerConfig },
    onStateChange: (state: GameState, analysis?: PositionAnalysis) => void
  ) {
    this._state = initialState;
    this.players = players;
    this.onStateChange = onStateChange;
  }

  get state() { return this._state; }

  async makeMove(pos: Position): Promise<void> {
    const currentPlayerConfig = this._state.currentPlayer === 'black' ? this.players.black : this.players.white;
    if (currentPlayerConfig.type !== 'human') {
      return;
    }

    this._state = applyMove(this._state, pos);
    this.onStateChange(this._state);
    void this.triggerBotMove();
  }

  async pass(): Promise<void> {
    const currentPlayerConfig = this._state.currentPlayer === 'black' ? this.players.black : this.players.white;
    if (currentPlayerConfig.type !== 'human') return;

    this._state = passTurn(this._state);
    this.onStateChange(this._state);
    void this.triggerBotMove();
  }

  async start(): Promise<void> {
    this.onStateChange(this._state);
    void this.triggerBotMove();
  }

  private async triggerBotMove(): Promise<void> {
    if (this._state.status === 'finished') return;

    const currentPlayerConfig = this._state.currentPlayer === 'black' ? this.players.black : this.players.white;
    if (currentPlayerConfig.type !== 'bot') {
       if (this._state.legalMoves.length === 0) {
          this._state = passTurn(this._state);
          this.onStateChange(this._state);
          void this.triggerBotMove();
       }
       return;
    }

    if (this.botTimeout) clearTimeout(this.botTimeout);

    const bot = currentPlayerConfig.player;
    if (this._state.legalMoves.length === 0) {
      this._state = passTurn(this._state);
      this.onStateChange(this._state);
      this.botTimeout = setTimeout(() => { void this.triggerBotMove(); }, currentPlayerConfig.thinkMs || 300);
      return;
    }

    try {
      const pos = await bot.chooseMove(this._state, currentPlayerConfig.thinkMs);
      this._state = applyMove(this._state, pos);
      this.onStateChange(this._state);
      this.botTimeout = setTimeout(() => { void this.triggerBotMove(); }, 300);
    } catch (e) {
      console.error('Bot failed to move', e);
    }
  }

  newGame(players: { black: PlayerConfig; white: PlayerConfig }, initialState: GameState) {
    if (this.botTimeout) clearTimeout(this.botTimeout);
    this.players = players;
    this._state = initialState;
    void this.start();
  }
}
