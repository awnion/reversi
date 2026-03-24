import { expect, test, describe } from 'bun:test';
import { GameController } from '../src/gameController';
import { createInitialState } from '@reversi/core';

describe('GameController', () => {
   test('initializes and manages state changes correctly for human players', async () => {
       const initial = createInitialState();
       let lastState = initial;
       const controller = new GameController(
           initial,
           { black: { type: 'human' }, white: { type: 'human' } },
           (state) => { lastState = state; }
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
                     chooseMove: async () => ({row:0, col:0}), 
                     analyzePosition: async () => ({bestMove:null, score:0, depth:0, moves:[]})
                 }, 
                 thinkMs: 100 
             }, 
             white: { type: 'human' } 
           },
           () => {}
       );
       await controller.makeMove({row: 0, col: 0});
       expect(controller.state.currentPlayer).toBe('black');
   });
});
