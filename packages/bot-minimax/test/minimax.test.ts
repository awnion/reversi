import { expect, test, describe } from 'bun:test';
import { createMinimaxBot } from '../src/index';
import { createInitialState } from '@reversi/core';

describe('MinimaxBot', () => {
    test('wrapper handles initialized bot and returns a valid starting move', async () => {
        const state = createInitialState();
        const bot = await createMinimaxBot('Test Minimax');
        expect(bot.name).toBe('Test Minimax');
        
        const move = await bot.chooseMove(state, 1000);
        expect(move).toBeDefined();
        const isLegal = state.legalMoves.find(m => m.row === move.row && m.col === move.col);
        expect(isLegal).toBeDefined();

        bot.destroy && bot.destroy();
    });
});
