import { describe, expect, test } from 'vitest';
import { page } from 'vitest/browser';

import { createInitialState } from '@reversi/core';
import { createReversiBoard } from '../src/reversiBoard';

describe('reversi board ui', () => {
  test('renders 64 squares and 4 starting discs', async () => {
    document.body.innerHTML = '<div id="board"></div>';
    const root = document.getElementById('board');
    if (!root) throw new Error('Missing board root');

    createReversiBoard(root, { state: createInitialState() });

    await expect.element(page.getByRole('button').nth(0)).toBeInTheDocument();
    expect(root.querySelectorAll('.rv-square')).toHaveLength(64);
    expect(root.querySelectorAll('.rv-disc')).toHaveLength(4);
  });
});
