import { expect, test } from '@playwright/test';

test('renders playable reversi shell', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Reversi' })).toBeVisible();
  await expect(page.locator('.rv-square')).toHaveCount(64);
});

test('choosing a bot does not start it; Play does', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#play-button')).toBeDisabled();

  await page.locator('#player-black .option', { hasText: 'Greedy' }).click();
  await expect(page.locator('#play-button')).toBeEnabled();
  await page.waitForTimeout(700);
  await expect(page.locator('.rv-disc')).toHaveCount(4);

  await page.locator('#play-button').click();
  await expect(page.locator('.rv-disc')).toHaveCount(5);
  await expect(page.locator('#play-button')).toHaveText('Pause');
});

test('a person can play right away: black to move with 4 legal moves', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.rv-hint')).toHaveCount(4);
  await expect(page.locator('#score-black')).toHaveText('2');
  await expect(page.locator('#score-white')).toHaveText('2');
});

test('pass button is hidden while legal moves exist', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#pass-prompt')).toBeHidden();
});

test('clicking a legal move plays it and switches turn', async ({ page }) => {
  await page.goto('/');

  // click a legal move hint
  const hint = page.locator('.rv-square[data-legal="true"]').first();
  await hint.click();

  // turn switches to white
  await expect(page.locator('#state-white')).toHaveText('to move');
  // black placed a disc, flipped at least one → black > 2
  await expect(page.locator('#score-black')).not.toHaveText('2');
});

test('clicking a non-legal square does nothing', async ({ page }) => {
  await page.goto('/');

  // (0,0) is not a legal move at start
  const corner = page.locator('.rv-square[data-row="0"][data-col="0"]');
  await corner.click();

  // still black's turn
  await expect(page.locator('#state-black')).toHaveText('to move');
});

test('disc count updates after each move', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('.rv-disc')).toHaveCount(4);

  await page.locator('.rv-square[data-legal="true"]').first().click();

  await expect(page.locator('.rv-disc')).toHaveCount(5);
});

test('multi-move sequence alternates players', async ({ page }) => {
  await page.goto('/');

  // black plays
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#state-white')).toHaveText('to move');

  // white plays
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#state-black')).toHaveText('to move');
});

test('legal move hints update after each move', async ({ page }) => {
  await page.goto('/');

  const initialHints = await page.locator('.rv-hint').count();
  expect(initialHints).toBe(4);

  await page.locator('.rv-square[data-legal="true"]').first().click();

  // hints should update (white's legal moves are different from black's opening)
  const newHints = await page.locator('.rv-hint').count();
  expect(newHints).toBeGreaterThan(0);
});

test('a bot chosen for white answers once the game is playing', async ({
  page,
}) => {
  await page.goto('/');

  await page.locator('#player-white .option', { hasText: 'Greedy' }).click();
  await page.locator('#play-button').click();
  await page.locator('.rv-square[data-legal="true"]').first().click();

  // The bot replies, so it is black's turn again with six discs on the board.
  await expect(page.locator('.rv-disc')).toHaveCount(6);
  await expect(page.locator('#state-black')).toHaveText('to move');
  await expect(page.locator('.rv-square[data-last="true"]')).toHaveCount(1);
});

test('while a bot is on the move the board is locked but shows its moves', async ({
  page,
}) => {
  await page.goto('/');

  await page.locator('#player-black .option', { hasText: 'Minimax' }).click();
  await page.locator('#play-button').click();

  // Black is thinking: the onlooker sees where it can go, but cannot move.
  await expect(page.locator('.rv-board')).toHaveAttribute(
    'data-locked',
    'true',
  );
  await expect(page.locator('.rv-hint')).toHaveCount(4);
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('.rv-disc')).toHaveCount(4);
});

test('pausing hands the move back to the person', async ({ page }) => {
  await page.goto('/');

  await page.locator('#player-black .option', { hasText: 'Minimax' }).click();
  await page.locator('#play-button').click();
  await page.locator('#play-button').click(); // pause while it is thinking
  await expect(page.locator('#play-button')).toHaveText('Play');
  await expect(page.locator('.rv-board')).toHaveAttribute(
    'data-locked',
    'false',
  );

  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('.rv-disc')).toHaveCount(5);

  // The move the bot was thinking about never lands.
  await page.waitForTimeout(1500);
  await expect(page.locator('.rv-disc')).toHaveCount(5);
  await expect(page.locator('#state-white')).toHaveText('to move');
});

test('network bots say when they were trained and their loss', async ({
  page,
}) => {
  await page.goto('/');
  await expect(
    page.locator('#player-black .option-detail[data-bot="zerorev"]'),
  ).toContainText(/\d{4}, loss \d\.\d\d/);
});

test('Gladiator0 is offered between AlphaZero and ZeroRev', async ({
  page,
}) => {
  await page.goto('/');
  const bots = page.locator('#player-black .option[data-bot]');
  const ids = await bots.evaluateAll((options) =>
    options.map((option) => option.getAttribute('data-bot')),
  );
  expect(ids.indexOf('gladiator0')).toBe(ids.indexOf('alphazero') + 1);
  expect(ids.indexOf('zerorev')).toBe(ids.indexOf('gladiator0') + 1);
  await expect(
    page.locator('#player-black .option[data-bot="gladiator0"]'),
  ).toContainText('Gladiator0');
  await expect(
    page.locator('#player-black .option-detail[data-bot="gladiator0"]'),
  ).toContainText(/\d{4}, loss \d\.\d\d/);
});

test('time per move: presets or a custom value, all in seconds', async ({
  page,
}) => {
  await page.goto('/');
  const time = page.locator('#think-black');
  const preset = (label: string) =>
    time.locator('button.think-cell', { hasText: new RegExp(`^${label}s$`) });
  // No searching bot on black yet: the control is shown but disabled.
  await expect(time).toBeVisible();
  await expect(preset('1')).toBeDisabled();

  await page.locator('#player-black .option', { hasText: 'Minimax' }).click();
  await expect(preset('1')).toBeEnabled();
  await expect(preset('1')).toHaveAttribute('aria-pressed', 'true');

  await preset('0.5').click();
  await expect(preset('0.5')).toHaveAttribute('aria-pressed', 'true');
  await expect(preset('1')).toHaveAttribute('aria-pressed', 'false');

  await page.locator('#think-custom-black').fill('0.25');
  await expect(preset('0.5')).toHaveAttribute('aria-pressed', 'false');
  await expect(time.locator('.think-custom')).toHaveAttribute(
    'data-active',
    'true',
  );

  // A quarter of a second per move: the bot answers well within a second.
  await page.locator('#play-button').click();
  await expect(page.locator('.rv-disc')).toHaveCount(5, { timeout: 1500 });
});

test('the scoreboard says whose move it is and that a bot is thinking', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('#state-black')).toHaveText('to move');
  await expect(page.locator('#state-white')).toHaveText('');

  await page.locator('#player-black .option', { hasText: 'Minimax' }).click();
  await page.locator('#play-button').click();
  await expect(page.locator('#state-black')).toHaveText('thinking');
});

test('the scoreboard names the winner when the game ends', async ({ page }) => {
  await page.goto('/');
  await page.locator('#player-black .option', { hasText: 'Greedy' }).click();
  await page.locator('#player-white .option', { hasText: 'Greedy' }).click();
  await page.locator('#play-button').click();

  // Two greedy bots finish a game in well under a minute.
  const states = page.locator('.player-state');
  await expect(states.filter({ hasText: /^(wins|draw)$/ }).first()).toBeVisible(
    {
      timeout: 45_000,
    },
  );
  const black = Number(await page.locator('#score-black').textContent());
  const white = Number(await page.locator('#score-white').textContent());
  const expected =
    black === white
      ? ['draw', 'draw']
      : black > white
        ? ['wins', '']
        : ['', 'wins'];
  await expect(page.locator('#state-black')).toHaveText(expected[0]);
  await expect(page.locator('#state-white')).toHaveText(expected[1]);
  await expect(page.locator('#play-button')).toBeDisabled();
});

test('no clock runs before the first move', async ({ page }) => {
  await page.goto('/');
  await page.waitForTimeout(400);
  await expect(page.locator('#clock-black')).toHaveAttribute(
    'data-idle',
    'true',
  );
  await expect(page.locator('#clock-white')).toHaveAttribute(
    'data-idle',
    'true',
  );

  // Once a move has been made the side on the move is timed.
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#clock-white')).toHaveAttribute(
    'data-idle',
    'false',
  );
  await expect(page.locator('#clock-white')).toHaveText(/^\d\d:\d\d\.\d$/);
});

test('clocks: time left for a thinking bot, time taken for the last move', async ({
  page,
}) => {
  await page.goto('/');
  await page.locator('#player-white .option', { hasText: 'Minimax' }).click();
  await page.locator('#think-custom-white').fill('2,5');
  await page.locator('#play-button').click();
  await page.locator('.rv-square[data-legal="true"]').first().click();

  // White is thinking with 2.5 s: its clock counts down in tenths.
  await expect(page.locator('#state-white')).toHaveText('thinking');
  await expect(page.locator('#clock-white')).toHaveText(/^00:0[012]\.\d$/);
  // Black's opening move was not timed: no clock runs before the first move.
  await expect(page.locator('#clock-black')).toHaveAttribute(
    'data-idle',
    'true',
  );

  // After white's move its clock shows the time it took, close to the budget.
  await expect(page.locator('#state-black')).toHaveText('to move', {
    timeout: 5000,
  });
  await expect(page.locator('#clock-white')).toHaveText(/^00:02\.\d$/);
});

test('the CPU meter lights up while a bot is searching', async ({ page }) => {
  await page.goto('/');
  const meter = page.locator('#cpu-meter');
  await expect(meter).toContainText('CPU');
  await expect(meter).toHaveAttribute('aria-label', /^CPU idle/);
  await expect(meter.locator('.cpu-cores i.busy')).toHaveCount(0);

  await page.locator('#player-black .option', { hasText: 'Minimax' }).click();
  await page.locator('#play-button').click();
  // One thread, or several where the page is cross-origin isolated.
  await expect(meter).toHaveAttribute('data-busy', /^[1-9]\d*$/);
  await expect(meter.locator('.cpu-cores i.busy')).not.toHaveCount(0);

  await page.locator('#play-button').click(); // pause
  await expect(meter.locator('.cpu-cores i.busy')).toHaveCount(0);
});

test('analysis switch: the engine annotates every position while it is on', async ({
  page,
}) => {
  await page.goto('/');
  const analysis = page.locator('#analysis-switch');
  await expect(analysis).toHaveAttribute('aria-checked', 'false');
  await expect(page.locator('.rv-note')).toHaveCount(0);

  await analysis.click();
  await expect(analysis).toHaveAttribute('aria-checked', 'true');
  // One note per legal move, exactly one marked as the best.
  await expect(page.locator('.rv-note')).toHaveCount(4);
  await expect(page.locator('.rv-note[data-best="true"]')).toHaveCount(1);
  // ZeroRev, the default, answers with shares of its search.
  await expect(page.locator('.rv-note').first()).toHaveText(/^\d+%$/);
  await expect(page.locator('#analysis-results')).toContainText('simulations');
  // The eval bar: ZeroRev's chance to win.
  const evalBar = page.locator('#eval-bar');
  await expect(evalBar).toBeVisible();
  await expect(page.locator('#eval-label')).toHaveText(/^\d+%$/);

  // Minimax answers with scores, and so does its bar.
  await page.locator('.analysis-engine', { hasText: 'Minimax' }).click();
  await expect(page.locator('.rv-note').first()).toHaveText(
    /^([+\u2212]\d+|0)$/,
  );
  await expect(page.locator('#eval-label')).toHaveText(/^\d+$/);
  await expect(page.locator('#analysis-results')).toContainText('moves deep');
  await page.locator('.analysis-engine', { hasText: 'ZeroRev' }).click();

  // After a move the new position is analyzed without asking again.
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#state-white')).toHaveText('to move');
  await expect(page.locator('.rv-note').first()).toHaveText(/^\d+%$/);

  // Off: the notes and the bar go away and stay away.
  await analysis.click();
  await expect(page.locator('.rv-note')).toHaveCount(0);
  await expect(evalBar).toBeHidden();
  await expect(page.locator('#analysis-results')).toHaveText('');
});

test('analysis found so far comes back when the switch is turned on again', async ({
  page,
}) => {
  await page.goto('/');
  const analysis = page.locator('#analysis-switch');
  await analysis.click();
  await expect(page.locator('.rv-note')).toHaveCount(4);
  const before = await page.locator('.rv-note').allTextContents();

  await analysis.click();
  await expect(page.locator('.rv-note')).toHaveCount(0);

  // Back on: the notes are there at once, without a new search.
  await analysis.click();
  await expect(page.locator('.rv-note')).toHaveCount(4, { timeout: 150 });
  expect(await page.locator('.rv-note').allTextContents()).toEqual(before);
  await expect(page.locator('#analysis-results')).toContainText('simulations');
});

test('the board sits evenly inside its frame', async ({ page }) => {
  await page.goto('/');

  const frame = await page.locator('.board-frame').boundingBox();
  const board = await page.locator('.rv-board').boundingBox();
  if (!frame || !board) throw new Error('board is not rendered');

  const margins = [
    board.x - frame.x,
    board.y - frame.y,
    frame.x + frame.width - (board.x + board.width),
    frame.y + frame.height - (board.y + board.height),
  ];
  for (const margin of margins) {
    expect(Math.abs(margin - margins[0])).toBeLessThan(0.5);
  }
  expect(Math.abs(board.width - board.height)).toBeLessThan(0.5);
});

test('mirror the board: columns run right to left, and it is remembered', async ({
  page,
}) => {
  await page.goto('/');
  const firstSquare = page.locator('.rv-square').first();
  await expect(firstSquare).toHaveAttribute('data-col', '0');

  const mirror = page.getByRole('button', { name: 'Mirror the board' });
  await expect(mirror).toHaveAttribute('aria-pressed', 'false');
  await mirror.click();
  await expect(mirror).toHaveAttribute('aria-pressed', 'true');
  await expect(firstSquare).toHaveAttribute('data-col', '7');

  // A move still lands on the square that was clicked.
  const legal = page.locator('.rv-square[data-legal="true"]').first();
  const col = await legal.getAttribute('data-col');
  const row = await legal.getAttribute('data-row');
  await legal.click();
  await expect(
    page.locator(`.rv-square[data-row="${row}"][data-col="${col}"]`),
  ).toHaveAttribute('data-last', 'true');

  await page.reload();
  await expect(
    page.getByRole('button', { name: 'Mirror the board' }),
  ).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.rv-square').first()).toHaveAttribute(
    'data-col',
    '7',
  );
});

test('a side with no moves is offered a pass on the board', async ({
  page,
}) => {
  await page.goto('/');
  // A fixed pseudo-random game that reaches a pass after 53 moves.
  let seed = 291;
  for (let i = 0; i < 53; i += 1) {
    const legal = await page
      .locator('.rv-square[data-legal="true"]')
      .evaluateAll((squares) =>
        squares.map((square) => [
          (square as HTMLElement).dataset.row,
          (square as HTMLElement).dataset.col,
        ]),
      );
    seed = (seed * 1103515245 + 12345) % 2147483648;
    const [row, col] = legal[seed % legal.length] ?? [];
    const square = page.locator(
      `.rv-square[data-row="${row}"][data-col="${col}"]`,
    );
    await square.click();
    await expect(square).toHaveAttribute('data-last', 'true');
  }

  const prompt = page.locator('#pass-prompt');
  await expect(prompt).toBeVisible();
  await expect(prompt).toContainText('has no moves');
  const mover = await page.locator('.rv-board').getAttribute('data-turn');
  await prompt.getByRole('button', { name: 'Pass' }).click();
  await expect(prompt).toBeHidden();
  await expect(page.locator('.rv-board')).not.toHaveAttribute(
    'data-turn',
    mover ?? '',
  );
});

test('a reload keeps the position on the board', async ({ page }) => {
  await page.goto('/');
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#score-black')).toHaveText('4');
  await expect(page.locator('#score-white')).toHaveText('1');

  await page.reload();
  await expect(page.locator('#score-black')).toHaveText('4');
  await expect(page.locator('#score-white')).toHaveText('1');
  await expect(page.locator('#player-white')).toHaveAttribute(
    'data-turn',
    'true',
  );

  // A new game is kept too.
  await page.getByRole('button', { name: 'New game' }).click();
  await page.reload();
  await expect(page.locator('#score-black')).toHaveText('2');
  await expect(page.locator('#score-white')).toHaveText('2');
});
