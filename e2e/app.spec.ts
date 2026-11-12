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
  await expect(page.locator('#pass-button')).toBeHidden();
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

test('time per move: presets in seconds or a custom value in ms', async ({
  page,
}) => {
  await page.goto('/');
  const time = page.locator('#think-black');
  const preset = (label: string) =>
    time.locator('button.think-cell', { hasText: new RegExp(`^${label}$`) });
  // No searching bot on black yet: the control is shown but disabled.
  await expect(time).toBeVisible();
  await expect(preset('1')).toBeDisabled();

  await page.locator('#player-black .option', { hasText: 'Minimax' }).click();
  await expect(preset('1')).toBeEnabled();
  await expect(preset('1')).toHaveAttribute('aria-pressed', 'true');

  await preset('0.5').click();
  await expect(preset('0.5')).toHaveAttribute('aria-pressed', 'true');
  await expect(preset('1')).toHaveAttribute('aria-pressed', 'false');

  await page.locator('#think-custom-black').fill('250');
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
  await expect(page.locator('#clock-black')).toHaveText('');
  await expect(page.locator('#clock-white')).toHaveText('');

  // Once a move has been made the side on the move is timed.
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#clock-white')).toHaveText(/^\d+ms$/);
});

test('clocks: time left for a thinking bot, time taken for the last move', async ({
  page,
}) => {
  await page.goto('/');
  await page.locator('#player-white .option', { hasText: 'Minimax' }).click();
  await page.locator('#think-custom-white').fill('2500');
  await page.locator('#play-button').click();
  await page.locator('.rv-square[data-legal="true"]').first().click();

  // White is thinking with 2.5 s: its clock counts down, thousands set apart.
  await expect(page.locator('#state-white')).toHaveText('thinking');
  await expect(page.locator('#clock-white .ms-thousands')).toHaveText(/^[12]$/);
  await expect(page.locator('#clock-white')).toHaveText(/^[12]\d{3}ms$/);
  // Black's opening move was not timed: no clock runs before the first move.
  await expect(page.locator('#clock-black')).toHaveText('');

  // After white's move its clock shows the time it took, close to the budget.
  await expect(page.locator('#state-black')).toHaveText('to move', {
    timeout: 5000,
  });
  await expect(page.locator('#clock-white')).toHaveText(/^2\d{3}ms$/);
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
