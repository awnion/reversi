import { expect, test } from '@playwright/test';

test('renders playable reversi shell', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Reversi' })).toBeVisible();
  await expect(page.locator('.rv-square')).toHaveCount(64);
});

test('initial state shows black turn with 4 legal moves', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#status')).toContainText('Turn: black');
  await expect(page.locator('#status')).toContainText('4 legal moves');
  await expect(page.locator('#score')).toContainText('Black 2 - 2 White');
});

test('pass button is disabled when legal moves exist', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#pass-button')).toBeDisabled();
});

test('clicking a legal move plays it and switches turn', async ({ page }) => {
  await page.goto('/');

  // click a legal move hint
  const hint = page.locator('.rv-square[data-legal="true"]').first();
  await hint.click();

  // turn switches to white
  await expect(page.locator('#status')).toContainText('Turn: white');
  // black placed a disc, flipped at least one → black > 2
  await expect(page.locator('#score')).not.toContainText('Black 2');
});

test('clicking a non-legal square does nothing', async ({ page }) => {
  await page.goto('/');

  // (0,0) is not a legal move at start
  const corner = page.locator('.rv-square[data-row="0"][data-col="0"]');
  await corner.click();

  // still black's turn
  await expect(page.locator('#status')).toContainText('Turn: black');
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
  await expect(page.locator('#status')).toContainText('Turn: white');

  // white plays
  await page.locator('.rv-square[data-legal="true"]').first().click();
  await expect(page.locator('#status')).toContainText('Turn: black');
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
