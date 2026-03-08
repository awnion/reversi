import { expect, test } from '@playwright/test';

test('renders playable reversi shell', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Reversi' })).toBeVisible();
  await expect(page.locator('.rv-square')).toHaveCount(64);
});
