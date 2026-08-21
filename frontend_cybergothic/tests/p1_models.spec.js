import { test, expect } from '@playwright/test';

test('model workspace supports runtime controls and asset preflight in fixture mode', async ({ page }) => {
  await page.goto('/#/models?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: 'Model workspace' })).toBeVisible();
  await expect(page.locator('.model-list > li')).toHaveCount(3);
  await expect(page.locator('.asset-list > li')).toHaveCount(2);
  await expect(page.locator('.runtime-banner')).toContainText('Qwen 1.8B Chat');

  await page.getByRole('button', { name: 'Run preflight' }).click();
  await expect(page.locator('.preflight-result')).toContainText('Gate passed');

  await page.getByRole('button', { name: 'Unload runtime' }).click();
  await expect(page.locator('.runtime-banner')).toContainText('No model loaded');
  await page.getByRole('button', { name: 'Load selected' }).click();
  await expect(page.locator('.runtime-banner')).toContainText('Qwen 1.8B Chat');
  await expect(page.locator('.model-panel--runtime .badge')).toContainText('LOADED');
});
