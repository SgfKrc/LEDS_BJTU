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

test('model workspace exposes asset governance and simulation state in fixture mode', async ({ page }) => {
  await page.goto('/#/models?fixtures=1');
  const governance = page.getByTestId('model-governance');
  await expect(governance).toBeVisible();
  await expect(governance).toContainText('Download manifest');
  await expect(governance).toContainText('master');
  await governance.getByRole('button', { name: 'Simulate deployment' }).click();
  await expect(governance).toContainText('fixture-plan-01');
  await governance.getByRole('button', { name: 'Prepare' }).click();
  await expect(governance).toContainText('ready');
  await governance.getByRole('button', { name: 'Activate' }).click();
  await expect(governance).toContainText('active');
});

test('model workspace can register and remove a custom fixture model', async ({ page }) => {
  await page.goto('/#/models?fixtures=1');
  const registry = page.locator('.model-panel--registry');
  await registry.locator('summary').click();
  await registry.getByLabel('Model ID').fill('custom-fixture-model');
  await registry.getByLabel('Name').fill('Custom Fixture Model');
  await registry.getByRole('button', { name: 'Register' }).click();
  await expect(registry).toContainText('Custom Fixture Model');
  const row = registry.locator('li').filter({ hasText: 'Custom Fixture Model' });
  await row.getByRole('button', { name: 'Remove' }).click();
  await expect(row).toHaveCount(0);
});
