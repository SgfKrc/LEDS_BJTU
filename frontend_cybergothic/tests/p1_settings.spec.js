import { test, expect } from '@playwright/test';

test('settings workspace exposes device controls and local RAG actions in fixture mode', async ({ page }) => {
  await page.goto('/#/settings?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: '设置' })).toBeVisible();
  await expect(page.getByTestId('settings-workspace')).toBeVisible();
  await expect(page.getByTestId('device-workspace')).toContainText('Performance laptop');
  await expect(page.locator('.gpu-picker button')).toHaveCount(2);

  await page.locator('.gpu-picker button').nth(1).click();
  await expect(page.locator('.gpu-picker button').nth(1)).toHaveClass(/is-active/);
  await page.getByRole('button', { name: 'Apply recommended configuration' }).click();
  await expect(page.getByText('Fixture recommended configuration applied', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Local RAG' }).click();
  await expect(page.getByTestId('rag-workspace')).toContainText('Sources');
  await page.getByRole('textbox', { name: 'RAG search' }).fill('runtime');
  await page.getByRole('button', { name: 'Search' }).click();
  await expect(page.locator('.rag-results article')).toHaveCount(1);
  await page.getByRole('button', { name: 'Rebuild FTS index' }).click();
  await expect(page.getByText('Fixture FTS index rebuilt', { exact: true })).toBeVisible();
});
