import { expect, test } from '@playwright/test';

test('settings uses page-level navigation with a bounded workspace and archive backdrop', async ({ page }) => {
  await page.goto('/#/settings?fixtures=1');

  await expect(page.getByTestId('settings-page')).toBeVisible();
  await expect(page.locator('canvas.clockwork-archive')).toBeVisible();
  await expect(page.locator('.settings-layout')).toBeVisible();
  await expect(page.locator('.settings-workspace__nav')).toHaveCount(0);
  await expect(page.locator('.settings-current__scroll')).toHaveCSS('overflow-y', 'auto');
  await expect(page.locator('.settings-details')).toContainText('运行上下文');

  await page.getByRole('button', { name: 'Local RAG' }).click();
  await expect(page.locator('.settings-current')).toHaveAttribute('data-section', 'rag');
  await expect(page.getByTestId('rag-workspace')).toContainText('Sources');

  await page.getByRole('button', { name: /连接状态/ }).click();
  await expect(page.locator('.settings-current')).toHaveAttribute('data-section', 'connection');
  await expect(page.locator('.settings-kvgrid')).toContainText('API 地址');

  await page.getByRole('button', { name: /日志令牌/ }).click();
  await page.locator('#log-token').fill('fixture-log-token');
  await expect(page.locator('.field__dirty')).toBeVisible();
  await expect(page.locator('.settings-unsaved')).toContainText('尚未保存');
  await page.getByRole('button', { name: '保存令牌' }).click();
  await expect(page.locator('.settings-saved')).toContainText('本地偏好已同步');
});
