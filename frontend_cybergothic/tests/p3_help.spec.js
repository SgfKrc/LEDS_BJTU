import { expect, test } from '@playwright/test';

test('help uses an indexed, bounded documentation workspace with a focused detail pane', async ({ page }) => {
  await page.goto('/#/help?fixtures=1');

  await expect(page.getByTestId('help-page')).toBeVisible();
  await expect(page.locator('canvas.stained-glass-scriptorium')).toBeVisible();
  await expect(page.locator('.help-workspace__scroll')).toHaveCSS('overflow-y', 'auto');
  await expect(page.locator('.help-details')).toContainText('当前条目');

  await page.getByRole('button', { name: '接口清单' }).click();
  await expect(page.locator('.help-workspace')).toHaveAttribute('data-section', 'api');
  await page.getByRole('button', { name: /GET \/api\/cluster\/queue/ }).click();
  await expect(page.locator('.help-detail-panel')).toContainText('仅主节点');

  await page.getByRole('button', { name: '常见问题' }).click();
  await page.getByRole('button', { name: '活动页读不到日志？' }).click();
  await expect(page.locator('.help-detail-panel')).toContainText('X-QLH-Log-Token');
});
