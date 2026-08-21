import { test, expect } from '@playwright/test';

test('对话历史支持刷新和删除单轮', async ({ page }) => {
  await page.goto('/#/workbench?fixtures=1');

  await expect(page.locator('.chat__eyebrow')).toContainText('default');
  await expect(page.locator('.msg--assistant')).toHaveCount(2);

  const refresh = page.getByRole('button', { name: '刷新会话历史' });
  await expect(refresh).toBeEnabled();
  await refresh.click();
  await expect(page.locator('.msg--assistant')).toHaveCount(2);

  await page.getByRole('button', { name: '删除这一轮对话' }).first().click();
  await expect(page.getByRole('button', { name: '确认' })).toBeVisible();
  await page.getByRole('button', { name: '确认' }).click();
  await expect(page.locator('.msg--assistant')).toHaveCount(1);
  await expect(page.getByText('演示会话已移除这一轮。')).toBeVisible();
});
