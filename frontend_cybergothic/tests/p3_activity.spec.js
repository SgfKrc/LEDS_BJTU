import { test, expect } from '@playwright/test';

test('activity workspace keeps feed, sessions, and detail context separate on desktop', async ({ page }) => {
  await page.goto('/#/activity?fixtures=1');

  await expect(page.getByTestId('activity-page')).toBeVisible();
  await expect(page.locator('.activity-rail')).toBeVisible();
  await expect(page.locator('.activity-feed')).toBeVisible();
  await expect(page.locator('.activity-details')).toBeVisible();
  await expect(page.locator('canvas.bell-tower-rain')).toHaveAttribute('aria-hidden', 'true');

  await page.locator('.timeline__body--action').first().click();
  await expect(page.locator('.activity-detail-panel')).toContainText('请求 ID');
  await page.locator('.activity-session').first().click();
  await expect(page.locator('.activity-detail-panel')).toContainText('会话 ID');
});

test('activity workspace moves selected log details into a drawer below desktop layout', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 });
  await page.goto('/#/activity?fixtures=1');
  await page.locator('.timeline__body--action').first().click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog')).toContainText('请求 ID');
});

test('activity workspace paginates a full log buffer', async ({ page }) => {
  const logs = Array.from({ length: 50 }, (_, index) => ({
    timestamp: `2026-08-22T10:${String(index).padStart(2, '0')}:00Z`,
    level: 'INFO',
    levelno: 20,
    name: 'test',
    message: `buffer event ${index + 1}`,
    seq: 1000 - index,
  }));
  await page.route('**/api/**', async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname.endsWith('/logs/recent')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ logs, count: logs.length, matched: logs.length, buffer_size: logs.length, buffer_capacity: 200 }),
      });
      return;
    }
    if (pathname.endsWith('/sessions')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [], active_session_id: '', total: 0, source: 'sqlite' }) });
      return;
    }
    await route.continue();
  });

  await page.goto('/#/activity');
  await expect(page.locator('.activity-pagination')).toContainText('第 1 / 3 页');
  await expect(page.locator('.timeline__item')).toHaveCount(24);
  await page.getByRole('button', { name: '下一页' }).click();
  await expect(page.locator('.activity-pagination')).toContainText('第 2 / 3 页');
  await expect(page.locator('.timeline__item')).toHaveCount(24);
});
