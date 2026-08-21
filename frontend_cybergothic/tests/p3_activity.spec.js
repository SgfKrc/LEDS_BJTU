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
