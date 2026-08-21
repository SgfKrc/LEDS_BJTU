import { expect, test } from '@playwright/test';

test('overview keeps summary, nodes, logs, and inspection context in bounded desktop panes', async ({ page }) => {
  await page.goto('/#/overview?fixtures=1');

  await expect(page.getByTestId('overview-page')).toBeVisible();
  await expect(page.locator('canvas.observatory-nave')).toBeVisible();
  await expect(page.locator('.overview-workspace__scroll')).toHaveCSS('overflow-y', 'auto');
  await expect(page.locator('.overview-details')).toContainText('当前详情');
  await expect(page.locator('.metricstrip .metric')).toHaveCount(4);

  await page.getByRole('button', { name: '节点' }).click();
  await expect(page.locator('.overview-workspace')).toHaveAttribute('data-workspace', 'nodes');
  await expect(page.locator('.overview-nodecard')).toHaveCount(3);
  await page.locator('.overview-nodecard button').nth(1).click();
  await expect(page.locator('.overview-detail-panel')).toContainText('TABLET-2TLUCNU8');

  await page.getByRole('button', { name: '最近活动' }).click();
  await expect(page.locator('.overview-workspace')).toHaveAttribute('data-workspace', 'activity');
  await page.locator('.overview-timeline .timeline__body--action').first().click();
  await expect(page.locator('.overview-detail-panel .codeblock')).toBeVisible();
});

test('overview moves selected node inspection into a drawer below the desktop layout', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 900 });
  await page.goto('/#/overview?fixtures=1');
  await page.getByRole('button', { name: '节点' }).click();
  await page.locator('.overview-nodecard button').first().click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog')).toContainText('localhost');
});
