import { test, expect } from '@playwright/test';

test('tasks workspace separates queue controls, workflow records, and detail on desktop', async ({ page }) => {
  await page.goto('/#/tasks?fixtures=1');

  await expect(page.getByTestId('tasks-page')).toBeVisible();
  await expect(page.locator('.tasks-rail')).toBeVisible();
  await expect(page.locator('.tasks-main')).toBeVisible();
  await expect(page.locator('.tasks-details')).toBeVisible();
  await expect(page.locator('canvas.gearworks-queue')).toHaveAttribute('aria-hidden', 'true');

  await page.locator('.tasks-workspace[data-workspace="queue"] .ttable__open').first().click();
  await expect(page.locator('.tasks-detail-panel')).toContainText('提示 tokens');
  await page.getByRole('button', { name: 'Workflows' }).click();
  await page.locator('.tasks-workspace[data-workspace="workflows"] .ttable__open').first().click();
  await expect(page.locator('.tasks-detail-panel')).toContainText('执行阶段');
});

test('tasks workspace sends workflow details to a drawer on tablet', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 });
  await page.goto('/#/tasks?fixtures=1');
  await page.getByRole('button', { name: 'Workflows' }).click();
  await page.locator('.tasks-workspace[data-workspace="workflows"] .ttable__open').first().click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog')).toContainText('执行阶段');
});

test('tasks client role does not request or expose primary queue controls', async ({ page }) => {
  let queueRequests = 0;
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/cluster/queue')) {
      queueRequests += 1;
      await route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ detail: 'master only' }) });
      return;
    }
    const payload = url.pathname.endsWith('/cluster/my-role')
      ? { node_role: 'client', node_id: 'client-01', is_master: false, is_client: true, max_nodes: 3, run_mode: 'distributed' }
      : url.pathname.endsWith('/workflows')
        ? { enabled: true, available: true, role: 'client', templates: [], providers: [], provider_status: [], provider_error: '', workflows: [] }
        : { status: 'ok', timestamp: Date.now() / 1000 };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) });
  });

  await page.goto('/#/tasks');
  await expect(page.locator('.tasks-controls')).toContainText('Queue authority');
  await expect(page.getByRole('button', { name: '暂停队列' })).toHaveCount(0);
  await expect(page.locator('.tasks-workspace')).toContainText('当前节点不使用主节点队列');
  expect(queueRequests).toBe(0);
});
