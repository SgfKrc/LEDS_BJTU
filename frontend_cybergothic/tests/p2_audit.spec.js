import { test, expect } from '@playwright/test';

test('audit ledger covers archive, node relay, and review workflows in fixture mode', async ({ page }) => {
  page.on('dialog', (dialog) => dialog.accept());
  await page.goto('/#/audit?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: 'Audit Ledger' })).toBeVisible();
  await expect(page.getByTestId('audit-page')).toBeVisible();
  await expect(page.locator('.audit-file-row')).toHaveCount(3);
  await expect(page.locator('.audit-stat-grid > div')).toHaveCount(3);
  await expect(page.locator('canvas.archive-clock')).toHaveAttribute('aria-hidden', 'true');

  await page.getByRole('button', { name: 'Preview' }).first().click();
  await expect(page.locator('.audit-preview')).toContainText('FILE PREVIEW');
  await page.getByRole('button', { name: 'Delete' }).first().click();
  await expect(page.locator('.audit-file-row')).toHaveCount(2);

  await page.locator('.audit-nav button', { hasText: 'Node relay' }).click();
  await expect(page.locator('.audit-node-grid > article')).toHaveCount(2);
  await expect(page.locator('.audit-relay-feed')).toContainText('heartbeat_sent');

  await page.locator('.audit-nav button', { hasText: 'Review' }).click();
  await expect(page.locator('.audit-ticket-list > article')).toHaveCount(2);
  await page.getByLabel('TARGET NODE ID').fill('client-lab-02');
  await page.getByLabel('REASON').fill('Maintenance rehearsal');
  await page.getByRole('button', { name: 'Create review' }).click();
  await expect(page.locator('.audit-ticket-list > article')).toHaveCount(3);
  await page.getByRole('button', { name: 'Approve' }).first().click();
  await expect(page.locator('.audit-ticket-list > article').first().locator('dd').first()).toHaveText('1');
});

test('audit node relay stays gated on a client role', async ({ page }) => {
  let aggregateRequests = 0;
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/cluster/nodes/log-aggregate') || url.pathname.endsWith('/logs/nodes-summary')) {
      aggregateRequests += 1;
      await route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ detail: 'master only' }) });
      return;
    }
    const payload = url.pathname.endsWith('/cluster/my-role')
      ? { node_role: 'client', node_id: 'client-01', is_master: false, is_client: true, max_nodes: 3, run_mode: 'distributed' }
      : url.pathname.endsWith('/logs') || url.pathname.endsWith('/logs/stats')
        ? { files: [], files_count: 0, files_total_bytes: 0, buffer_size: 0, buffer_capacity: 100, buffer_total_seen: 0, buffer_dropped_estimate: 0, levels: {}, loggers: {}, nodes: {} }
        : url.pathname.endsWith('/cluster/review/tickets')
          ? { tickets: [], count: 0 }
          : url.pathname.endsWith('/cluster/review/can-vote')
            ? { node_id: 'client-01', can_vote: false, reason: 'client cannot vote' }
            : { status: 'ok', timestamp: Date.now() / 1000 };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) });
  });

  await page.goto('/#/audit');
  await page.locator('.audit-nav button', { hasText: 'Node relay' }).click();
  await expect(page.locator('.audit-readonly-callout')).toContainText('master-only');
  expect(aggregateRequests).toBe(0);
});
