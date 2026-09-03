import { test, expect } from '@playwright/test';

test('cluster admin exposes master controls and fixture topology', async ({ page }) => {
  await page.goto('/#/cluster?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: 'Cluster Admin' })).toBeVisible();
  await expect(page.getByTestId('cluster-admin-page')).toBeVisible();
  await expect(page.locator('.cluster-status-card')).toHaveCount(4);
  await expect(page.locator('.cluster-table tbody tr')).toHaveCount(3);
  await expect(page.locator('canvas.cluster-constellation')).toHaveAttribute('aria-hidden', 'true');
  await expect(page.locator('canvas.cluster-constellation')).toHaveAttribute('data-node-count', '3');
  const paintedPixels = await page.locator('canvas.cluster-constellation').evaluate((canvas) => {
    const context = canvas.getContext('2d');
    if (!context || canvas.width === 0 || canvas.height === 0) return 0;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let index = 3; index < pixels.length; index += 4) {
      if (pixels[index] > 0) count += 1;
    }
    return count;
  });
  expect(paintedPixels).toBeGreaterThan(0);

  await page.locator('.cluster-capacity input').fill('4');
  await page.getByRole('button', { name: 'Apply capacity' }).click();
  await expect(page.locator('.toasthost .toast')).toContainText('Fixture capacity updated to 4');

  await page.getByLabel('NODE ID *').fill('client-lab-01');
  await page.getByRole('button', { name: 'Reserve node' }).click();
  await expect(page.locator('.cluster-table tbody tr')).toHaveCount(4);
  await expect(page.locator('.cluster-table')).toContainText('client-lab-01');
});

test('cluster admin is read-only on a client role and does not request invite data', async ({ page }) => {
  let inviteRequests = 0;
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/cluster/invite')) {
      inviteRequests += 1;
      await route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ detail: 'master only' }) });
      return;
    }
    const payload = url.pathname.endsWith('/cluster/my-role')
      ? { node_role: 'client', node_id: 'client-01', is_master: false, is_client: true, max_nodes: 3, run_mode: 'distributed' }
      : url.pathname.endsWith('/cluster/nodes')
        ? { nodes: [{ node_id: 'client-01', role: 'client', node_type: 'pc', state: 'online', address: '100.64.0.2:8888', hostname: 'client-01', device_info: {}, network_type: 'tailscale', connected_at: 0, last_heartbeat: Math.floor(Date.now() / 1000), avg_rtt_ms: 12, last_rtt_ms: 12, task_count: 0, error_count: 0, is_available: true }], count: 1, online_count: 1, offline_count: 0 }
        : url.pathname.endsWith('/cluster/status')
          ? { run_mode: 'distributed', nodes_ready: true, nodes: {}, current_task: null, tcp_server: null, pipeline: null, pipeline_queue: null, network_path: null }
          : url.pathname.endsWith('/cluster/config')
            ? { max_nodes: 3, network: { heartbeat_interval_s: 5 } }
            : url.pathname.endsWith('/cluster/master-health')
              ? { master_online: true, last_seen_seconds_ago: 1, stale: false, master_host: '100.64.0.1', master_port: 8888 }
              : { status: 'ok', timestamp: Date.now() / 1000 };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) });
  });

  await page.goto('/#/cluster');
  await expect(page.getByRole('heading', { level: 1, name: 'Cluster Admin' })).toBeVisible();
  await expect(page.locator('.cluster-health')).toContainText('ONLINE');
  await expect(page.locator('.cluster-readonly-callout')).toContainText('Read-only cluster view');
  await expect(page.locator('.cluster-register')).toHaveCount(0);
  await expect(page.locator('.cluster-table__actions')).toContainText('read only');
  expect(inviteRequests).toBe(0);
});

test('cluster admin exposes capacity, layer and sidecar contract projections', async ({ page }) => {
  await page.goto('/#/cluster?fixtures=1');

  await expect(page.locator('#cluster-plans')).toBeVisible();
  await expect(page.locator('.cluster-plan-status')).toContainText('ADMITTED');
  await expect(page.locator('.cluster-layer-list .cluster-layer-row')).toHaveCount(2);
  await expect(page.locator('.cluster-endpoint-list .cluster-endpoint-row')).toContainText('Local API');
  await expect(page.locator('.cluster-sidecar-list .cluster-sidecar-row')).toHaveCount(2);
  await expect(page.locator('.cluster-contract-summary')).toContainText('experimental gate only');
});
