import { test, expect } from '@playwright/test';

const DEFENSE_ROUTE = '/#/cluster?fixtures=1&defense=topology';

test('DEF-P4 shows a redacted readiness and layer-allocation snapshot', async ({ page }) => {
  await page.goto(DEFENSE_ROUTE);

  await expect(page.getByRole('heading', { level: 1, name: 'Cluster Topology Defense' })).toBeVisible();
  const snapshot = page.getByTestId('defense-topology-snapshot');
  await expect(snapshot).toBeVisible();
  await expect(snapshot.getByRole('note', { name: 'Snapshot claim boundary' })).toContainText('FIXTURE / REDACTED / NOT LIVE');
  await expect(snapshot.getByRole('note', { name: 'Snapshot claim boundary' })).toContainText('no real model · no physical dual-host claim');

  await expect(snapshot.locator('.cluster-defense__check')).toHaveCount(3);
  await expect(snapshot.locator('.cluster-defense__node')).toHaveCount(3);
  await expect(snapshot.locator('.cluster-defense__node[data-state="ready"]')).toHaveCount(2);
  await expect(snapshot.locator('.cluster-defense__node[data-state="offline"]')).toHaveCount(1);
  await expect(snapshot.locator('.cluster-defense__layer-row')).toHaveCount(2);
  await expect(snapshot).toContainText('node-master');
  await expect(snapshot).toContainText('node-worker-a');
  await expect(snapshot).toContainText('L0–L7');
  await expect(snapshot).toContainText('L8–L23');

  const evidenceText = await snapshot.innerText();
  expect(evidenceText).not.toMatch(/(?:\d{1,3}\.){3}\d{1,3}/);
  expect(evidenceText).not.toContain('TABLET-2TLUCNU8');
  expect(evidenceText).not.toContain('RTX 4060');
  await expect(page.locator('.cluster-table')).toHaveCount(0);
  await expect(page.locator('canvas.cluster-constellation')).toHaveAttribute('data-node-count', '3');
});

test('DEF-P4 snapshot remains within a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(DEFENSE_ROUTE);

  await expect(page.getByTestId('defense-topology-snapshot')).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  const nodes = page.locator('.cluster-defense__node');
  await expect(nodes).toHaveCount(3);
  const boxes = await nodes.evaluateAll((items) => items.map((item) => item.getBoundingClientRect()));
  expect(boxes[1].top).toBeGreaterThan(boxes[0].bottom);
});
