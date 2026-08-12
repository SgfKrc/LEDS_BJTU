import { expect, test } from '@playwright/test';

import { installWorkspaceApi } from './fixtures/workspace-api.js';

async function readSettings(page) {
  return page.evaluate(() => JSON.parse(localStorage.getItem('qlh-settings') || '{}'));
}

async function openSettings(page) {
  await page.getByTitle('系统设置').first().click();
  await expect(page.getByRole('heading', { name: /系统设置/ })).toBeVisible();
}

test('main-node SQLite settings fill missing browser fields without replacing local intent', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('qlh-settings', JSON.stringify({ temperature: 0.9 }));
  });
  const requests = await installWorkspaceApi(page, {
    'GET /api/user/settings': {
      body: {
        source: 'sqlite',
        settings: { maxNewTokens: 2048, temperature: 0.3, topP: 0.8 },
      },
    },
  });

  await page.goto('/');
  await expect.poll(async () => readSettings(page)).toMatchObject({
    maxNewTokens: 2048,
    temperature: 0.9,
    topP: 0.8,
  });
  await openSettings(page);
  await expect(page.locator('.setting-number-input')).toHaveValue('2048');

  expect(requests.some((request) => request.key === 'GET /api/user/settings')).toBe(true);
  expect(requests.some((request) => request.key === 'PUT /api/user/settings')).toBe(false);
});

test('task graph controls enforce full streaming and persist the selected worker policy', async ({ page }) => {
  const requests = await installWorkspaceApi(page);

  await page.goto('/');
  await openSettings(page);
  const streamingToggle = page.getByTitle('完整模式 — 点击切换快速模式');
  await streamingToggle.click();
  await expect(page.getByTitle('快速模式 — 点击切换完整模式')).toBeVisible();

  await page.getByRole('button', { name: '任务链实验' }).click();
  await expect(page.getByTitle('完整模式 — 点击切换快速模式')).toBeDisabled();
  await page.getByRole('button', { name: '自动 Worker' }).click();

  await expect.poll(() => requests.filter((request) => (
    request.key === 'PUT /api/user/settings'
    && request.json?.settings?.taskGraphRemoteMode === 'auto'
  )).length).toBeGreaterThan(0);
  await expect.poll(async () => readSettings(page)).toMatchObject({
    executionMode: 'task_graph',
    streamingMode: 'full',
    taskGraphRemoteMode: 'auto',
  });
});

test('distributed inference failure is surfaced and rolls the optimistic setting back', async ({ page }) => {
  const requests = await installWorkspaceApi(page, {
    'PUT /api/cluster/config/distributed-inference': {
      status: 503,
      body: { detail: 'scheduler unavailable' },
    },
  });

  await page.goto('/');
  await openSettings(page);
  const section = page.getByText('启用分布式推理优化')
    .locator('xpath=ancestor::div[contains(@class,"sidebar-section")]');
  const toggle = section.locator('button.setting-toggle-btn');
  await toggle.click();

  await expect(page.locator('.toast.error')).toContainText('后端同步失败: scheduler unavailable');
  await expect(toggle).toHaveAttribute('title', '已关闭 — 点击启用');
  await expect.poll(async () => (await readSettings(page)).distributedInference).toBe(false);
  expect(requests.some((request) => (
    request.key === 'PUT /api/cluster/config/distributed-inference'
    && request.json?.enabled === true
  ))).toBe(true);
});
