import { test, expect } from '@playwright/test';

test('工作台 fixture 会话栏支持新建、切换和重命名', async ({ page }) => {
  await page.goto('/#/workbench?fixtures=1');
  await expect(page.locator('.chat-sessions')).toBeVisible();
  await expect(page.locator('.chat-sessions__row')).toHaveCount(2);

  await page.getByRole('button', { name: '新建会话' }).click();
  await expect(page.locator('.chat-sessions__row')).toHaveCount(3);
  await expect(page.locator('.chat__eyebrow')).toContainText('fixture-');

  await page.locator('.chat-sessions__select', { hasText: '提交信息生成' }).click();
  await expect(page.locator('.chat__eyebrow')).toContainText('sess_commit');

  await page.getByRole('button', { name: '重命名 提交信息生成' }).click();
  const input = page.getByRole('textbox', { name: '会话名称' });
  await input.fill('任务提交记录');
  await page.getByRole('button', { name: '保存会话名称' }).click();
  await expect(page.locator('.chat-sessions__select', { hasText: '任务提交记录' })).toBeVisible();
});

test('client 角色的 Overview 不被主节点队列 403 阻断', async ({ page }) => {
  await page.route('**/api/cluster/my-role', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        node_role: 'client',
        node_id: 'client_local',
        is_master: false,
        is_client: true,
        max_nodes: 1,
        run_mode: 'single',
      }),
    }),
  );
  await page.route('**/api/cluster/queue', (route) =>
    route.fulfill({
      status: 403,
      contentType: 'application/json',
      body: JSON.stringify({ detail: '仅主节点可查看请求队列' }),
    }),
  );

  await page.goto('/#/overview');
  await expect(page.getByRole('heading', { level: 1, name: '集群概览' })).toBeVisible();
  await expect(page.locator('.metric').filter({ hasText: '队列深度' })).toContainText('单机模式不适用');
  await page.getByRole('button', { name: '流水线准入' }).click();
  await expect(page.getByText('单机模式不使用主节点队列')).toBeVisible();
  await expect(page.getByRole('heading', { level: 1, name: '集群概览' })).toBeVisible();
});

test('容量接口返回 unavailable 精简响应时 Overview 仍可渲染', async ({ page }) => {
  await page.route('**/api/cluster/my-role', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        node_role: 'master',
        node_id: 'master',
        is_master: true,
        is_client: false,
        max_nodes: 1,
        run_mode: 'single',
      }),
    }),
  );
  await page.route('**/api/cluster/pipeline-capacity', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'unavailable',
        admitted: false,
        reason_code: 'pipeline_descriptor_unavailable',
        assignments: [],
      }),
    }),
  );

  await page.goto('/#/overview?fixtures=0');
  await expect(page.getByRole('heading', { level: 1, name: '集群概览' })).toBeVisible();
  await page.getByRole('button', { name: '流水线准入' }).click();
  await expect(page.locator('.capacity__facts')).toBeVisible();
  await expect(page.locator('.capacity__facts')).toContainText('无');
});
