import { test, expect } from '@playwright/test';

const baseStatus = {
  model_loaded: false,
  pipeline_prepared: true,
  current_quant: 'fp16',
  model_name: 'Qwen-1_8B',
  model_path: 'models/qwen-1_8b',
  active_model_id: null,
  engine: 'pipeline',
  run_mode: 'single',
  node_role: 'master',
  node_id: 'local',
  max_nodes: 1,
  conversation_turns: 0,
  gpu: { name: 'Test GPU', total_mb: 8192, allocated_mb: 0, reserved_mb: 0, utilization: 0 },
  kv_cache: { total_tokens: 0, max_tokens: 8192, allocated_pages: 0, free_pages: 64, max_pages: 64, page_size: 128, utilization: 0, estimated_memory_mb: 0, rounds: 0, total_time_s: 0 },
  device: { tier: 'desktop', tier_label: '测试设备', score: 80, recommendations: [], warnings: [] },
};

function installRoleRoutes(page, roleName) {
  const counters = { queue: 0, capacity: 0 };

  page.route('**/api/**', async (route) => {
    const request = route.request();
    if (request.method() !== 'GET') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
      return;
    }

    const path = new URL(request.url()).pathname;
    let body = {};
    if (path === '/api/cluster/my-role') {
      body = roleName === 'unset'
        ? {}
        : {
            node_role: roleName,
            node_id: `${roleName}_local`,
            is_master: roleName === 'master',
            is_client: roleName === 'client',
            max_nodes: 1,
            run_mode: 'single',
          };
    } else if (path === '/api/status') {
      body = { ...baseStatus, node_role: roleName === 'unset' ? '' : roleName };
    } else if (path === '/api/cluster/nodes') {
      body = { nodes: [], count: 0, online_count: 0, offline_count: 0 };
    } else if (path === '/api/cluster/queue') {
      counters.queue += 1;
      body = {
        running: true,
        strategy: 'fifo',
        paused: false,
        current_task: null,
        queue_size: 0,
        q0_depth: 0,
        q1_depth: 0,
        q2_depth: 0,
        q0: [],
        q1: [],
        q2: [],
        completed_count: 0,
        max_size: 16,
        preempt_stats: { count: 0 },
      };
    } else if (path === '/api/cluster/pipeline-capacity') {
      counters.capacity += 1;
      body = {
        model_id: 'qwen-1_8b',
        model_type: 'qwen',
        total_layers: 24,
        raw_model_bytes: 1024,
        candidate_node_count: roleName === 'master' ? 1 : 0,
        status: roleName === 'master' ? 'ready' : 'unavailable',
        admitted: roleName === 'master',
        reason_code: roleName === 'master' ? '' : 'single_mode',
        reason: roleName === 'master' ? 'ready' : 'single mode',
        plan_id: roleName === 'master' ? 'plan-test' : '',
        assignments: [],
        control_only_nodes: [],
        participating_node_count: roleName === 'master' ? 1 : 0,
        single_node_full_model_candidates: [],
        prepared_node_count: roleName === 'master' ? 1 : 0,
        ready_node_count: roleName === 'master' ? 1 : 0,
        worker_count: 0,
        computed_at: 0,
      };
    } else if (path === '/api/logs/recent') {
      body = { logs: [], count: 0, matched: 0, buffer_size: 0, buffer_capacity: 100 };
    } else if (path === '/api/health') {
      body = { status: 'ok' };
    }

    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });

  return counters;
}

test('Overview 启动矩阵覆盖 master、client 和未设置角色', async ({ browser }) => {
  for (const roleName of ['master', 'client', 'unset']) {
    const context = await browser.newContext();
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    const counters = installRoleRoutes(page, roleName);

    await page.goto('/#/overview?fixtures=0');
    await expect(page.getByRole('heading', { level: 1, name: '集群概览' })).toBeVisible();
    await expect(page.locator('.pagehead__desc')).not.toContainText('无法读取集群状态');

    if (roleName === 'master') {
      await expect(page.locator('.metric').filter({ hasText: '队列深度' })).not.toContainText('单机模式不适用');
      await page.getByRole('button', { name: '流水线准入' }).click();
      await expect(page.locator('.capacity__facts')).toBeVisible();
      expect(counters.queue).toBeGreaterThan(0);
    } else {
      await expect(page.locator('.metric').filter({ hasText: '队列深度' })).toContainText('单机模式不适用');
      await page.getByRole('button', { name: '流水线准入' }).click();
      await expect(page.getByText('单机模式不使用主节点队列')).toBeVisible();
      await page.waitForTimeout(350);
      expect(counters.queue).toBe(0);
    }

    await page.locator('.topbar .iconbtn').nth(1).click();
    await expect(page.getByRole('heading', { level: 1, name: '集群概览' })).toBeVisible();
    expect(pageErrors).toEqual([]);
    await context.close();
  }
});
