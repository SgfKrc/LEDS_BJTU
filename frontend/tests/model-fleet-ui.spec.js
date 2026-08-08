import { expect, test } from '@playwright/test';

const artifact = {
  artifact_id: `sha256:${'a'.repeat(64)}`,
  reference: { namespace: 'user', name: 'deepseek-q4', tag: 'latest' },
  format: 'gguf',
  engine: 'llama_cpp',
  family: 'deepseek',
  quantization: 'Q4_K_M',
  requirements: { runtime_profile: 'llm-cpu-v1' },
  storage: { file_count: 1, total_bytes: 4_294_967_296 },
  runtime_check: {
    status: 'ready', node_id: 'local', checked_at: '2026-08-08T12:00:00.000Z',
  },
  runnable: true,
};

test('local model fleet workspace manages artifacts, imports, pulls, and user proxy', async ({ page }) => {
  let savedProxy = null;
  let importPayload = null;
  let pullPayload = null;

  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const json = (body, status = 200) => route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });

    if (path === '/api/models/artifacts') {
      return json({
        node_id: 'local', artifacts: [artifact],
        summary: { total: 1, ready: 1, stale: 0, attention: 0, unchecked: 0, total_bytes: artifact.storage.total_bytes },
      });
    }
    if (path === '/api/models/pull' && method === 'GET') return json({ jobs: [] });
    if (path === '/api/models/pull' && method === 'POST') {
      pullPayload = request.postDataJSON();
      return json({ status: 'created', job_id: 'pull_fixture', state: 'queued' }, 202);
    }
    if (path === '/api/models/imports' && method === 'POST') {
      importPayload = request.postDataJSON();
      return json({ status: 'imported', runnable: true, report: { runtime_check: { status: 'ready' } } }, 201);
    }
    if (path === '/api/models/network') {
      return json({
        proxy: savedProxy
          ? { configured: true, source: 'user', endpoint: savedProxy }
          : { configured: false, source: 'direct', endpoint: null },
        user_proxy: savedProxy ? { schema_version: 1, url: savedProxy, updated_at: '2026-08-08T12:00:00.000Z' } : null,
      });
    }
    if (path === '/api/models/network/proxy' && method === 'PUT') {
      savedProxy = request.postDataJSON().url;
      return json({ status: 'saved', user_proxy: { url: savedProxy } });
    }
    if (path === '/api/models/network/proxy' && method === 'DELETE') {
      savedProxy = null;
      return json({ status: 'cleared' });
    }
    if (path === '/api/models/runtime-checks/retry') return json({ runnable: true, runtime_check: { status: 'ready' } });
    if (path === '/api/models/runtime-checks' && method === 'DELETE') return json({ status: 'invalidated', count: 1 });
    if (path === '/api/cluster/my-role') return json({ is_master: true, node_id: 'local' });
    if (path === '/api/cluster/distributed-inference/config') return json({ enabled: false });
    if (path === '/api/status') return json({ model_loaded: false });
    if (path === '/api/user/settings') return json({ settings: {} });
    if (path === '/api/models' && method === 'GET') return json({ models: [], active_model_id: null });
    if (path === '/api/logs/recent') return json({ logs: [], buffer_capacity: 200, total_seen: 0 });
    if (path === '/api/logs') return json({ files: [] });
    if (path === '/api/logs/stats') return json({ files_count: 0, total_size: 0 });
    if (path === '/api/sessions') return json({ sessions: [], active_session_id: null, total: 0 });
    return json({});
  });

  await page.goto('/');
  await page.locator('button[title="系统设置"]').first().click();
  const workspace = page.getByTestId('model-fleet-workspace');
  await expect(workspace).toBeVisible();
  await expect(workspace.getByText('user/deepseek-q4')).toBeVisible();
  await expect(workspace.getByText('可运行', { exact: true })).toBeVisible();
  await expect(workspace.getByText('4 GB')).toBeVisible();
  await workspace.scrollIntoViewIfNeeded();
  await page.screenshot({ path: '../build/model-fleet-workspace.png' });

  await workspace.getByRole('tab', { name: '导入与拉取' }).click();
  await workspace.getByTestId('fleet-import-path').fill('C:/models/deepseek.gguf');
  await workspace.getByTestId('fleet-import-name').fill('deepseek-local');
  await workspace.getByRole('button', { name: '导入并检查' }).click();
  await expect.poll(() => importPayload).toEqual({
    source_path: 'C:/models/deepseek.gguf', namespace: 'user', name: 'deepseek-local', tag: 'latest',
  });

  await workspace.getByRole('tab', { name: '导入与拉取' }).click();
  await workspace.getByTestId('fleet-pull-repo').fill('Qwen/Qwen2.5-7B-Instruct-GGUF');
  await workspace.getByRole('button', { name: '开始拉取' }).click();
  await expect.poll(() => pullPayload?.source?.repo_id).toBe('Qwen/Qwen2.5-7B-Instruct-GGUF');

  await workspace.getByRole('tab', { name: '网络' }).click();
  await workspace.getByTestId('fleet-proxy-url').fill('http://127.0.0.1:7897');
  await workspace.getByRole('button', { name: '保存代理' }).click();
  await expect.poll(() => savedProxy).toBe('http://127.0.0.1:7897');
  await workspace.getByRole('button', { name: '清除代理' }).click();
  await expect.poll(() => savedProxy).toBeNull();
});
