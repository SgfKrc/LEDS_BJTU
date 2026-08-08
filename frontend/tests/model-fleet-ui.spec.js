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
  let resolvePayload = null;
  let credentialSecret = null;
  let credentials = [];
  let acceptances = [];
  let sources = [{
    schema_version: 1, source_id: 'hf-official', name: 'Hugging Face',
    provider: 'huggingface', endpoint: 'https://huggingface.co', credential_ref: null,
    priority: 10, enabled: true, builtin: true,
  }];

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
    if (path === '/api/models/sources' && method === 'GET') return json({ sources });
    if (path === '/api/models/sources/reset' && method === 'POST') return json({ status: 'reset', sources });
    if (path.startsWith('/api/models/sources/') && method === 'PUT') {
      const sourceId = decodeURIComponent(path.slice('/api/models/sources/'.length));
      const payload = request.postDataJSON();
      const source = { schema_version: 1, source_id: sourceId, ...payload, builtin: false };
      sources = [...sources.filter((entry) => entry.source_id !== sourceId), source]
        .sort((a, b) => a.priority - b.priority);
      return json({ status: 'saved', source });
    }
    if (path.startsWith('/api/models/sources/') && method === 'DELETE') {
      const sourceId = decodeURIComponent(path.slice('/api/models/sources/'.length));
      sources = sources.filter((entry) => entry.source_id !== sourceId);
      return json({ status: 'deleted', source_id: sourceId });
    }
    if (path === '/api/models/resolve' && method === 'POST') {
      resolvePayload = request.postDataJSON();
      const selected = sources.find((source) => source.source_id === resolvePayload.source_id);
      return json({
        schema_version: 1, status: 'ready', source: selected,
        access: { gated: false, credential_required: false, credential_available: false, acceptance_required: false },
        repo_id: resolvePayload.repo_id, requested_revision: resolvePayload.requested_revision,
        resolved_revision: 'b'.repeat(40), files: [{ path: 'model.gguf', size: 1024, sha256: 'c'.repeat(64) }],
        total_bytes: 1024, existing_bytes: 0, disk_required_bytes: 1024, disk_available_bytes: 10_000,
      });
    }
    if (path === '/api/models/credentials' && method === 'GET') return json({ credentials });
    if (path.startsWith('/api/models/credentials/') && method === 'PUT') {
      const credentialId = decodeURIComponent(path.slice('/api/models/credentials/'.length));
      credentialSecret = request.postDataJSON().secret;
      const credential = {
        credential_ref: `os:qlh/${credentialId}`, exists: true,
        protection: 'windows-dpapi-current-user', updated_at: '2026-08-09T12:00:00.000Z',
      };
      credentials = [...credentials.filter((entry) => entry.credential_ref !== credential.credential_ref), credential];
      return json({ status: 'saved', credential });
    }
    if (path.startsWith('/api/models/credentials/') && method === 'DELETE') {
      const credentialId = decodeURIComponent(path.slice('/api/models/credentials/'.length));
      credentials = credentials.filter((entry) => entry.credential_ref !== `os:qlh/${credentialId}`);
      return json({ status: 'deleted', credential_ref: `os:qlh/${credentialId}` });
    }
    if (path === '/api/models/licenses/acceptances' && method === 'GET') return json({ acceptances });
    if (path === '/api/models/licenses/acceptances' && method === 'POST') {
      const payload = request.postDataJSON();
      const acceptance = {
        schema_version: 1, repo_id: payload.repo_id, license_id: payload.license_id,
        accepted_at: '2026-08-09T12:00:00.000Z', accepted_by: 'local_user',
      };
      acceptances = [...acceptances, acceptance];
      return json({ status: 'accepted', acceptance });
    }
    if (path === '/api/models/licenses/acceptances' && method === 'DELETE') {
      const payload = request.postDataJSON();
      acceptances = acceptances.filter((entry) => entry.repo_id !== payload.repo_id || entry.license_id !== payload.license_id);
      return json({ status: 'revoked' });
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
  await expect(workspace.getByRole('tab', { name: '工件' })).toHaveAttribute('aria-selected', 'true');

  await workspace.getByRole('tab', { name: '导入与拉取' }).click();
  await workspace.getByTestId('fleet-pull-repo').fill('Qwen/Qwen2.5-7B-Instruct-GGUF');
  await workspace.getByRole('button', { name: '解析检查' }).click();
  await expect(workspace.getByTestId('fleet-preflight')).toContainText('可以拉取');
  await expect.poll(() => resolvePayload?.source_id).toBe('hf-official');
  await workspace.getByRole('button', { name: '确认拉取' }).click();
  await expect.poll(() => pullPayload?.source?.repo_id).toBe('Qwen/Qwen2.5-7B-Instruct-GGUF');
  expect(pullPayload.source_id).toBe('hf-official');
  await expect(workspace.getByRole('tab', { name: '任务' })).toHaveAttribute('aria-selected', 'true');

  await workspace.getByRole('tab', { name: '来源与凭据' }).click();
  await workspace.getByTestId('fleet-credential-id').fill('hf-main');
  await workspace.getByTestId('fleet-credential-secret').fill('hf_fixture_secret');
  await workspace.getByRole('button', { name: '保存凭据' }).click();
  await expect.poll(() => credentialSecret).toBe('hf_fixture_secret');
  await expect(workspace.locator('.fleet-credential-row strong').filter({ hasText: 'os:qlh/hf-main' })).toBeVisible();
  await expect(workspace.getByTestId('fleet-credential-secret')).toHaveValue('');

  await workspace.getByTestId('fleet-source-id').fill('team-mirror');
  await workspace.getByTestId('fleet-source-name').fill('Team Mirror');
  await workspace.getByTestId('fleet-source-endpoint').fill('https://mirror.example');
  await workspace.getByLabel('凭据', { exact: true }).selectOption('os:qlh/hf-main');
  await workspace.getByRole('button', { name: '保存来源' }).click();
  await expect.poll(() => sources.some((source) => source.source_id === 'team-mirror' && source.credential_ref === 'os:qlh/hf-main')).toBe(true);

  await workspace.getByTestId('fleet-license-repo').fill('org/gated-model');
  await workspace.getByTestId('fleet-license-id').fill('llama3');
  await workspace.getByRole('button', { name: '接受许可' }).click();
  await expect.poll(() => acceptances.some((entry) => entry.repo_id === 'org/gated-model')).toBe(true);
  await workspace.scrollIntoViewIfNeeded();
  await page.screenshot({ path: '../build/model-fleet-sources-credentials.png' });

  await workspace.getByRole('tab', { name: '网络' }).click();
  await workspace.getByTestId('fleet-proxy-url').fill('http://127.0.0.1:7897');
  await workspace.getByRole('button', { name: '保存代理' }).click();
  await expect.poll(() => savedProxy).toBe('http://127.0.0.1:7897');
  await workspace.getByRole('button', { name: '清除代理' }).click();
  await expect.poll(() => savedProxy).toBeNull();

  await page.setViewportSize({ width: 390, height: 844 });
  await workspace.getByRole('tab', { name: '来源与凭据' }).click();
  await workspace.scrollIntoViewIfNeeded();
  await expect(workspace.getByTestId('fleet-source-id')).toBeVisible();
  await expect(workspace.getByTestId('fleet-credential-id')).toBeVisible();
  await page.screenshot({ path: '../build/model-fleet-sources-credentials-mobile.png' });
});
