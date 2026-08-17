import { expect, test } from '@playwright/test';

import { installWorkspaceApi } from './fixtures/workspace-api.js';

test('UX-01 provides a line-icon navigation rail on desktop and narrow screens', async ({ page }, testInfo) => {
  await installWorkspaceApi(page);
  await page.goto('/');

  const navigation = page.getByRole('navigation', { name: '主导航' });
  await expect(navigation).toBeVisible();
  await expect(navigation.getByRole('button', { name: '对话' })).toBeVisible();
  await expect(navigation.locator('svg').first()).toBeVisible();
  await expect(page.locator('.sidebar-system-status')).toContainText('STANDBY');
  await page.screenshot({ path: testInfo.outputPath('ux-01-desktop.png'), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('.sidebar')).toHaveClass(/collapsed/);
  await expect(page.locator('.sidebar')).toHaveCSS('width', '56px');
  await expect(page.getByRole('button', { name: '对话' }).last()).toBeVisible();
  const mainBox = await page.locator('.main-area').boundingBox();
  expect(mainBox?.width || 0).toBeGreaterThan(280);
  await page.screenshot({ path: testInfo.outputPath('ux-01-mobile.png'), fullPage: true });
});

test('UX-02 groups administration controls into focused workspaces', async ({ page }, testInfo) => {
  await installWorkspaceApi(page);
  await page.goto('/');

  await page.getByRole('button', { name: '后台管理' }).click();
  const workspaces = page.getByRole('tablist', { name: '后台管理工作区' });
  await expect(workspaces).toBeVisible();
  await expect(page.getByRole('tab', { name: '概览' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('heading', { name: '集群概览' })).toBeVisible();

  await page.getByRole('tab', { name: '节点' }).click();
  await expect(page.getByRole('heading', { name: '手动注册从节点' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '集群概览' })).toBeHidden();

  await page.getByRole('tab', { name: '运行时' }).click();
  await expect(page.getByRole('heading', { name: '分布式推理' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '手动注册从节点' })).toBeHidden();
  await page.screenshot({ path: testInfo.outputPath('ux-02-runtime.png'), fullPage: true });
});

test('UX-02 keeps the dark workspace neutral instead of using a green primary palette', async ({ page }, testInfo) => {
  await page.addInitScript(() => localStorage.setItem('qlh-theme', 'dark'));
  await installWorkspaceApi(page);
  await page.goto('/');

  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('.app-layout')).toHaveCSS('background-color', 'rgb(11, 11, 12)');
  await page.getByRole('button', { name: '后台管理' }).click();
  await page.screenshot({ path: testInfo.outputPath('ux-02-dark-overview.png'), fullPage: true });
});

test('UX-03 separates settings and image asset controls from their primary workflows', async ({ page }, testInfo) => {
  await installWorkspaceApi(page);
  await page.goto('/');

  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();
  await expect(page.getByRole('tablist', { name: '系统设置工作区' })).toBeVisible();
  await page.getByRole('tab', { name: '模型与设备' }).click();
  await expect(page.getByTestId('model-fleet-workspace')).toBeVisible();
  await page.getByRole('tab', { name: '日志' }).click();
  const logTabs = page.getByRole('tablist', { name: '日志工作区' });
  await expect(logTabs).toBeVisible();
  await logTabs.getByRole('tab', { name: '文件' }).click();
  await expect(logTabs.getByRole('tab', { name: '文件' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('heading', { name: '日志文件' })).toBeVisible();
  await expect(page.locator('.log-stats-bar')).not.toContainText('NaN');
  await page.screenshot({ path: testInfo.outputPath('ux-03-settings-logs.png'), fullPage: true });

  await page.getByLabel('关闭系统设置').click();
  await page.getByRole('button', { name: '图像工作区' }).click();
  const imageTabs = page.getByRole('tablist', { name: '图像工作区' });
  await expect(imageTabs).toBeVisible();
  await imageTabs.getByRole('tab', { name: /模型资产/ }).click();
  await expect(page.getByTestId('diffusion-model-path')).toBeVisible();
  await imageTabs.getByRole('tab', { name: '创作' }).click();
  await expect(page.getByRole('button', { name: '管理模型' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('ux-03-diffusion-generate.png'), fullPage: true });
});

test('UX-03R reconciles registered models with local fleet assets and edits custom registrations', async ({ page }) => {
  let models = [
    {
      model_id: 'future-model', name: 'Future Model', is_available: false,
      is_builtin: true, is_experimental: true, model_type: 'safetensors',
      recommended_vram_gb: 16, max_context: 32768,
    },
    {
      model_id: 'custom-ready', name: 'Custom Ready', is_available: true,
      is_builtin: false, is_experimental: true, model_type: 'gguf',
      gguf_path: 'models/custom-ready.gguf', recommended_vram_gb: 4,
      max_context: 4096, description: '可编辑的本地注册模型',
    },
  ];
  let savedModel = null;
  const artifact = {
    artifact_id: `sha256:${'a'.repeat(64)}`,
    reference: { namespace: 'local', name: 'qwen3-4b', tag: 'latest' },
    source: { repo_id: 'Qwen/Qwen3-4B' },
    format: 'safetensors', engine: 'qwen3_sidecar', family: 'qwen3',
    context_length: 32768, runtime_check: { status: 'ready' }, runnable: true,
    storage: { file_count: 3, total_bytes: 8_000_000_000 },
  };

  await installWorkspaceApi(page, {
    'GET /api/models': () => ({ body: { models, active_model_id: null } }),
    'GET /api/models/artifacts': {
      body: {
        node_id: 'local', artifacts: [artifact],
        summary: { total: 1, ready: 1, stale: 0, attention: 0, unchecked: 0, total_bytes: artifact.storage.total_bytes },
      },
    },
    'POST /api/models/registry': (request) => {
      savedModel = request.json;
      models = models.map((model) => (
        model.model_id === request.json.model_id ? { ...model, ...request.json } : model
      ));
      return { body: { status: 'registered', model_id: request.json.model_id } };
    },
  });
  await page.goto('/');
  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();
  await page.getByRole('tab', { name: '模型与设备' }).click();

  const runtimeControl = page.getByLabel('Sidecar 运行时控制面');
  await expect(runtimeControl).toContainText('Qwen3 Sidecar');
  await expect(runtimeControl).toContainText('Gemma 4 Pipeline Sidecar');
  await expect(runtimeControl).toContainText('需要任务合同');

  const cards = page.locator('.experimental-model-card');
  await expect(cards).toHaveCount(3);
  await expect(cards.nth(0)).toContainText('Custom Ready');
  await expect(cards.nth(1)).toContainText('Qwen/Qwen3-4B');
  await expect(cards.nth(1)).toContainText('工件可运行');
  await expect(cards.nth(1).getByRole('button', { name: '由任务路由加载' })).toBeDisabled();
  await expect(cards.nth(2)).toContainText('Future Model');
  await expect(cards.nth(2)).toContainText('未下载');

  await cards.nth(0).getByRole('button', { name: '编辑' }).click();
  await expect(page.getByRole('heading', { name: '编辑自定义模型' })).toBeVisible();
  const modelId = page.locator('.add-model-form input').first();
  await expect(modelId).toHaveValue('custom-ready');
  await expect(modelId).toBeDisabled();
  await page.locator('.add-model-form input').nth(1).fill('Custom Ready Updated');
  await page.getByRole('button', { name: '保存修改' }).click();
  await expect.poll(() => savedModel?.name).toBe('Custom Ready Updated');
  expect(savedModel?.model_id).toBe('custom-ready');
});

test('UX-03R2 discovers root models assets without exposing legacy model switching', async ({ page }) => {
  const localAsset = {
    model_id: 'qwen3-5-9b',
    name: 'Qwen/Qwen3.5-9B',
    huggingface_id: 'Qwen/Qwen3.5-9B',
    model_type: 'both',
    available_formats: ['safetensors', 'gguf'],
    max_context: 262144,
    runtime_profile: 'qwen3_sidecar',
    runtime_status: 'inventory_only',
    runtime_action: 'qwen3_preflight',
    runtime_hint: '本地资产已发现；当前仅安全登记，尚无可执行的 Qwen3 Sidecar 加载控制面，不使用旧单机加载器。',
  };
  await installWorkspaceApi(page, {
    'GET /api/models': {
      body: {
        models: [{
          model_id: 'future-model', name: 'Future Model', is_available: false,
          is_builtin: true, is_experimental: true, model_type: 'safetensors',
        }],
        active_model_id: null,
      },
    },
    'GET /api/models/local-assets': {
      body: { assets: [localAsset], summary: { total: 1, total_bytes: 12_000_000_000 } },
    },
    'POST /api/models/local-assets/qwen3-5-9b/preflight': {
      body: {
        model_id: 'qwen3-5-9b', gate_passed: true,
        status: 'ready_for_qwen3_smoke', read_only: true, starts_sidecar: false,
      },
    },
  });
  await page.goto('/');
  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();
  await page.getByRole('tab', { name: '模型与设备' }).click();

  const cards = page.locator('.experimental-model-card');
  await expect(cards).toHaveCount(2);
  await expect(cards.nth(0)).toContainText('Qwen/Qwen3.5-9B');
  await expect(cards.nth(0)).toContainText('本地资产');
  await expect(cards.nth(0)).toContainText('safetensors + gguf');
  await cards.nth(0).getByRole('button', { name: 'Sidecar 预检' }).click();
  await expect(cards.nth(0)).toContainText('Sidecar 预检通过');
  await expect(cards.nth(0)).toContainText('尚未启动模型加载');
  await expect(cards.nth(1)).toContainText('Future Model');
  await expect(cards.nth(1)).toContainText('未下载');
});

test('MODEL-RUNTIME-UI-03 exposes contract evidence and a safe recovery hint', async ({ page }) => {
  const contractId = 'c'.repeat(64);
  await installWorkspaceApi(page, {
    'GET /api/models/local-assets': {
      body: {
        assets: [{
          model_id: 'qwen3-4b', name: 'Qwen/Qwen3-4B', model_type: 'safetensors',
          model_path: 'models/qwen3-4b', available_formats: ['safetensors'],
          runtime_profile: 'qwen3_sidecar', runtime_status: 'inventory_only',
        }],
        summary: { total: 1, total_bytes: 8_000_000_000 },
      },
    },
    'GET /api/cluster/model-runtime/contracts': {
      body: {
        schema_version: 1,
        contracts: [{
          contract_id: contractId, profile: 'qwen3_sidecar', model_id: 'qwen3-4b',
          plan_id: 'plan-audit-1', generation: 3, segment_count: 2,
          execution: {
            event_count: 3,
            recovery_action: 'retry_prepare',
            recent_events: [{
              at: 1_785_000_000, action: 'prepare_failed', phase: 'aborted', reason_code: 'runtimeerror',
            }],
          },
        }],
      },
    },
  });
  await page.goto('/');
  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();
  await page.getByRole('tab', { name: '模型与设备' }).click();

  const runtimeControl = page.getByLabel('Sidecar 运行时控制面');
  await expect(runtimeControl).toContainText('可重新检查后准备');
  await expect(runtimeControl).toContainText('准备被拒绝');
  await expect(runtimeControl).toContainText('runtimeerror');
  await expect(page.locator('.experimental-model-card').first()).toContainText('准备被拒绝');
  await expect(runtimeControl).not.toContainText('private');
});

test('UX-03R2 keeps execution mode text legible while hovered', async ({ page }) => {
  await installWorkspaceApi(page);
  await page.goto('/');
  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();

  const standard = page.getByRole('group', { name: '对话执行模式' }).getByRole('button', { name: '标准' });
  await standard.hover();
  const colors = await standard.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return { background: style.backgroundColor, color: style.color };
  });
  expect(colors.color).not.toBe(colors.background);
});

test('UX-03R3 keeps the robot account entry interactive in legacy mode', async ({ page }) => {
  await installWorkspaceApi(page);
  await page.goto('/');

  await page.getByRole('button', { name: '打开账户与安全' }).click();
  await expect(page.getByRole('heading', { name: '账户与安全' })).toBeVisible();
  await expect(page.getByText('认证服务尚未启用')).toBeVisible();
  await page.getByRole('button', { name: '返回对话' }).click();
  await expect(page.getByRole('button', { name: '发送消息' })).toBeVisible();
});

test('UX-03B keeps chat tools compact and partitions account management workflows', async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    sessionStorage.setItem('qlh-auth-session-token', 'ux-03b-session');
  });
  await installWorkspaceApi(page, {
    'GET /api/auth/capability': { body: { required: true, mode: 'local_totp', service: 'control-svc' } },
    'GET /api/auth/session': {
      body: {
        session_id: 'current-session', expires_at: '2026-08-20T00:00:00.000Z',
        user: { user_id: 'owner-1', username: 'owner', display_name: 'Owner', role: 'owner' },
      },
    },
    'GET /api/auth/sessions': {
      body: { sessions: [{ session_id: 'current-session', current: true, active: true, created_at: '2026-08-17T00:00:00.000Z', last_seen_at: '2026-08-17T00:00:00.000Z' }] },
    },
    'GET /api/users': {
      body: { users: [{ user_id: 'owner-1', username: 'owner', display_name: 'Owner', role: 'owner', status: 'active', totp_state: 'active', active_session_count: 1, aggregate_version: 1 }] },
    },
    'GET /api/auth/tailscale/bindings': { body: { bindings: [] } },
  });
  await page.goto('/');

  await expect(page.getByRole('heading', { name: '对话' })).toBeVisible();
  await expect(page.getByRole('button', { name: '系统设置' }).first().locator('svg')).toBeVisible();
  await expect(page.getByRole('button', { name: '发送消息' }).locator('svg')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('ux-03b-chat.png'), fullPage: true });

  await page.getByRole('button', { name: '打开账户与安全' }).click();
  const accountTabs = page.getByRole('tablist', { name: '账户与安全工作区' });
  await expect(accountTabs).toBeVisible();
  await expect(accountTabs.getByRole('tab', { name: '安全与会话' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('heading', { name: '当前会话' })).toBeVisible();

  await accountTabs.getByRole('tab', { name: '组网绑定' }).click();
  await expect(page.getByRole('heading', { name: 'Tailscale 组网绑定' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '当前会话' })).toBeHidden();

  await accountTabs.getByRole('tab', { name: '用户管理' }).click();
  await expect(page.getByRole('heading', { name: '本地主节点用户' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Tailscale 组网绑定' })).toBeHidden();
  await page.screenshot({ path: testInfo.outputPath('ux-03b-account-users.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(accountTabs).toBeVisible();
  await expect(page.getByRole('button', { name: '创建用户' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('ux-03b-account-users-mobile.png'), fullPage: true });
});

test('UX-06 keeps desktop and mobile workspaces keyboard reachable', async ({ page }, testInfo) => {
  await installWorkspaceApi(page);
  await page.goto('/');

  const navigation = page.locator('.sidebar');
  for (const viewport of [
    { name: 'desktop', width: 1280, height: 960 },
    { name: 'mobile', width: 390, height: 844 },
  ]) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    if (viewport.name === 'mobile') {
      await expect(navigation).toHaveClass(/collapsed/);
      await expect(navigation).toHaveCSS('width', '56px');
    }
    await expect(page.locator('.app-layout')).toBeVisible();

    const main = await page.locator('.main-area').boundingBox();
    expect(main).not.toBeNull();
    expect(main.x).toBeGreaterThanOrEqual(0);
    expect(main.x + main.width).toBeLessThanOrEqual(viewport.width + 1);
    const documentWidth = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(documentWidth.scrollWidth).toBeLessThanOrEqual(documentWidth.clientWidth + 1);

    const primaryAction = navigation.locator('button:visible').first();
    await expect(primaryAction).toBeVisible();
    const semantics = await primaryAction.evaluate((element) => ({
      tabIndex: element.tabIndex,
      label: element.getAttribute('aria-label'),
      disabled: element.matches(':disabled'),
    }));
    expect(semantics.tabIndex).toBeGreaterThanOrEqual(0);
    expect(semantics.label).not.toBeNull();
    expect(semantics.disabled).toBe(false);
    await page.screenshot({ path: testInfo.outputPath(`ux-06-${viewport.name}.png`), fullPage: true });
  }
});

test('UX-07 keeps chat avatars local and upload tools legible in light mode', async ({ page }, testInfo) => {
  let modelReady = false;
  await page.addInitScript(() => localStorage.setItem('qlh-theme', 'light'));
  await installWorkspaceApi(page, {
    'GET /api/status': () => ({
      body: modelReady
        ? {
          model_loaded: false,
          current_quant: null,
          external: { enabled: true, reachable: true, model: 'fixture', data_scope: 'opt_in' },
        }
        : { model_loaded: false, current_quant: null, gpu: {}, kv_cache: {} },
    }),
    'GET /api/presets': { body: { presets: [] } },
    'POST /api/sessions': {
      status: 201,
      body: { id: 'avatar-session', title: 'Avatar test', message_count: 0, active: true },
    },
    'GET /api/sessions/avatar-session': {
      body: { session_id: 'avatar-session', title: 'Avatar test', messages: [] },
    },
    'POST /api/chat': {
      body: { role: 'assistant', content: '头像主题回归完成。', metrics: {}, followups: [] },
    },
  });
  await page.goto('/');

  const fileUpload = page.getByRole('button', { name: '上传文本文件' });
  await expect(fileUpload).toBeDisabled();
  const lightUploadStyle = await fileUpload.evaluate((element) => {
    const style = window.getComputedStyle(element);
    const icon = element.querySelector('svg');
    const iconStyle = icon ? window.getComputedStyle(icon) : null;
    return {
      color: style.color,
      background: style.backgroundColor,
      opacity: style.opacity,
      iconColor: iconStyle?.color,
      iconStroke: iconStyle?.stroke,
      iconOpacity: iconStyle?.opacity,
    };
  });
  expect(lightUploadStyle.opacity).toBe('1');
  expect(lightUploadStyle.color).not.toBe(lightUploadStyle.background);
  expect(lightUploadStyle.iconColor).toBe(lightUploadStyle.color);
  expect(lightUploadStyle.iconStroke).toBe(lightUploadStyle.color);
  expect(lightUploadStyle.iconOpacity).toBe('1');

  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();
  await page.getByRole('tab', { name: '外观' }).click();
  const avatarInput = page.getByTestId('chat-user-avatar-input');
  await avatarInput.setInputFiles({
    name: 'owner.png',
    mimeType: 'image/png',
    buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL+XwAAAABJRU5ErkJggg==', 'base64'),
  });
  const avatarPreview = page.getByTestId('user-avatar-preview');
  await expect(avatarPreview.getByAltText('当前对话头像')).toHaveAttribute('src', /^data:image\/png;base64,/);
  await expect.poll(() => page.evaluate(() => localStorage.getItem('qlh-user-avatar-v1'))).toMatch(/^data:image\/png;base64,/);
  await page.screenshot({ path: testInfo.outputPath('ux-07-light-avatar.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }))).toEqual({ scrollWidth: 390, clientWidth: 390 });
  await page.screenshot({ path: testInfo.outputPath('ux-07-light-avatar-mobile.png'), fullPage: true });
  await page.setViewportSize({ width: 1280, height: 960 });
  await page.getByLabel('关闭系统设置').click();
  await page.screenshot({ path: testInfo.outputPath('ux-07-light-workspace.png'), fullPage: true });

  modelReady = true;
  await page.reload();
  const chatInput = page.locator('textarea[placeholder*="输入消息"]');
  await expect(chatInput).toBeEnabled();
  await chatInput.fill('请验证头像');
  await page.getByRole('button', { name: '发送消息' }).click();
  await expect(page.getByText('头像主题回归完成。')).toBeVisible();
  await expect(page.locator('.message.user .message-avatar-image')).toHaveAttribute('src', /^data:image\/png;base64,/);
  await expect.poll(() => page.locator('.message.assistant .message-avatar-image').getAttribute('src')).toContain('qlh-light.jpg');

  await page.getByRole('navigation', { name: '主导航' }).getByRole('button', { name: '系统设置' }).click();
  await page.getByRole('tab', { name: '外观' }).click();
  await page.getByRole('group', { name: '主题模式' }).getByRole('button', { name: '深色' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect.poll(() => page.locator('.message.assistant .message-avatar-image').getAttribute('src')).toContain('qlh.jpg');
  await expect(page.locator('.message.assistant .message-avatar-image')).not.toHaveAttribute('src', /qlh-light\.jpg/);
  await page.screenshot({ path: testInfo.outputPath('ux-07-dark-chat.png'), fullPage: true });

  await page.getByRole('button', { name: '默认' }).click();
  await expect(avatarPreview.getByAltText('当前对话头像')).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => localStorage.getItem('qlh-user-avatar-v1'))).toBeNull();
});
