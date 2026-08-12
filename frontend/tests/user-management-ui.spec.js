import { expect, test } from '@playwright/test';

test('owner manages local users, Auth App provisioning, and login sessions', async ({ page }) => {
  const secret = 'JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP';
  const recoveryCodes = Array.from({ length: 10 }, (_, index) => `SAFE-${String(index + 1).padStart(4, '0')}-CODE`);
  const users = [{
    user_id: 'user-owner', username: 'owner', display_name: 'Main owner', role: 'owner',
    status: 'active', aggregate_version: 1, totp_state: 'active', active_session_count: 2,
  }];
  const sessions = [
    { session_id: 'session-current', user_id: 'user-owner', current: true, active: true, created_at: '2026-08-10T08:00:00.000Z', last_seen_at: '2026-08-10T09:00:00.000Z', expires_at: '2026-08-10T20:00:00.000Z', revoked_at: null },
    { session_id: 'session-other', user_id: 'user-owner', current: false, active: true, created_at: '2026-08-09T08:00:00.000Z', last_seen_at: '2026-08-10T08:30:00.000Z', expires_at: '2026-08-10T20:00:00.000Z', revoked_at: null },
  ];
  const bindings = [];

  await page.addInitScript(() => {
    window.sessionStorage.setItem('qlh-auth-session-token', 'owner-browser-token');
  });

  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request();
    const parsed = new URL(request.url());
    const path = parsed.pathname;
    const json = (body, status = 200) => route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
    if (path === '/api/auth/capability') return json({ required: true, enforced: true, mode: 'local_totp', policy_version: 'n1a-v1' });
    if (path === '/api/auth/session') return json({ session_id: 'session-current', expires_at: '2026-08-10T20:00:00.000Z', user: users[0] });
    if (path === '/api/auth/tailscale/local-status') return json({
      local_status: {
        available: true, state: 'ready', reason_code: null, source: 'tailscale_status_json',
        observed_at: '2026-08-10T09:00:00.000Z', requires_confirmation: true,
        candidate: {
          tailnet_id: 'tailnet-main', tailnet_id_source: 'magic_dns_suffix',
          tailnet_display_name: 'Main tailnet', tailscale_user_id: 'ts-user-owner',
          node_id: 'node-owner', hostname: 'owner-pc', dns_name: 'owner-pc.tailnet-main.',
          addresses: ['100.64.0.2', 'fd7a:115c:a1e0::2'],
        },
      },
    });
    if (path === '/api/auth/tailscale/bindings' && request.method() === 'GET') return json({ bindings });
    if (path === '/api/auth/tailscale/bindings' && request.method() === 'POST') {
      const binding = {
        binding_id: `binding-${bindings.length + 1}`, user_id: 'user-owner', tailnet_id: null,
        tailscale_user_id: null, node_id: null, state: 'pending', authorization_method: request.postDataJSON().authorization_method,
        aggregate_version: 1, prepared_at: '2026-08-10T09:00:00.000Z', updated_at: '2026-08-10T09:00:00.000Z',
        confirmed_at: null, revoked_at: null, last_verified_at: null,
      };
      bindings.push(binding);
      return json({ status: 'pending', binding }, 201);
    }
    if (path.match(/^\/api\/auth\/tailscale\/bindings\/[^/]+\/confirm$/) && request.method() === 'POST') {
      const binding = bindings.find((entry) => path.includes(entry.binding_id));
      const payload = request.postDataJSON();
      const previous = bindings.find((entry) => entry.state === 'active');
      if (previous) previous.state = 'revoked';
      Object.assign(binding, {
        tailnet_id: payload.tailnet_id,
        tailscale_user_id: payload.tailscale_user_id,
        node_id: payload.node_id || null,
        state: 'active', confirmed_at: '2026-08-10T09:02:00.000Z', updated_at: '2026-08-10T09:02:00.000Z',
      });
      return json({ status: 'active', binding });
    }
    if (path.match(/^\/api\/auth\/tailscale\/bindings\/[^/]+\/revoke$/) && request.method() === 'POST') {
      const binding = bindings.find((entry) => path.includes(entry.binding_id));
      binding.state = 'revoked';
      binding.revoked_at = '2026-08-10T09:03:00.000Z';
      binding.updated_at = binding.revoked_at;
      return json({ status: 'revoked', binding });
    }
    if (path === '/api/auth/sessions' && request.method() === 'GET') return json({ sessions });
    if (path.startsWith('/api/auth/sessions/') && request.method() === 'DELETE') {
      expect(request.headers().authorization).toBe('Bearer owner-browser-token');
      const session = sessions.find((entry) => path.endsWith(entry.session_id));
      session.active = false;
      session.revoked_at = '2026-08-10T09:05:00.000Z';
      return json({ status: 'revoked', session });
    }
    if (path === '/api/users' && request.method() === 'GET') return json({ users });
    if (path === '/api/users' && request.method() === 'POST') {
      const payload = request.postDataJSON();
      expect(request.headers().authorization).toBe('Bearer owner-browser-token');
      const user = {
        user_id: 'user-member', username: payload.username, display_name: payload.display_name,
        role: payload.role, status: 'active', aggregate_version: 1,
        totp_state: 'none', active_session_count: 0,
      };
      users.push(user);
      return json({ status: 'created', user }, 201);
    }
    if (path === '/api/users/user-member' && request.method() === 'PATCH') {
      const payload = request.postDataJSON();
      Object.assign(users[1], {
        ...(payload.display_name !== undefined ? { display_name: payload.display_name } : {}),
        ...(payload.role ? { role: payload.role } : {}),
        ...(payload.status ? { status: payload.status } : {}),
        aggregate_version: users[1].aggregate_version + 1,
      });
      return json({ status: 'updated', user: users[1] });
    }
    if (path === '/api/auth/users/user-member/totp') return json({
      status: 'pending',
      provisioning: {
        user_id: 'user-member', authenticator_id: 'totp-member', secret,
        otpauth_uri: `otpauth://totp/QLH%3Amember-one?secret=${secret}&issuer=QLH&algorithm=SHA1&digits=6&period=30`,
        qr_payload: `otpauth://totp/QLH%3Amember-one?secret=${secret}&issuer=QLH&algorithm=SHA1&digits=6&period=30`,
        algorithm: 'SHA1', digits: 6, period_seconds: 30,
      },
    }, 201);
    if (path === '/api/auth/totp/verify') {
      expect(request.postDataJSON()).toMatchObject({ user_id: 'user-member', authenticator_id: 'totp-member', code: '123456' });
      users[1].totp_state = 'active';
      return json({ status: 'active', user: users[1], recovery_codes: recoveryCodes });
    }
    if (path === '/api/auth/recovery-codes/rotate') return json({ status: 'rotated', recovery_codes: recoveryCodes });
    if (path === '/api/cluster/my-role') return json({ is_master: true, node_id: 'local' });
    if (path === '/api/cluster/config/distributed-inference') return json({ enabled: false });
    if (path === '/api/status') return json({ model_loaded: false });
    if (path === '/api/user/settings') return json({ settings: {} });
    if (path === '/api/sessions') return json({ sessions: [] });
    return json({});
  });

  await page.goto('/');
  await page.getByRole('button', { name: '账户' }).click();
  await expect(page.getByRole('heading', { name: '账户与安全' })).toBeVisible();
  await expect(page.getByText('2 个活跃')).toBeVisible();
  await expect(page.getByText('尚未绑定')).toBeVisible();
  await page.getByRole('button', { name: '读取本机状态' }).click();
  await expect(page.getByText('Main tailnet')).toBeVisible();
  await expect(page.getByText(/fd7a:115c:a1e0::2/)).toBeVisible();
  await page.screenshot({ path: '../build/auth-tailscale-local-status.png', fullPage: true });
  await page.getByRole('button', { name: '发起绑定' }).click();
  await expect(page.getByLabel('tailnet ID')).toHaveValue('tailnet-main');
  await expect(page.getByLabel('Tailscale 用户 ID')).toHaveValue('ts-user-owner');
  await expect(page.getByLabel('节点 ID')).toHaveValue('node-owner');
  await page.getByRole('button', { name: '确认绑定' }).click();
  await expect(page.getByText('当前已绑定')).toBeVisible();
  await expect(page.getByText('tailnet-main', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '官方 CLI', exact: true }).click();
  await page.getByRole('button', { name: '发起换网' }).click();
  await page.getByLabel('tailnet ID').fill('tailnet-new');
  await page.getByLabel('Tailscale 用户 ID').fill('ts-user-owner-new');
  await page.getByRole('button', { name: '确认绑定' }).click();
  await expect(page.getByText('tailnet-new', { exact: true })).toBeVisible();
  await expect(page.getByText(/查看历史绑定/)).toBeVisible();
  await page.screenshot({ path: '../build/auth-tailscale-binding.png', fullPage: true });
  await page.getByRole('button', { name: '撤销当前绑定' }).click();
  await expect(page.getByText('尚未绑定')).toBeVisible();
  const otherSession = page.locator('.session-row').filter({ hasText: '其他登录' });
  await otherSession.getByRole('button', { name: '撤销' }).click();
  await expect(otherSession.getByText('已结束')).toBeVisible();

  await page.getByLabel('用户名').fill('member-one');
  await page.getByLabel('显示名称').fill('Member One');
  await page.getByRole('button', { name: '创建用户' }).click();
  const memberRow = page.locator('.user-row').filter({ hasText: 'member-one' });
  await expect(memberRow).toBeVisible();
  await memberRow.getByRole('button', { name: '配置 Auth App' }).click();
  await expect(page.getByAltText('Auth App 配置二维码')).toBeVisible();
  await page.getByRole('button', { name: '字符串', exact: true }).click();
  await expect(page.getByText(secret, { exact: true })).toBeVisible();
  await page.getByLabel('输入 Auth App 验证码').fill('123456');
  await page.getByRole('button', { name: '确认启用' }).click();
  await expect(page.getByText(recoveryCodes[0])).toBeVisible();
  await page.screenshot({ path: '../build/auth-user-management.png', fullPage: true });

  await page.getByRole('button', { name: '关闭' }).click();
  await memberRow.getByRole('button', { name: '编辑' }).click();
  await page.getByLabel('显示名称').last().fill('Research Member');
  await page.getByRole('button', { name: '保存' }).click();
  await expect(memberRow.getByText('Research Member')).toBeVisible();
  await memberRow.getByRole('button', { name: '暂停' }).click();
  await expect(memberRow.getByText('暂停', { exact: true })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('.account-workspace')).toBeVisible();
  await page.screenshot({ path: '../build/auth-user-management-mobile.png', fullPage: true });
});
