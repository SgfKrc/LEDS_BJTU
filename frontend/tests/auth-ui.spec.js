import { expect, test } from '@playwright/test';

test('first owner configures Auth App, receives recovery codes, enters, and logs out', async ({ page }) => {
  const secret = 'JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP';
  const recoveryCodes = Array.from({ length: 10 }, (_, index) => `RC${String(index + 1).padStart(2, '0')}-ABCD-EFGH`);
  let loggedIn = false;

  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const json = (body, status = 200) => route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
    if (path === '/api/auth/capability') {
      return json({ required: true, mode: 'local_totp', service: 'control-svc' });
    }
    if (path === '/api/auth/bootstrap') {
      return json({
        status: 'pending',
        provisioning: {
          user_id: 'user-owner',
          authenticator_id: 'totp-owner',
          secret,
          otpauth_uri: `otpauth://totp/QLH%3Aowner?secret=${secret}&issuer=QLH&algorithm=SHA1&digits=6&period=30`,
          qr_payload: `otpauth://totp/QLH%3Aowner?secret=${secret}&issuer=QLH&algorithm=SHA1&digits=6&period=30`,
          algorithm: 'SHA1',
          digits: 6,
          period_seconds: 30,
        },
      }, 201);
    }
    if (path === '/api/auth/totp/verify') {
      expect(request.postDataJSON()).toMatchObject({
        user_id: 'user-owner', authenticator_id: 'totp-owner', code: '123456',
      });
      return json({ status: 'active', user: { username: 'owner' }, recovery_codes: recoveryCodes });
    }
    if (path === '/api/auth/login') {
      loggedIn = true;
      return json({
        access_token: 'browser-session-token',
        session_id: 'session-owner',
        expires_at: '2026-08-11T00:00:00.000Z',
        user: { user_id: 'user-owner', username: 'owner', display_name: 'Main owner', role: 'owner' },
      });
    }
    if (path === '/api/auth/logout') {
      expect(request.headers().authorization).toBe('Bearer browser-session-token');
      loggedIn = false;
      return json({ status: 'logged_out' });
    }
    if (path === '/api/auth/session') {
      return loggedIn
        ? json({ session_id: 'session-owner', user: { username: 'owner', role: 'owner' } })
        : json({ detail: '本地主节点会话无效' }, 401);
    }
    if (path === '/api/cluster/my-role') return json({ is_master: true, node_id: 'local' });
    if (path === '/api/cluster/config/distributed-inference') return json({ enabled: false });
    if (path === '/api/status') return json({ model_loaded: false });
    if (path === '/api/user/settings') return json({ settings: {} });
    if (path === '/api/sessions') return json({ sessions: [] });
    return json({});
  });

  await page.goto('/');
  await expect(page.getByRole('heading', { name: '登录' })).toBeVisible();
  await page.getByRole('button', { name: '初始化主节点' }).click();
  await page.getByLabel('用户名').fill('owner');
  await page.getByLabel('显示名称').fill('Main owner');
  await page.getByRole('button', { name: '创建并配置 Auth App' }).click();
  await expect(page.getByAltText('Auth App 配置二维码')).toBeVisible();
  await page.getByRole('button', { name: '字符串' }).click();
  await expect(page.getByText(secret, { exact: true })).toBeVisible();
  await page.getByLabel('验证码').fill('123456');
  await page.screenshot({ path: '../build/auth-provisioning.png' });
  await page.getByRole('button', { name: '确认并生成恢复码' }).click();
  await expect(page.getByRole('heading', { name: '恢复码' })).toBeVisible();
  await expect(page.getByText(recoveryCodes[0])).toBeVisible();
  await page.getByRole('button', { name: '进入主节点' }).click();
  await expect(page.getByTitle('退出登录')).toBeVisible();
  await page.getByTitle('退出登录').click();
  await expect(page.getByRole('heading', { name: '登录' })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('.auth-panel')).toBeVisible();
  await page.screenshot({ path: '../build/auth-login-mobile.png' });
});

test('legacy monolith without an auth capability keeps the existing workspace available', async ({ page }) => {
  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/capability') {
      return route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Not Found' }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    });
  });
  await page.goto('/');
  await expect(page.locator('.app-layout')).toBeVisible();
  await expect(page.getByRole('heading', { name: '登录' })).toHaveCount(0);
  await expect(page.getByTitle('退出登录')).toHaveCount(0);
});

test('standalone primary-node compatibility capability keeps the workspace available', async ({ page }) => {
  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/capability') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          required: false,
          enforced: false,
          available: false,
          mode: 'local_primary_node',
          reason_code: 'auth_control_plane_unavailable',
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    });
  });

  await page.addInitScript(() => sessionStorage.setItem('qlh-auth-session-token', 'stale-gateway-token'));
  await page.goto('/');

  await expect(page.locator('.app-layout')).toBeVisible();
  await expect(page.getByRole('heading', { name: '认证服务不可用' })).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => sessionStorage.getItem('qlh-auth-session-token'))).toBeNull();
});

test('auth capability errors fail closed', async ({ page }) => {
  await page.route('**/api/auth/capability', (route) => route.fulfill({
    status: 502,
    contentType: 'application/json',
    body: JSON.stringify({ detail: 'control-svc unavailable' }),
  }));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '认证服务不可用' })).toBeVisible();
  await expect(page.getByRole('button', { name: '重新连接' })).toBeVisible();
  await expect(page.locator('.app-layout')).toHaveCount(0);
});
