import { test, expect } from '@playwright/test';

test('account workspace covers auth session, Tailscale binding, and user management in fixture mode', async ({ page }) => {
  await page.goto('/#/account?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: 'Account & Security' })).toBeVisible();
  await expect(page.getByTestId('account-page')).toBeVisible();
  await expect(page.locator('.account-profile')).toContainText('QLH Operator');
  await expect(page.locator('.account-session-row')).toHaveCount(2);
  await expect(page.locator('canvas.iron-gate')).toHaveAttribute('aria-hidden', 'true');

  await page.locator('.account-nav button', { hasText: 'Tailscale' }).click();
  await expect(page.locator('.account-binding-current')).toContainText('tailnet-qlh-demo');
  await page.getByRole('button', { name: 'Inspect local status' }).click();
  await page.getByRole('button', { name: 'Prepare binding' }).click();
  await expect(page.locator('.account-binding-form')).toBeVisible();
  await page.getByLabel('TAILNET ID').fill('tailnet-qlh-demo');
  await page.getByLabel('TAILSCALE USER ID').fill('ts-user-operator');
  await page.getByRole('button', { name: 'Confirm binding' }).click();
  await expect(page.locator('.account-main .account-panel .sectionhead .badge').first()).toContainText('BOUND');

  await page.locator('.account-nav button', { hasText: 'Users' }).click();
  await expect(page.locator('.account-user-row')).toHaveCount(3);
  await page.getByLabel('USERNAME').fill('auditor');
  await page.getByLabel('DISPLAY NAME').fill('Audit User');
  await page.getByRole('button', { name: 'Create user' }).click();
  await expect(page.locator('.account-user-row')).toHaveCount(4);
  await page.getByRole('button', { name: 'Provision Auth App for auditor' }).click();
  await expect(page.getByRole('heading', { level: 2, name: 'Auth App for auditor' })).toBeVisible();
  await page.getByRole('button', { name: 'Discard provisioning material' }).click();

  await page.locator('.account-nav button', { hasText: 'Security' }).click();
  await page.getByLabel('CURRENT AUTH APP CODE').fill('123456');
  await page.getByRole('button', { name: 'Rotate recovery codes' }).click();
  await expect(page.getByRole('heading', { level: 2, name: /Recovery codes/ })).toBeVisible();

  await page.getByRole('button', { name: 'Sign out' }).click();
  await expect(page.getByRole('heading', { level: 2, name: 'Sign in to continue' })).toBeVisible();
  await page.getByLabel('USERNAME').fill('operator');
  await page.getByLabel('AUTH APP CODE').fill('123456');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.locator('.account-profile')).toContainText('Fixture Operator');
});

test('account page reports missing auth service instead of rendering a fake session', async ({ page }) => {
  await page.route('**/api/auth/capability', async (route) => {
    await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'Not Found' }) });
  });
  await page.goto('/#/account');
  await expect(page.getByRole('heading', { level: 1, name: 'Account & Security' })).toBeVisible();
  await expect(page.locator('.emptystate--error')).toContainText('Authentication capability unavailable');
  await expect(page.locator('.account-profile')).toHaveCount(0);
});

test('account page distinguishes unavailable auth control plane from disabled auth policy', async ({ page }) => {
  await page.route('**/api/auth/capability', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ required: true, enforced: true, available: false, reason_code: 'auth_control_plane_unavailable' }),
    });
  });
  await page.goto('/#/account');
  await expect(page.getByRole('heading', { level: 2, name: 'Authentication control plane unavailable' })).toBeVisible();
  await expect(page.locator('.account-disabled')).toContainText('Start the configured control-svc or gateway');
  await expect(page.locator('.account-disabled')).not.toContainText('Account controls will appear when the Auth service is enabled');
});

test('first owner receives local Auth App QR and recovery codes before sign in', async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/capability') {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ required: true, bootstrap_available: true }) });
      return;
    }
    if (path === '/api/auth/bootstrap') {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ status: 'pending', provisioning: { user_id: 'user-owner', authenticator_id: 'totp-owner', secret: 'JBSWY3DPEHPK3PXP', qr_payload: 'otpauth://totp/QLH:owner?secret=JBSWY3DPEHPK3PXP&issuer=QLH', otpauth_uri: 'otpauth://totp/QLH:owner?secret=JBSWY3DPEHPK3PXP&issuer=QLH' } }) });
      return;
    }
    if (path === '/api/auth/totp/verify') {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ status: 'active', recovery_codes: ['A1C2-E3F4', 'B5D6-G7H8'] }) });
      return;
    }
    await route.fulfill({ contentType: 'application/json', body: '{}' });
  });

  await page.goto('/#/account');
  await expect(page.getByRole('heading', { level: 2, name: 'Initialize the owner account' })).toBeVisible();
  await page.getByLabel('USERNAME').fill('owner');
  await page.getByRole('button', { name: 'Create Auth App setup' }).click();
  await expect(page.getByAltText('Auth App provisioning QR')).toBeVisible();
  await page.getByRole('button', { name: 'String' }).click();
  await expect(page.locator('.account-secret-row').first()).toContainText('JBSWY3DPEHPK3PXP');
  await page.getByLabel('AUTH APP CODE').fill('123456');
  await page.getByRole('button', { name: 'Verify and activate' }).click();
  await expect(page.getByRole('heading', { level: 2, name: 'Recovery codes for owner' })).toBeVisible();
  await expect(page.locator('.account-recovery-grid')).toContainText('A1C2-E3F4');
  await page.getByRole('button', { name: 'Continue to sign in' }).click();
  await expect(page.getByRole('heading', { level: 2, name: 'Sign in to continue' })).toBeVisible();
});
