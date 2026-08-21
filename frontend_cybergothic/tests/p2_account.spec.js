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
