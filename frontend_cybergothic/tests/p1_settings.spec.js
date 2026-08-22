import { test, expect } from '@playwright/test';

test('settings workspace exposes device controls and local RAG actions in fixture mode', async ({ page }) => {
  await page.goto('/#/settings?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: '设置' })).toBeVisible();
  await expect(page.getByTestId('settings-workspace')).toBeVisible();
  await expect(page.getByTestId('device-workspace')).toContainText('Performance laptop');
  await expect(page.locator('.gpu-picker button')).toHaveCount(2);

  await page.locator('.gpu-picker button').nth(1).click();
  await expect(page.locator('.gpu-picker button').nth(1)).toHaveClass(/is-active/);
  await page.getByRole('button', { name: 'Apply recommended configuration' }).click();
  await expect(page.getByText('Fixture recommended configuration applied', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Local RAG' }).click();
  await expect(page.getByTestId('rag-workspace')).toContainText('Sources');
  await page.getByRole('textbox', { name: 'RAG search' }).fill('runtime');
  await page.getByRole('button', { name: 'Search' }).click();
  await expect(page.locator('.rag-results article')).toHaveCount(1);
  await page.getByRole('button', { name: 'Rebuild FTS index' }).click();
  await expect(page.getByText('Fixture FTS index rebuilt', { exact: true })).toBeVisible();
});

test('local RAG exposes capacity, ANN gate, and resumable embedding jobs in fixture mode', async ({ page }) => {
  await page.goto('/#/settings?fixtures=1');

  await page.getByRole('button', { name: 'Local RAG' }).click();
  await expect(page.getByTestId('rag-capacity')).toContainText('2,841');
  await expect(page.getByTestId('rag-ann')).toContainText('bounded_cosine_within_scan_budget');

  const jobs = page.getByTestId('rag-embedding-jobs');
  await jobs.getByLabel('Model SHA256').fill('a'.repeat(64));
  await jobs.getByRole('button', { name: 'Create job' }).click();
  await expect(jobs).toContainText('fixture-rag-job-01');

  await jobs.getByRole('button', { name: 'Run batch' }).click();
  await expect(jobs).toContainText('RUNNING');
  await jobs.getByRole('button', { name: 'Cancel' }).click();
  await expect(jobs).toContainText('CANCELLED');
});

test('settings diagnostics exposes local storage and fail-closed experiment gate', async ({ page }) => {
  await page.goto('/#/settings?fixtures=1');

  await page.getByRole('button', { name: '运行诊断' }).click();
  const diagnostics = page.getByTestId('settings-diagnostics');
  await expect(diagnostics).toBeVisible();
  await expect(diagnostics).toContainText('sqlite');
  await expect(diagnostics).toContainText('local_only');
  await expect(diagnostics).toContainText('FAIL CLOSED');
  await diagnostics.getByRole('button', { name: 'Send client diagnostic' }).click();
  await expect(page.getByText('Fixture client diagnostic recorded', { exact: true })).toBeVisible();
});
