import { test, expect } from '@playwright/test';

test('模型下载页支持仓库搜索并从结果创建下载任务', async ({ page }) => {
  await page.goto('/#/downloads?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: '模型下载' })).toBeVisible();
  await expect(page.locator('.downloads-page__bg')).toBeVisible();
  await expect(page.locator('.downloads-panel[data-reveal]')).toHaveCount(3);
  await page.getByRole('textbox', { name: '搜索模型仓库' }).fill('qwen');
  await page.getByRole('button', { name: '搜索' }).click();

  await expect(page.locator('.downloads-results')).toBeVisible();
  await expect(page.locator('.downloads-results')).toContainText('HUGGING FACE');
  await expect(page.locator('.downloads-results')).toContainText('MODELSCOPE');
  await expect(page.locator('.downloads-search__meta')).toContainText('直连返回');

  await page.locator('.downloads-results').getByRole('button', { name: '安装' }).first().click();
  await expect(page.locator('.downloads-jobs')).toContainText('Qwen3-4B-GGUF');
  await expect(page.locator('.toasthost .toast')).toContainText('已排队');
});

test('模型下载页在窄屏将搜索结果表转换为可读条目', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/#/downloads?fixtures=1');
  await page.getByRole('textbox', { name: '搜索模型仓库' }).fill('qwen');
  await page.getByRole('button', { name: '搜索' }).click();

  const cell = page.locator('.downloads-results tbody td[data-label="模型"]').first();
  await expect(cell).toBeVisible();
  await expect(cell).toHaveAttribute('data-label', '模型');
  await expect(cell).toContainText('Qwen3');
});
