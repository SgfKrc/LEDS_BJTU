import { test, expect } from '@playwright/test';

test('生图工坊 fixture 可提交任务并显示生图列表', async ({ page }) => {
  await page.goto('/#/image?fixtures=1');

  await expect(page.getByRole('heading', { level: 1, name: '生图工坊' })).toBeVisible();
  await expect(page.locator('.image-studio__layout')).toBeVisible();
  await expect(page.locator('.image-assets li')).toHaveCount(1);

  const prompt = page.locator('textarea').first();
  await prompt.fill('gothic mechanical cathedral with a luminous clock');
  const generate = page.getByRole('button', { name: '开始生成' });
  await expect(generate).toBeEnabled();
  await generate.click();

  await expect(page.locator('.image-jobs > li')).toHaveCount(1);
  await expect(page.locator('.image-job__prompt')).toContainText('gothic mechanical cathedral');
  await expect(page.locator('.image-studio__details')).toContainText('已完成');
});
