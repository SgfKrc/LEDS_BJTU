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

test('生图工坊接入资产检查登记、许可证门和本地导入动作', async ({ page }) => {
  await page.goto('/#/image?fixtures=1');

  const tools = page.getByTestId('image-asset-tools');
  await tools.getByLabel('计算 SHA256').check();
  await tools.getByRole('button', { name: '检查' }).click();
  await expect(tools).toContainText('fixture-inspected-artifact');

  await tools.getByRole('button', { name: '登记' }).click();
  await expect(tools).toContainText('fixture-registered-artifact');
  await tools.getByLabel('我确认许可证和来源').check();
  await tools.getByRole('button', { name: '下载' }).click();
  await expect(tools).toContainText('queued');
  await tools.getByRole('button', { name: '导入' }).click();
  await expect(tools).toContainText('imported');
});

test('生图工坊支持输入图像编辑和分布式 workflow 结果', async ({ page }) => {
  await page.goto('/#/image?fixtures=1');
  await page.getByRole('button', { name: '图生图' }).click();
  await page.getByLabel('输入 Blob ID').fill('fixture-input-image');
  await page.locator('textarea').first().fill('change the lighting to neon violet');
  await page.getByRole('button', { name: '开始编辑' }).click();
  await expect(page.locator('.image-studio__details')).toContainText('已完成');

  await page.getByRole('button', { name: '分布式' }).click();
  await page.locator('textarea').first().fill('distributed gothic observatory');
  await page.getByRole('button', { name: '提交分布式' }).click();
  await expect(page.getByTestId('distributed-result')).toContainText('fixture_remote_diffusion');
  await expect(page.getByTestId('distributed-result')).toContainText('fixture-wf-');
});
