import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP_ICON = path.join(
  HERE,
  '..',
  'node_modules',
  'playwright-core',
  'lib',
  'server',
  'chromium',
  'appIcon.png',
);

test.skip(process.env.QLH_GEMMA4_REAL !== '1', 'requires local Ollama gemma4:12b');

test('Gemma 4 browser image chat completes through the QLH external provider', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: '添加图片' })).toBeEnabled();
  await expect(page.getByText('GEMMA4:12B', { exact: true })).toBeVisible();
  await page.locator('.session-new-btn').click();
  await expect(page.locator('.empty-state')).toBeVisible();
  await expect(page.getByRole('button', { name: '添加图片' })).toBeEnabled();

  await page.locator('input[accept="image/png,image/jpeg,image/webp"]').setInputFiles(APP_ICON);
  await expect(page.getByRole('status', { name: '待发送图片' })).toContainText('1/4');
  await page.locator('textarea[placeholder*="输入消息"]').fill('请用一句话描述这张图片。');
  const assistantBubbles = page.locator('.message.assistant .bubble-content');
  const assistantCount = await assistantBubbles.count();
  await page.getByTitle('发送消息').click();

  await expect.poll(() => assistantBubbles.count()).toBeGreaterThan(assistantCount);
  const assistant = assistantBubbles.last();
  await expect(assistant).toHaveText(/.{6,}/);
  await expect(assistant).not.toContainText('错误:');
  await page.waitForTimeout(1000);
  const response = await assistant.textContent();
  expect(response.trim().length).toBeGreaterThan(12);
  console.log(`Gemma 4 response: ${response}`);
  await page.screenshot({ path: '../build/gemma4-real-browser.png', fullPage: true });
});
