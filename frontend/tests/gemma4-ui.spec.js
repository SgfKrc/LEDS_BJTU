import { expect, test } from '@playwright/test';

const PNG = {
  name: 'theater.png',
  mimeType: 'image/png',
  buffer: Buffer.from('89504e470d0a1a0a66697874757265', 'hex'),
};

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });
}

test('Gemma 4 chat selects, previews, removes, and sends bounded images', async ({ page }) => {
  let chatPayload = null;

  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();

    if (path === '/api/auth/capability') return json(route, { detail: 'Not Found' }, 404);
    if (path === '/api/cluster/my-role') return json(route, { is_master: true, node_id: 'local' });
    if (path === '/api/cluster/distributed-inference/config') return json(route, { enabled: false });
    if (path === '/api/workflows') return json(route, { workflows: [], available: false });
    if (path === '/api/status') {
      return json(route, {
        model_loaded: false,
        current_quant: null,
        external: {
          enabled: true,
          reachable: true,
          model: 'gemma4:12b',
          data_scope: 'opt_in',
        },
      });
    }
    if (path === '/api/user/settings') return json(route, { settings: {} });
    if (path === '/api/presets') return json(route, { presets: [] });
    if (path === '/api/sessions' && method === 'GET') {
      return json(route, { sessions: [], active_session_id: null, total: 0 });
    }
    if (path === '/api/sessions' && method === 'POST') {
      return json(route, { id: 'sess_gemma4', title: 'Gemma 4', message_count: 0, active: true }, 201);
    }
    if (path.startsWith('/api/sessions/') && path.endsWith('/activate')) {
      return json(route, { session_id: 'sess_gemma4', title: 'Gemma 4', messages: [] });
    }
    if (path === '/api/chat' && method === 'POST') {
      chatPayload = request.postDataJSON();
      return json(route, {
        role: 'assistant',
        content: 'Gemma 4 已读取这张图片。',
        metrics: { engine: 'external_api', model: 'gemma4:12b' },
        followups: [],
      });
    }
    if (path === '/api/chat/clear') return json(route, { status: 'cleared' });
    if (path === '/api/chat/generations/') return json(route, { status: 'cancelled' });
    return json(route, {});
  });

  await page.goto('/');
  const imageInput = page.locator('input[accept="image/png,image/jpeg,image/webp"]');
  await expect(page.getByRole('button', { name: '添加图片' })).toBeEnabled();

  await imageInput.setInputFiles([
    PNG,
    { ...PNG, name: 'second.png' },
  ]);
  await expect(page.getByRole('status', { name: '待发送图片' })).toContainText('2/4');
  await expect(page.getByAltText('theater.png')).toBeVisible();
  await expect(page.getByAltText('second.png')).toBeVisible();

  await page.getByRole('button', { name: '移除图片 second.png' }).click();
  await expect(page.getByAltText('second.png')).toHaveCount(0);

  await page.locator('textarea[placeholder*="输入消息"]').fill('请描述图像内容');
  await page.getByTitle('发送消息').click();
  await expect.poll(() => chatPayload).toMatchObject({
    message: '请描述图像内容',
    allow_external: true,
    prefer_external: true,
    execution_mode: 'auto',
  });
  expect(chatPayload.image_data_urls).toHaveLength(1);
  expect(chatPayload.image_data_urls[0]).toMatch(/^data:image\/png;base64,/);
  await expect(page.getByText('Gemma 4 已读取这张图片。')).toBeVisible();
  const stored = await page.evaluate(() => JSON.stringify(localStorage));
  expect(stored).not.toContain('data:image');

  await page.setViewportSize({ width: 390, height: 844 });
  await imageInput.setInputFiles(PNG);
  await expect(page.getByAltText('theater.png')).toBeVisible();
  await page.screenshot({ path: '../build/gemma4-chat-mobile.png', fullPage: true });
});
