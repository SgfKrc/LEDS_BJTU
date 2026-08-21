import { test, expect } from '@playwright/test';

test('API client 保留 HTTP、网络和超时错误语义', async ({ page }) => {
  await page.goto('/#/help?fixtures=1');

  const result = await page.evaluate(async () => {
    const api = await import('/src/data/api.ts');
    const originalFetch = window.fetch;
    const originalSetTimeout = window.setTimeout;
    const results = {};

    async function capture(label, fetchImpl) {
      window.fetch = fetchImpl;
      try {
        await api.fetchSystemStatus();
      } catch (error) {
        results[label] = {
          kind: error.kind,
          status: error.status,
          retryable: error.retryable,
          text: api.describeError(error),
        };
      }
    }

    for (const [label, status] of [
      ['unauthorized', 401],
      ['forbidden', 403],
      ['notFound', 404],
      ['conflict', 409],
      ['server', 503],
    ]) {
      await capture(label, async () => new Response(JSON.stringify({ detail: label }), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }));
    }

    await capture('network', async () => {
      throw new TypeError('offline');
    });

    // 将计时器压缩为 0ms，验证真实 timeout 分支而不让测试等待 15 秒。
    window.setTimeout = ((callback) => originalSetTimeout(callback, 0));
    await capture('timeout', (_input, init) => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(new DOMException('timeout', 'AbortError')), { once: true });
    }));

    window.fetch = originalFetch;
    window.setTimeout = originalSetTimeout;
    return results;
  });

  expect(result.unauthorized).toMatchObject({ kind: 'unauthorized', status: 401, retryable: false });
  expect(result.unauthorized.text).toContain('登录已失效');
  expect(result.forbidden).toMatchObject({ kind: 'forbidden', status: 403, retryable: false });
  expect(result.notFound).toMatchObject({ kind: 'not_found', status: 404, retryable: false });
  expect(result.conflict).toMatchObject({ kind: 'conflict', status: 409, retryable: false });
  expect(result.server).toMatchObject({ kind: 'server', status: 503, retryable: true });
  expect(result.network).toMatchObject({ kind: 'network', status: 0, retryable: true });
  expect(result.timeout).toMatchObject({ kind: 'timeout', status: 0, retryable: true });
  expect(result.timeout.text).toContain('请求超时');
});
