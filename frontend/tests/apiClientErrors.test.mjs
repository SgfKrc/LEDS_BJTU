import assert from 'node:assert/strict';
import test from 'node:test';

import {
  fetchAuthCapability,
  fetchManagedUsers,
  getAuthSessionToken,
  setAuthSessionToken,
} from '../src/api/client.js';

/**
 * API 客户端错误规范化与 fail-closed 测试。
 *
 * 覆盖 request() 的错误路径：非 2xx 时的 detail/code/request_id 提取、
 * 非 JSON 响应体回退、网络失败传播、Bearer 附加规则。
 */

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

function withHttpMocks({ fetchImpl }) {
  const originalWindow = global.window;
  const originalFetch = global.fetch;
  global.window = { sessionStorage: memoryStorage() };
  global.fetch = fetchImpl;
  return () => {
    global.fetch = originalFetch;
    global.window = originalWindow;
  };
}

function jsonResponse(body, status, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

test('non-ok with string detail surfaces status and detail without request id', async () => {
  const restore = withHttpMocks({
    fetchImpl: async () =>
      jsonResponse({ detail: '服务暂不可用' }, 500),
  });
  try {
    await assert.rejects(
      fetchAuthCapability(),
      (error) => {
        assert.equal(error.status, 500);
        assert.equal(error.detail, '服务暂不可用');
        assert.equal(error.code, null);
        assert.equal(error.requestId, null);
        assert.equal(error.path, '/auth/capability');
        assert.equal(error.message.includes('request_id'), false);
        return true;
      },
    );
  } finally {
    restore();
  }
});

test('non-ok with object detail extracts code and message', async () => {
  const restore = withHttpMocks({
    fetchImpl: async () =>
      jsonResponse(
        { detail: { code: 'AUTH_REQUIRED', message: '请先登录' } },
        401,
      ),
  });
  try {
    await assert.rejects(
      fetchManagedUsers(),
      (error) => {
        assert.equal(error.status, 401);
        assert.equal(error.code, 'AUTH_REQUIRED');
        assert.equal(error.detail, '请先登录');
        return true;
      },
    );
  } finally {
    restore();
  }
});

test('non-json response body falls back to raw text detail', async () => {
  const restore = withHttpMocks({
    fetchImpl: async () =>
      new Response('<html>Bad Gateway</html>', { status: 502 }),
  });
  try {
    await assert.rejects(
      fetchAuthCapability(),
      (error) => {
        assert.equal(error.status, 502);
        assert.equal(error.detail, '<html>Bad Gateway</html>');
        assert.equal(error.code, null);
        return true;
      },
    );
  } finally {
    restore();
  }
});

test('X-Request-ID header is surfaced and appended to message', async () => {
  const restore = withHttpMocks({
    fetchImpl: async () =>
      jsonResponse({ detail: '内部错误' }, 500, { 'X-Request-ID': 'req-abc-123' }),
  });
  try {
    await assert.rejects(
      fetchAuthCapability(),
      (error) => {
        assert.equal(error.requestId, 'req-abc-123');
        assert.ok(error.message.includes('request_id: req-abc-123'));
        return true;
      },
    );
  } finally {
    restore();
  }
});

test('network failure propagates the original error (fail-closed)', async () => {
  const restore = withHttpMocks({
    fetchImpl: async () => {
      throw new TypeError('fetch failed');
    },
  });
  try {
    await assert.rejects(fetchAuthCapability(), TypeError);
  } finally {
    restore();
  }
});

test('successful response returns parsed data', async () => {
  const restore = withHttpMocks({
    fetchImpl: async () =>
      jsonResponse({ required: true, mode: 'local_totp' }, 200),
  });
  try {
    const capability = await fetchAuthCapability();
    assert.deepEqual(capability, { required: true, mode: 'local_totp' });
  } finally {
    restore();
  }
});

test('auth capability request does not attach a bearer token', async () => {
  const requests = [];
  const restore = withHttpMocks({
    fetchImpl: async (url, options = {}) => {
      requests.push({ url: String(url), options });
      return jsonResponse({ required: false }, 200);
    },
  });
  try {
    setAuthSessionToken('should-not-be-sent');
    await fetchAuthCapability();
    assert.equal(getAuthSessionToken(), 'should-not-be-sent');
    assert.equal(requests[0].options.headers.Authorization, undefined);
  } finally {
    restore();
  }
});

test('protected call attaches bearer only when a token exists', async () => {
  const requests = [];
  const restore = withHttpMocks({
    fetchImpl: async (url, options = {}) => {
      requests.push({ url: String(url), options });
      return jsonResponse({ status: 'ok' }, 200);
    },
  });
  try {
    await fetchManagedUsers();
    assert.equal(requests[0].options.headers.Authorization, undefined);

    setAuthSessionToken('token-123');
    await fetchManagedUsers();
    assert.equal(requests[1].options.headers.Authorization, 'Bearer token-123');
  } finally {
    restore();
  }
});
