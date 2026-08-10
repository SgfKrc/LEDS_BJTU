import assert from 'node:assert/strict';
import test from 'node:test';

import {
  AUTH_TOKEN_STORAGE_KEY,
  bootstrapAuthOwner,
  fetchAuthCapability,
  fetchAuthSession,
  fetchLocalTailscaleStatus,
  fetchTailscaleBindings,
  fetchStatus,
  getAuthSessionToken,
  loginAuth,
  logoutAuth,
  prepareTailscaleBinding,
  confirmTailscaleBinding,
  revokeTailscaleBinding,
  verifyAuthProvisioning,
} from '../src/api/client.js';

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

test('auth client keeps the bearer token in session storage and applies it to protected API calls', async () => {
  const originalWindow = global.window;
  const originalFetch = global.fetch;
  const requests = [];
  global.window = { sessionStorage: memoryStorage() };
  global.fetch = async (url, options = {}) => {
    requests.push({ url: String(url), options });
    if (String(url).endsWith('/auth/login')) {
      return new Response(JSON.stringify({
        access_token: 'local-session-token',
        session_id: 'session-1',
        expires_at: '2026-08-11T00:00:00.000Z',
        user: { user_id: 'user-1', username: 'owner', role: 'owner' },
      }), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    return new Response(JSON.stringify({ status: 'ok' }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  try {
    await fetchAuthCapability();
    await bootstrapAuthOwner({ username: ' owner ', displayName: ' Main owner ' });
    await verifyAuthProvisioning({ userId: 'user-1', authenticatorId: 'totp-1', code: '123456' });
    await loginAuth({ username: 'owner', code: '123456' });
    assert.equal(getAuthSessionToken(), 'local-session-token');
    assert.equal(global.window.sessionStorage.getItem(AUTH_TOKEN_STORAGE_KEY), 'local-session-token');
    await fetchAuthSession();
    await fetchStatus();
    await logoutAuth();
    assert.equal(getAuthSessionToken(), '');
  } finally {
    global.fetch = originalFetch;
    global.window = originalWindow;
  }

  assert.deepEqual(requests.map((entry) => [entry.options.method || 'GET', entry.url]), [
    ['GET', '/api/auth/capability'],
    ['POST', '/api/auth/bootstrap'],
    ['POST', '/api/auth/totp/verify'],
    ['POST', '/api/auth/login'],
    ['GET', '/api/auth/session'],
    ['GET', '/api/status'],
    ['POST', '/api/auth/logout'],
  ]);
  assert.equal(requests[0].options.headers.Authorization, undefined);
  assert.equal(requests[1].options.headers.Authorization, undefined);
  assert.equal(requests[3].options.headers.Authorization, undefined);
  assert.equal(requests[4].options.headers.Authorization, 'Bearer local-session-token');
  assert.equal(requests[5].options.headers.Authorization, 'Bearer local-session-token');
  assert.equal(requests[6].options.headers.Authorization, 'Bearer local-session-token');
  assert.deepEqual(JSON.parse(requests[1].options.body), {
    username: 'owner', display_name: 'Main owner',
  });
  assert.equal(requests.every((entry) => !String(entry.options.body || '').includes('local-session-token')), true);
});

test('auth client sends a recovery code instead of a TOTP code', async () => {
  const originalWindow = global.window;
  const originalFetch = global.fetch;
  let body = null;
  global.window = { sessionStorage: memoryStorage() };
  global.fetch = async (_url, options = {}) => {
    body = JSON.parse(options.body);
    return new Response(JSON.stringify({ access_token: 'recovery-session' }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  try {
    await loginAuth({ username: 'owner', recoveryCode: 'ABCD-EFGH-IJKL' });
  } finally {
    global.fetch = originalFetch;
    global.window = originalWindow;
  }
  assert.deepEqual(body, { username: 'owner', recovery_code: 'ABCD-EFGH-IJKL' });
});

test('tailscale binding client keeps stable identity fields on the control API', async () => {
  const originalWindow = global.window;
  const originalFetch = global.fetch;
  const requests = [];
  global.window = { sessionStorage: memoryStorage() };
  global.window.sessionStorage.setItem(AUTH_TOKEN_STORAGE_KEY, 'binding-token');
  global.fetch = async (url, options = {}) => {
    requests.push({ url: String(url), options });
    return new Response(JSON.stringify({ bindings: [], binding: { binding_id: 'binding-1' } }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  try {
    await fetchLocalTailscaleStatus();
    await fetchTailscaleBindings();
    await fetchTailscaleBindings('user-member');
    await prepareTailscaleBinding({ authorizationMethod: 'tailscale_cli' });
    await prepareTailscaleBinding({ userId: 'user-member', authorizationMethod: 'local_status' });
    await confirmTailscaleBinding('binding-1', {
      tailnetId: ' tailnet-main ',
      tailscaleUserId: ' ts-user-1 ',
      nodeId: ' node-1 ',
    });
    await revokeTailscaleBinding('binding-1');
  } finally {
    global.fetch = originalFetch;
    global.window = originalWindow;
  }

  assert.deepEqual(requests.map((entry) => [entry.options.method || 'GET', entry.url]), [
    ['GET', '/api/auth/tailscale/local-status'],
    ['GET', '/api/auth/tailscale/bindings'],
    ['GET', '/api/auth/users/user-member/tailscale'],
    ['POST', '/api/auth/tailscale/bindings'],
    ['POST', '/api/auth/users/user-member/tailscale'],
    ['POST', '/api/auth/tailscale/bindings/binding-1/confirm'],
    ['POST', '/api/auth/tailscale/bindings/binding-1/revoke'],
  ]);
  assert.deepEqual(JSON.parse(requests[3].options.body), { authorization_method: 'tailscale_cli' });
  assert.deepEqual(JSON.parse(requests[5].options.body), {
    tailnet_id: 'tailnet-main',
    tailscale_user_id: 'ts-user-1',
    node_id: 'node-1',
  });
  assert.equal(requests.every((entry) => entry.options.headers.Authorization === 'Bearer binding-token'), true);
});
