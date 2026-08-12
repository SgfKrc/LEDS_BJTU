import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import test from 'node:test';

import { WORKSPACE_API_FIXTURES } from './fixtures/workspace-api.js';

/**
 * Q22：Playwright fixture 与后端契约对齐边界。
 *
 * 单向不变量：所有 mock fixture 使用的 API 端点必须存在于 client.js 的
 * 消费集合中——fixture 不得"发明"client 不调用的端点（那说明 fixture 已
 * 漂移或 client 重构后未同步）；client 消费的端点允许比 fixture 多
 * （未被 UI 触发的端点不需要 mock）。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

function clientRequestPaths() {
  const source = readFileSync(path.join(ROOT, 'src', 'api', 'client.js'), 'utf-8');
  const paths = new Set();
  // request('...') 与 request(`...`)（模板字符串）都纳入；
  // 去掉模板插值（$...）、query（?...）与尾部斜杠，只留静态路径段
  for (const match of source.matchAll(/request\(\s*[`'"]([^`'"]+)[`'"]/g)) {
    const raw = match[1];
    const staticPath = raw.split('$')[0].split('?')[0].replace(/^\//, '').replace(/\/$/, '');
    if (staticPath) paths.add(staticPath);
  }
  return paths;
}

function fixturePaths() {
  const paths = new Set();
  for (const key of Object.keys(WORKSPACE_API_FIXTURES)) {
    const [, pathPart] = key.split(' ');
    paths.add(pathPart.replace(/^\/api\//, '').replace(/\/$/, ''));
  }
  return paths;
}

function specApiPaths() {
  const paths = new Set();
  const specs = [
    'auth-ui.spec.js',
    'model-fleet-ui.spec.js',
    'user-management-ui.spec.js',
    'gemma4-ui.spec.js',
    'settings-ui.spec.js',
  ];
  for (const name of specs) {
    const source = readFileSync(path.join(HERE, name), 'utf-8');
    for (const match of source.matchAll(/['"`]\/api\/([a-z0-9_./${}-]+)['"`]/gi)) {
      const raw = match[1].split('$')[0].split('?')[0].replace(/\/$/, '');
      if (raw) paths.add(raw);
    }
  }
  return paths;
}

function isConsumed(specPath, clientPaths) {
  // 模板参数化豁免：spec 的 `users/user-member` 由 client 的 `/users/${id}` 消费
  const segments = specPath.split('/');
  for (let i = segments.length; i >= 1; i -= 1) {
    if (clientPaths.has(segments.slice(0, i).join('/'))) return true;
  }
  return false;
}

test('workspace fixture endpoints are all consumed by the API client', () => {
  const client = clientRequestPaths();
  const fixtures = fixturePaths();
  const orphan = [...fixtures].filter((entry) => !client.has(entry));
  assert.deepEqual(
    orphan,
    [],
    `fixture 端点不在 client.js 消费集合中（fixture 漂移或 client 重构未同步）: ${orphan.join(', ')}`,
  );
});

test('playwright spec inline endpoints are all consumed by the API client', () => {
  const client = clientRequestPaths();
  const specPaths = specApiPaths();
  const orphan = [...specPaths].filter((entry) => !isConsumed(entry, client));
  assert.deepEqual(
    orphan,
    [],
    `spec 内联 mock 端点不在 client.js 消费集合中: ${orphan.join(', ')}`,
  );
});

test('key workspace fixtures carry the exact fields the client consumes', () => {
  // 契约快照：client.js 对这些端点解析的字段（与后端契约一致）
  const capability = WORKSPACE_API_FIXTURES['GET /api/auth/capability'];
  assert.equal(capability.status, 404, '默认未认证环境 capability 应为 404');

  const myRole = WORKSPACE_API_FIXTURES['GET /api/cluster/my-role'].body;
  assert.equal(myRole.is_master, true);
  assert.equal(myRole.node_role, 'master');
  assert.equal(typeof myRole.node_id, 'string');

  const settings = WORKSPACE_API_FIXTURES['GET /api/user/settings'].body;
  assert.deepEqual(Object.keys(settings).sort(), ['settings', 'source']);
  assert.equal(settings.source, 'sqlite');

  const status = WORKSPACE_API_FIXTURES['GET /api/status'].body;
  for (const field of ['model_loaded', 'current_quant', 'gpu', 'kv_cache']) {
    assert.ok(field in status, `status fixture 缺少字段 ${field}`);
  }

  const artifacts = WORKSPACE_API_FIXTURES['GET /api/models/artifacts'].body;
  assert.deepEqual(Object.keys(artifacts).sort(), ['artifacts', 'node_id', 'summary']);
  assert.ok('total' in artifacts.summary, 'artifacts.summary 缺少 total');
});

test('models workspace fixtures cover the fleet UI critical paths', () => {
  for (const key of [
    'GET /api/models/artifacts',
    'GET /api/models/pull',
    'GET /api/models/network',
    'GET /api/models/sources',
    'GET /api/models/credentials',
    'GET /api/models/licenses/acceptances',
  ]) {
    assert.ok(key in WORKSPACE_API_FIXTURES, `缺少模型工作区 fixture: ${key}`);
  }
});
