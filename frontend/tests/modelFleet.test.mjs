import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildLocalImportRequest,
  buildModelPullRequest,
  isModelPullActive,
  modelPullProgressPercent,
  modelRuntimeLabel,
} from '../src/modelFleetState.js';
import {
  cancelModelPull,
  clearModelProxy,
  createModelPull,
  fetchModelArtifacts,
  fetchModelNetwork,
  fetchModelPullJobs,
  importLocalModel,
  invalidateModelRuntimeCheck,
  retryModelRuntimeCheck,
  saveModelProxy,
} from '../src/api/client.js';

test('model fleet state bounds progress and distinguishes terminal jobs', () => {
  assert.equal(modelPullProgressPercent({ progress: { total_bytes: 10, downloaded_bytes: 13 } }), 100);
  assert.equal(modelPullProgressPercent({ progress: { total_bytes: 0, downloaded_bytes: 1 } }), 0);
  assert.equal(isModelPullActive({ state: 'downloading' }), true);
  assert.equal(isModelPullActive({ state: 'registered' }), false);
  assert.equal(modelRuntimeLabel({ status: 'ready' }), '可运行');
  assert.equal(modelRuntimeLabel(null), '未检查');
});

test('model fleet request builders normalize local import and pull fields', () => {
  assert.deepEqual(buildLocalImportRequest({
    sourcePath: ' C:/models/demo.gguf ', namespace: ' local ', name: ' demo ', tag: ' v1 ',
  }), {
    source_path: 'C:/models/demo.gguf', namespace: 'local', name: 'demo', tag: 'v1',
  });
  assert.deepEqual(buildModelPullRequest({
    provider: 'gguf_huggingface', repoId: ' org/demo ', revision: ' main ', allowPatterns: '*.gguf, *.json\n',
  }), {
    source: {
      provider: 'gguf_huggingface', repo_id: 'org/demo', requested_revision: 'main',
      allow_patterns: ['*.gguf', '*.json'],
    },
    cancel_policy: 'keep_partial',
  });
});

test('model fleet API client uses the control-plane routes and query encoding', async () => {
  const originalFetch = global.fetch;
  const requests = [];
  global.fetch = async (url, options = {}) => {
    requests.push({ url: String(url), method: options.method || 'GET', body: options.body || null });
    return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
  };
  try {
    await fetchModelArtifacts();
    await fetchModelPullJobs();
    await createModelPull({ source: { repo_id: 'org/demo' } });
    await cancelModelPull('job/1');
    await importLocalModel({ source_path: 'C:/demo.gguf' });
    await retryModelRuntimeCheck({ namespace: 'user', name: 'demo', tag: 'latest' });
    await invalidateModelRuntimeCheck({
      artifactId: 'sha256:abc', nodeId: 'local', runtimeProfile: 'llm-cpu-v1', reason: 'recheck now',
    });
    await fetchModelNetwork();
    await saveModelProxy('http://127.0.0.1:7897');
    await clearModelProxy();
  } finally {
    global.fetch = originalFetch;
  }

  assert.deepEqual(requests.map((request) => [request.method, request.url]), [
    ['GET', '/api/models/artifacts'],
    ['GET', '/api/models/pull'],
    ['POST', '/api/models/pull'],
    ['DELETE', '/api/models/pull/job%2F1'],
    ['POST', '/api/models/imports'],
    ['POST', '/api/models/runtime-checks/retry'],
    ['DELETE', '/api/models/runtime-checks?artifact_id=sha256%3Aabc&node_id=local&runtime_profile=llm-cpu-v1&reason=recheck+now'],
    ['GET', '/api/models/network'],
    ['PUT', '/api/models/network/proxy'],
    ['DELETE', '/api/models/network/proxy'],
  ]);
  assert.deepEqual(JSON.parse(requests[2].body), { source: { repo_id: 'org/demo' } });
  assert.deepEqual(JSON.parse(requests[8].body), { url: 'http://127.0.0.1:7897' });
});
