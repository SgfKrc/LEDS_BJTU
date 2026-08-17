import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildLocalImportRequest,
  buildModelPullRequest,
  buildModelResolveRequest,
  buildModelSourceRequest,
  credentialIdFromRef,
  isModelPullActive,
  modelPullRequestKey,
  modelPullProgressPercent,
  modelRuntimeLabel,
} from '../src/modelFleetState.js';
import {
  cancelModelPull,
  acceptModelLicense,
  clearModelProxy,
  createModelPull,
  deleteModelCredential,
  deleteModelSource,
  fetchModelArtifacts,
  fetchModelCredentials,
  fetchModelLicenseAcceptances,
  fetchLocalModelAssets,
  fetchModelRuntimeContracts,
  bindModelRuntimeContract,
  beginModelRuntimeSidecar,
  fetchModelNetwork,
  fetchModelPullJobs,
  fetchModelSources,
  importLocalModel,
  invalidateModelRuntimeCheck,
  resetModelSources,
  resolveModelPull,
  revokeModelLicense,
  retryModelRuntimeCheck,
  saveModelCredential,
  saveModelProxy,
  saveModelSource,
} from '../src/api/client.js';
import { mergeModelCatalog } from '../src/modelCatalogState.js';

test('model catalog merges fleet artifacts and orders local assets before undownloaded candidates', () => {
  const catalog = mergeModelCatalog([
    {
      model_id: 'future-model', name: 'Future Model', is_available: false,
      is_builtin: true, is_experimental: true, model_type: 'safetensors',
    },
    {
      model_id: 'custom-ready', name: 'Custom Ready', is_available: true,
      is_builtin: false, is_experimental: true, model_type: 'gguf',
    },
  ], [{
    artifact_id: `sha256:${'a'.repeat(64)}`,
    reference: { namespace: 'local', name: 'qwen3-4b', tag: 'latest' },
    source: { repo_id: 'Qwen/Qwen3-4B' },
    format: 'safetensors', engine: 'qwen3_sidecar', family: 'qwen3',
    context_length: 32768,
    runtime_check: { status: 'ready' }, runnable: true,
  }], 'custom-ready');

  assert.deepEqual(catalog.map((model) => model.model_id), [
    'custom-ready', 'fleet:local/qwen3-4b:latest', 'future-model',
  ]);
  assert.equal(catalog[0].is_editable, true);
  assert.equal(catalog[1].name, 'Qwen/Qwen3-4B');
  assert.equal(catalog[1].has_local_asset, true);
  assert.equal(catalog[1].fleet_runnable, true);
  assert.equal(catalog[1].is_available, false);
});

test('model catalog annotates a matching registry model instead of duplicating its fleet artifact', () => {
  const catalog = mergeModelCatalog([{
    model_id: 'qwen3-4b', name: 'Qwen3 4B', huggingface_id: 'Qwen/Qwen3-4B',
    is_available: false, is_builtin: false, is_experimental: true,
  }], [{
    reference: { namespace: 'user', name: 'qwen3-4b', tag: 'v1' },
    source: { repo_id: 'Qwen/Qwen3-4B' },
    format: 'safetensors', engine: 'qwen3_sidecar', runnable: false,
  }]);

  assert.equal(catalog.length, 1);
  assert.equal(catalog[0].model_id, 'qwen3-4b');
  assert.equal(catalog[0].catalog_source, 'registry+fleet');
  assert.equal(catalog[0].has_local_asset, true);
  assert.equal(catalog[0].is_editable, true);
});

test('model catalog exposes root models assets before unavailable candidates without enabling legacy loading', () => {
  const catalog = mergeModelCatalog([{
    model_id: 'future-model', name: 'Future Model', is_available: false,
    is_builtin: true, is_experimental: true,
  }], [], '', [{
    model_id: 'qwen3-5-9b',
    name: 'Qwen/Qwen3.5-9B',
    huggingface_id: 'Qwen/Qwen3.5-9B',
    model_type: 'both', available_formats: ['safetensors', 'gguf'],
    max_context: 262144, runtime_profile: 'qwen3_sidecar',
    runtime_status: 'inventory_only', runtime_action: 'qwen3_preflight',
    runtime_hint: '本地资产已发现；当前仅安全登记，尚无可执行的 Qwen3 Sidecar 加载控制面，不使用旧单机加载器。',
  }]);

  assert.deepEqual(catalog.map((model) => model.model_id), ['local:qwen3-5-9b', 'future-model']);
  assert.equal(catalog[0].is_local_discovered_asset, true);
  assert.equal(catalog[0].has_local_asset, true);
  assert.equal(catalog[0].is_available, false);
  assert.equal(catalog[0].preferred_engine, 'specialized_runtime');
  assert.equal(catalog[0].runtime_status, 'inventory_only');
  assert.deepEqual(catalog[0].available_formats, ['safetensors', 'gguf']);
});

test('model catalog annotates a matching registry entry with local root assets', () => {
  const catalog = mergeModelCatalog([{
    model_id: 'qwen3-4b', name: 'Qwen3 4B', huggingface_id: 'Qwen/Qwen3-4B',
    is_available: false, is_builtin: false, is_experimental: true,
  }], [], '', [{
    model_id: 'qwen3-4b', name: 'Qwen/Qwen3-4B',
    huggingface_id: 'Qwen/Qwen3-4B', available_formats: ['safetensors', 'gguf'],
    model_type: 'both', max_context: 40960,
  }]);

  assert.equal(catalog.length, 1);
  assert.equal(catalog[0].model_id, 'qwen3-4b');
  assert.equal(catalog[0].catalog_source, 'registry+local-asset');
  assert.equal(catalog[0].is_local_discovered_asset, true);
  assert.deepEqual(catalog[0].available_formats, ['safetensors', 'gguf']);
});

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
    sourceId: ' hf-official ', provider: 'gguf_huggingface', repoId: ' org/demo ', revision: ' main ', allowPatterns: '*.gguf, *.json\n',
  }), {
    source_id: 'hf-official',
    source: {
      provider: 'gguf_huggingface', repo_id: 'org/demo', requested_revision: 'main',
      allow_patterns: ['*.gguf', '*.json'],
    },
    cancel_policy: 'keep_partial',
  });
  const resolveForm = {
    sourceId: 'hf-official', repoId: ' org/demo ', revision: ' main ', allowPatterns: '*.json',
  };
  assert.deepEqual(buildModelResolveRequest(resolveForm), {
    source_id: 'hf-official', repo_id: 'org/demo', requested_revision: 'main',
    allow_patterns: ['*.json'],
  });
  assert.equal(modelPullRequestKey(resolveForm), JSON.stringify(buildModelResolveRequest(resolveForm)));
  assert.deepEqual(buildModelSourceRequest({
    sourceId: ' team ', name: ' Team Mirror ', provider: 'huggingface',
    endpoint: ' https://mirror.example/ ', credentialRef: ' os:qlh/team ', priority: '5', enabled: true,
  }), {
    source_id: 'team',
    payload: {
      name: 'Team Mirror', provider: 'huggingface', endpoint: 'https://mirror.example',
      credential_ref: 'os:qlh/team', priority: 5, enabled: true,
    },
  });
  assert.equal(credentialIdFromRef('os:qlh/team'), 'team');
  assert.equal(credentialIdFromRef('os:other/team'), '');
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
    await fetchModelSources();
    await saveModelSource('team/source', { name: 'Team' });
    await deleteModelSource('team/source');
    await resetModelSources();
    await resolveModelPull({ source_id: 'team', repo_id: 'org/demo' });
    await fetchModelCredentials();
    await saveModelCredential('hf/main', 'secret');
    await deleteModelCredential('hf/main');
    await fetchModelLicenseAcceptances();
    await acceptModelLicense({ repo_id: 'org/demo', license_id: 'apache-2.0' });
    await revokeModelLicense({ repo_id: 'org/demo', license_id: 'apache-2.0' });
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
    ['GET', '/api/models/sources'],
    ['PUT', '/api/models/sources/team%2Fsource'],
    ['DELETE', '/api/models/sources/team%2Fsource'],
    ['POST', '/api/models/sources/reset'],
    ['POST', '/api/models/resolve'],
    ['GET', '/api/models/credentials'],
    ['PUT', '/api/models/credentials/hf%2Fmain'],
    ['DELETE', '/api/models/credentials/hf%2Fmain'],
    ['GET', '/api/models/licenses/acceptances'],
    ['POST', '/api/models/licenses/acceptances'],
    ['DELETE', '/api/models/licenses/acceptances'],
  ]);
  assert.deepEqual(JSON.parse(requests[2].body), { source: { repo_id: 'org/demo' } });
  assert.deepEqual(JSON.parse(requests[8].body), { url: 'http://127.0.0.1:7897' });
  assert.deepEqual(JSON.parse(requests[16].body), { secret: 'secret' });
  assert.deepEqual(JSON.parse(requests[19].body), {
    repo_id: 'org/demo', license_id: 'apache-2.0', accepted: true,
  });
});

test('local asset API client uses the inference inventory route', async () => {
  const originalFetch = global.fetch;
  const requests = [];
  global.fetch = async (url, options = {}) => {
    requests.push({ url: String(url), method: options.method || 'GET' });
    return new Response(JSON.stringify({ assets: [], summary: { total: 0, total_bytes: 0 } }), { status: 200 });
  };
  try {
    await fetchLocalModelAssets();
  } finally {
    global.fetch = originalFetch;
  }

  assert.deepEqual(requests, [{ url: '/api/models/local-assets', method: 'GET' }]);
});

test('runtime contract client binds and begins by contract id without sending contract content', async () => {
  const originalFetch = global.fetch;
  const requests = [];
  global.fetch = async (url, options = {}) => {
    requests.push({
      url: String(url),
      method: options.method || 'GET',
      body: options.body || '',
    });
    return new Response(JSON.stringify({ contracts: [], status: 'bound' }), { status: 200 });
  };
  try {
    await fetchModelRuntimeContracts();
    await bindModelRuntimeContract('qwen3_sidecar', 'qwen3-4b');
    await beginModelRuntimeSidecar('qwen3_sidecar', null, 'contract-1');
  } finally {
    global.fetch = originalFetch;
  }

  assert.equal(requests[0].url, '/api/cluster/model-runtime/contracts');
  assert.deepEqual(JSON.parse(requests[1].body), {
    profile: 'qwen3_sidecar', model_id: 'qwen3-4b',
  });
  assert.equal(requests[2].url, '/api/cluster/model-runtime/sidecars/begin');
  assert.deepEqual(JSON.parse(requests[2].body), {
    profile: 'qwen3_sidecar', contract: null, contract_id: 'contract-1',
  });
});
