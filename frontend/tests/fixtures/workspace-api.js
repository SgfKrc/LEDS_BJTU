const EMPTY_MODEL_INVENTORY = {
  node_id: 'local',
  artifacts: [],
  summary: {
    total: 0,
    ready: 0,
    stale: 0,
    attention: 0,
    unchecked: 0,
    total_bytes: 0,
  },
};

export const WORKSPACE_API_FIXTURES = Object.freeze({
  'GET /api/auth/capability': { status: 404, body: { detail: 'Not Found' } },
  'GET /api/cluster/my-role': {
    body: { node_role: 'master', node_id: 'local', is_master: true, is_client: false },
  },
  'GET /api/cluster/config/distributed-inference': { body: { enabled: false } },
  'PUT /api/cluster/config/distributed-inference': {
    body: { status: 'ok', enabled: true },
  },
  'GET /api/workflows': {
    body: {
      enabled: true,
      available: true,
      role: 'master',
      workflows: [],
      worker_protocol: { experiment_enabled: true },
    },
  },
  'GET /api/status': {
    body: { model_loaded: false, current_quant: null, gpu: {}, kv_cache: {} },
  },
  'GET /api/user/settings': { body: { settings: {}, source: 'sqlite' } },
  'PUT /api/user/settings': { body: { status: 'ok', source: 'sqlite' } },
  'GET /api/sessions': { body: { sessions: [] } },
  'GET /api/models': { body: { models: [], active_model_id: null } },
  'GET /api/models/available': {
    body: { available_engines: [], current: null, current_engine: null },
  },
  'GET /api/device/profile': {
    body: {
      tier: 'edge',
      tier_label: '边缘设备',
      score_total: 40,
      score_breakdown: { gpu: 0, ram: 24, cpu: 16 },
      cpu: { physical_cores: 4, freq_max_mhz: 3200, model_name: 'Fixture CPU' },
      ram: { total_gb: 16 },
      gpu: null,
      gpus: [],
      recommendations: {},
      warnings: [],
    },
  },
  'GET /api/models/artifacts': { body: EMPTY_MODEL_INVENTORY },
  'GET /api/models/pull': { body: { jobs: [] } },
  'GET /api/models/network': {
    body: { proxy: { source: 'direct', endpoint: null }, user_proxy: null },
  },
  'GET /api/models/sources': { body: { sources: [] } },
  'GET /api/models/credentials': { body: { credentials: [] } },
  'GET /api/models/licenses/acceptances': { body: { acceptances: [] } },
  'GET /api/logs': { body: { files: [] } },
  'GET /api/logs/recent': { body: { entries: [], total: 0 } },
  'GET /api/logs/stats': { body: { totals: {} } },
  'GET /api/logs/nodes-summary': { body: { nodes: [] } },
});

function fixtureKey(request) {
  const url = new URL(request.url());
  return `${request.method()} ${url.pathname}`;
}

async function requestJson(request) {
  const text = request.postData();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (_) {
    return text;
  }
}

export async function installWorkspaceApi(page, overrides = {}) {
  const requests = [];

  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const key = fixtureKey(request);
    const record = {
      key,
      method: request.method(),
      path: url.pathname,
      query: Object.fromEntries(url.searchParams),
      json: await requestJson(request),
    };
    requests.push(record);

    const configured = overrides[key]
      ?? overrides[url.pathname]
      ?? WORKSPACE_API_FIXTURES[key]
      ?? { body: {} };
    const response = typeof configured === 'function'
      ? await configured(record)
      : configured;

    await route.fulfill({
      status: response?.status ?? 200,
      contentType: 'application/json',
      headers: response?.headers,
      body: JSON.stringify(response?.body ?? {}),
    });
  });

  return requests;
}
