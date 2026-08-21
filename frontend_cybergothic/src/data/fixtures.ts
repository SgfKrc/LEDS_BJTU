/**
 * 本地 fixture — 与真实接口完全相同的数据形状（§4.3）。
 *
 * 用途：后端未启动时的离线预览、以及各页面状态矩阵的手工验证。
 * 演示数据只允许出现在本文件，不得散落到组件内部。
 * 通过 URL 参数 `?fixtures=1` 或 localStorage 开关启用，默认关闭。
 */

import type {
  AuthCapabilityResponse,
  AuthSessionResponse,
  AuthSessionsResponse,
  AuthUser,
  AvailableModelsResponse,
  ClusterConfigResponse,
  ClusterInviteResponse,
  ClusterStatusResponse,
  ClusterNodesResponse,
  ConversationResponse,
  CurrentModelResponse,
  DeviceProfileResponse,
  DiffusionArtifactsResponse,
  DiffusionAssetsResponse,
  DiffusionCapabilitiesResponse,
  LocalModelAssetsResponse,
  LocalTailscaleStatusResponse,
  CanVoteResponse,
  LogFilesResponse,
  LogStatsResponse,
  NodeLogAggregateResponse,
  NodesLogSummaryResponse,
  ManagedUsersResponse,
  ModelsResponse,
  MasterHealthResponse,
  PipelineCapacityResponse,
  QueueResponse,
  RagHealthResponse,
  RagSearchResponse,
  RecentLogsResponse,
  ReviewTicketsResponse,
  SessionsResponse,
  SystemStatusResponse,
  TailscaleBindingsResponse,
  WorkflowsResponse,
} from './types';

const FIXTURE_FLAG_KEY = 'qlh_cg_use_fixtures';

/**
 * 读取 fixtures 参数。因为路由用 hash，`?fixtures=1` 既可能写在 hash 前
 * （/?fixtures=1#/tasks），也可能被顺手写在 hash 后（/#/tasks?fixtures=1）。
 * 两种都要认，否则用户按帮助页说明操作却「没反应」。
 */
function readFixtureParam(): string | null {
  const fromSearch = new URLSearchParams(window.location.search).get('fixtures');
  if (fromSearch !== null) return fromSearch;
  const hash = window.location.hash;
  const q = hash.indexOf('?');
  if (q === -1) return null;
  return new URLSearchParams(hash.slice(q + 1)).get('fixtures');
}

/** fixture 模式：URL 参数优先，其次 localStorage，默认使用真实接口。 */
export function fixturesEnabled(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    const param = readFixtureParam();
    if (param === '1' || param === 'true') return true;
    if (param === '0' || param === 'false') return false;
    return window.localStorage?.getItem(FIXTURE_FLAG_KEY) === '1';
  } catch {
    return false;
  }
}

export function setFixturesEnabled(on: boolean): void {
  try {
    if (on) window.localStorage?.setItem(FIXTURE_FLAG_KEY, '1');
    else window.localStorage?.removeItem(FIXTURE_FLAG_KEY);
  } catch {
    // 存储不可用时忽略，刷新后回落到真实接口。
  }
}

const NOW = Date.now() / 1000;

export const authCapabilityFixture: AuthCapabilityResponse = {
  required: true,
  mode: 'local_primary_node',
  bootstrap_available: false,
};

export const authOwnerFixture: AuthUser = {
  user_id: 'user_owner_01',
  username: 'operator',
  display_name: 'QLH Operator',
  role: 'owner',
  status: 'active',
  totp_state: 'enabled',
  active_session_count: 2,
  aggregate_version: 4,
};

export const authSessionFixture: AuthSessionResponse = {
  session_id: 'auth_session_current',
  expires_at: '2026-08-22T12:00:00.000Z',
  user: authOwnerFixture,
};

export const authSessionsFixture: AuthSessionsResponse = {
  sessions: [
    { session_id: 'auth_session_current', user_id: authOwnerFixture.user_id, active: true, current: true, created_at: '2026-08-20T12:28:05.340855Z', last_seen_at: '2026-08-21T09:28:05.340855Z', expires_at: '2026-08-22T12:00:00.000Z' },
    { session_id: 'auth_session_tablet', user_id: authOwnerFixture.user_id, active: true, current: false, created_at: '2026-08-19T08:20:00.000Z', last_seen_at: '2026-08-21T08:55:00.000Z', expires_at: '2026-08-22T08:20:00.000Z' },
  ],
};

export const managedUsersFixture: ManagedUsersResponse = {
  users: [
    authOwnerFixture,
    { user_id: 'user_member_02', username: 'researcher', display_name: 'Researcher', role: 'member', status: 'active', totp_state: 'enabled', active_session_count: 1, aggregate_version: 2 },
    { user_id: 'user_member_03', username: 'tablet', display_name: 'Tablet Client', role: 'member', status: 'suspended', totp_state: 'pending', active_session_count: 0, aggregate_version: 1 },
  ],
};

export const tailscaleBindingsFixture: TailscaleBindingsResponse = {
  bindings: [
    { binding_id: 'binding_qlh_01', user_id: authOwnerFixture.user_id, tailnet_id: 'tailnet-qlh-demo', tailscale_user_id: 'ts-user-operator', node_id: 'master', state: 'active', authorization_method: 'local_status', confirmed_at: '2026-08-20T12:30:00.000Z', updated_at: '2026-08-20T12:30:00.000Z' },
  ],
};

export const localTailscaleStatusFixture: LocalTailscaleStatusResponse = {
  local_status: {
    state: 'ready',
    available: true,
    candidate: { tailnet_id: 'tailnet-qlh-demo', tailnet_display_name: 'QLH Demo Tailnet', tailscale_user_id: 'ts-user-operator', node_id: 'master', hostname: 'qlh-master', addresses: ['100.90.76.108'] },
  },
};

export const statusFixture: SystemStatusResponse = {
  model_loaded: false,
  pipeline_prepared: true,
  current_quant: 'fp16',
  model_name: 'Qwen-1_8B',
  model_path: 'models/qwen-1_8b',
  active_model_id: null,
  engine: 'pipeline',
  run_mode: 'distributed',
  node_role: 'master',
  node_id: 'master',
  max_nodes: 3,
  conversation_turns: 28,
  gpu: {
    name: 'NVIDIA GeForce RTX 4060 Laptop GPU',
    total_mb: 8188,
    allocated_mb: 1024.5,
    reserved_mb: 1536.0,
    utilization: 12.5,
  },
  kv_cache: {
    total_tokens: 18432,
    max_tokens: 65536,
    allocated_pages: 144,
    free_pages: 368,
    max_pages: 512,
    page_size: 128,
    utilization: 0.2812,
    estimated_memory_mb: 1152.0,
    rounds: 28,
    total_time_s: 412.6,
  },
  device: {
    tier: 'laptop',
    tier_label: '笔记本 / 中端独显',
    score: 58.8,
    recommendations: ['建议使用 int4 量化以降低显存占用', '可启用流水线并行分担层权重'],
    warnings: ['可用内存低于 6GB，长上下文可能触发交换'],
  },
  pipeline_descriptor: {
    model_id: 'qwen-1_8b',
    total_layers: 24,
    weight_bytes: 3673657344,
    model_type: 'qwen',
    pipeline_runtime_supported: true,
    runtime_block_reason: '',
  },
};

export const nodesFixture: ClusterNodesResponse = {
  nodes: [
    {
      node_id: 'master',
      role: 'master',
      node_type: 'pc',
      state: 'online',
      address: '100.90.76.108:8888',
      hostname: 'localhost',
      device_info: {
        tier: 'laptop',
        tier_label: '笔记本 / 中端独显',
        score_total: 58.8,
        cpu: {
          model_name: '13th Gen Intel(R) Core(TM) i9-13900H',
          physical_cores: 14,
          logical_cores: 20,
          usage_percent: 18.8,
        },
        ram: { total_gb: 15.6, available_gb: 4.7, used_gb: 10.9, percent_used: 69.7 },
        gpu: { name: 'NVIDIA GeForce RTX 4060 Laptop GPU', vram_total_gb: 8, vram_free_gb: 8 },
      },
      network_type: 'tailscale',
      connected_at: NOW - 7200,
      last_heartbeat: NOW - 2,
      avg_rtt_ms: 0.4,
      last_rtt_ms: 0.3,
      task_count: 42,
      error_count: 0,
      is_available: true,
    },
    {
      node_id: 'client_TABLET-2TLUCNU8',
      role: 'client',
      node_type: 'pc',
      state: 'online',
      address: '100.71.12.44:8888',
      hostname: 'TABLET-2TLUCNU8',
      device_info: {
        tier: 'tablet',
        tier_label: '平板 / 集显',
        score_total: 21.4,
        cpu: { model_name: 'Intel(R) Core(TM) i5-1035G1', physical_cores: 4, logical_cores: 8, usage_percent: 31.2 },
        ram: { total_gb: 7.8, available_gb: 2.1, used_gb: 5.7, percent_used: 73.1 },
        gpu: { name: 'Intel(R) UHD Graphics', vram_total_gb: 0, vram_free_gb: 0 },
      },
      network_type: 'tailscale',
      connected_at: NOW - 3600,
      last_heartbeat: NOW - 4,
      avg_rtt_ms: 24.6,
      last_rtt_ms: 19.8,
      task_count: 7,
      error_count: 1,
      is_available: true,
    },
    {
      node_id: 'client_PHONE-A52',
      role: 'client',
      node_type: 'android',
      state: 'offline',
      address: '100.64.8.9:8889',
      hostname: 'PHONE-A52',
      device_info: {
        tier: 'phone',
        tier_label: '手机 / SoC',
        score_total: 9.2,
        cpu: { model_name: 'Snapdragon 720G', physical_cores: 8, logical_cores: 8, usage_percent: 0 },
        ram: { total_gb: 5.6, available_gb: 3.4, used_gb: 2.2, percent_used: 39.3 },
      },
      network_type: 'tailscale',
      connected_at: NOW - 18000,
      last_heartbeat: NOW - 640,
      avg_rtt_ms: 88.4,
      last_rtt_ms: 0,
      task_count: 0,
      error_count: 3,
      is_available: false,
    },
  ],
  count: 3,
  online_count: 2,
  offline_count: 1,
};

export const clusterStatusFixture: ClusterStatusResponse = {
  run_mode: 'distributed',
  nodes_ready: true,
  nodes: Object.fromEntries(nodesFixture.nodes.map((node) => [node.node_id, node])),
  current_task: { task_id: 'task_9f31c2', state: 'running' },
  tcp_server: { host: '100.90.76.108', port: 8888, connected: true },
  pipeline: { prepared: true, model_id: 'qwen-1_8b' },
  pipeline_queue: { paused: false, queue_size: 3 },
  network_path: { type: 'tailscale', healthy: true },
};

export const clusterConfigFixture: ClusterConfigResponse = {
  max_nodes: 3,
  network: { server_ip: '100.90.76.108', server_port: 8888, heartbeat_interval_s: 5 },
  model: { quant_type: 'int4', page_size: 512, max_seq_len: 8192 },
  node_role: 'master',
  node_id: 'master',
};

export const clusterInviteFixture: ClusterInviteResponse = {
  master_host: '100.90.76.108',
  master_port: 8888,
  node_count: 3,
  max_nodes: 3,
  has_capacity: false,
  identity_verified: true,
  identity_reason: 'local identity matches cluster record',
  mac_addresses: ['AA:BB:CC:DD:EE:FF'],
};

export const masterHealthFixture: MasterHealthResponse = {
  master_online: true,
  last_seen_seconds_ago: 0,
  stale: false,
  master_host: '100.90.76.108',
  master_port: 8888,
  source: 'self',
};

export const queueFixture: QueueResponse = {
  running: true,
  strategy: 'mlfq',
  paused: false,
  current_task: {
    task_id: 'task_9f31c2',
    session_id: 'default',
    priority: 0,
    wait_s: 1.2,
    estimated_s: 6.4,
    prompt_tokens: 96,
  },
  queue_size: 3,
  q0_depth: 1,
  q1_depth: 1,
  q2_depth: 1,
  q0: [{ task_id: 'task_a1b2c3', session_id: 'default', priority: 0, wait_s: 2.4, estimated_s: 3.1, prompt_tokens: 84, aged: false }],
  q1: [{ task_id: 'task_d4e5f6', session_id: 'sess_commit', priority: 1, wait_s: 41.8, estimated_s: 12.6, prompt_tokens: 402, aged: false }],
  q2: [{ task_id: 'task_7890ab', session_id: 'default', priority: 2, wait_s: 138.2, estimated_s: 28.9, prompt_tokens: 1284, aged: true }],
  completed_count: 42,
  max_size: 100,
  preempt_stats: { count: 3, last_time: NOW - 220, total_overhead_ms: 184.5 },
};

export const workflowsFixture: WorkflowsResponse = {
  enabled: true,
  available: true,
  local_available: true,
  local_provider_ready: true,
  role: 'master',
  templates: ['dual_candidate'],
  providers: ['local_full_model'],
  provider_status: [
    {
      provider_id: 'local_full_model',
      provider_kind: 'local_full_model',
      supported_stage_types: ['full_inference', 'aggregate', 'image_prompt'],
      max_concurrency: 1,
      active_reservations: 0,
      healthy: true,
      available: true,
      node_id: 'master',
      updated_at: NOW - 12,
    },
  ],
  provider_error: '',
  worker_protocol: {
    phase: 'TC-N2.4',
    admission_state: 'n2_4_experiment_enabled_not_connected',
    connected_worker_count: 0,
    control_plane_ready: true,
    control_plane_connected: false,
    task_dispatch_enabled: false,
  },
  journal: { available: true, record_count: 12 },
  workflows: [
    {
      workflow_id: 'wf_20260820_0012',
      session_id: 'default',
      template: 'dual_candidate',
      state: 'succeeded',
      created_at: NOW - 480,
      updated_at: NOW - 420,
      stages: [
        { stage_id: 'cand_a', stage_type: 'full_inference', state: 'succeeded', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 478, finished_at: NOW - 450 },
        { stage_id: 'cand_b', stage_type: 'full_inference', state: 'succeeded', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 450, finished_at: NOW - 430 },
        { stage_id: 'agg', stage_type: 'aggregate', state: 'succeeded', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 430, finished_at: NOW - 420 },
      ],
    },
    {
      workflow_id: 'wf_20260820_0013',
      session_id: 'sess_commit',
      template: 'dual_candidate',
      state: 'running',
      created_at: NOW - 60,
      updated_at: NOW - 4,
      stages: [
        { stage_id: 'cand_a', stage_type: 'full_inference', state: 'succeeded', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 58, finished_at: NOW - 30 },
        { stage_id: 'cand_b', stage_type: 'full_inference', state: 'running', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 30 },
        { stage_id: 'agg', stage_type: 'aggregate', state: 'pending' },
      ],
    },
    {
      workflow_id: 'wf_20260820_0011',
      session_id: 'default',
      template: 'dual_candidate',
      state: 'failed',
      created_at: NOW - 1800,
      updated_at: NOW - 1740,
      error: 'PROVIDER_UNAVAILABLE: 本地全模型提供者被占用',
      stages: [
        { stage_id: 'cand_a', stage_type: 'full_inference', state: 'succeeded', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 1798, finished_at: NOW - 1770 },
        { stage_id: 'cand_b', stage_type: 'full_inference', state: 'failed', provider_id: 'local_full_model', node_id: 'master', started_at: NOW - 1770, error: 'PROVIDER_UNAVAILABLE' },
      ],
    },
  ],
};

export const capacityFixture: PipelineCapacityResponse = {
  model_id: 'qwen-1_8b',
  model_type: 'qwen',
  total_layers: 24,
  raw_model_bytes: 3673657344,
  candidate_node_count: 2,
  status: 'rejected',
  admitted: false,
  reason_code: 'INSUFFICIENT_PARTICIPANTS',
  reason: '可参与节点不足，已回落到单机全模型',
  plan_id: '148a97054b34445121efb500b515733b',
  assignments: [],
  control_only_nodes: ['client_TABLET-2TLUCNU8'],
  participating_node_count: 1,
  single_node_full_model_candidates: ['master', 'client_TABLET-2TLUCNU8'],
  prepared_node_count: 0,
  ready_node_count: 0,
  worker_count: 0,
  computed_at: NOW - 300,
};

export const ragFixture: RagHealthResponse = {
  schema: 'qlh.rag_store.v2',
  status: 'ok',
  journal_mode: 'wal',
  source_count: 4,
  document_count: 128,
  chunk_count: 2841,
  fts_chunk_count: 2841,
  embedding_count: 2841,
  query_event_count: 96,
};

export const ragSearchFixture: RagSearchResponse = {
  mode: 'fts',
  provider: null,
  results: [
    { chunk_id: 'chunk_overview_01', source_id: 'runtime-notes', relative_ref: 'docs/runtime.md', revision: 3, ordinal: 12, snippet: 'The local runtime keeps model loading, device selection, and queue admission as separate control surfaces.' },
    { chunk_id: 'chunk_overview_02', source_id: 'frontend-plan', relative_ref: 'docs/frontend-plan.md', revision: 2, ordinal: 8, snippet: 'Use explicit empty and unavailable states when the backend is offline or the current role cannot access an endpoint.' },
  ],
  count: 2,
  storage: 'sqlite',
};

export const sessionsFixture: SessionsResponse = {
  sessions: [
    { id: 'default', title: '新对话', message_count: 56, created_at: '2026-08-18T12:28:05.340855Z', updated_at: '2026-08-20T15:28:56.588493Z' },
    { id: 'sess_commit', title: '提交信息生成', message_count: 4, created_at: '2026-08-20T14:38:13.739033Z', updated_at: '2026-08-20T15:28:44.048113Z' },
  ],
  active_session_id: 'default',
  total: 2,
  source: 'sqlite',
};

export const diffusionCapabilitiesFixture: DiffusionCapabilitiesResponse = {
  state: 'loaded',
  loaded: true,
  loaded_artifact: {
    artifact_id: 'sd15-demo',
    name: 'SD 1.5 Demo',
    artifact: { artifact_kind: 'sd15_pipeline', loadable: true, precision: 'fp16' },
  },
  capabilities: { txt2img: true, img2img: false, inpaint: false },
  dependencies: { torch: true, diffusers: true, transformers: true },
  presets: [
    {
      preset_id: 'sd15_demo',
      model_id: 'sd15-demo',
      prompt: 'gothic observatory at night',
      negative_prompt: 'blurry, low quality',
      width: 512,
      height: 512,
      steps: 24,
      guidance_scale: 7.5,
      scheduler: 'ddim',
      seeds: [1742],
    },
  ],
};

export const diffusionArtifactsFixture: DiffusionArtifactsResponse = {
  artifacts: [
    {
      artifact_id: 'sd15-demo',
      name: 'SD 1.5 Demo',
      registered_at: NOW - 7200,
      artifact: { artifact_kind: 'sd15_pipeline', loadable: true, precision: 'fp16', size_bytes: 3673657344 },
    },
  ],
};

export const diffusionAssetsFixture: DiffusionAssetsResponse = {
  assets: [
    {
      asset_id: 'sd15-demo',
      name: 'SD 1.5 Demo Pipeline',
      artifact_id: 'sd15-demo',
      artifact_kind: 'sd15_pipeline',
      description: '用于离线预览的本地 SD 1.5 资产。',
      installed: true,
      present_bytes: 3673657344,
      total_bytes: 3673657344,
    },
  ],
};

export const modelsFixture: ModelsResponse = {
  active_model_id: 'qwen-1_8b',
  models: [
    {
      model_id: 'qwen-1_8b',
      name: 'Qwen 1.8B Chat',
      model_type: 'qwen',
      is_builtin: true,
      recommended_vram_gb: 3.5,
      max_context: 8192,
      quant_types: ['fp16', 'int8', 'int4', 'Q4_K_M'],
      description: 'Compact local assistant model for chat and workflow tasks.',
      location: 'builtin',
      is_available: true,
      available_formats: ['gguf', 'safetensors'],
      has_gguf: true,
      has_safetensors: true,
      supported_engines: ['llama_cpp', 'pytorch'],
      preferred_engine: 'llama_cpp',
      default_quant_type: 'Q4_K_M',
      requires_cuda: false,
      expected_paths: ['models/qwen-1_8b'],
    },
    {
      model_id: 'qwen3-8b-sidecar',
      name: 'Qwen3 8B Sidecar',
      model_type: 'qwen3',
      is_builtin: false,
      is_experimental: true,
      recommended_vram_gb: 10,
      max_context: 32768,
      quant_types: ['int4'],
      description: 'Inventory-only asset. Requires the sidecar runtime gate.',
      location: 'local asset',
      is_available: false,
      unavailable_reason: 'Sidecar asset is registered for preflight only.',
      available_formats: ['safetensors'],
      has_safetensors: true,
      has_gguf: false,
      supported_engines: [],
      preferred_engine: 'auto',
      default_quant_type: 'int4',
      requires_cuda: true,
      expected_paths: ['models/qwen3-8b-sidecar'],
    },
    {
      model_id: 'deepseek-r1-7b',
      name: 'DeepSeek R1 7B',
      model_type: 'deepseek',
      is_builtin: false,
      is_experimental: true,
      recommended_vram_gb: 8,
      max_context: 16384,
      quant_types: ['int4'],
      description: 'Registered model awaiting a local weight package.',
      location: 'external',
      is_available: false,
      unavailable_reason: 'Model weights are not present on this node.',
      available_formats: [],
      has_safetensors: false,
      has_gguf: false,
      supported_engines: [],
      preferred_engine: 'auto',
      default_quant_type: 'int4',
      requires_cuda: true,
      expected_paths: ['models/deepseek-r1-7b'],
    },
  ],
};

export const availableModelsFixture: AvailableModelsResponse = {
  current: 'Q4_K_M',
  current_engine: 'llama_cpp',
  models: [
    { id: 'gguf', name: 'GGUF 量化', engine: 'llama_cpp', description: 'CPU/集显友好', is_available: true },
    { id: 'int4', name: 'INT4 量化', engine: 'pytorch', memory_gb: 1.8, speed_tok_s: 29, is_available: true },
    { id: 'int8', name: 'INT8 量化', engine: 'pytorch', memory_gb: 2.3, speed_tok_s: 10, is_available: true },
  ],
  available_engines: [
    { id: 'llama_cpp', name: 'llama.cpp + GGUF', description: 'GGUF quantized runtime.', requires_cuda: false },
    { id: 'pytorch', name: 'PyTorch + Safetensors', description: 'Safetensors runtime.', requires_cuda: false },
  ],
};

export const currentModelFixture: CurrentModelResponse = {
  loaded: true,
  pipeline_prepared: false,
  quant_type: 'Q4_K_M',
  model_id: 'qwen-1_8b',
  model_name: 'Qwen 1.8B Chat',
  model_path: 'models/qwen-1_8b/Qwen-1_8B-Chat.Q4_K_M.gguf',
  engine: 'llama_cpp',
};

export const localModelAssetsFixture: LocalModelAssetsResponse = {
  assets: [
    {
      model_id: 'qwen3-8b-sidecar',
      name: 'Qwen3 8B Sidecar',
      huggingface_id: 'Qwen/Qwen3-8B',
      model_type: 'safetensors',
      available_formats: ['safetensors'],
      model_path: 'models/qwen3-8b-sidecar',
      max_context: 32768,
      architectures: ['Qwen3ForCausalLM'],
      total_bytes: 8589934592,
      asset_ids: ['qwen3-8b-sidecar'],
      source_paths: ['models/qwen3-8b-sidecar'],
      manifest_paths: ['models/qwen3-8b-sidecar/.qlh-model-asset.json'],
      integrity: 'manifest_verified',
      runtime_profile: 'qwen3_sidecar',
      runtime_hint: 'Inventory only until the sidecar preflight gate passes.',
      runtime_status: 'inventory_only',
      runtime_action: 'qwen3_preflight',
    },
    {
      model_id: 'qwen-1_8b',
      name: 'Qwen 1.8B Chat',
      model_type: 'gguf',
      available_formats: ['gguf'],
      model_path: '',
      gguf_path: 'models/qwen-1_8b/Qwen-1_8B-Chat.Q4_K_M.gguf',
      max_context: 8192,
      architectures: ['QWenForCausalLM'],
      total_bytes: 1929379840,
      asset_ids: ['qwen-1_8b-gguf'],
      source_paths: ['models/qwen-1_8b'],
      manifest_paths: [],
      integrity: 'filesystem_discovered',
      runtime_profile: 'manual_runtime_selection',
      runtime_hint: 'Available through the classic model loader.',
      runtime_status: 'inventory_only',
      runtime_action: null,
    },
  ],
  summary: { total: 2, total_bytes: 10519314432 },
};

export const deviceProfileFixture: DeviceProfileResponse = {
  tier: 'laptop',
  tier_label: 'Performance laptop',
  tier_icon: 'GPU',
  score_total: 78,
  score_breakdown: { gpu: 42, ram: 23, cpu: 13 },
  cpu: { model_name: 'AMD Ryzen 7 7840HS', physical_cores: 8, logical_cores: 16, freq_max_mhz: 5100 },
  ram: { total_gb: 32, available_gb: 18.6, used_gb: 13.4, percent_used: 41.9 },
  gpu: { name: 'NVIDIA GeForce RTX 4060 Laptop GPU', gpu_type: 'dedicated', is_integrated: false, cuda_available: true, vram_total_gb: 8, vram_free_gb: 6.2 },
  gpus: [
    { name: 'NVIDIA GeForce RTX 4060 Laptop GPU', gpu_type: 'dedicated', is_integrated: false, cuda_available: true, vram_total_gb: 8, vram_free_gb: 6.2 },
    { name: 'AMD Radeon 780M', gpu_type: 'integrated', is_integrated: true, cuda_available: false, vram_total_gb: 0, vram_free_gb: 0, mps_available: false },
  ],
  selected_gpu_index: 0,
  recommendations: ['Use INT4 for the best memory/speed balance.', 'Keep the dedicated GPU selected for local PyTorch inference.'],
  warnings: ['Switching GPU takes effect after the next model reload.'],
};

export const conversationFixture: ConversationResponse = {
  messages: [
    {
      role: 'user',
      content: '当前集群能跑分布式推理吗？',
      created_at: '2026-08-20T15:27:10.101000Z',
    },
    {
      role: 'assistant',
      content:
        '暂时不能。流水线准入检查判定「可参与节点不足」：注册了 2 个节点，但只有 1 个满足参与条件，已回落到单机全模型。补齐第二个可参与节点后会自动切回分布式。',
      created_at: '2026-08-20T15:27:14.882000Z',
      // 字段名与后端 done 事件一致，演示数据不能教出错的契约
      metrics: {
        tokens_per_second: 18.4,
        generated_tokens: 96,
        completion_tokens: 96,
        usage: { prompt_tokens: 41, completion_tokens: 96, total_tokens: 137 },
        engine: 'llama_cpp',
        execution_mode: 'fallback_full_model',
        mode: 'fallback_full_model',
        route: 'master_pipeline_fallback_full_model',
        fallback: true,
        fallback_reason: 'insufficient_participants',
        distributed_used: false,
        serving_node_id: 'master',
        workers_used: [],
      },
    },
    {
      role: 'user',
      content: '那 task_graph 那条 ERROR 是什么？',
      created_at: '2026-08-20T15:28:50.400000Z',
    },
    {
      role: 'assistant',
      content:
        'cand_b 这个 stage 拿不到 provider（PROVIDER_UNAVAILABLE），双候选校验只剩一路，聚合阶段因此降级。左侧「运行日志」筛到 ERROR 能看到原始记录。',
      created_at: '2026-08-20T15:28:56.588000Z',
      metrics: {
        tokens_per_second: 21.7,
        generated_tokens: 74,
        completion_tokens: 74,
        usage: { prompt_tokens: 58, completion_tokens: 74, total_tokens: 132 },
        engine: 'llama_cpp',
        execution_mode: 'fallback_full_model',
        mode: 'fallback_full_model',
        route: 'master_pipeline_fallback_full_model',
        fallback: true,
        fallback_reason: 'insufficient_participants',
        distributed_used: false,
        serving_node_id: 'master',
        workers_used: [],
      },
    },
  ],
  count: 4,
  source: 'sqlite',
};

export const logsFixture: RecentLogsResponse = {
  logs: [
    { timestamp: '2026-08-20 23:29:45', level: 'INFO', levelno: 20, name: 'api_server', message: 'event=http_request method=GET path=/api/cluster/pipeline-capacity status=200 duration_ms=3', request_id: '35b1585250b44c27', node_id: 'master', seq: 2745 },
    { timestamp: '2026-08-20 23:29:12', level: 'WARNING', levelno: 30, name: 'scheduler', message: 'event=pipeline_capacity_rejected reason=insufficient_participants candidates=2', request_id: '-', node_id: 'master', seq: 2702 },
    { timestamp: '2026-08-20 23:28:56', level: 'ERROR', levelno: 40, name: 'task_graph', message: 'event=stage_failed stage_id=cand_b code=PROVIDER_UNAVAILABLE', request_id: 'c81f0a4d9e2b7f36', node_id: 'master', seq: 2688 },
    { timestamp: '2026-08-20 23:28:44', level: 'INFO', levelno: 20, name: 'scheduler', message: 'event=node_heartbeat node_id=client_TABLET-2TLUCNU8 rtt_ms=19.8', request_id: '-', node_id: 'master', seq: 2671 },
  ],
  count: 4,
  matched: 2746,
  buffer_size: 2746,
  buffer_capacity: 5000,
  truncated: true,
};

export const logFilesFixture: LogFilesResponse = {
  files: [
    { name: 'qlh-api-20260821.log', size: 184_320, modified: '2026-08-21T09:29:45.000Z' },
    { name: 'qlh-scheduler-20260820.log', size: 98_304, modified: '2026-08-20T23:58:22.000Z' },
    { name: 'qlh-worker-tablet-20260820.log', size: 42_018, modified: '2026-08-20T20:06:10.000Z' },
  ],
};

export const logStatsFixture: LogStatsResponse = {
  files_count: 3,
  files_total_bytes: 324_642,
  buffer_size: 2746,
  buffer_capacity: 5000,
  buffer_total_seen: 2879,
  buffer_dropped_estimate: 133,
  levels: { INFO: 2381, WARNING: 241, ERROR: 116, DEBUG: 8 },
  loggers: { api_server: 1287, scheduler: 934, task_graph: 298, cluster_transport: 227 },
  nodes: { master: 2288, client_TABLET_2TLUCNU8: 458 },
  node_id: 'master',
  device_ip: '100.90.76.108',
};

export const nodesLogSummaryFixture: NodesLogSummaryResponse = {
  local: {
    node_id: 'master',
    role: 'master',
    state: 'online',
    files_count: 3,
    files_total_bytes: 324_642,
    buffer_size: 2746,
    buffer_capacity: 5000,
    buffer_total_seen: 2879,
    buffer_dropped_estimate: 133,
    levels: logStatsFixture.levels,
  },
  workers: [
    {
      node_id: 'client_TABLET-2TLUCNU8',
      role: 'client',
      state: 'online',
      files_count: 1,
      files_total_bytes: 42_018,
      buffer_size: 418,
      buffer_capacity: 1000,
      buffer_total_seen: 438,
      buffer_dropped_estimate: 20,
      levels: { INFO: 381, WARNING: 29, ERROR: 8 },
    },
  ],
  total_workers: 1,
};

export const nodesLogAggregateFixture: NodeLogAggregateResponse = {
  local: { node_id: 'master', logs: logsFixture.logs },
  workers: [
    {
      node_id: 'client_TABLET-2TLUCNU8',
      count: 2,
      logs: [
        { timestamp: '2026-08-20 23:29:37', level: 'INFO', levelno: 20, name: 'worker', message: 'event=heartbeat_sent rtt_ms=19.8', node_id: 'client_TABLET-2TLUCNU8', seq: 82 },
        { timestamp: '2026-08-20 23:26:14', level: 'WARNING', levelno: 30, name: 'worker', message: 'event=thermal_limit throttle=true', node_id: 'client_TABLET-2TLUCNU8', seq: 77 },
      ],
    },
  ],
  limit: 50,
  filters: {},
  total_workers: 1,
};

export const canVoteFixture: CanVoteResponse = {
  node_id: 'master',
  can_vote: true,
  reason: 'Current node is an eligible reviewer.',
};

export const reviewTicketsFixture: ReviewTicketsResponse = {
  tickets: [
    {
      ticket_id: 'review_20260821_01',
      status: 'pending',
      created_at: NOW - 2_400,
      created_by: 'master',
      target_node_id: 'client_TABLET-2TLUCNU8',
      transfer_reason: 'Planned maintenance window for the current primary node.',
      score: 1,
      expires_at: NOW + 86_400,
      votes: [{ voter_node_id: 'master', value: 1, timestamp: NOW - 1_800, comment: 'Approved for the maintenance window.' }],
    },
    {
      ticket_id: 'review_20260818_02',
      status: 'approved',
      created_at: NOW - 250_000,
      resolved_at: NOW - 180_000,
      created_by: 'master',
      target_node_id: 'client_STUDIO-01',
      transfer_reason: 'Recovery drill.',
      score: 2,
      expires_at: NOW - 190_000,
      votes: [
        { voter_node_id: 'master', value: 1, timestamp: NOW - 240_000, comment: 'Approved.' },
        { voter_node_id: 'client_STUDIO-01', value: 1, timestamp: NOW - 230_000, comment: 'Ready.' },
      ],
    },
  ],
  count: 2,
};
