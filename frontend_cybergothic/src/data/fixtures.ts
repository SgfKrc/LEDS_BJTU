/**
 * 本地 fixture — 与真实接口完全相同的数据形状（§4.3）。
 *
 * 用途：后端未启动时的离线预览、以及各页面状态矩阵的手工验证。
 * 演示数据只允许出现在本文件，不得散落到组件内部。
 * 通过 URL 参数 `?fixtures=1` 或 localStorage 开关启用，默认关闭。
 */

import type {
  ClusterNodesResponse,
  ConversationResponse,
  PipelineCapacityResponse,
  QueueResponse,
  RagHealthResponse,
  RecentLogsResponse,
  SessionsResponse,
  SystemStatusResponse,
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

export const sessionsFixture: SessionsResponse = {
  sessions: [
    { id: 'default', title: '新对话', message_count: 56, created_at: '2026-08-18T12:28:05.340855Z', updated_at: '2026-08-20T15:28:56.588493Z' },
    { id: 'sess_commit', title: '提交信息生成', message_count: 4, created_at: '2026-08-20T14:38:13.739033Z', updated_at: '2026-08-20T15:28:44.048113Z' },
  ],
  active_session_id: 'default',
  total: 2,
  source: 'sqlite',
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
