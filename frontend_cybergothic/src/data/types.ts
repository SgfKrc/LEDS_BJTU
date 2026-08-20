/**
 * 后端数据契约 — 字段以 2026-08-20 实际抓取的 FastAPI 响应为准。
 *
 * 只声明本前端实际消费的字段；后端返回的其他字段保持透传但不做类型约束，
 * 避免 API 扩展时被迫改动视觉组件（§5.1 分层要求）。
 */

// ---- 通用状态语义 ----

export type LoadState = 'idle' | 'loading' | 'ready' | 'error';

export type StatusTone = 'ok' | 'warn' | 'danger' | 'info' | 'idle';

export interface ApiErrorShape {
  detail: string;
  status: number;
  requestId: string | null;
  path: string;
}

// ---- /api/cluster/nodes ----

export interface NodeGpu {
  name?: string;
  vram_total_gb?: number;
  vram_free_gb?: number;
  cuda_available?: boolean;
  compute_capability?: string;
  gpu_type?: string;
  index?: number;
}

export interface NodeDeviceInfo {
  tier?: string;
  tier_label?: string;
  score_total?: number;
  cpu?: {
    model_name?: string;
    physical_cores?: number;
    logical_cores?: number;
    usage_percent?: number;
  };
  ram?: {
    total_gb?: number;
    available_gb?: number;
    used_gb?: number;
    percent_used?: number;
  };
  gpu?: NodeGpu;
  gpus?: NodeGpu[];
}

export interface ClusterNode {
  node_id: string;
  role: string;
  node_type: string;
  state: string;
  address: string;
  hostname: string;
  device_info: NodeDeviceInfo;
  network_type: string;
  connected_at: number;
  last_heartbeat: number;
  avg_rtt_ms: number;
  last_rtt_ms: number;
  task_count: number;
  error_count: number;
  is_available: boolean;
}

export interface ClusterNodesResponse {
  nodes: ClusterNode[];
  count: number;
  online_count: number;
  offline_count: number;
}

// ---- /api/cluster/status ----

export interface ClusterStatusResponse {
  run_mode: string;
  nodes_ready: boolean;
  nodes: Record<string, ClusterNode>;
  current_task: Record<string, unknown> | null;
  tcp_server: Record<string, unknown> | null;
  pipeline: Record<string, unknown> | null;
  pipeline_queue: Record<string, unknown> | null;
  network_path: Record<string, unknown> | null;
}

// ---- /api/status ----

export interface SystemStatusResponse {
  model_loaded: boolean;
  pipeline_prepared: boolean;
  current_quant: string | null;
  model_name: string;
  model_path: string;
  active_model_id: string | null;
  engine: string;
  run_mode: string;
  node_role: string;
  node_id: string;
  max_nodes: number;
  conversation_turns: number;
  gpu: {
    name?: string;
    total_mb?: number;
    allocated_mb?: number;
    reserved_mb?: number;
    utilization?: number;
  };
  kv_cache: {
    total_tokens: number;
    max_tokens: number;
    allocated_pages: number;
    free_pages: number;
    max_pages: number;
    page_size: number;
    utilization: number;
    estimated_memory_mb: number;
    rounds: number;
    total_time_s: number;
  };
  device: {
    tier?: string;
    tier_label?: string;
    score?: number;
    recommendations?: string[];
    warnings?: string[];
  } | null;
  pipeline_descriptor?: {
    model_id?: string;
    total_layers?: number;
    weight_bytes?: number;
    model_type?: string;
    pipeline_runtime_supported?: boolean;
    runtime_block_reason?: string;
  };
}

// ---- /api/cluster/queue ----

export interface QueueTask {
  task_id?: string;
  session_id?: string;
  priority?: number;
  wait_s?: number;
  estimated_s?: number;
  aged?: boolean;
  prompt_tokens?: number;
}

export interface QueueResponse {
  running: boolean;
  strategy: string;
  paused: boolean;
  current_task: QueueTask | null;
  queue_size: number;
  q0_depth: number;
  q1_depth: number;
  q2_depth: number;
  q0: QueueTask[];
  q1: QueueTask[];
  q2: QueueTask[];
  completed_count: number;
  max_size: number;
  preempt_stats?: {
    count?: number;
    last_time?: number;
    total_overhead_ms?: number;
  };
}

// ---- /api/workflows ----

export interface WorkflowStage {
  stage_id?: string;
  stage_type?: string;
  state?: string;
  provider_id?: string;
  node_id?: string;
  started_at?: number;
  finished_at?: number;
  error?: string;
}

export interface WorkflowRecord {
  workflow_id: string;
  session_id?: string;
  template?: string;
  state?: string;
  created_at?: number;
  updated_at?: number;
  stages?: WorkflowStage[];
  error?: string;
}

export interface ProviderStatus {
  provider_id: string;
  provider_kind?: string;
  supported_stage_types?: string[];
  max_concurrency?: number;
  active_reservations?: number;
  healthy?: boolean;
  available?: boolean;
  node_id?: string;
  updated_at?: number;
}

export interface WorkflowsResponse {
  enabled: boolean;
  available: boolean;
  local_available?: boolean;
  local_provider_ready?: boolean;
  role: string;
  templates: string[];
  providers: string[];
  provider_status: ProviderStatus[];
  provider_error: string;
  worker_protocol?: {
    phase?: string;
    admission_state?: string;
    connected_worker_count?: number;
    control_plane_ready?: boolean;
    control_plane_connected?: boolean;
    task_dispatch_enabled?: boolean;
  };
  journal?: {
    available?: boolean;
    path?: string;
    record_count?: number;
  };
  workflows: WorkflowRecord[];
}

// ---- /api/cluster/pipeline-capacity ----

export interface PipelineCapacityResponse {
  model_id: string;
  model_type: string;
  total_layers: number;
  raw_model_bytes: number;
  candidate_node_count: number;
  status: string;
  admitted: boolean;
  reason_code: string;
  reason: string;
  plan_id: string;
  assignments: Array<{
    node_id?: string;
    start_layer?: number;
    end_layer?: number;
    layer_count?: number;
  }>;
  control_only_nodes: string[];
  participating_node_count: number;
  single_node_full_model_candidates: string[];
  prepared_node_count: number;
  ready_node_count: number;
  worker_count: number;
  computed_at: number;
}

// ---- /api/logs/recent ----

export interface LogEntry {
  timestamp: string;
  level: string;
  levelno: number;
  name: string;
  message: string;
  request_id?: string;
  node_id?: string;
  seq: number;
}

export interface RecentLogsResponse {
  logs: LogEntry[];
  count: number;
  matched: number;
  buffer_size: number;
  buffer_capacity: number;
  truncated?: boolean;
}

// ---- /api/sessions ----

export interface SessionSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface SessionsResponse {
  sessions: SessionSummary[];
  active_session_id: string;
  total: number;
  source: string;
}

// ---- /api/rag/health ----

export interface RagHealthResponse {
  schema: string;
  status: string;
  journal_mode: string;
  source_count: number;
  document_count: number;
  chunk_count: number;
  fts_chunk_count: number;
  embedding_count: number;
  query_event_count: number;
}

// ---- /api/cluster/my-role ----

export interface MyRoleResponse {
  node_role: string;
  node_id: string;
  is_master: boolean;
  is_client: boolean;
  max_nodes: number;
  run_mode: string;
}

// ---- 对话（/api/conversations、/api/chat/stream） ----

/** 后端 metrics 字段随路由方式变化，这里只声明界面会展示的部分。 */
/**
 * /api/chat/stream 的 done 事件里的 metrics。
 *
 * 字段名以实际响应为准（后端 done 事件抓包核对过）：token 数是
 * generated_tokens / completion_tokens，累计用量在嵌套的 usage 里，
 * 没有 new_tokens / elapsed_seconds / node_count 这些字段。
 * 索引签名开着，用不到的字段（layer_assignments 等）不必逐个声明。
 */
export interface ChatMetrics {
  tokens_per_second?: number;
  generated_tokens?: number;
  completion_tokens?: number;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  route?: string;
  execution_mode?: string;
  mode?: string;
  engine?: string;
  fallback?: boolean;
  fallback_reason?: string;
  distributed_used?: boolean;
  serving_node_id?: string;
  workers_used?: string[];
  generation_id?: string;
  request_id?: string;
  [key: string]: unknown;
}

export interface ConversationMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at?: string;
  metrics?: ChatMetrics;
}

export interface ConversationResponse {
  messages: ConversationMessage[];
  count: number;
  source: string;
}

/**
 * POST /api/chat/upload 的响应。
 *
 * 后端只接文本类扩展名（ALLOWED_TEXT_EXTENSIONS），上限 5 MB / 5000 行，
 * 超行数时截断并置 truncated=true，正文已解码好放在 content 里。
 */
export interface ChatUploadResponse {
  filename: string;
  extension: string;
  language: string;
  char_count: number;
  word_count: number;
  line_count: number;
  total_lines: number;
  truncated: boolean;
  truncated_lines: number;
  size_bytes: number;
  content: string;
}

/** 输入区里挂着的一个附件（已上传完成，正文在手上）。 */
export interface ChatAttachment {
  id: string;
  filename: string;
  language: string;
  lineCount: number;
  sizeBytes: number;
  truncated: boolean;
  truncatedLines: number;
  content: string;
}

/** 界面用的消息模型：比后端多了流式态和本地生成的 id。 */
export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** 深度思考内容，可折叠展示。 */
  thinking?: string;
  /** 正在流式接收中；用于打字光标和「停止」按钮。 */
  streaming?: boolean;
  /** 该轮失败的原因；失败轮仍留在列表里，便于重试。 */
  error?: string;
  metrics?: ChatMetrics;
  createdAt?: string;
  /**
   * 这一轮随消息发出的附件摘要。
   * 只留文件名和行数：正文已经拼进 content 发给模型了，
   * 在气泡里再铺一遍会把几千行日志糊满整栏。
   */
  attachments?: { filename: string; lineCount: number }[];
}

// ---- 视图层聚合模型 ----

/** Overview 首屏一次性拉取的聚合快照。 */
export interface OverviewSnapshot {
  status: SystemStatusResponse | null;
  nodes: ClusterNodesResponse | null;
  queue: QueueResponse | null;
  workflows: WorkflowsResponse | null;
  capacity: PipelineCapacityResponse | null;
  rag: RagHealthResponse | null;
  /** 部分接口失败时记录，用于 inline 降级提示而不是整页报错。 */
  partialErrors: string[];
}
