/**
 * 后端数据契约 — 字段以 2026-08-20 实际抓取的 FastAPI 响应为准。
 *
 * 只声明本前端实际消费的字段；后端返回的其他字段保持透传但不做类型约束，
 * 避免 API 扩展时被迫改动视觉组件（§5.1 分层要求）。
 */

// ---- 通用状态语义 ----

export type LoadState = 'idle' | 'loading' | 'ready' | 'error';

/** 请求失败的稳定语义，页面不应通过解析中文错误文案来判断状态。 */
export type ApiErrorKind =
  | 'aborted'
  | 'network'
  | 'timeout'
  | 'unauthorized'
  | 'forbidden'
  | 'not_found'
  | 'conflict'
  | 'rate_limited'
  | 'server'
  | 'http'
  | 'unknown';

export type StatusTone = 'ok' | 'warn' | 'danger' | 'info' | 'idle';

export interface ApiErrorShape {
  detail: string;
  status: number;
  requestId: string | null;
  path: string;
  kind?: ApiErrorKind;
  retryable?: boolean;
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

// ---- /api/cluster/config, /api/cluster/invite, /api/cluster/master-health ----

export interface ClusterConfigResponse {
  max_nodes?: number;
  network?: {
    server_ip?: string;
    server_port?: number;
    heartbeat_interval_s?: number;
    [key: string]: unknown;
  };
  model?: Record<string, unknown>;
  layers?: Record<string, unknown>;
  task_stats?: Record<string, unknown>;
  node_role?: string;
  node_id?: string;
  [key: string]: unknown;
}

export interface ClusterInviteResponse {
  master_host?: string;
  master_port?: number;
  node_count?: number;
  max_nodes?: number;
  has_capacity?: boolean;
  identity_verified?: boolean;
  identity_reason?: string;
  mac_addresses?: string[];
  [key: string]: unknown;
}

export interface MasterHealthResponse {
  master_online: boolean;
  last_seen_seconds_ago?: number;
  stale?: boolean;
  master_host?: string;
  master_port?: number;
  source?: string;
  [key: string]: unknown;
}

export interface ClusterMutationResponse {
  status?: string;
  node_id?: string;
  max_nodes?: number;
  reason?: string;
  message?: string;
  [key: string]: unknown;
}

// ---- /api/auth*, /api/users, /api/auth/tailscale ----

export interface AuthCapabilityResponse {
  required: boolean;
  available?: boolean;
  enforced?: boolean;
  service?: string;
  reason_code?: string;
  mode?: string;
  bootstrap_available?: boolean;
  [key: string]: unknown;
}

export interface AuthUser {
  user_id: string;
  username: string;
  display_name?: string;
  role: 'owner' | 'admin' | 'member' | string;
  status?: 'active' | 'suspended' | 'revoked' | string;
  totp_state?: string;
  active_session_count?: number;
  aggregate_version?: number;
  [key: string]: unknown;
}

export interface AuthSessionResponse {
  session_id: string;
  expires_at?: string;
  user: AuthUser;
  [key: string]: unknown;
}

export interface AuthSessionRecord {
  session_id: string;
  user_id?: string;
  active?: boolean;
  current?: boolean;
  created_at?: string;
  last_seen_at?: string;
  expires_at?: string;
  [key: string]: unknown;
}

export interface AuthSessionsResponse {
  sessions: AuthSessionRecord[];
  [key: string]: unknown;
}

export interface ManagedUsersResponse {
  users: AuthUser[];
  [key: string]: unknown;
}

export interface AuthMutationResponse {
  status?: string;
  message?: string;
  user?: AuthUser;
  recovery_codes?: string[];
  provisioning?: {
    user_id?: string;
    authenticator_id?: string;
    secret?: string;
    qr_payload?: string;
    otpauth_uri?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface AuthProvisioning {
  user_id: string;
  authenticator_id: string;
  secret: string;
  qr_payload: string;
  otpauth_uri: string;
  algorithm?: string;
  digits?: number;
  period_seconds?: number;
  [key: string]: unknown;
}

export interface TailscaleBinding {
  binding_id: string;
  user_id?: string;
  tailnet_id?: string;
  tailscale_user_id?: string;
  node_id?: string;
  state?: 'active' | 'pending' | 'revoked' | 'expired' | string;
  authorization_method?: string;
  confirmed_at?: string;
  updated_at?: string;
  [key: string]: unknown;
}

export interface TailscaleBindingsResponse {
  bindings: TailscaleBinding[];
  [key: string]: unknown;
}

export interface LocalTailscaleStatusResponse {
  local_status?: {
    state?: string;
    available?: boolean;
    candidate?: {
      tailnet_id?: string;
      tailnet_display_name?: string;
      tailscale_user_id?: string;
      node_id?: string;
      hostname?: string;
      addresses?: string[];
      [key: string]: unknown;
    } | null;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
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
  provider?: string;
  requested_provider?: string;
  selected_provider?: string;
  provider_node_id?: string;
  provider_kind?: string;
  depends_on?: string[];
  input_bindings?: Record<string, {
    dependency_stage_id?: string;
    output_key?: string;
    [key: string]: unknown;
  }>;
  winner_attempt_id?: string;
  retry_count?: number;
  same_provider_retry_count?: number;
  last_retry_error_code?: string;
  result_rejection_count?: number;
  last_result_rejection_reason?: string;
  output_available?: boolean;
  output_sha256?: string;
  output_size_bytes?: number;
  started_at?: number;
  finished_at?: number;
  duration_seconds?: number;
  error?: string;
  attempts?: WorkflowAttempt[];
  [key: string]: unknown;
}

export interface WorkflowAttempt {
  attempt_id?: string;
  provider?: string;
  provider_kind?: string;
  provider_node_id?: string;
  reservation_id?: string;
  reservation_active?: boolean;
  lease_id?: string;
  lease_epoch?: number;
  lease_expires_at?: number;
  lease_enforced?: boolean;
  state?: string;
  started_at?: number;
  finished_at?: number;
  duration_seconds?: number;
  error?: string;
  result_metadata?: Record<string, unknown>;
  result_sha256?: string;
  [key: string]: unknown;
}

export interface WorkflowRecord {
  workflow_id: string;
  session_id?: string;
  request_id?: string;
  template?: string;
  state?: string;
  final_stage_id?: string;
  created_at?: number;
  started_at?: number;
  result_ready_at?: number;
  finished_at?: number;
  duration_seconds?: number;
  updated_at?: number;
  stage_count?: number;
  completed_stage_count?: number;
  failed_stage_count?: number;
  attempt_count?: number;
  retry_count?: number;
  same_provider_retry_count?: number;
  result_rejection_count?: number;
  cancel_requested?: boolean;
  stages?: WorkflowStage[];
  error?: string;
  observability?: {
    state?: string;
    result_ready?: boolean;
    terminal?: boolean;
    partial_result?: boolean;
    recovered_after_restart?: boolean;
    recovery_reason?: string;
    retry_count?: number;
    same_provider_retry_count?: number;
    reassignment_count?: number;
    retrying?: boolean;
    result_rejection_count?: number;
    last_result_rejection_reason?: string;
    actual_providers?: string[];
    actual_nodes?: string[];
    [key: string]: unknown;
  };
  journal?: {
    available?: boolean;
    record_count?: number;
    retention_days?: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
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

// ---- /api/logs, /api/logs/stats, /api/logs/nodes-summary ----

export interface LogFileRecord {
  name: string;
  size: number;
  modified: string;
  [key: string]: unknown;
}

export interface LogFilesResponse {
  files: LogFileRecord[];
  [key: string]: unknown;
}

export interface LogFileContentResponse {
  name: string;
  content: string;
  truncated?: boolean;
  [key: string]: unknown;
}

export interface LogStatsResponse {
  files_count: number;
  files_total_bytes: number;
  buffer_size: number;
  buffer_capacity: number;
  buffer_total_seen: number;
  buffer_dropped_estimate: number;
  levels: Record<string, number>;
  loggers: Record<string, number>;
  nodes: Record<string, number>;
  node_id?: string;
  device_ip?: string;
  [key: string]: unknown;
}

export interface ClientErrorReport {
  source?: string;
  message: string;
  url?: string;
  line?: number;
  col?: number;
  stack?: string;
  user_agent?: string;
  extra?: Record<string, unknown>;
}

export interface StorageHealthResponse {
  local?: {
    status?: string;
    backend?: string;
    writable?: boolean;
    path?: string;
    journal_mode?: string;
    synchronous?: string;
    error?: string;
    [key: string]: unknown;
  };
  remote?: {
    status?: string;
    backend?: string;
    mode?: string;
    [key: string]: unknown;
  };
  projection?: { pending_events?: number; oldest_event_age_seconds?: number | null; [key: string]: unknown };
  export?: { pending_items?: number; oldest_item_age_seconds?: number | null; [key: string]: unknown };
  effective_mode?: string;
  retirement?: { status?: string; prepared_at?: string | null; retired_at?: string | null; [key: string]: unknown };
  [key: string]: unknown;
}

export interface SpeculativeCapabilityResponse {
  enabled: boolean;
  configured?: boolean;
  available: boolean;
  execution_mode?: string;
  local_only?: boolean;
  data_scope?: string;
  reason_code?: string;
  verify_model?: string;
  gamma?: number;
  max_rounds?: number;
  [key: string]: unknown;
}

export interface SpeculativeExperimentResponse {
  content?: string;
  finish_reason?: string;
  metrics?: Record<string, unknown>;
  rounds?: number;
  request_id?: string | null;
  [key: string]: unknown;
}

export interface RagSource {
  source_id: string;
  owner_scope?: string;
  display_name?: string;
  relative_ref?: string;
  revision?: number;
  document_count?: number;
  chunk_count?: number;
  [key: string]: unknown;
}

export interface RagSourcesResponse {
  sources: RagSource[];
  storage?: string;
  [key: string]: unknown;
}

export interface NodeLogSummary {
  node_id: string;
  role?: string;
  state?: string;
  files_count?: number;
  files_total_bytes?: number;
  buffer_size?: number;
  buffer_capacity?: number;
  buffer_total_seen?: number;
  buffer_dropped_estimate?: number;
  levels?: Record<string, number>;
  error?: string;
  [key: string]: unknown;
}

export interface NodesLogSummaryResponse {
  local: NodeLogSummary;
  workers: NodeLogSummary[];
  total_workers: number;
  [key: string]: unknown;
}

export interface NodeLogAggregateResponse {
  local: { node_id: string; logs: LogEntry[]; [key: string]: unknown };
  workers: Array<{ node_id: string; logs: LogEntry[]; count?: number; error?: string; [key: string]: unknown }>;
  limit: number;
  filters?: Record<string, unknown>;
  total_workers: number;
  [key: string]: unknown;
}

// ---- /api/cluster/review/* ----

export interface ReviewTicket {
  ticket_id: string;
  target_node_id?: string;
  transfer_reason?: string;
  status?: string;
  created_at?: string | number;
  expires_at?: string | number;
  resolved_at?: string | number | null;
  created_by?: string;
  score?: number;
  votes?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface ReviewTicketsResponse {
  tickets: ReviewTicket[];
  count: number;
  [key: string]: unknown;
}

export interface CanVoteResponse {
  node_id?: string;
  can_vote: boolean;
  reason?: string;
  [key: string]: unknown;
}

export interface ReviewMutationResponse {
  status?: string;
  ticket_id?: string;
  count?: number;
  expired?: ReviewTicket[];
  [key: string]: unknown;
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

// ---- /api/device/profile ----

export interface DeviceGpuProfile {
  name?: string;
  gpu_type?: string;
  is_integrated?: boolean;
  cuda_available?: boolean;
  vram_total_gb?: number;
  vram_free_gb?: number;
  mps_available?: boolean;
  [key: string]: unknown;
}

export interface DeviceProfileResponse {
  tier?: string;
  tier_label?: string;
  tier_icon?: string;
  score_total?: number;
  score_breakdown?: { gpu?: number; ram?: number; cpu?: number; [key: string]: unknown };
  cpu?: {
    model_name?: string;
    physical_cores?: number;
    logical_cores?: number;
    freq_max_mhz?: number;
    [key: string]: unknown;
  };
  ram?: {
    total_gb?: number;
    available_gb?: number;
    used_gb?: number;
    percent_used?: number;
    [key: string]: unknown;
  };
  gpu?: DeviceGpuProfile;
  gpus?: DeviceGpuProfile[];
  selected_gpu_index?: number;
  recommendations?: string[];
  warnings?: string[];
  [key: string]: unknown;
}

export interface DeviceAutoConfigureResponse {
  status?: string;
  tier?: string;
  score?: number;
  applied_config?: Record<string, unknown>;
  recommendations?: string[];
  warnings?: string[];
  [key: string]: unknown;
}

export interface RagSearchResult {
  chunk_id?: string;
  source_id?: string;
  relative_ref?: string;
  revision?: number;
  ordinal?: number;
  snippet?: string;
  [key: string]: unknown;
}

export interface RagSearchResponse {
  mode?: string;
  provider?: string | null;
  results: RagSearchResult[];
  count?: number;
  storage?: string;
  [key: string]: unknown;
}

export interface RagRebuildResponse {
  status?: string;
  fts_chunk_count?: number;
  storage?: string;
  [key: string]: unknown;
}

export interface RagCapacityResponse {
  dimensions?: number;
  active_chunk_count?: number;
  embedding_count?: number;
  current_identity_embedding_count?: number;
  stale_embedding_count?: number;
  estimated_vector_bytes?: number;
  database_bytes?: number;
  wal_bytes?: number;
  estimated_total_bytes?: number;
  max_index_chunks?: number;
  [key: string]: unknown;
}

export interface RagAnnDecision {
  decision?: 'GO' | 'NO_GO' | string;
  reason?: string;
  corpus_chunks?: number;
  scan_budget?: number;
  extension?: string;
  extension_available?: boolean;
  production_approved?: boolean;
  [key: string]: unknown;
}

export interface RagAnnDecisionResponse {
  storage?: string;
  decision?: RagAnnDecision;
  [key: string]: unknown;
}

export interface RagEmbeddingJob {
  job_id: string;
  kind?: string;
  state?: 'queued' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed' | string;
  cursor?: { total?: number; indexed?: number; position?: number; batch_size?: number; model_id?: string; [key: string]: unknown };
  error_code?: string | null;
  attempts?: number;
  lease_active?: boolean;
  created_at?: number | string;
  updated_at?: number | string;
  [key: string]: unknown;
}

export interface DiffusionDistributedResponse {
  job_id?: string;
  state?: string;
  workflow_id?: string;
  artifact_id?: string;
  blob_id?: string;
  images?: Array<Record<string, unknown>>;
  distributed?: boolean;
  status?: string;
  workflow?: Record<string, unknown>;
  result?: {
    image?: { blob_id?: string; url?: string; content_type?: string; size_bytes?: number; [key: string]: unknown };
    images?: Array<{ blob_id?: string; url?: string; [key: string]: unknown }>;
    metrics?: Record<string, unknown>;
    [key: string]: unknown;
  };
  provider_id?: string;
  node_id?: string;
  cancellation?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DiffusionArtifactInspectResponse extends DiffusionArtifact {
  path?: string;
}

export interface DiffusionAssetActionResponse {
  status?: string;
  asset_id?: string;
  valid?: boolean;
  path?: string;
  artifact_id?: string;
  artifact?: Record<string, unknown>;
  job?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface DiffusionBlobUploadResponse {
  blob_id: string;
  content_type?: string;
  size_bytes?: number;
  sha256?: string;
  [key: string]: unknown;
}

// ---- Stable Diffusion 工作区 ----

export interface DiffusionPreset {
  preset_id: string;
  model_id?: string;
  prompt?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  scheduler?: string;
  seeds?: number[];
  safety_checker_required?: boolean;
}

export interface DiffusionArtifact {
  artifact_id: string;
  name: string;
  registered_at?: number;
  artifact?: {
    artifact_kind?: string;
    loadable?: boolean;
    path?: string;
    precision?: string;
    model_id?: string;
    size_bytes?: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface DiffusionJob {
  job_id: string;
  state: string;
  artifact_id?: string;
  created_at?: number;
  started_at?: number;
  completed_at?: number;
  error?: string;
  error_code?: string;
  cancel_requested?: boolean;
  progress?: {
    step?: number;
    total?: number;
    percent?: number;
    [key: string]: unknown;
  };
  parameters?: {
    prompt?: string;
    negative_prompt?: string;
    seed?: number;
    width?: number;
    height?: number;
    steps?: number;
    guidance_scale?: number;
    scheduler?: string;
    [key: string]: unknown;
  };
  metrics?: {
    elapsed_seconds?: number;
    [key: string]: unknown;
  };
  blob?: {
    blob_id?: string;
    content_type?: string;
    size_bytes?: number;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
}

export interface DiffusionCapabilitiesResponse {
  state: string;
  loaded: boolean;
  loaded_artifact?: DiffusionArtifact | null;
  engine_config?: Record<string, unknown> | null;
  capabilities?: Record<string, unknown> | null;
  active_job?: DiffusionJob | null;
  registered_artifacts?: number;
  jobs?: number;
  dependencies?: Record<string, boolean>;
  last_error?: string | null;
  presets?: DiffusionPreset[];
  [key: string]: unknown;
}

export interface DiffusionArtifactsResponse {
  artifacts: DiffusionArtifact[];
}

export interface DiffusionAsset {
  asset_id: string;
  name?: string;
  artifact_id?: string;
  artifact_kind?: string;
  description?: string;
  installed?: boolean;
  present_bytes?: number;
  total_bytes?: number;
  license?: string;
  job?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface DiffusionAssetsResponse {
  assets: DiffusionAsset[];
}

// ---- /api/models* ----

export interface ModelSummary {
  model_id: string;
  name: string;
  model_type?: string;
  is_builtin?: boolean;
  is_experimental?: boolean;
  recommended_vram_gb?: number;
  max_context?: number;
  quant_types?: string[];
  description?: string;
  huggingface_id?: string;
  location?: string;
  model_path?: string;
  gguf_path?: string;
  is_available?: boolean;
  unavailable_reason?: string;
  available_formats?: string[];
  has_safetensors?: boolean;
  has_gguf?: boolean;
  expected_paths?: string[];
  supported_engines?: string[];
  preferred_engine?: string;
  default_quant_type?: string;
  requires_cuda?: boolean;
  [key: string]: unknown;
}

export interface ModelsResponse {
  models: ModelSummary[];
  active_model_id?: string | null;
  [key: string]: unknown;
}

export interface ModelEngine {
  id: string;
  name?: string;
  description?: string;
  model_size_gb?: number | null;
  requires_cuda?: boolean;
  [key: string]: unknown;
}

export interface AvailableModelsResponse {
  models: Array<{
    id?: string;
    name?: string;
    description?: string;
    engine?: string;
    memory_gb?: number | null;
    speed_tok_s?: number | null;
    is_available?: boolean;
    [key: string]: unknown;
  }>;
  current?: string | null;
  current_engine?: string | null;
  available_engines: ModelEngine[];
  [key: string]: unknown;
}

export interface CurrentModelResponse {
  loaded: boolean;
  pipeline_prepared?: boolean;
  quant_type?: string | null;
  model_id?: string | null;
  model_name?: string;
  model_path?: string;
  engine?: string | null;
  descriptor?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface LocalModelAsset {
  model_id: string;
  name?: string;
  huggingface_id?: string;
  model_type?: string;
  available_formats?: string[];
  model_path?: string;
  gguf_path?: string;
  max_context?: number;
  architectures?: string[];
  total_bytes?: number;
  asset_ids?: string[];
  source_paths?: string[];
  manifest_paths?: string[];
  integrity?: string;
  runtime_profile?: string;
  runtime_hint?: string;
  runtime_status?: string;
  runtime_action?: string | null;
  [key: string]: unknown;
}

export interface LocalModelAssetsResponse {
  assets: LocalModelAsset[];
  summary?: {
    total?: number;
    total_bytes?: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface GgufModelRecord {
  filename: string;
  size_bytes?: number;
  size_mb?: number;
  sha256?: string;
  download_url?: string;
  [key: string]: unknown;
}

export interface GgufModelsResponse {
  models: GgufModelRecord[];
  exists?: boolean;
  count?: number;
  [key: string]: unknown;
}

export interface PreparePipelineResponse {
  status?: string;
  model_id?: string;
  pipeline_prepared?: boolean;
  [key: string]: unknown;
}

export interface ModelRegistryResponse {
  models: ModelSummary[];
  [key: string]: unknown;
}

export interface ModelDownloadManifest {
  model_id?: string;
  sha256?: string;
  artifact_sha256?: string;
  total_layers?: number;
  files?: Array<{
    path?: string;
    size_bytes?: number;
    sha256?: string;
    [key: string]: unknown;
  }>;
  count?: number;
  [key: string]: unknown;
}

export interface ModelPipelineAssignmentResponse {
  model_id?: string;
  assignments?: Array<Record<string, unknown>>;
  segments?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface DeploymentSimulationRecord {
  node_id?: string;
  status?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface DeploymentSimulationPlan {
  schema_version?: number;
  plan_id: string;
  artifact_id: string;
  runtime_profile: string;
  required_capabilities?: string[];
  target_nodes?: string[];
  actual_nodes?: string[];
  status?: string;
  epoch?: number;
  records?: DeploymentSimulationRecord[];
  created_at?: string;
  updated_at?: string;
  [key: string]: unknown;
}

export interface DeploymentSimulationResponse {
  status?: string;
  plan?: DeploymentSimulationPlan;
  [key: string]: unknown;
}

export interface ClusterProfile {
  profile_id: string;
  name: string;
  cluster_id: string;
  master_endpoint?: { scheme?: string; host?: string; port?: number };
  status?: string;
  node_role?: string;
  last_verified_at?: string | number | null;
  [key: string]: unknown;
}

export interface ClusterProfilesResponse { profiles: ClusterProfile[]; [key: string]: unknown }
export interface ClusterCurrentProfileResponse { profile: ClusterProfile | null; [key: string]: unknown }
export interface ClusterDiscoveryResponse { candidates: Array<Record<string, unknown>>; [key: string]: unknown }
export interface ClusterEndpointsResponse { endpoints: Array<Record<string, unknown>>; [key: string]: unknown }

export interface ClusterLayersResponse {
  total?: number;
  strategy?: string;
  assignments: Array<Record<string, unknown>>;
  computed_at?: number | null;
  cache_key?: string;
  [key: string]: unknown;
}

export interface ModelRuntimeSidecarProfile {
  display_name?: string;
  runtime_environment?: string;
  preflight_supported?: boolean;
  requires_task_contract?: boolean;
  production_admitted?: boolean;
  supported_actions?: string[];
  session?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ModelRuntimeSidecarStatusResponse {
  schema_version?: number;
  role?: string;
  control_available?: boolean;
  production_admitted?: boolean;
  profiles: Record<string, ModelRuntimeSidecarProfile>;
  [key: string]: unknown;
}

export interface ModelRuntimeContractsResponse {
  schema_version?: number;
  contracts: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface ModelPreflightResponse {
  schema_version?: number;
  operation?: string;
  model_id: string;
  runtime_profile?: string | null;
  read_only?: boolean;
  starts_sidecar?: boolean;
  gate_passed?: boolean;
  status?: string;
  preflight?: Record<string, unknown> | null;
  errors?: Array<{ code?: string; message?: string; [key: string]: unknown }>;
  [key: string]: unknown;
}

// ---- 模型一键下载（/api/models/presets、/api/models/downloads） ----

export interface ModelPreset {
  id: string;
  display: string;
  kind: 'gguf' | 'safetensors' | string;
  default_engine?: string;
  default_quant?: string;
  default_model_id?: string;
  hf_repo?: string;
  ms_path?: string;
  file_pattern?: string;
  expected_sha256?: string;
  resource_gate?: {
    min_ram_gb?: number;
    min_vram_gb?: number;
    min_disk_gb?: number;
    allow_cpu?: boolean;
    [key: string]: unknown;
  };
  description?: string;
  installable: boolean;
  blocked_reasons: Record<string, string>;
  [key: string]: unknown;
}

export interface ModelPresetsResponse {
  presets: ModelPreset[];
  [key: string]: unknown;
}

export interface ModelDownloadJob {
  job_id: string;
  status:
    | 'queued'
    | 'downloading'
    | 'verifying'
    | 'registering'
    | 'ready'
    | 'failed'
    | 'cancelled';
  progress: number;
  source: string;
  target?: string;
  model_id?: string;
  preset_id?: string;
  engine?: string;
  quant?: string;
  total_bytes?: number;
  downloaded_bytes?: number;
  error_code?: string | null;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
  finished_at?: string | null;
  [key: string]: unknown;
}

export interface ModelDownloadResponse {
  job: ModelDownloadJob;
  status?: string;
  [key: string]: unknown;
}

export interface ModelDownloadsResponse {
  jobs: ModelDownloadJob[];
  [key: string]: unknown;
}

export interface ModelSearchAttempt {
  provider: 'hf' | 'ms' | string;
  transport?: 'direct' | 'proxy' | string;
  status?: 'ok' | 'failed' | string;
  code?: string;
  [key: string]: unknown;
}

export interface ModelSearchResult {
  id: string;
  source: 'hf' | 'ms' | string;
  display_name?: string;
  description?: string;
  tasks?: string[];
  tags?: string[];
  downloads?: number | null;
  likes?: number | null;
  size_bytes?: number;
  last_modified?: string;
  license?: string;
  private?: boolean;
  gated?: boolean;
  url?: string;
  [key: string]: unknown;
}

export interface ModelSearchResponse {
  query: string;
  source: 'hf' | 'ms' | 'all' | string;
  provider: 'hf' | 'ms' | string;
  results: ModelSearchResult[];
  total?: number;
  fallback_used?: boolean;
  attempts?: ModelSearchAttempt[];
  [key: string]: unknown;
}

export interface CreateModelDownloadPayload {
  preset_id?: string;
  source?: string;
  target?: string;
  model_id?: string;
  engine?: string;
  quant?: string;
  use_modelscope?: boolean;
  proxy?: string;
  expected_sha256?: string;
  gguf_path?: string;
  allow_cpu?: boolean;
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
