/**
 * 数据访问层 — 唯一与后端通信的入口。
 *
 * 视觉组件不允许直接 fetch；页面通过 hooks 消费这里的函数（§5.1）。
 * 鉴权沿用既有 frontend 的约定：sessionStorage 中的 bearer token（若存在）。
 */

import type {
  ChatMetrics,
  ChatUploadResponse,
  ClusterConfigResponse,
  ClusterInviteResponse,
  ClusterMutationResponse,
  ClusterNodesResponse,
  ClusterStatusResponse,
  AuthCapabilityResponse,
  AuthMutationResponse,
  AuthSessionResponse,
  AuthSessionsResponse,
  DiffusionArtifactInspectResponse,
  DiffusionArtifactsResponse,
  DiffusionAssetActionResponse,
  DiffusionAssetsResponse,
  DiffusionBlobUploadResponse,
  DiffusionCapabilitiesResponse,
  DiffusionJob,
  AvailableModelsResponse,
  ApiErrorKind,
  CurrentModelResponse,
  DeviceAutoConfigureResponse,
  DeviceProfileResponse,
  LocalModelAssetsResponse,
  LocalTailscaleStatusResponse,
  CanVoteResponse,
  LogFileContentResponse,
  LogFilesResponse,
  LogStatsResponse,
  NodeLogAggregateResponse,
  NodesLogSummaryResponse,
  ReviewMutationResponse,
  ReviewTicketsResponse,
  ManagedUsersResponse,
  ModelPreflightResponse,
  ModelRegistryResponse,
  ModelDownloadManifest,
  ModelPipelineAssignmentResponse,
  DeploymentSimulationResponse,
  ModelsResponse,
  MasterHealthResponse,
  ConversationResponse,
  MyRoleResponse,
  ModelPresetsResponse,
  ModelDownloadResponse,
  ModelDownloadsResponse,
  CreateModelDownloadPayload,
  ModelSearchResponse,
  ClientErrorReport,
  StorageHealthResponse,
  SpeculativeCapabilityResponse,
  SpeculativeExperimentResponse,
  ClusterCurrentProfileResponse,
  ClusterDiscoveryResponse,
  ClusterEndpointsResponse,
  ClusterLayersResponse,
  ClusterProfilesResponse,
  DiffusionDistributedResponse,
  GgufModelsResponse,
  PipelineCapacityResponse,
  ModelRuntimeContractsResponse,
  ModelRuntimeSidecarStatusResponse,
  QueueResponse,
  RagRebuildResponse,
  RagHealthResponse,
  RagSourcesResponse,
  RagSearchResponse,
  RagCapacityResponse,
  RagAnnDecisionResponse,
  RagEmbeddingJob,
  RecentLogsResponse,
  SessionsResponse,
  SystemStatusResponse,
  TailscaleBindingsResponse,
  WorkflowsResponse,
  WorkflowRecord,
  PreparePipelineResponse,
} from './types';

const BASE = '/api';

/** 与既有 frontend/src/api/client.js 保持同一个 key，便于两套 UI 共存。 */
const AUTH_TOKEN_STORAGE_KEY = 'qlh-auth-session-token';
const LOG_ADMIN_TOKEN_STORAGE_KEY = 'qlh_log_admin_token';
const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;

function kindForStatus(status: number): ApiErrorKind {
  if (status === 401) return 'unauthorized';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'not_found';
  if (status === 408) return 'timeout';
  if (status === 409) return 'conflict';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'server';
  return 'http';
}

function retryableForKind(kind: ApiErrorKind): boolean {
  return kind === 'network' || kind === 'timeout' || kind === 'rate_limited' || kind === 'server';
}

interface RequestSignal {
  signal: AbortSignal;
  timedOut: () => boolean;
  cleanup: () => void;
}

function createRequestSignal(signal: AbortSignal | undefined, timeoutMs: number): RequestSignal {
  const controller = new AbortController();
  let timedOut = false;
  let timer = 0;

  const onAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  if (timeoutMs > 0) {
    timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
  }

  return {
    signal: controller.signal,
    timedOut: () => timedOut,
    cleanup: () => {
      if (timer) window.clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    },
  };
}

export class ApiError extends Error {
  status: number;
  requestId: string | null;
  path: string;
  kind: ApiErrorKind;
  retryable: boolean;

  constructor(
    message: string,
    opts: {
      status?: number;
      requestId?: string | null;
      path?: string;
      kind?: ApiErrorKind;
      retryable?: boolean;
    } = {},
  ) {
    const requestId = opts.requestId ?? null;
    super(requestId ? `${message} (request_id: ${requestId})` : message);
    this.name = 'ApiError';
    this.status = opts.status ?? 0;
    this.requestId = requestId;
    this.path = opts.path ?? '';
    this.kind = opts.kind ?? (this.status > 0 ? kindForStatus(this.status) : 'network');
    this.retryable = opts.retryable ?? retryableForKind(this.kind);
  }
}

function readStorage(storage: 'session' | 'local', key: string): string {
  try {
    if (typeof window === 'undefined') return '';
    const store = storage === 'session' ? window.sessionStorage : window.localStorage;
    return store?.getItem(key) || '';
  } catch {
    // 存储被禁用时按未登录处理。
    return '';
  }
}

function normalizeDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail) return detail;
  if (detail && typeof detail === 'object') {
    const maybe = detail as { message?: unknown };
    if (typeof maybe.message === 'string' && maybe.message) return maybe.message;
    try {
      return JSON.stringify(detail);
    } catch {
      return fallback;
    }
  }
  return fallback;
}

interface RequestOptions {
  signal?: AbortSignal;
  method?: string;
  body?: string;
  headers?: Record<string, string>;
  /** 日志类接口需要额外的管理 token。 */
  withLogToken?: boolean;
  /** 首部响应超时；流式正文由调用方自己的 AbortController 管理。 */
  timeoutMs?: number;
}

async function fetchResponse(
  url: string,
  init: RequestInit,
  opts: { signal?: AbortSignal; timeoutMs?: number; path: string },
): Promise<Response> {
  const requestSignal = createRequestSignal(
    opts.signal,
    opts.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
  );
  try {
    return await fetch(url, { ...init, signal: requestSignal.signal });
  } catch (err) {
    if (opts.signal?.aborted && !requestSignal.timedOut()) throw err;
    if (requestSignal.timedOut()) {
      throw new ApiError(`请求超时（${Math.round((opts.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS) / 1000)} 秒）`, {
        status: 0,
        path: opts.path,
        kind: 'timeout',
        retryable: true,
      });
    }
    throw new ApiError('无法连接后端，请检查服务是否启动或网络连接。', {
      status: 0,
      path: opts.path,
      kind: 'network',
      retryable: true,
    });
  } finally {
    requestSignal.cleanup();
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { signal, withLogToken, headers, timeoutMs, ...rest } = options;
  const token = readStorage('session', AUTH_TOKEN_STORAGE_KEY);
  const logToken = withLogToken ? readStorage('local', LOG_ADMIN_TOKEN_STORAGE_KEY) : '';

  const res = await fetchResponse(
    `${BASE}${path}`,
    {
      ...rest,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(logToken ? { 'X-QLH-Log-Token': logToken } : {}),
      },
    },
    { signal, timeoutMs, path },
  );

  const text = await res.text();
  let data: Record<string, unknown> = {};
  if (text) {
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      data = { detail: text };
    }
  }

  const requestId =
    res.headers.get('X-Request-ID') || (data.request_id as string | undefined) || null;

  if (!res.ok) {
    throw new ApiError(normalizeDetail(data.detail, `HTTP ${res.status}`), {
      status: res.status,
      requestId,
      path,
      kind: kindForStatus(res.status),
      retryable: retryableForKind(kindForStatus(res.status)),
    });
  }

  return data as T;
}

async function requestBlob(path: string, options: RequestOptions = {}): Promise<Blob> {
  const { signal, withLogToken, headers, timeoutMs, ...rest } = options;
  const token = readStorage('session', AUTH_TOKEN_STORAGE_KEY);
  const logToken = withLogToken ? readStorage('local', LOG_ADMIN_TOKEN_STORAGE_KEY) : '';
  const res = await fetchResponse(`${BASE}${path}`, {
    ...rest,
    headers: {
      ...headers,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(logToken ? { 'X-QLH-Log-Token': logToken } : {}),
    },
  }, { signal, timeoutMs, path });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json() as { detail?: unknown };
      detail = normalizeDetail(body.detail, detail);
    } catch { /* binary endpoints may return plain text */ }
    throw new ApiError(detail, { status: res.status, path, kind: kindForStatus(res.status) });
  }
  return res.blob();
}

// ---- 只读快照接口（Overview / Nodes / Activity 使用） ----

export const fetchSystemStatus = (signal?: AbortSignal) =>
  request<SystemStatusResponse>('/status', { signal });

export const fetchClusterNodes = (signal?: AbortSignal) =>
  request<ClusterNodesResponse>('/cluster/nodes', { signal });

export const fetchClusterStatus = (signal?: AbortSignal) =>
  request<ClusterStatusResponse>('/cluster/status', { signal });

export const fetchClusterConfig = (signal?: AbortSignal) =>
  request<ClusterConfigResponse>('/cluster/config', { signal });

export const fetchClusterInvite = (signal?: AbortSignal) =>
  request<ClusterInviteResponse>('/cluster/invite', { signal });

export const fetchMasterHealth = (signal?: AbortSignal) =>
  request<MasterHealthResponse>('/cluster/master-health', { signal });

export const updateMaxNodes = (maxNodes: number) =>
  request<ClusterMutationResponse>('/cluster/config/max-nodes', {
    method: 'PUT',
    body: JSON.stringify({ max_nodes: maxNodes }),
  });

export const registerClusterNode = (payload: {
  node_id: string;
  hostname?: string;
  address?: string;
  network_type?: string;
  node_type?: string;
}) =>
  request<ClusterMutationResponse>('/cluster/nodes/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  });

export const deregisterClusterNode = (nodeId: string) =>
  request<ClusterMutationResponse>(`/cluster/nodes/${encodeURIComponent(nodeId)}/deregister`, {
    method: 'POST',
  });

export const deleteClusterNode = (nodeId: string) =>
  request<ClusterMutationResponse>(`/cluster/nodes/${encodeURIComponent(nodeId)}`, {
    method: 'DELETE',
  });

export const fetchMyRole = (signal?: AbortSignal) =>
  request<MyRoleResponse>('/cluster/my-role', { signal });

// ---- Authentication, account, and Tailscale ----

export const fetchAuthCapability = (signal?: AbortSignal) =>
  request<AuthCapabilityResponse>('/auth/capability', { signal });

export const fetchAuthSession = (signal?: AbortSignal) =>
  request<AuthSessionResponse>('/auth/session', { signal });

export const loginAuth = (username: string, code: string, recoveryCode = '') =>
  request<AuthSessionResponse & { access_token?: string }>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({
      username: username.trim(),
      ...(recoveryCode.trim() ? { recovery_code: recoveryCode.trim() } : { code: code.trim() }),
    }),
  }).then((result) => {
    setAuthToken(result.access_token || '');
    return result;
  });

export const bootstrapAuth = (payload: { username: string; display_name?: string }) =>
  request<AuthMutationResponse>('/auth/bootstrap', {
    method: 'POST',
    body: JSON.stringify({
      username: payload.username.trim(),
      ...(payload.display_name?.trim() ? { display_name: payload.display_name.trim() } : {}),
    }),
  });

export const verifyAuthTotp = (payload: { user_id: string; authenticator_id: string; code: string }) =>
  request<AuthMutationResponse>('/auth/totp/verify', {
    method: 'POST',
    body: JSON.stringify({ ...payload, code: payload.code.trim() }),
  });

export const rotateRecoveryCodes = (code: string) =>
  request<AuthMutationResponse>('/auth/recovery-codes/rotate', {
    method: 'POST',
    body: JSON.stringify({ code: code.trim() }),
  });

export const provisionUserTotp = (userId: string) =>
  request<AuthMutationResponse>(`/auth/users/${encodeURIComponent(userId)}/totp`, {
    method: 'POST',
  });

export const logoutAuth = () =>
  request<AuthMutationResponse>('/auth/logout', { method: 'POST' }).finally(() => setAuthToken(''));

export const fetchAuthSessions = (userId = '', signal?: AbortSignal) =>
  request<AuthSessionsResponse>(`/auth/sessions${userId ? `?user_id=${encodeURIComponent(userId)}` : ''}`, { signal });

export const revokeAuthSession = (sessionId: string) =>
  request<AuthMutationResponse>(`/auth/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });

export const fetchManagedUsers = (signal?: AbortSignal) =>
  request<ManagedUsersResponse>('/users', { signal });

export const createManagedUser = (payload: { username: string; display_name?: string; role?: string }) =>
  request<AuthMutationResponse>('/users', { method: 'POST', body: JSON.stringify(payload) });

export const updateManagedUser = (userId: string, payload: { expected_version?: number; display_name?: string; role?: string; status?: string }) =>
  request<AuthMutationResponse>(`/users/${encodeURIComponent(userId)}`, { method: 'PATCH', body: JSON.stringify({ expected_version: payload.expected_version, ...payload }) });

export const fetchTailscaleBindings = (userId = '', signal?: AbortSignal) =>
  request<TailscaleBindingsResponse>(userId ? `/auth/users/${encodeURIComponent(userId)}/tailscale` : '/auth/tailscale/bindings', { signal });

// ---- Cluster profiles, discovery, and advertised endpoints ----

export const fetchClusterProfiles = (signal?: AbortSignal) =>
  request<ClusterProfilesResponse>('/cluster/profiles', { signal });

export const fetchCurrentClusterProfile = (signal?: AbortSignal) =>
  request<ClusterCurrentProfileResponse>('/cluster/profiles/current', { signal });

export const verifyClusterProfile = (masterEndpoint: string, name = '') =>
  request<Record<string, unknown>>('/cluster/profiles/verify', {
    method: 'POST', body: JSON.stringify({ master_endpoint: masterEndpoint, ...(name ? { name } : {}) }),
  });

export const createClusterProfile = (payload: { cluster_id: string; name: string; master_endpoint: string; node_role?: string }) =>
  request<Record<string, unknown>>('/cluster/profiles', { method: 'POST', body: JSON.stringify(payload) });

export const activateClusterProfile = (profileId: string) =>
  request<Record<string, unknown>>(`/cluster/profiles/${encodeURIComponent(profileId)}/activate`, { method: 'POST' });

export const deleteClusterProfile = (profileId: string) =>
  request<Record<string, unknown>>(`/cluster/profiles/${encodeURIComponent(profileId)}`, { method: 'DELETE' });

export const fetchClusterDiscoveryCandidates = (tailscalePeers: string[] = [], signal?: AbortSignal) => {
  const query = tailscalePeers.length ? `?tailscale_peers=${encodeURIComponent(tailscalePeers.join(','))}` : '';
  return request<ClusterDiscoveryResponse>(`/cluster/discovery/candidates${query}`, { signal });
};

export const fetchClusterEndpoints = (signal?: AbortSignal) =>
  request<ClusterEndpointsResponse>('/cluster/endpoints', { signal });

export const verifyClusterEndpoint = (payload: { scheme?: string; host: string; port?: number }) =>
  request<Record<string, unknown>>('/cluster/endpoints/verify', { method: 'POST', body: JSON.stringify(payload) });

export const fetchClusterLayers = (signal?: AbortSignal) =>
  request<ClusterLayersResponse>('/cluster/layers', { signal });

export const fetchModelRuntimeSidecarStatus = (signal?: AbortSignal) =>
  request<ModelRuntimeSidecarStatusResponse>('/cluster/model-runtime/sidecars', { signal });

export const fetchModelRuntimeContracts = (signal?: AbortSignal) =>
  request<ModelRuntimeContractsResponse>('/cluster/model-runtime/contracts', { signal });

export const releaseModelRuntimeSidecar = (profile: 'qwen3_sidecar' | 'gemma4_pipeline') =>
  request<Record<string, unknown>>('/cluster/model-runtime/sidecars/release', {
    method: 'POST', body: JSON.stringify({ profile }),
  });

export const cancelModelRuntimeSidecar = (profile: 'qwen3_sidecar' | 'gemma4_pipeline') =>
  request<Record<string, unknown>>(`/cluster/model-runtime/sidecars/${encodeURIComponent(profile)}`, { method: 'DELETE' });

export const fetchQwen3LocalChain = (signal?: AbortSignal) =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain', { signal });

export const beginQwen3LocalChain = (contract: Record<string, unknown>) =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain/begin', { method: 'POST', body: JSON.stringify({ contract }) });

export const runQwen3LocalPrefill = (payload: { input_ref: string; batch_size: number; sequence_length: number }) =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain/prefill', { method: 'POST', body: JSON.stringify(payload) });

export const runQwen3LocalDecode = (payload: { input_ref: string; batch_size: number; sequence_length: number }) =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain/decode', { method: 'POST', body: JSON.stringify(payload) });

export const verifyQwen3LocalParity = (payload: { reference_prefill: string; reference_decode: string; rtol?: number; atol?: number }) =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain/parity', { method: 'POST', body: JSON.stringify(payload) });

export const releaseQwen3LocalChain = () =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain/release', { method: 'POST' });

export const cancelQwen3LocalChain = () =>
  request<Record<string, unknown>>('/cluster/qwen3/local-chain', { method: 'DELETE' });

export const fetchLocalTailscaleStatus = (signal?: AbortSignal) =>
  request<LocalTailscaleStatusResponse>('/auth/tailscale/local-status', { signal });

export const prepareTailscaleBinding = (authorizationMethod: string, userId = '') =>
  request<AuthMutationResponse>(userId ? `/auth/users/${encodeURIComponent(userId)}/tailscale` : '/auth/tailscale/bindings', {
    method: 'POST',
    body: JSON.stringify({ authorization_method: authorizationMethod }),
  });

export const confirmTailscaleBinding = (bindingId: string, payload: { tailnet_id: string; tailscale_user_id: string; node_id?: string }) =>
  request<AuthMutationResponse>(`/auth/tailscale/bindings/${encodeURIComponent(bindingId)}/confirm`, { method: 'POST', body: JSON.stringify(payload) });

export const revokeTailscaleBinding = (bindingId: string) =>
  request<AuthMutationResponse>(`/auth/tailscale/bindings/${encodeURIComponent(bindingId)}/revoke`, { method: 'POST' });

export const fetchQueue = (signal?: AbortSignal) =>
  request<QueueResponse>('/cluster/queue', { signal });

export const fetchWorkflows = (limit = 20, signal?: AbortSignal) =>
  request<WorkflowsResponse>(`/workflows?limit=${encodeURIComponent(limit)}`, { signal });

export const fetchWorkflow = (workflowId: string, signal?: AbortSignal) =>
  request<WorkflowRecord>(`/workflows/${encodeURIComponent(workflowId)}`, { signal });

export const fetchPipelineCapacity = async (signal?: AbortSignal): Promise<PipelineCapacityResponse> => {
  // 后端在模型描述符尚未就绪时返回精简的 unavailable 响应，只包含
  // status/admitted/reason_code/assignments；在数据层补齐集合字段，避免
  // 页面把「能力暂不可用」误当成完整计划并读取 undefined.length。
  const data = await request<Partial<PipelineCapacityResponse>>('/cluster/pipeline-capacity', { signal });
  const normalized = {
    model_id: '',
    model_type: '',
    total_layers: 0,
    raw_model_bytes: 0,
    candidate_node_count: 0,
    status: 'unavailable',
    admitted: false,
    reason_code: '',
    reason: '',
    plan_id: '',
    assignments: [],
    control_only_nodes: [],
    participating_node_count: 0,
    single_node_full_model_candidates: [],
    prepared_node_count: 0,
    ready_node_count: 0,
    worker_count: 0,
    computed_at: 0,
    ...data,
  };
  return {
    ...normalized,
    assignments: Array.isArray(normalized.assignments) ? normalized.assignments : [],
    control_only_nodes: Array.isArray(normalized.control_only_nodes) ? normalized.control_only_nodes : [],
    single_node_full_model_candidates: Array.isArray(normalized.single_node_full_model_candidates)
      ? normalized.single_node_full_model_candidates
      : [],
  };
};

export const fetchRagHealth = (signal?: AbortSignal) =>
  request<RagHealthResponse>('/rag/health', { signal });

export const fetchRagSources = (ownerScope = '', signal?: AbortSignal) =>
  request<RagSourcesResponse>(`/rag/sources${ownerScope ? `?owner_scope=${encodeURIComponent(ownerScope)}` : ''}`, { signal });

export const deleteRagSource = (sourceId: string) =>
  request<{ status?: string; source_id?: string }>('/rag/sources/' + encodeURIComponent(sourceId), { method: 'DELETE' });

export const fetchSessions = (limit = 20, signal?: AbortSignal) =>
  request<SessionsResponse>(`/sessions?limit=${encodeURIComponent(limit)}`, { signal });

export const createSession = (title = '新对话') =>
  request<{ id: string; title: string; message_count?: number; active?: boolean }>('/sessions', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });

export const renameSession = (sessionId: string, title: string) =>
  request<Partial<import('./types').SessionSummary> & { id: string }>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: 'PUT',
      body: JSON.stringify({ title }),
    },
  );

export const deleteSession = (sessionId: string) =>
  request<{ status?: string; session_id?: string }>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE' },
  );

export const activateSession = (sessionId: string) =>
  request<{ session_id: string; messages?: unknown[]; count?: number }>(
    `/sessions/${encodeURIComponent(sessionId)}/activate`,
    { method: 'POST' },
  );

export const fetchRecentLogs = (
  params: { limit?: number; level?: string } = {},
  signal?: AbortSignal,
) => {
  const qs = new URLSearchParams();
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.level) qs.set('level', params.level);
  const q = qs.toString();
  return request<RecentLogsResponse>(`/logs/recent${q ? `?${q}` : ''}`, {
    signal,
    withLogToken: true,
  });
};

export const fetchLogFiles = (signal?: AbortSignal) =>
  request<LogFilesResponse>('/logs', { signal, withLogToken: true });

export const fetchLogContent = (filename: string, signal?: AbortSignal) =>
  request<LogFileContentResponse>(`/logs/${encodeURIComponent(filename)}`, { signal, withLogToken: true });

export const deleteLogFile = (filename: string) =>
  request<{ status?: string; name?: string }>(`/logs/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    withLogToken: true,
  });

export const fetchLogStats = (signal?: AbortSignal) =>
  request<LogStatsResponse>('/logs/stats', { signal, withLogToken: true });

export const fetchNodesLogSummary = (signal?: AbortSignal) =>
  request<NodesLogSummaryResponse>('/logs/nodes-summary', { signal, withLogToken: true });

export const fetchNodesLogAggregate = (
  params: { limit?: number; level?: string; name?: string } = {},
  signal?: AbortSignal,
) => {
  const qs = new URLSearchParams();
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.level) qs.set('level', params.level);
  if (params.name) qs.set('name', params.name);
  const query = qs.toString();
  return request<NodeLogAggregateResponse>(`/cluster/nodes/log-aggregate${query ? `?${query}` : ''}`, {
    signal,
    withLogToken: true,
  });
};

export const downloadLogFile = (filename: string, signal?: AbortSignal) =>
  requestBlob(`/logs/download?name=${encodeURIComponent(filename)}`, { signal, withLogToken: true });

export const exportLogs = (signal?: AbortSignal) =>
  requestBlob('/logs/export', { signal, withLogToken: true, timeoutMs: 60_000 });

export const reportClientError = (payload: ClientErrorReport) =>
  request<{ status?: string }>('/logs/client-error', { method: 'POST', body: JSON.stringify(payload), withLogToken: true });

export const fetchReviewTickets = (status = '', signal?: AbortSignal) =>
  request<ReviewTicketsResponse>(`/cluster/review/tickets${status ? `?status=${encodeURIComponent(status)}` : ''}`, { signal });

export const checkCanVote = (signal?: AbortSignal) =>
  request<CanVoteResponse>('/cluster/review/can-vote', { signal });

export const createReviewTicket = (targetNodeId: string, reason: string, timeoutHours = 48) =>
  request<ReviewMutationResponse>('/cluster/review/create', {
    method: 'POST',
    body: JSON.stringify({ target_node_id: targetNodeId, reason, timeout_hours: timeoutHours }),
  });

export const castReviewVote = (ticketId: string, vote: -1 | 0 | 1, comment = '') =>
  request<ReviewMutationResponse>('/cluster/review/vote', {
    method: 'POST',
    body: JSON.stringify({ ticket_id: ticketId, vote, comment }),
  });

export const pollReviewMail = () =>
  request<Record<string, unknown>>('/cluster/review/mail-poll', { method: 'POST' });

export const expireReviewCheck = () =>
  request<ReviewMutationResponse>('/cluster/review/expire-check', { method: 'POST' });

export const deleteReviewTicket = (ticketId: string) =>
  request<ReviewMutationResponse>(`/cluster/review/tickets/${encodeURIComponent(ticketId)}`, { method: 'DELETE' });

export const deleteResolvedReviewTickets = () =>
  request<ReviewMutationResponse>('/cluster/review/tickets', { method: 'DELETE' });

export const fetchHealth = (signal?: AbortSignal) =>
  request<{ status: string; timestamp: number }>('/health', { signal });

// ---- Device and local RAG workspace ----

export const fetchDeviceProfile = (signal?: AbortSignal) =>
  request<DeviceProfileResponse>('/device/profile', { signal });

export const autoConfigureDevice = () =>
  request<DeviceAutoConfigureResponse>('/device/auto-configure', { method: 'POST' });

export const selectGpu = (gpuIndex: number) =>
  request<DeviceProfileResponse & { selected_gpu?: DeviceProfileResponse['gpu']; warning?: string }>(
    '/device/select-gpu',
    { method: 'POST', body: JSON.stringify({ gpu_index: gpuIndex }) },
  );

export const searchRag = (query: string, options: { mode?: string; access_scope?: string; limit?: number } = {}) =>
  request<RagSearchResponse>('/rag/search', {
    method: 'POST',
    body: JSON.stringify({ query, ...options }),
  });

export const rebuildRagIndex = () =>
  request<RagRebuildResponse>('/rag/rebuild', {
    method: 'POST',
    body: JSON.stringify({ include_embeddings: false }),
  });

export const fetchRagCapacity = (dimensions = 768, signal?: AbortSignal) =>
  request<RagCapacityResponse>(`/rag/capacity?dimensions=${encodeURIComponent(dimensions)}`, { signal });

export const fetchRagAnnDecision = (scanBudget = 1024, signal?: AbortSignal) =>
  request<RagAnnDecisionResponse>(`/rag/ann-decision?scan_budget=${encodeURIComponent(scanBudget)}`, { signal });

export const createRagEmbeddingJob = (payload: {
  provider?: 'ollama';
  model_id?: string;
  model_sha256: string;
  source_id?: string;
  batch_size?: number;
}) => request<RagEmbeddingJob>('/rag/embedding-jobs', {
  method: 'POST',
  body: JSON.stringify({ provider: 'ollama', model_id: 'nomic-embed-text:latest', ...payload }),
});

export const fetchRagEmbeddingJob = (jobId: string, signal?: AbortSignal) =>
  request<RagEmbeddingJob>(`/rag/embedding-jobs/${encodeURIComponent(jobId)}`, { signal });

export const runRagEmbeddingJob = (jobId: string, payload: {
  model_id?: string;
  expected_dimensions?: number;
  max_batches?: number;
  lease_seconds?: number;
  max_retries?: number;
} = {}) => request<RagEmbeddingJob>(`/rag/embedding-jobs/${encodeURIComponent(jobId)}/run`, {
  method: 'POST',
  body: JSON.stringify({ model_id: 'nomic-embed-text:latest', ...payload }),
});

export const cancelRagEmbeddingJob = (jobId: string) =>
  request<RagEmbeddingJob>(`/rag/embedding-jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });

export const fetchStorageHealth = (signal?: AbortSignal) =>
  request<StorageHealthResponse>('/storage/health', { signal });

// ---- Stable Diffusion 工作区 ----

export const fetchDiffusionCapabilities = (signal?: AbortSignal) =>
  request<DiffusionCapabilitiesResponse>('/diffusion/capabilities', { signal });

export const fetchDiffusionArtifacts = (signal?: AbortSignal) =>
  request<DiffusionArtifactsResponse>('/diffusion/artifacts', { signal });

export const fetchDiffusionAssets = (signal?: AbortSignal) =>
  request<DiffusionAssetsResponse>('/diffusion/assets/catalog', { signal });

export const inspectDiffusionArtifact = (path: string, computeHash = false) =>
  request<DiffusionArtifactInspectResponse>('/diffusion/artifacts/inspect', {
    method: 'POST',
    body: JSON.stringify({ path, compute_hash: computeHash }),
  });

export const registerDiffusionArtifact = (payload: {
  path: string;
  artifact_id?: string;
  name?: string;
  compute_hash?: boolean;
}) => request<DiffusionArtifactInspectResponse>('/diffusion/artifacts/register', {
  method: 'POST',
  body: JSON.stringify(payload),
});

export const fetchDiffusionAssetStatus = (assetId: string, signal?: AbortSignal) =>
  request<DiffusionAssetActionResponse>(`/diffusion/assets/${encodeURIComponent(assetId)}/status`, { signal });

export const downloadDiffusionAsset = (assetId: string, options: {
  licenseAccepted: boolean;
  useLocalProxyFallback?: boolean;
}) => request<DiffusionAssetActionResponse>(`/diffusion/assets/${encodeURIComponent(assetId)}/download`, {
  method: 'POST',
  body: JSON.stringify({
    license_accepted: options.licenseAccepted,
    use_local_proxy_fallback: options.useLocalProxyFallback ?? true,
  }),
});

export const importDiffusionAsset = (payload: {
  assetId: string;
  path: string;
  licenseAccepted: boolean;
}) => request<DiffusionAssetActionResponse>('/diffusion/assets/import', {
  method: 'POST',
  body: JSON.stringify({
    asset_id: payload.assetId,
    path: payload.path,
    license_accepted: payload.licenseAccepted,
  }),
});

export const loadDiffusionArtifact = (artifactId: string, profile = 'balanced') =>
  request<DiffusionCapabilitiesResponse>('/diffusion/load', {
    method: 'POST',
    body: JSON.stringify({ artifact_id: artifactId, profile, safety_checker_required: true }),
  });

export const unloadDiffusionArtifact = () =>
  request<DiffusionCapabilitiesResponse>('/diffusion/unload', { method: 'POST' });

export const generateDiffusionImage = (params: {
  preset_id?: string;
  prompt: string;
  negative_prompt?: string;
  seed?: number;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  scheduler?: string;
}) =>
  request<DiffusionJob>('/diffusion/generate', {
    method: 'POST',
    body: JSON.stringify(params),
  });

export const editDiffusionImage = (params: {
  mode: 'img2img' | 'reference' | 'inpaint' | 'instruction';
  source_blob_id: string;
  mask_blob_id?: string;
  prompt?: string;
  negative_prompt?: string;
  seed?: number;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  scheduler?: string;
  strength?: number;
  instruction?: string;
  edit_adapter_id?: string;
  conditioning_scale?: number;
  image_guidance_scale?: number;
}) => request<DiffusionJob>('/diffusion/edit', {
  method: 'POST',
  body: JSON.stringify(params),
});

export async function uploadDiffusionBlob(file: File, purpose = 'input_image', signal?: AbortSignal) {
  const token = readStorage('session', AUTH_TOKEN_STORAGE_KEY);
  const form = new FormData();
  form.append('purpose', purpose);
  form.append('file', file);
  const path = '/diffusion/blobs';
  const res = await fetchResponse(`${BASE}${path}`, {
    method: 'POST',
    body: form,
    ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
  }, { signal, timeoutMs: 60_000, path });
  const text = await res.text();
  let data: Record<string, unknown> = {};
  if (text) {
    try { data = JSON.parse(text) as Record<string, unknown>; } catch { data = { detail: text }; }
  }
  const requestId = res.headers.get('X-Request-ID') || (data.request_id as string | undefined) || null;
  if (!res.ok) {
    const kind = kindForStatus(res.status);
    throw new ApiError(normalizeDetail(data.detail, `HTTP ${res.status}`), {
      status: res.status, requestId, path, kind, retryable: retryableForKind(kind),
    });
  }
  return data as DiffusionBlobUploadResponse;
}

export const fetchDiffusionJob = (jobId: string, signal?: AbortSignal) =>
  request<DiffusionJob>(`/diffusion/jobs/${encodeURIComponent(jobId)}`, { signal });

export const cancelDiffusionJob = (jobId: string) =>
  request<{ accepted?: boolean; job?: DiffusionJob }>(
    `/diffusion/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' },
  );

export const deleteDiffusionBlob = (blobId: string) =>
  request<{ deleted?: boolean; blob_id?: string }>(`/diffusion/blobs/${encodeURIComponent(blobId)}`, { method: 'DELETE' });

export const generateDistributedDiffusion = (params: Record<string, unknown>) =>
  request<DiffusionDistributedResponse>('/diffusion/distributed/generate', { method: 'POST', body: JSON.stringify(params) });

export const generateDistributedDiffusionGrid = (params: Record<string, unknown> & { seeds: number[] }) =>
  request<DiffusionDistributedResponse>('/diffusion/distributed/grid', { method: 'POST', body: JSON.stringify(params) });

export const generateDistributedDiffusionMixed = (params: { message: string; text_provider_id?: string; text_model_id?: string }) =>
  request<DiffusionDistributedResponse>('/diffusion/distributed/mixed', { method: 'POST', body: JSON.stringify(params) });

export const downloadDistributedWorkflowBlob = (workflowId: string, blobId: string, signal?: AbortSignal) =>
  requestBlob(`/diffusion/distributed/workflows/${encodeURIComponent(workflowId)}/blobs/${encodeURIComponent(blobId)}`, { signal });

// ---- Model and local asset workspace ----

export const fetchModels = (signal?: AbortSignal) =>
  request<ModelsResponse>('/models', { signal });

export const fetchAvailableModels = (signal?: AbortSignal) =>
  request<AvailableModelsResponse>('/models/available', { signal });

export const fetchCurrentModel = (signal?: AbortSignal) =>
  request<CurrentModelResponse>('/models/current', { signal });

export const fetchLocalModelAssets = (signal?: AbortSignal) =>
  request<LocalModelAssetsResponse>('/models/local-assets', { signal });

export const fetchModelRegistry = (signal?: AbortSignal) =>
  request<ModelRegistryResponse>('/models/registry', { signal });

export interface RegisterModelPayload {
  model_id: string;
  name: string;
  model_type?: 'safetensors' | 'gguf' | 'both';
  model_path?: string;
  gguf_path?: string;
  recommended_vram_gb?: number;
  max_context?: number;
  huggingface_id?: string;
  description?: string;
}

export const registerModel = (payload: RegisterModelPayload) =>
  request<{ status?: string; model_id?: string }>('/models/registry', {
    method: 'POST',
    body: JSON.stringify(payload),
  });

export const unregisterModel = (modelId: string) =>
  request<{ status?: string; model_id?: string }>(`/models/registry/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  });

export const preflightLocalModelAsset = (modelId: string) =>
  request<ModelPreflightResponse>(`/models/local-assets/${encodeURIComponent(modelId)}/preflight`, {
    method: 'POST',
  });

export const prepareModelPipeline = (modelId: string, quantType = 'fp16') =>
  request<PreparePipelineResponse>('/models/prepare-pipeline', {
    method: 'POST', body: JSON.stringify({ model_id: modelId, quant_type: quantType }),
  });

export const fetchGgufModels = (signal?: AbortSignal) =>
  request<GgufModelsResponse>('/models/gguf', { signal });

export const downloadGgufModel = (filename: string, signal?: AbortSignal) =>
  requestBlob(`/models/download/${encodeURIComponent(filename)}`, { signal, timeoutMs: 120_000 });

export const fetchDownloadableModelManifest = (modelId = '', signal?: AbortSignal) =>
  request<ModelDownloadManifest>(`/models/downloadable${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''}`, { signal });

export const fetchModelPipelineAssignment = (modelId: string, params: Record<string, string | number | boolean> = {}, signal?: AbortSignal) => {
  const query = new URLSearchParams(Object.entries(params).reduce<Record<string, string>>((out, [key, value]) => { out[key] = String(value); return out; }, {})).toString();
  return request<ModelPipelineAssignmentResponse>(`/models/pipeline-assignment/${encodeURIComponent(modelId)}${query ? `?${query}` : ''}`, { signal });
};

export const downloadModelFile = (modelId: string, relativePath: string, signal?: AbortSignal) =>
  requestBlob(`/models/files/${encodeURIComponent(modelId)}/${relativePath.split('/').map(encodeURIComponent).join('/')}`, { signal, timeoutMs: 120_000 });

export const loadModel = (
  engine: string,
  quantType: string,
  useCompile = false,
  modelId?: string,
) =>
  request<CurrentModelResponse>('/models/load', {
    method: 'POST',
    body: JSON.stringify({
      engine: engine || 'auto',
      quant_type: quantType || 'int4',
      use_compile: useCompile,
      ...(modelId ? { model_id: modelId } : {}),
    }),
  });

export const unloadModel = () =>
  request<CurrentModelResponse>('/models/unload', {
    method: 'POST',
    body: JSON.stringify({}),
  });

// ---- 模型一键下载（P0A）----

export const fetchModelPresets = (signal?: AbortSignal) =>
  request<ModelPresetsResponse>('/models/presets', { signal });

export const createModelDownload = (payload: CreateModelDownloadPayload) =>
  request<ModelDownloadResponse>('/models/downloads', {
    method: 'POST',
    body: JSON.stringify(payload),
  });

export const fetchModelDownloads = (signal?: AbortSignal) =>
  request<ModelDownloadsResponse>('/models/downloads', { signal });

export const searchModelRepositories = (
  params: { q: string; source?: 'hf' | 'ms' | 'all'; page?: number; limit?: number; proxy?: string },
  signal?: AbortSignal,
) => {
  const qs = new URLSearchParams({ q: params.q });
  if (params.source) qs.set('source', params.source);
  if (params.page) qs.set('page', String(params.page));
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.proxy) qs.set('proxy', params.proxy);
  return request<ModelSearchResponse>(`/models/search?${qs.toString()}`, { signal });
};

export const fetchModelDownload = (jobId: string, signal?: AbortSignal) =>
  request<ModelDownloadResponse>(`/models/downloads/${encodeURIComponent(jobId)}`, { signal });

export const cancelModelDownload = (jobId: string) =>
  request<ModelDownloadResponse & { cancelled?: boolean }>(
    `/models/downloads/${encodeURIComponent(jobId)}`,
    { method: 'DELETE' },
  );

// ---- 写操作（Tasks 页队列控制） ----

export const pauseQueue = () => request<{ success?: boolean }>('/cluster/queue/pause', { method: 'POST' });

export const resumeQueue = () => request<{ success?: boolean }>('/cluster/queue/resume', { method: 'POST' });

export const setQueueStrategy = (strategy: 'fifo' | 'mlfq') =>
  request<{ success?: boolean; strategy?: string }>('/cluster/queue/strategy', {
    method: 'POST',
    body: JSON.stringify({ strategy }),
  });

export const cancelQueueTask = (taskId: string) =>
  request<{ success?: boolean }>(`/cluster/queue/task/${encodeURIComponent(taskId)}`, {
    method: 'DELETE',
  });

export const cancelWorkflow = (workflowId: string) =>
  request<Record<string, unknown>>(`/workflows/${encodeURIComponent(workflowId)}/cancel`, {
    method: 'POST',
  });

export const cleanupWorkflowJournal = (params: { max_age_days?: number; max_records?: number } = {}) => {
  const query = new URLSearchParams();
  if (params.max_age_days !== undefined) query.set('max_age_days', String(params.max_age_days));
  if (params.max_records !== undefined) query.set('max_records', String(params.max_records));
  return request<Record<string, unknown>>(`/workflows/journal/cleanup${query.toString() ? `?${query}` : ''}`, { method: 'POST' });
};

export const shutdownSystem = (reason = 'operator_requested') =>
  request<{ ok?: boolean; message?: string }>('/system/shutdown', { method: 'POST', body: JSON.stringify({ reason }) });

export const runSpeculativeExperiment = (payload: { message?: string; execution_mode?: 'speculative_assisted'; allow_external?: boolean; [key: string]: unknown }) =>
  request<SpeculativeExperimentResponse>('/experimental/speculative', { method: 'POST', body: JSON.stringify(payload) });

export const fetchSpeculativeCapability = (signal?: AbortSignal) =>
  request<SpeculativeCapabilityResponse>('/experimental/speculative/capability', { signal });

export const createDeploymentSimulation = (payload: { artifact_id: string; runtime_profile: string; required_capabilities?: string[]; nodes: Array<Record<string, unknown>> }) =>
  request<DeploymentSimulationResponse>('/models/deployment-simulations', { method: 'POST', body: JSON.stringify(payload) });

export const fetchDeploymentSimulation = (planId: string, signal?: AbortSignal) =>
  request<DeploymentSimulationResponse>(`/models/deployment-simulations/${encodeURIComponent(planId)}`, { signal });

export const prepareDeploymentSimulation = (planId: string, nodes: Array<Record<string, unknown>> = []) =>
  request<DeploymentSimulationResponse>(`/models/deployment-simulations/${encodeURIComponent(planId)}/prepare`, { method: 'POST', body: JSON.stringify({ nodes }) });

export const activateDeploymentSimulation = (planId: string) =>
  request<DeploymentSimulationResponse>(`/models/deployment-simulations/${encodeURIComponent(planId)}/activate`, { method: 'POST' });

export const rollbackDeploymentSimulation = (planId: string) =>
  request<DeploymentSimulationResponse>(`/models/deployment-simulations/${encodeURIComponent(planId)}/rollback`, { method: 'POST' });

// ---- 令牌读写（Settings 页使用；与既有 frontend 共用同一存储键） ----

/** 日志接口所需的管理令牌，存于 localStorage。 */
// ---- 对话（工作台右栏） ----

export const fetchConversation = (sessionId = 'default', limit = 200, signal?: AbortSignal) =>
  request<ConversationResponse>(
    `/conversations?session_id=${encodeURIComponent(sessionId)}&limit=${limit}`,
    { signal },
  );

export const clearChat = (sessionId = 'default') =>
  request<{ success?: boolean }>(
    `/chat/clear?session_id=${encodeURIComponent(sessionId)}`,
    { method: 'POST' },
  );

/** 删除会话中的一整轮消息（用户 + 助手）— DELETE /api/sessions/:id/turns/:index。 */
export const deleteConversationTurn = (sessionId = 'default', turnIndex: number) =>
  request<{ status?: string; session_id?: string; turn_index?: number; remaining_turns?: number }>(
    `/sessions/${encodeURIComponent(sessionId)}/turns/${turnIndex}`,
    { method: 'DELETE' },
  );

export const cancelGeneration = (generationId: string) =>
  request<{ success?: boolean }>(
    `/chat/generations/${encodeURIComponent(generationId)}/cancel`,
    { method: 'POST' },
  );

/**
 * 上传文本附件 — POST /api/chat/upload。
 *
 * 不走上面的 request()：那个 helper 固定发 Content-Type: application/json，
 * 而 multipart 的 boundary 必须由浏览器自己写进头里，手动指定会让后端解析失败。
 * 字段名 file 对齐后端签名（api_server.py 的 `file: UploadFile = File(...)`）。
 */
export async function uploadChatFile(
  file: File,
  signal?: AbortSignal,
): Promise<ChatUploadResponse> {
  const token = readStorage('session', AUTH_TOKEN_STORAGE_KEY);
  const form = new FormData();
  form.append('file', file);

  const path = '/chat/upload';
  const res = await fetchResponse(
    `${BASE}${path}`,
    {
      method: 'POST',
      body: form,
      ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
    },
    { signal, timeoutMs: 30_000, path },
  );

  const text = await res.text();
  let data: Record<string, unknown> = {};
  if (text) {
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      data = { detail: text };
    }
  }

  const requestId =
    res.headers.get('X-Request-ID') || (data.request_id as string | undefined) || null;

  if (!res.ok) {
    throw new ApiError(normalizeDetail(data.detail, `HTTP ${res.status}`), {
      status: res.status,
      requestId,
      path,
      kind: kindForStatus(res.status),
      retryable: retryableForKind(kindForStatus(res.status)),
    });
  }

  return data as unknown as ChatUploadResponse;
}

export interface StreamChatOptions {
  sessionId?: string;
  /** interactive=真流式逐 token + 完成时提交会话事务，最适合聊天页。 */
  streamingMode?: 'full' | 'fast' | 'interactive';
  showThinking?: boolean;
  temperature?: number;
  maxNewTokens?: number;
  generationId?: string;
  signal?: AbortSignal;
  /** 逐 token 回调；参数是增量和累积文本。 */
  onToken?: (token: string, full: string) => void;
}

export interface StreamChatResult {
  content: string;
  thinkingContent: string;
  followups: string[];
  metrics: ChatMetrics;
}

/**
 * SSE 流式对话 — 直接用 fetch 读 ReadableStream。
 *
 * 不走上面的 request()：那个 helper 会把整个响应体读成 JSON，
 * 流式响应必须边读边解析。事件格式见后端 /api/chat/stream 的 docstring：
 *   data: {"token": "你"}
 *   data: {"done": true, "response": "...", "followups": [...], "metrics": {...}}
 */
export async function streamChat(
  message: string,
  options: StreamChatOptions = {},
): Promise<StreamChatResult> {
  const {
    sessionId = 'default',
    streamingMode = 'interactive',
    showThinking = false,
    temperature = 0.7,
    maxNewTokens = 1024,
    generationId,
    signal,
    onToken,
  } = options;

  const token = readStorage('session', AUTH_TOKEN_STORAGE_KEY);
  const path = '/chat/stream';
  const res = await fetchResponse(
    `${BASE}${path}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        streaming_mode: streamingMode,
        show_thinking: showThinking,
        temperature,
        max_new_tokens: maxNewTokens,
        ...(generationId ? { generation_id: generationId } : {}),
      }),
    },
    // 只限制建立连接的时间，不限制生成正文的时长。
    { signal, timeoutMs: 10_000, path },
  );

  const requestId = res.headers.get('X-Request-ID');
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = JSON.parse(await res.text()) as Record<string, unknown>;
      detail = normalizeDetail(data.detail, detail);
    } catch {
      /* 响应体不是 JSON，保留 HTTP 状态描述 */
    }
    throw new ApiError(detail, {
      status: res.status,
      requestId,
      path,
      kind: kindForStatus(res.status),
      retryable: retryableForKind(kindForStatus(res.status)),
    });
  }
  if (!res.body) {
    throw new ApiError('后端未返回流式响应体', { status: 0, requestId, path, kind: 'network' });
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let full = '';
  let done: Record<string, unknown> | null = null;

  /** 解析一行 `data: {...}`；返回 true 表示这是终止事件。 */
  const handleLine = (line: string): void => {
    if (!line.startsWith('data: ')) return;
    let event: Record<string, unknown>;
    try {
      event = JSON.parse(line.slice(6)) as Record<string, unknown>;
    } catch {
      return; // 分块边界上的半行，丢掉即可
    }
    if (event.done) {
      done = event;
      return;
    }
    if (typeof event.token === 'string') {
      full += event.token;
      onToken?.(event.token, full);
    }
  };

  try {
    for (;;) {
      const { done: streamDone, value } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) handleLine(line);
    }
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new ApiError('流式连接中断，请稍后重试。', {
      status: 0,
      requestId,
      path,
      kind: 'network',
      retryable: true,
    });
  }
  if (buffer) handleLine(buffer);

  if (done === null) {
    throw new ApiError('流式响应中断，未收到完成事件', {
      status: 0,
      requestId,
      path,
      kind: 'network',
      retryable: true,
    });
  }

  const final: Record<string, unknown> = done;
  if (typeof final.error === 'string' && final.error) {
    throw new ApiError(final.error, {
      status: 200,
      requestId: (final.request_id as string | undefined) ?? requestId,
      path: '/chat/stream',
    });
  }

  return {
    content: (final.response as string | undefined) ?? full,
    thinkingContent:
      (final.thinking_content as string | undefined) ?? (final.thinking as string | undefined) ?? '',
    followups: (final.followups as string[] | undefined) ?? [],
    metrics: (final.metrics as ChatMetrics | undefined) ?? {},
  };
}

export function getLogToken(): string {
  return readStorage('local', LOG_ADMIN_TOKEN_STORAGE_KEY);
}

export function setLogToken(token: string): void {
  try {
    if (typeof window === 'undefined') return;
    if (token) window.localStorage.setItem(LOG_ADMIN_TOKEN_STORAGE_KEY, token);
    else window.localStorage.removeItem(LOG_ADMIN_TOKEN_STORAGE_KEY);
  } catch {
    // 存储不可用时静默失败，调用方通过再次读取判断是否生效。
  }
}

/** 登录会话令牌，存于 sessionStorage；只读，登录流程仍由既有前端负责。 */
export function getAuthToken(): string {
  return readStorage('session', AUTH_TOKEN_STORAGE_KEY);
}

export function setAuthToken(token: string): void {
  try {
    if (typeof window === 'undefined') return;
    if (token) window.sessionStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    else window.sessionStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // Storage may be unavailable in private or embedded contexts.
  }
}

/** 把任意异常转成可展示的中文文案，避免页面直接渲染 Error 对象。 */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    switch (err.kind) {
      case 'unauthorized':
        return `登录已失效，请重新登录。${err.message ? `（${err.message}）` : ''}`;
      case 'forbidden':
        return `无权限：${err.message}`;
      case 'not_found':
        return `接口不存在：${err.path}`;
      case 'conflict':
        return `状态冲突：${err.message}`;
      case 'timeout':
        return `${err.message || '请求超时'}，请稍后重试。`;
      case 'rate_limited':
        return `请求过于频繁，请稍后重试。${err.message ? `（${err.message}）` : ''}`;
      case 'server':
        return `后端暂时不可用：${err.message}`;
      case 'network':
        return err.message || '无法连接后端，请检查服务是否启动或网络连接。';
      default:
        return err.message;
    }
  }
  if (err instanceof DOMException && err.name === 'AbortError') return '';
  if (err instanceof TypeError) return '无法连接后端，请检查服务是否启动或网络连接。';
  if (err instanceof Error) return err.message;
  return '未知错误';
}

export function getErrorKind(err: unknown): ApiErrorKind {
  if (err instanceof ApiError) return err.kind;
  if (err instanceof DOMException && err.name === 'AbortError') return 'aborted';
  if (err instanceof TypeError) return 'network';
  return 'unknown';
}

export function isRetryableError(err: unknown): boolean {
  if (err instanceof ApiError) return err.retryable;
  return getErrorKind(err) === 'network';
}
