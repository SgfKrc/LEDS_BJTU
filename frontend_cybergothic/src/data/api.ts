/**
 * 数据访问层 — 唯一与后端通信的入口。
 *
 * 视觉组件不允许直接 fetch；页面通过 hooks 消费这里的函数（§5.1）。
 * 鉴权沿用既有 frontend 的约定：sessionStorage 中的 bearer token（若存在）。
 */

import type {
  ChatMetrics,
  ChatUploadResponse,
  ClusterNodesResponse,
  ConversationResponse,
  MyRoleResponse,
  PipelineCapacityResponse,
  QueueResponse,
  RagHealthResponse,
  RecentLogsResponse,
  SessionsResponse,
  SystemStatusResponse,
  WorkflowsResponse,
} from './types';

const BASE = '/api';

/** 与既有 frontend/src/api/client.js 保持同一个 key，便于两套 UI 共存。 */
const AUTH_TOKEN_STORAGE_KEY = 'qlh-auth-session-token';
const LOG_ADMIN_TOKEN_STORAGE_KEY = 'qlh_log_admin_token';

export class ApiError extends Error {
  status: number;
  requestId: string | null;
  path: string;

  constructor(
    message: string,
    opts: { status?: number; requestId?: string | null; path?: string } = {},
  ) {
    const requestId = opts.requestId ?? null;
    super(requestId ? `${message} (request_id: ${requestId})` : message);
    this.name = 'ApiError';
    this.status = opts.status ?? 0;
    this.requestId = requestId;
    this.path = opts.path ?? '';
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
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { signal, withLogToken, headers, ...rest } = options;
  const token = readStorage('session', AUTH_TOKEN_STORAGE_KEY);
  const logToken = withLogToken ? readStorage('local', LOG_ADMIN_TOKEN_STORAGE_KEY) : '';

  const res = await fetch(`${BASE}${path}`, {
    ...rest,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(logToken ? { 'X-QLH-Log-Token': logToken } : {}),
    },
    ...(signal ? { signal } : {}),
  });

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
    });
  }

  return data as T;
}

// ---- 只读快照接口（Overview / Nodes / Activity 使用） ----

export const fetchSystemStatus = (signal?: AbortSignal) =>
  request<SystemStatusResponse>('/status', { signal });

export const fetchClusterNodes = (signal?: AbortSignal) =>
  request<ClusterNodesResponse>('/cluster/nodes', { signal });

export const fetchMyRole = (signal?: AbortSignal) =>
  request<MyRoleResponse>('/cluster/my-role', { signal });

export const fetchQueue = (signal?: AbortSignal) =>
  request<QueueResponse>('/cluster/queue', { signal });

export const fetchWorkflows = (limit = 20, signal?: AbortSignal) =>
  request<WorkflowsResponse>(`/workflows?limit=${encodeURIComponent(limit)}`, { signal });

export const fetchPipelineCapacity = (signal?: AbortSignal) =>
  request<PipelineCapacityResponse>('/cluster/pipeline-capacity', { signal });

export const fetchRagHealth = (signal?: AbortSignal) =>
  request<RagHealthResponse>('/rag/health', { signal });

export const fetchSessions = (limit = 20, signal?: AbortSignal) =>
  request<SessionsResponse>(`/sessions?limit=${encodeURIComponent(limit)}`, { signal });

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

export const fetchHealth = (signal?: AbortSignal) =>
  request<{ status: string; timestamp: number }>('/health', { signal });

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

  const res = await fetch(`${BASE}/chat/upload`, {
    method: 'POST',
    body: form,
    ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
    ...(signal ? { signal } : {}),
  });

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
      path: '/chat/upload',
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
  const res = await fetch(`${BASE}/chat/stream`, {
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
    ...(signal ? { signal } : {}),
  });

  const requestId = res.headers.get('X-Request-ID');
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = JSON.parse(await res.text()) as Record<string, unknown>;
      detail = normalizeDetail(data.detail, detail);
    } catch {
      /* 响应体不是 JSON，保留 HTTP 状态描述 */
    }
    throw new ApiError(detail, { status: res.status, requestId, path: '/chat/stream' });
  }
  if (!res.body) {
    throw new ApiError('后端未返回流式响应体', { status: 0, requestId, path: '/chat/stream' });
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

  for (;;) {
    const { done: streamDone, value } = await reader.read();
    if (streamDone) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) handleLine(line);
  }
  if (buffer) handleLine(buffer);

  if (done === null) {
    throw new ApiError('流式响应中断，未收到完成事件', {
      status: 0,
      requestId,
      path: '/chat/stream',
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

/** 把任意异常转成可展示的中文文案，避免页面直接渲染 Error 对象。 */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 403) return `无权限：${err.message}`;
    if (err.status === 404) return `接口不存在：${err.path}`;
    if (err.status === 0) return `无法连接后端：${err.message}`;
    return err.message;
  }
  if (err instanceof DOMException && err.name === 'AbortError') return '';
  if (err instanceof TypeError) return '网络请求失败，后端可能未启动。';
  if (err instanceof Error) return err.message;
  return '未知错误';
}
