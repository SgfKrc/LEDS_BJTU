/**
 * API 客户端 — 与 FastAPI 后端通信
 */

const BASE = '/api';
const LOG_ADMIN_TOKEN_STORAGE_KEY = 'qlh_log_admin_token';

function getLogTokenStorage() {
  try {
    if (typeof window === 'undefined') return null;
    return window.localStorage || null;
  } catch (_) {
    return null;
  }
}

export function getLogAdminToken() {
  try {
    return getLogTokenStorage()?.getItem(LOG_ADMIN_TOKEN_STORAGE_KEY) || '';
  } catch (_) {
    return '';
  }
}

export function setLogAdminToken(token) {
  try {
    const storage = getLogTokenStorage();
    if (!storage) return;
    const normalized = (token || '').trim();
    if (normalized) {
      storage.setItem(LOG_ADMIN_TOKEN_STORAGE_KEY, normalized);
    } else {
      storage.removeItem(LOG_ADMIN_TOKEN_STORAGE_KEY);
    }
  } catch (_) {
    // Ignore storage failures; local log access still works without a token.
  }
}

function withLogAdminHeaders(headers = {}) {
  const token = getLogAdminToken();
  return token ? { ...headers, 'X-QLH-Log-Token': token } : headers;
}

function makeApiError(message, { status = 0, requestId = null, path = '' } = {}) {
  const suffix = requestId ? ` (request_id: ${requestId})` : '';
  const error = new Error(`${message}${suffix}`);
  error.detail = message;
  error.status = status;
  error.requestId = requestId;
  error.path = path;
  if (requestId) {
    console.error('API request failed', { path, status, requestId, detail: message });
  }
  return error;
}

function normalizeErrorDetail(detail, fallback) {
  if (typeof detail === 'string' && detail) return detail;
  if (detail && typeof detail === 'object') {
    if (typeof detail.message === 'string' && detail.message) return detail.message;
    try { return JSON.stringify(detail); } catch (_) {}
  }
  return fallback;
}

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const { signal, ...rest } = options;
  const res = await fetch(url, {
    ...rest,                                                         // 先展开 rest，允许 headers 覆盖
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...(signal ? { signal } : {}),
  });
  const text = await res.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (_) {
      data = { detail: text };
    }
  }
  const requestId = res.headers.get('X-Request-ID') || data.request_id || null;
  if (!res.ok) {
    throw makeApiError(normalizeErrorDetail(data.detail, `HTTP ${res.status}`), {
      status: res.status,
      requestId,
      path,
    });
  }
  return data;
}

export async function fetchStatus() {
  return request('/status');
}

export async function fetchCurrentModel() {
  return request('/models/current');
}

export async function fetchAvailableModels() {
  return request('/models/available');
}

export async function loadModel(engine, quantType, useCompile = false, modelId = null) {
  return request('/models/load', {
    method: 'POST',
    body: JSON.stringify({
      engine: engine || 'llama_cpp',
      quant_type: quantType,
      use_compile: useCompile,
      ...(modelId ? { model_id: modelId } : {}),
    }),
  });
}

// ---- P3: 多模型实验支持 ----

export async function fetchModels() {
  return request('/models');
}

export async function switchModel(modelId, quantType = 'int4', engine = 'auto') {
  return request('/models/switch', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, quant_type: quantType, engine }),
  });
}

export async function fetchModelRegistry() {
  return request('/models/registry');
}

export async function registerModel(config) {
  return request('/models/registry', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function unregisterModel(modelId) {
  return request(`/models/registry/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  });
}

export async function sendMessage(message, opts = {}) {
  return request('/chat', {
    method: 'POST',
    ...(opts.signal ? { signal: opts.signal } : {}),
    body: JSON.stringify({
      message,
      session_id: opts.sessionId || null,
      max_new_tokens: opts.maxNewTokens || 1024,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.9,
      show_thinking: opts.showThinking || false,
      execution_mode: opts.executionMode || 'auto',
      task_graph_template: opts.taskGraphTemplate || 'dual_candidate',
      task_graph_auto_remote: opts.taskGraphAutoRemote === true,
      workflow_id: opts.workflowId || null,
      generation_id: opts.generationId || null,
    }),
  });
}

/**
 * SSE 流式发送消息 — 调用 /api/chat/stream，解析 Server-Sent Events。
 *
 * @param {string} message - 用户消息
 * @param {object} opts
 * @param {string} opts.streamingMode - 'full'（默认，完整功能）| 'fast'（真流式逐token）
 * @param {function} opts.onToken - fast 模式回调 (token: string, fullText: string)
 * @returns {Promise<{content, thinking_content, metrics, followups}>}
 */
export async function sendMessageStream(message, opts = {}) {
  const url = `${BASE}/chat/stream`;
  const streamingMode = opts.streamingMode || 'full';
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal || null,
    body: JSON.stringify({
      message,
      session_id: opts.sessionId || null,
      max_new_tokens: opts.maxNewTokens || 1024,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.9,
      show_thinking: opts.showThinking || false,
      streaming_mode: streamingMode,
      execution_mode: opts.executionMode || 'auto',
      task_graph_template: opts.taskGraphTemplate || 'dual_candidate',
      task_graph_auto_remote: opts.taskGraphAutoRemote === true,
      workflow_id: opts.workflowId || null,
      generation_id: opts.generationId || null,
    }),
  });

  const requestId = res.headers.get('X-Request-ID') || null;
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const d = await res.json();
      detail = normalizeErrorDetail(d.detail, detail);
    } catch (_) {}
    throw makeApiError(detail, { status: res.status, requestId, path: '/chat/stream' });
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullResponse = '';
  let finalResult = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const event = JSON.parse(line.slice(6));
        if (event.done) {
          finalResult = event;
        } else if (event.token) {
          fullResponse += event.token;
          if (opts.onToken) opts.onToken(event.token, fullResponse);
        }
      } catch (_) { /* skip incomplete chunks */ }
    }
  }

  // 处理 buffer 中可能残留的最后一条
  if (buffer.startsWith('data: ')) {
    try {
      const event = JSON.parse(buffer.slice(6));
      if (event.done) finalResult = event;
    } catch (_) {}
  }

  const finalRequestId = finalResult?.request_id || finalResult?.metrics?.request_id || requestId;
  if (!finalResult) {
    throw makeApiError('未收到流式响应', { status: 0, requestId, path: '/chat/stream' });
  }
  if (finalResult.error) {
    throw makeApiError(finalResult.error, {
      status: 200,
      requestId: finalRequestId,
      path: '/chat/stream',
    });
  }

  return {
    content: finalResult.response || fullResponse,
    thinking_content: finalResult.thinking_content || finalResult.thinking || null,
    metrics: finalResult.metrics || {},
    followups: finalResult.followups || [],
  };
}

export async function clearChat(sessionId = 'default') {
  return request(`/chat/clear?session_id=${encodeURIComponent(sessionId || 'default')}`, {
    method: 'POST',
  });
}

export async function fetchTaskGraphStatus(sessionId = '') {
  const query = new URLSearchParams({ limit: '1' });
  if (sessionId) query.set('session_id', sessionId);
  return request(`/workflows?${query.toString()}`);
}

export async function fetchWorkflow(workflowId, options = {}) {
  return request(`/workflows/${encodeURIComponent(workflowId)}`, options);
}

export async function cancelWorkflow(workflowId) {
  return request(`/workflows/${encodeURIComponent(workflowId)}/cancel`, {
    method: 'POST',
  });
}

export async function cancelChatGeneration(generationId) {
  return request(`/chat/generations/${encodeURIComponent(generationId)}/cancel`, {
    method: 'POST',
  });
}

export async function healthCheck() {
  return request('/health');
}

export async function fetchDeviceProfile() {
  return request('/device/profile');
}

export async function autoConfigure() {
  return request('/device/auto-configure', { method: 'POST' });
}

export async function selectGpu(gpuIndex) {
  return request('/device/select-gpu', {
    method: 'POST',
    body: JSON.stringify({ gpu_index: gpuIndex }),
  });
}

export async function fetchPresets() {
  return request('/presets');
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const url = `${BASE}/chat/upload`;
  const res = await fetch(url, { method: 'POST', body: formData });
  const text = await res.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (_) {
      data = { detail: text };
    }
  }
  const requestId = res.headers.get('X-Request-ID') || data.request_id || null;
  if (!res.ok) {
    throw makeApiError(data.detail || `HTTP ${res.status}`, {
      status: res.status,
      requestId,
      path: '/chat/upload',
    });
  }
  return data;
}

// ---- 集群管理 ----

export async function fetchClusterStatus() {
  return request('/cluster/status');
}

export async function fetchClusterNodes() {
  return request('/cluster/nodes');
}

export async function deregisterNode(nodeId) {
  return request(`/cluster/nodes/${encodeURIComponent(nodeId)}/deregister`, {
    method: 'POST',
  });
}

export async function deleteClusterNode(nodeId) {
  return request(`/cluster/nodes/${encodeURIComponent(nodeId)}`, {
    method: 'DELETE',
  });
}

export async function fetchClusterConfig() {
  return request('/cluster/config');
}

export async function fetchMyRole() {
  return request('/cluster/my-role');
}

export async function updateMaxNodes(maxNodes) {
  return request('/cluster/config/max-nodes', {
    method: 'PUT',
    body: JSON.stringify({ max_nodes: maxNodes }),
  });
}

// ---- 对话历史（数据库持久化） ----

export async function fetchConversations(sessionId = 'default', limit = 200) {
  return request(`/conversations?session_id=${encodeURIComponent(sessionId)}&limit=${limit}`);
}

export async function deleteConversations(sessionId = 'default') {
  return request(`/conversations?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}

export async function fetchDbHealth() {
  return request('/db/health');
}

// ---- 会话管理（多会话支持） ----

export async function createSession(title = '新对话', firstMessage = null) {
  const body = { title };
  if (firstMessage) body.first_message = firstMessage;
  return request('/sessions', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function fetchSessions(limit = 50, offset = 0) {
  return request(`/sessions?limit=${limit}&offset=${offset}`);
}

export async function fetchSession(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`);
}

export async function renameSession(sessionId, title) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PUT',
    body: JSON.stringify({ title }),
  });
}

export async function deleteSession(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}

export async function activateSession(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}/activate`, {
    method: 'POST',
  });
}

// ---- 单轮对话删除 ----

export async function deleteTurn(sessionId, turnIndex) {
  return request(`/sessions/${encodeURIComponent(sessionId)}/turns/${turnIndex}`, {
    method: 'DELETE',
  });
}

// ---- 节点连接 ----

export async function fetchInviteInfo() {
  return request('/cluster/invite');
}

export async function connectToMaster(masterHost, masterPort, switchToClient = false) {
  return request('/cluster/connect', {
    method: 'POST',
    body: JSON.stringify({
      master_host: masterHost,
      master_port: masterPort,
      switch_to_client: switchToClient,
    }),
  });
}

// ---- 主节点自动发现（数据库查询） ----

export async function discoverMaster() {
  return request('/cluster/discover');
}

export async function resetMasterIdentity(confirm = 'reset') {
  return request('/cluster/reset-identity', {
    method: 'POST',
    body: JSON.stringify({ confirm }),
  });
}

export async function manualRegisterNode(
  nodeId,
  hostname = '',
  address = '',
  networkType = 'unknown',
  nodeType = 'pc',
) {
  return request('/cluster/nodes/register', {
    method: 'POST',
    body: JSON.stringify({
      node_id: nodeId,
      hostname: hostname,
      address: address,
      network_type: networkType,
      node_type: nodeType,
    }),
  });
}

export async function fetchMasterHealth() {
  return request('/cluster/master-health');
}

export async function testEmailNotification() {
  return request('/cluster/email-test', { method: 'POST' });
}

// ---- 分布式推理开关 ----

export async function fetchDistributedInferenceConfig() {
  return request('/cluster/config/distributed-inference');
}

export async function updateDistributedInferenceConfig(enabled) {
  return request('/cluster/config/distributed-inference', {
    method: 'PUT',
    body: JSON.stringify({ enabled }),
  });
}

// ---- 动态模型分层 ----

export async function fetchLayerAssignment() {
  return request('/cluster/layers');
}

export async function updateLayerAssignment(assignments) {
  return request('/cluster/layers', {
    method: 'PUT',
    body: JSON.stringify({ assignments }),
  });
}

export async function resetLayerAssignments() {
  return request('/cluster/layers', { method: 'DELETE' });
}

// ---- 角色转让 ----

export async function transferMasterRole(targetNodeId) {
  return request('/cluster/transfer-master', {
    method: 'POST',
    body: JSON.stringify({ target_node_id: targetNodeId }),
  });
}

export async function fetchTransferLogs() {
  return request('/cluster/transfer-logs');
}

// ---- 备用主节点 ----

export async function fetchSpareMaster() {
  return request('/cluster/spare-master');
}

export async function designateSpareMaster(nodeId) {
  return request('/cluster/spare-master', {
    method: 'POST',
    body: JSON.stringify({ target_node_id: nodeId }),
  });
}

export async function removeSpareMaster() {
  return request('/cluster/spare-master', { method: 'DELETE' });
}

export async function fetchSpareMasterLogs() {
  return request('/cluster/spare-master/logs');
}

// ---- 用户偏好设置云同步 ----

export async function fetchUserSettings() {
  return request('/user/settings');
}

export async function updateUserSettings(settings) {
  return request('/user/settings', {
    method: 'PUT',
    body: JSON.stringify({ settings }),
  });
}

// ---- 对话云同步状态 ----

export async function fetchConversationSyncStatus() {
  return request('/conversations/sync-status');
}

// ---- 日志管理 ----

export async function fetchLogFiles() {
  return request('/logs', { headers: withLogAdminHeaders() });
}

export async function fetchLogContent(filename) {
  return request(`/logs/${encodeURIComponent(filename)}`, { headers: withLogAdminHeaders() });
}

export async function deleteLogFile(filename) {
  return request(`/logs/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: withLogAdminHeaders(),
  });
}

export async function deleteAllLogFiles() {
  return request('/logs', {
    method: 'DELETE',
    headers: withLogAdminHeaders(),
  });
}

export async function fetchRecentLogs(params = {}) {
  const qs = new URLSearchParams();
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.level) qs.set('level', params.level);
  if (params.name) qs.set('name', params.name);
  if (params.node_id) qs.set('node_id', params.node_id);
  if (params.request_id) qs.set('request_id', params.request_id);
  const q = qs.toString();
  return request(`/logs/recent${q ? '?' + q : ''}`, { headers: withLogAdminHeaders() });
}

export async function fetchLogStats() {
  return request('/logs/stats', { headers: withLogAdminHeaders() });
}

export function getLogDownloadUrl(filename) {
  return `${BASE}/logs/download?name=${encodeURIComponent(filename)}`;
}

export async function downloadLogFileBlob(filename) {
  const path = `/logs/download?name=${encodeURIComponent(filename)}`;
  const res = await fetch(`${BASE}${path}`, {
    headers: withLogAdminHeaders({ Accept: 'text/plain' }),
  });

  if (!res.ok) {
    const text = await res.text();
    let data = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (_) {
        data = { detail: text };
      }
    }
    const requestId = res.headers.get('X-Request-ID') || data.request_id || null;
    throw makeApiError(data.detail || `HTTP ${res.status}`, {
      status: res.status,
      requestId,
      path,
    });
  }

  return {
    blob: await res.blob(),
    filename,
  };
}

export async function fetchNodeRecentLogs(nodeId, params = {}) {
  const qs = new URLSearchParams();
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.level) qs.set('level', params.level);
  if (params.name) qs.set('name', params.name);
  if (params.timeout) qs.set('timeout', String(params.timeout));
  const q = qs.toString();
  return request(`/logs/node/${encodeURIComponent(nodeId)}/recent${q ? '?' + q : ''}`, {
    headers: withLogAdminHeaders(),
  });
}

export async function fetchNodesLogSummary() {
  return request('/logs/nodes-summary', { headers: withLogAdminHeaders() });
}

// ---- P3: 主节点转让审查 ----

export async function createReviewTicket(targetNodeId, reason, timeoutHours) {
  return request('/cluster/review/create', {
    method: 'POST',
    body: JSON.stringify({
      target_node_id: targetNodeId,
      reason: reason || '',
      timeout_hours: timeoutHours || 48,
    }),
  });
}

export async function castVote(ticketId, vote, comment) {
  return request('/cluster/review/vote', {
    method: 'POST',
    body: JSON.stringify({
      ticket_id: ticketId,
      vote,
      comment: comment || '',
    }),
  });
}

export async function fetchReviewTickets(status) {
  const params = status ? `?status=${encodeURIComponent(status)}` : '';
  return request(`/cluster/review/tickets${params}`);
}

export async function fetchReviewTicket(ticketId) {
  return request(`/cluster/review/tickets/${encodeURIComponent(ticketId)}`);
}

export async function checkCanVote() {
  return request('/cluster/review/can-vote');
}

export async function expireReviewCheck() {
  return request('/cluster/review/expire-check', { method: 'POST' });
}

export async function deleteReviewTicket(ticketId) {
  return request(`/cluster/review/tickets/${encodeURIComponent(ticketId)}`, {
    method: 'DELETE',
  });
}

export async function deleteResolvedReviewTickets() {
  return request('/cluster/review/tickets', { method: 'DELETE' });
}

// ---- 请求队列管理 (Phase 3) ----

export async function fetchQueue() {
  return request('/cluster/queue');
}


// ============================================================
// L5: 前端错误上报
// ============================================================

let _errorReporterInstalled = false;

/**
 * 安装全局前端错误上报处理器。
 * 捕获 window.onerror 和 unhandledrejection 并发送到后端诊断日志。
 * 仅在生产环境生效（import.meta.env.PROD），且仅安装一次。
 */
export function installErrorReporter() {
  if (_errorReporterInstalled) return;
  _errorReporterInstalled = true;

  // 仅在非开发模式下启用自动上报（DEV 下错误在 console 可见）
  if (import.meta.env.DEV) return;

  const sendError = (report) => {
    const url = `${BASE}/logs/client-error`;
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: (report.message || '').slice(0, 500),
        source: report.source || 'unknown',
        stack: (report.stack || '').slice(0, 2000),
        url: report.url || '',
        line: report.line || 0,
        col: report.col || 0,
        user_agent: navigator.userAgent || '',
        extra: report.extra || {},
      }),
    }).catch(() => { /* 上报失败不触发进一步错误 */ });
  };

  // 全局未捕获异常
  window.addEventListener('error', (e) => {
    sendError({
      source: 'window.onerror',
      message: e.message || 'Unknown error',
      stack: e.error?.stack || '',
      url: e.filename || window.location.href,
      line: e.lineno || 0,
      col: e.colno || 0,
    });
  });

  // 未处理的 Promise 拒绝
  window.addEventListener('unhandledrejection', (e) => {
    sendError({
      source: 'unhandledrejection',
      message: e.reason?.message || String(e.reason || 'Unhandled rejection'),
      stack: e.reason?.stack || '',
      url: window.location.href,
    });
  });
}

export async function setQueueStrategy(strategy) {
  return request('/cluster/queue/strategy', {
    method: 'POST',
    body: JSON.stringify({ strategy }),
  });
}

export async function pauseQueue() {
  return request('/cluster/queue/pause', { method: 'POST' });
}

export async function resumeQueue() {
  return request('/cluster/queue/resume', { method: 'POST' });
}

export async function clearQueue() {
  return request('/cluster/queue/clear', { method: 'POST' });
}

export async function cancelQueueTask(taskId) {
  return request(`/cluster/queue/task/${encodeURIComponent(taskId)}`, {
    method: 'DELETE',
  });
}
