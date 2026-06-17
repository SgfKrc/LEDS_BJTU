/**
 * API 客户端 — 与 FastAPI 后端通信
 */

const BASE = '/api';

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
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

export async function loadModel(engine, quantType, useCompile = false) {
  return request('/models/load', {
    method: 'POST',
    body: JSON.stringify({ engine: engine || 'auto', quant_type: quantType, use_compile: useCompile }),
  });
}

export async function sendMessage(message, opts = {}) {
  return request('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      session_id: opts.sessionId || null,
      max_new_tokens: opts.maxNewTokens || 512,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.9,
      show_thinking: opts.showThinking || false,
    }),
  });
}

export async function clearChat() {
  return request('/chat/clear', { method: 'POST' });
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
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
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

export async function connectToMaster(masterHost, masterPort) {
  return request('/cluster/connect', {
    method: 'POST',
    body: JSON.stringify({ master_host: masterHost, master_port: masterPort }),
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

export async function manualRegisterNode(nodeId, hostname = '', address = '', networkType = 'unknown') {
  return request('/cluster/nodes/register', {
    method: 'POST',
    body: JSON.stringify({
      node_id: nodeId,
      hostname: hostname,
      address: address,
      network_type: networkType,
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
