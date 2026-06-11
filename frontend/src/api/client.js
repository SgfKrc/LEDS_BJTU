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

export async function loadModel(quantType, useCompile = false) {
  return request('/models/load', {
    method: 'POST',
    body: JSON.stringify({ quant_type: quantType, use_compile: useCompile }),
  });
}

export async function sendMessage(message, opts = {}) {
  return request('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      max_new_tokens: opts.maxNewTokens || 512,
      temperature: opts.temperature ?? 0.7,
      top_p: opts.topP ?? 0.9,
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
