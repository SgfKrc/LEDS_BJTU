export const MODEL_PULL_TERMINAL_STATES = new Set([
  'registered', 'failed', 'cancelled', 'rejected', 'quarantined', 'rolled_back',
]);

export function modelPullProgressPercent(job) {
  const total = Number(job?.progress?.total_bytes || 0);
  const downloaded = Number(job?.progress?.downloaded_bytes || 0);
  if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(downloaded)) return 0;
  return Math.max(0, Math.min(100, Math.round((downloaded / total) * 100)));
}

export function isModelPullActive(job) {
  return Boolean(job?.state) && !MODEL_PULL_TERMINAL_STATES.has(job.state);
}

export function modelRuntimeLabel(runtimeCheck) {
  const status = runtimeCheck?.status || 'unchecked';
  const labels = {
    ready: '可运行',
    load_failed: '加载失败',
    resource_rejected: '资源不足',
    stale: '需要复检',
    unchecked: '未检查',
  };
  return labels[status] || status;
}

export function buildLocalImportRequest(form) {
  return {
    source_path: String(form?.sourcePath || '').trim(),
    namespace: String(form?.namespace || 'user').trim() || 'user',
    name: String(form?.name || '').trim(),
    tag: String(form?.tag || 'latest').trim() || 'latest',
  };
}

export function buildModelPullRequest(form) {
  const allowPatterns = normalizeAllowPatterns(form?.allowPatterns);
  return {
    source_id: String(form?.sourceId || '').trim(),
    source: {
      provider: form?.provider === 'huggingface' ? 'huggingface' : 'gguf_huggingface',
      repo_id: String(form?.repoId || '').trim(),
      requested_revision: String(form?.revision || 'main').trim() || 'main',
      allow_patterns: allowPatterns,
    },
    cancel_policy: 'keep_partial',
  };
}

export function buildModelResolveRequest(form) {
  return {
    source_id: String(form?.sourceId || '').trim(),
    repo_id: String(form?.repoId || '').trim(),
    requested_revision: String(form?.revision || 'main').trim() || 'main',
    allow_patterns: normalizeAllowPatterns(form?.allowPatterns),
  };
}

export function modelPullRequestKey(form) {
  return JSON.stringify(buildModelResolveRequest(form));
}

export function buildModelSourceRequest(form) {
  const credentialRef = String(form?.credentialRef || '').trim();
  const parsedPriority = Number(form?.priority);
  return {
    source_id: String(form?.sourceId || '').trim(),
    payload: {
      name: String(form?.name || '').trim(),
      provider: form?.provider === 'modelscope' ? 'modelscope' : 'huggingface',
      endpoint: String(form?.endpoint || '').trim().replace(/\/+$/, ''),
      credential_ref: credentialRef || null,
      priority: Number.isInteger(parsedPriority) && parsedPriority >= 0 ? parsedPriority : 100,
      enabled: form?.enabled !== false,
    },
  };
}

export function credentialIdFromRef(credentialRef) {
  const value = String(credentialRef || '').trim();
  return value.startsWith('os:qlh/') ? value.slice('os:qlh/'.length) : '';
}

function normalizeAllowPatterns(value) {
  const allowPatterns = String(value || '')
    .split(/[\n,]/)
    .map((value) => value.trim())
    .filter(Boolean);
  return allowPatterns.length > 0 ? allowPatterns : null;
}
