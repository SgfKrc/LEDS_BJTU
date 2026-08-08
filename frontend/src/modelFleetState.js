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
  const allowPatterns = String(form?.allowPatterns || '')
    .split(/[\n,]/)
    .map((value) => value.trim())
    .filter(Boolean);
  return {
    source: {
      provider: form?.provider === 'huggingface' ? 'huggingface' : 'gguf_huggingface',
      repo_id: String(form?.repoId || '').trim(),
      requested_revision: String(form?.revision || 'main').trim() || 'main',
      allow_patterns: allowPatterns.length > 0 ? allowPatterns : null,
    },
    cancel_policy: 'keep_partial',
  };
}
