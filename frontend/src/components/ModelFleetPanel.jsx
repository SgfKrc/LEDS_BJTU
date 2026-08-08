import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  buildLocalImportRequest,
  buildModelPullRequest,
  isModelPullActive,
  modelPullProgressPercent,
  modelRuntimeLabel,
} from '../modelFleetState';

const TABS = [
  ['artifacts', '工件'],
  ['acquire', '导入与拉取'],
  ['jobs', '任务'],
  ['network', '网络'],
];

const EMPTY_INVENTORY = {
  node_id: 'local',
  artifacts: [],
  summary: { total: 0, ready: 0, stale: 0, attention: 0, unchecked: 0, total_bytes: 0 },
};

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const displayed = size >= 10 || unit === 0 ? Math.round(size) : Number(size.toFixed(1));
  return `${displayed} ${units[unit]}`;
}

function formatTime(value) {
  if (!value) return '未检查';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '未检查' : date.toLocaleString();
}

function runtimeClass(runtimeCheck) {
  return runtimeCheck?.status || 'unchecked';
}

export default function ModelFleetPanel({ onToast }) {
  const [tab, setTab] = useState('artifacts');
  const [inventory, setInventory] = useState(EMPTY_INVENTORY);
  const [jobs, setJobs] = useState([]);
  const [network, setNetwork] = useState({ proxy: { source: 'direct', endpoint: null }, user_proxy: null });
  const [proxyDraft, setProxyDraft] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [importForm, setImportForm] = useState({
    sourcePath: '', namespace: 'user', name: '', tag: 'latest',
  });
  const [pullForm, setPullForm] = useState({
    provider: 'gguf_huggingface', repoId: '', revision: 'main', allowPatterns: '*.gguf',
  });

  const refresh = useCallback(async ({ preserveProxy = false, quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    const api = await import('../api/client');
    const results = await Promise.allSettled([
      api.fetchModelArtifacts(),
      api.fetchModelPullJobs(),
      api.fetchModelNetwork(),
    ]);
    const failures = [];
    if (results[0].status === 'fulfilled') setInventory(results[0].value);
    else failures.push(results[0].reason);
    if (results[1].status === 'fulfilled') setJobs(results[1].value.jobs || []);
    else failures.push(results[1].reason);
    if (results[2].status === 'fulfilled') {
      setNetwork(results[2].value);
      if (!preserveProxy) setProxyDraft(results[2].value.user_proxy?.url || '');
    } else {
      failures.push(results[2].reason);
    }
    setError(failures.length === 3 ? (failures[0]?.message || '模型控制面不可用') : '');
    if (!quiet) setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const hasActiveJobs = useMemo(() => jobs.some(isModelPullActive), [jobs]);
  useEffect(() => {
    if (!hasActiveJobs) return undefined;
    const timer = window.setInterval(() => refresh({ preserveProxy: true, quiet: true }), 2000);
    return () => window.clearInterval(timer);
  }, [hasActiveJobs, refresh]);

  const run = useCallback(async (key, action, successMessage, options = {}) => {
    setBusy(key);
    try {
      const result = await action();
      if (successMessage) onToast?.({ type: options.warning ? 'warning' : 'success', msg: successMessage(result) });
      await refresh({ preserveProxy: options.preserveProxy === true, quiet: true });
      return result;
    } catch (actionError) {
      onToast?.({ type: 'error', msg: actionError.message || '操作失败' });
      return null;
    } finally {
      setBusy('');
    }
  }, [onToast, refresh]);

  const submitImport = async (event) => {
    event.preventDefault();
    const payload = buildLocalImportRequest(importForm);
    if (!payload.source_path || !payload.name) {
      onToast?.({ type: 'error', msg: '导入路径和工件名称为必填项' });
      return;
    }
    const result = await run('import', async () => {
      const api = await import('../api/client');
      return api.importLocalModel(payload);
    }, (value) => value?.runnable ? '本地工件已导入并通过运行时检查' : '本地工件已导入，等待运行时检查处理', {
      warning: true,
    });
    if (result) setTab('artifacts');
  };

  const submitPull = async (event) => {
    event.preventDefault();
    const payload = buildModelPullRequest(pullForm);
    if (!payload.source.repo_id) {
      onToast?.({ type: 'error', msg: '仓库 ID 为必填项' });
      return;
    }
    const result = await run('pull', async () => {
      const api = await import('../api/client');
      return api.createModelPull(payload);
    }, (value) => `拉取任务已进入 ${value?.state || '队列'}`);
    if (result) setTab('jobs');
  };

  const retryRuntime = (artifact) => run(`retry:${artifact.artifact_id}`, async () => {
    const api = await import('../api/client');
    return api.retryModelRuntimeCheck(artifact.reference);
  }, (value) => value?.runnable ? '运行时检查通过' : '运行时检查未通过');

  const invalidateRuntime = (artifact) => run(`invalidate:${artifact.artifact_id}`, async () => {
    const api = await import('../api/client');
    return api.invalidateModelRuntimeCheck({
      artifactId: artifact.artifact_id,
      runtimeProfile: artifact.requirements?.runtime_profile,
      reason: 'user_requested_recheck',
    });
  }, () => '运行时检查已标记为需要复检');

  const cancelPull = (jobId) => run(`cancel:${jobId}`, async () => {
    const api = await import('../api/client');
    return api.cancelModelPull(jobId);
  }, () => '拉取任务已取消');

  const saveProxy = (event) => {
    event.preventDefault();
    if (!proxyDraft.trim()) {
      onToast?.({ type: 'error', msg: '代理地址不能为空；请使用清除操作关闭用户代理' });
      return;
    }
    run('proxy-save', async () => {
      const api = await import('../api/client');
      return api.saveModelProxy(proxyDraft);
    }, () => '模型下载代理已保存');
  };

  const clearProxy = () => run('proxy-clear', async () => {
    const api = await import('../api/client');
    return api.clearModelProxy();
  }, () => '用户模型下载代理已清除');

  return (
    <section className="sidebar-section model-fleet-workspace" data-testid="model-fleet-workspace">
      <div className="fleet-section-header">
        <h3>模型工件</h3>
        <button
          type="button"
          className="fleet-icon-button"
          onClick={() => refresh({ preserveProxy: true })}
          disabled={loading || Boolean(busy)}
          title="刷新模型工件状态"
          aria-label="刷新模型工件状态"
        >
          刷新
        </button>
      </div>

      <div className="fleet-tabs" role="tablist" aria-label="模型工件控制台">
        {TABS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            className={`fleet-tab${tab === id ? ' active' : ''}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {error && <div className="fleet-error" role="status">{error}</div>}

      {tab === 'artifacts' && (
        <div className="fleet-panel" role="tabpanel">
          <div className="fleet-summary" aria-label="工件汇总">
            <span>总计 {inventory.summary?.total || 0}</span>
            <span className="ready">可运行 {inventory.summary?.ready || 0}</span>
            <span className="attention">待处理 {(inventory.summary?.attention || 0) + (inventory.summary?.stale || 0)}</span>
            <span>未检查 {inventory.summary?.unchecked || 0}</span>
          </div>
          {loading ? (
            <div className="fleet-empty">正在读取本地工件…</div>
          ) : inventory.artifacts?.length ? (
            <div className="fleet-artifact-list">
              {inventory.artifacts.map((artifact) => {
                const runtimeCheck = artifact.runtime_check;
                const status = runtimeClass(runtimeCheck);
                return (
                  <div className="fleet-artifact-row" key={`${artifact.reference?.namespace}/${artifact.reference?.name}:${artifact.reference?.tag}`}>
                    <div className="fleet-artifact-main">
                      <div className="fleet-artifact-title">
                        <strong>{artifact.reference?.namespace}/{artifact.reference?.name}</strong>
                        <span className={`fleet-status ${status}`}>{modelRuntimeLabel(runtimeCheck)}</span>
                      </div>
                      <div className="fleet-artifact-meta">
                        <span>{artifact.reference?.tag}</span>
                        <span>{artifact.format}</span>
                        <span>{artifact.quantization || artifact.family}</span>
                        <span>{formatBytes(artifact.storage?.total_bytes)}</span>
                      </div>
                      <div className="fleet-artifact-detail">
                        {artifact.requirements?.runtime_profile || '未声明运行时'} · {formatTime(runtimeCheck?.checked_at)}
                      </div>
                    </div>
                    <div className="fleet-row-actions">
                      <button
                        type="button"
                        className="setting-btn secondary"
                        onClick={() => retryRuntime(artifact)}
                        disabled={Boolean(busy)}
                        title="重新执行本机运行时检查"
                      >
                        {busy === `retry:${artifact.artifact_id}` ? '检查中…' : '复检'}
                      </button>
                      {runtimeCheck && runtimeCheck.status !== 'stale' && (
                        <button
                          type="button"
                          className="setting-btn secondary"
                          onClick={() => invalidateRuntime(artifact)}
                          disabled={Boolean(busy)}
                          title="标记当前运行时检查失效"
                        >
                          {busy === `invalidate:${artifact.artifact_id}` ? '处理中…' : '标记失效'}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="fleet-empty">没有已注册工件</div>
          )}
        </div>
      )}

      {tab === 'acquire' && (
        <div className="fleet-panel fleet-acquire" role="tabpanel">
          <form className="fleet-form" onSubmit={submitImport}>
            <h4>本机导入</h4>
            <label htmlFor="fleet-import-path">路径</label>
            <input id="fleet-import-path" data-testid="fleet-import-path" value={importForm.sourcePath} onChange={(event) => setImportForm((value) => ({ ...value, sourcePath: event.target.value }))} />
            <div className="fleet-form-grid">
              <div><label htmlFor="fleet-import-namespace">命名空间</label><input id="fleet-import-namespace" value={importForm.namespace} onChange={(event) => setImportForm((value) => ({ ...value, namespace: event.target.value }))} /></div>
              <div><label htmlFor="fleet-import-name">名称</label><input id="fleet-import-name" data-testid="fleet-import-name" value={importForm.name} onChange={(event) => setImportForm((value) => ({ ...value, name: event.target.value }))} /></div>
              <div><label htmlFor="fleet-import-tag">标签</label><input id="fleet-import-tag" value={importForm.tag} onChange={(event) => setImportForm((value) => ({ ...value, tag: event.target.value }))} /></div>
            </div>
            <button type="submit" className="setting-btn primary" disabled={Boolean(busy)}>{busy === 'import' ? '导入中…' : '导入并检查'}</button>
          </form>

          <form className="fleet-form" onSubmit={submitPull}>
            <h4>远端拉取</h4>
            <div className="fleet-form-grid">
              <div>
                <label htmlFor="fleet-pull-provider">格式</label>
                <select id="fleet-pull-provider" value={pullForm.provider} onChange={(event) => setPullForm((value) => ({ ...value, provider: event.target.value }))}>
                  <option value="gguf_huggingface">GGUF</option>
                  <option value="huggingface">Safetensors</option>
                </select>
              </div>
              <div><label htmlFor="fleet-pull-revision">Revision</label><input id="fleet-pull-revision" value={pullForm.revision} onChange={(event) => setPullForm((value) => ({ ...value, revision: event.target.value }))} /></div>
            </div>
            <label htmlFor="fleet-pull-repo">仓库 ID</label>
            <input id="fleet-pull-repo" data-testid="fleet-pull-repo" placeholder="org/repository" value={pullForm.repoId} onChange={(event) => setPullForm((value) => ({ ...value, repoId: event.target.value }))} />
            <label htmlFor="fleet-pull-patterns">文件匹配</label>
            <input id="fleet-pull-patterns" value={pullForm.allowPatterns} onChange={(event) => setPullForm((value) => ({ ...value, allowPatterns: event.target.value }))} />
            <button type="submit" className="setting-btn primary" disabled={Boolean(busy)}>{busy === 'pull' ? '创建中…' : '开始拉取'}</button>
          </form>
        </div>
      )}

      {tab === 'jobs' && (
        <div className="fleet-panel" role="tabpanel">
          {jobs.length ? (
            <div className="fleet-job-list">
              {jobs.map((job) => {
                const percent = modelPullProgressPercent(job);
                const active = isModelPullActive(job);
                return (
                  <div className="fleet-job-row" key={job.job_id}>
                    <div className="fleet-job-title">
                      <strong>{job.source?.repo_id || job.job_id}</strong>
                      <span className={`fleet-status job-${job.state}`}>{job.state}</span>
                    </div>
                    <div className="fleet-progress-track" aria-label={`拉取进度 ${percent}%`}><span style={{ width: `${percent}%` }} /></div>
                    <div className="fleet-job-meta">
                      <span>{percent}% · {job.progress?.files_done || 0}/{job.progress?.files_total || 0} 文件</span>
                      <span>{job.progress?.current_file || job.error?.message || '等待处理'}</span>
                    </div>
                    <div className="fleet-row-actions">
                      {active && <button type="button" className="setting-btn danger-ghost" onClick={() => cancelPull(job.job_id)} disabled={Boolean(busy)}>{busy === `cancel:${job.job_id}` ? '取消中…' : '取消'}</button>}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : <div className="fleet-empty">没有拉取任务</div>}
        </div>
      )}

      {tab === 'network' && (
        <div className="fleet-panel" role="tabpanel">
          <div className="fleet-network-status">
            <span>当前连接</span>
            <strong>{network.proxy?.source === 'direct' ? '直连' : network.proxy?.source === 'user' ? '用户代理' : '环境代理'}</strong>
            <code>{network.proxy?.endpoint || 'direct'}</code>
          </div>
          <form className="fleet-form" onSubmit={saveProxy}>
            <label htmlFor="fleet-proxy-url">用户代理</label>
            <input id="fleet-proxy-url" data-testid="fleet-proxy-url" placeholder="http://127.0.0.1:7897" value={proxyDraft} onChange={(event) => setProxyDraft(event.target.value)} />
            <div className="fleet-row-actions">
              <button type="submit" className="setting-btn primary" disabled={Boolean(busy)}>{busy === 'proxy-save' ? '保存中…' : '保存代理'}</button>
              <button type="button" className="setting-btn secondary" onClick={clearProxy} disabled={Boolean(busy) || !network.user_proxy}>{busy === 'proxy-clear' ? '清除中…' : '清除代理'}</button>
            </div>
          </form>
        </div>
      )}
    </section>
  );
}
