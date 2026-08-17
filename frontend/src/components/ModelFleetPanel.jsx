import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  buildLocalImportRequest,
  buildModelPullRequest,
  buildModelResolveRequest,
  buildModelSourceRequest,
  credentialIdFromRef,
  isModelPullActive,
  modelPullRequestKey,
  modelPullProgressPercent,
  modelRuntimeLabel,
} from '../modelFleetState';

const TABS = [
  ['artifacts', '工件'],
  ['acquire', '导入与拉取'],
  ['jobs', '任务'],
  ['security', '来源与凭据'],
  ['network', '网络'],
];

const EMPTY_INVENTORY = {
  node_id: 'local',
  artifacts: [],
  summary: { total: 0, ready: 0, stale: 0, attention: 0, unchecked: 0, total_bytes: 0 },
};

const EMPTY_SOURCE_FORM = {
  sourceId: '', name: '', provider: 'huggingface', endpoint: 'https://huggingface.co',
  credentialRef: '', priority: 100, enabled: true,
};

const EMPTY_CREDENTIAL_FORM = { credentialId: '', secret: '' };
const EMPTY_LICENSE_FORM = { repoId: '', licenseId: '' };

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

function preflightLabel(status) {
  return ({
    ready: '可以拉取',
    insufficient_storage: '磁盘不足',
    credential_required: '需要凭据',
    license_required: '需要许可',
  })[status] || status || '未解析';
}

export default function ModelFleetPanel({ onToast, onInventoryChange }) {
  const [tab, setTab] = useState('artifacts');
  const [inventory, setInventory] = useState(EMPTY_INVENTORY);
  const [jobs, setJobs] = useState([]);
  const [sources, setSources] = useState([]);
  const [credentials, setCredentials] = useState([]);
  const [acceptances, setAcceptances] = useState([]);
  const [network, setNetwork] = useState({ proxy: { source: 'direct', endpoint: null }, user_proxy: null });
  const [proxyDraft, setProxyDraft] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [importForm, setImportForm] = useState({
    sourcePath: '', namespace: 'user', name: '', tag: 'latest',
  });
  const [pullForm, setPullForm] = useState({
    sourceId: '', provider: 'gguf_huggingface', repoId: '', revision: 'main', allowPatterns: '*.gguf',
  });
  const [preflight, setPreflight] = useState(null);
  const [sourceForm, setSourceForm] = useState(EMPTY_SOURCE_FORM);
  const [credentialForm, setCredentialForm] = useState(EMPTY_CREDENTIAL_FORM);
  const [licenseForm, setLicenseForm] = useState(EMPTY_LICENSE_FORM);

  const refresh = useCallback(async ({ preserveProxy = false, quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    const api = await import('../api/client');
    const results = await Promise.allSettled([
      api.fetchModelArtifacts(),
      api.fetchModelPullJobs(),
      api.fetchModelNetwork(),
      api.fetchModelSources(),
      api.fetchModelCredentials(),
      api.fetchModelLicenseAcceptances(),
    ]);
    const failures = [];
    if (results[0].status === 'fulfilled') {
      const nextInventory = results[0].value;
      setInventory(nextInventory);
      onInventoryChange?.(nextInventory.artifacts || []);
    }
    else failures.push(results[0].reason);
    if (results[1].status === 'fulfilled') setJobs(results[1].value.jobs || []);
    else failures.push(results[1].reason);
    if (results[2].status === 'fulfilled') {
      setNetwork(results[2].value);
      if (!preserveProxy) setProxyDraft(results[2].value.user_proxy?.url || '');
    } else {
      failures.push(results[2].reason);
    }
    if (results[3].status === 'fulfilled') {
      const nextSources = results[3].value.sources || [];
      setSources(nextSources);
      setPullForm((value) => {
        const currentAvailable = nextSources.some(
          (source) => source.source_id === value.sourceId && source.enabled && source.provider === 'huggingface',
        );
        if (currentAvailable) return value;
        const preferred = nextSources.find((source) => source.enabled && source.provider === 'huggingface');
        return { ...value, sourceId: preferred?.source_id || '' };
      });
    } else failures.push(results[3].reason);
    if (results[4].status === 'fulfilled') setCredentials(results[4].value.credentials || []);
    else failures.push(results[4].reason);
    if (results[5].status === 'fulfilled') setAcceptances(results[5].value.acceptances || []);
    else failures.push(results[5].reason);
    setError(failures.length === results.length ? (failures[0]?.message || '模型控制面不可用') : '');
    if (!quiet) setLoading(false);
  }, [onInventoryChange]);

  useEffect(() => { refresh(); }, [refresh]);

  const hasActiveJobs = useMemo(() => jobs.some(isModelPullActive), [jobs]);
  const preflightReady = useMemo(() => (
    preflight?.request_key === modelPullRequestKey(pullForm) && preflight?.status === 'ready'
  ), [preflight, pullForm]);
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
    if (!payload.source_id || !payload.source.repo_id) {
      onToast?.({ type: 'error', msg: '来源和仓库 ID 为必填项' });
      return;
    }
    if (!preflightReady) {
      onToast?.({ type: 'error', msg: '当前来源与文件清单尚未通过解析检查' });
      return;
    }
    const result = await run('pull', async () => {
      const api = await import('../api/client');
      return api.createModelPull(payload);
    }, (value) => `拉取任务已进入 ${value?.state || '队列'}`);
    if (result) setTab('jobs');
  };

  const updatePullForm = (patch) => {
    setPullForm((value) => ({ ...value, ...patch }));
    setPreflight(null);
  };

  const resolvePull = async () => {
    const payload = buildModelResolveRequest(pullForm);
    if (!payload.source_id || !payload.repo_id) {
      onToast?.({ type: 'error', msg: '来源和仓库 ID 为必填项' });
      return;
    }
    const requestKey = modelPullRequestKey(pullForm);
    const result = await run('resolve', async () => {
      const api = await import('../api/client');
      return api.resolveModelPull(payload);
    }, (value) => `解析结果：${preflightLabel(value?.status)}`, { warning: true });
    if (result) setPreflight({ ...result, request_key: requestKey });
  };

  const submitSource = async (event) => {
    event.preventDefault();
    const request = buildModelSourceRequest(sourceForm);
    if (!request.source_id || !request.payload.name || !request.payload.endpoint) {
      onToast?.({ type: 'error', msg: '来源 ID、名称和地址为必填项' });
      return;
    }
    const result = await run('source-save', async () => {
      const api = await import('../api/client');
      return api.saveModelSource(request.source_id, request.payload);
    }, () => '模型来源已保存');
    if (result) {
      setSourceForm({ ...EMPTY_SOURCE_FORM });
      setPreflight(null);
    }
  };

  const editSource = (source) => setSourceForm({
    sourceId: source.source_id,
    name: source.name,
    provider: source.provider,
    endpoint: source.endpoint,
    credentialRef: source.credential_ref || '',
    priority: source.priority,
    enabled: source.enabled,
  });

  const deleteSource = (source) => run(`source-delete:${source.source_id}`, async () => {
    const api = await import('../api/client');
    return api.deleteModelSource(source.source_id);
  }, () => source.builtin ? '内置来源已禁用' : '模型来源已删除').then((result) => {
    if (result) {
      setSourceForm({ ...EMPTY_SOURCE_FORM });
      setPreflight(null);
    }
  });

  const resetSources = () => run('source-reset', async () => {
    const api = await import('../api/client');
    return api.resetModelSources();
  }, () => '模型来源已恢复默认').then((result) => {
    if (result) {
      setSourceForm({ ...EMPTY_SOURCE_FORM });
      setPreflight(null);
    }
  });

  const submitCredential = async (event) => {
    event.preventDefault();
    const credentialId = credentialForm.credentialId.trim();
    if (!credentialId || !credentialForm.secret) {
      onToast?.({ type: 'error', msg: '凭据 ID 和密钥为必填项' });
      return;
    }
    const result = await run('credential-save', async () => {
      const api = await import('../api/client');
      return api.saveModelCredential(credentialId, credentialForm.secret);
    }, () => '凭据已写入操作系统安全存储');
    if (result) setCredentialForm({ credentialId, secret: '' });
  };

  const deleteCredential = (credential) => {
    const credentialId = credentialIdFromRef(credential.credential_ref);
    if (!credentialId) return;
    run(`credential-delete:${credentialId}`, async () => {
      const api = await import('../api/client');
      return api.deleteModelCredential(credentialId);
    }, () => '凭据已删除');
  };

  const submitLicense = async (event) => {
    event.preventDefault();
    const payload = {
      repo_id: licenseForm.repoId.trim(), license_id: licenseForm.licenseId.trim(),
    };
    if (!payload.repo_id || !payload.license_id) {
      onToast?.({ type: 'error', msg: '仓库 ID 和许可 ID 为必填项' });
      return;
    }
    const result = await run('license-accept', async () => {
      const api = await import('../api/client');
      return api.acceptModelLicense(payload);
    }, () => '模型许可已接受');
    if (result) setLicenseForm({ ...EMPTY_LICENSE_FORM });
  };

  const revokeLicense = (acceptance) => run(
    `license-revoke:${acceptance.repo_id}:${acceptance.license_id}`,
    async () => {
      const api = await import('../api/client');
      return api.revokeModelLicense({
        repo_id: acceptance.repo_id, license_id: acceptance.license_id,
      });
    },
    () => '模型许可接受记录已撤销',
  );

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
                <label htmlFor="fleet-pull-source">来源</label>
                <select id="fleet-pull-source" data-testid="fleet-pull-source" value={pullForm.sourceId} onChange={(event) => updatePullForm({ sourceId: event.target.value })}>
                  <option value="">选择来源</option>
                  {sources.filter((source) => source.enabled && source.provider === 'huggingface').map((source) => (
                    <option key={source.source_id} value={source.source_id}>{source.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="fleet-pull-provider">格式</label>
                <select id="fleet-pull-provider" value={pullForm.provider} onChange={(event) => updatePullForm({ provider: event.target.value })}>
                  <option value="gguf_huggingface">GGUF</option>
                  <option value="huggingface">Safetensors</option>
                </select>
              </div>
              <div><label htmlFor="fleet-pull-revision">Revision</label><input id="fleet-pull-revision" value={pullForm.revision} onChange={(event) => updatePullForm({ revision: event.target.value })} /></div>
            </div>
            <label htmlFor="fleet-pull-repo">仓库 ID</label>
            <input id="fleet-pull-repo" data-testid="fleet-pull-repo" placeholder="org/repository" value={pullForm.repoId} onChange={(event) => updatePullForm({ repoId: event.target.value })} />
            <label htmlFor="fleet-pull-patterns">文件匹配</label>
            <input id="fleet-pull-patterns" value={pullForm.allowPatterns} onChange={(event) => updatePullForm({ allowPatterns: event.target.value })} />
            {preflight && (
              <div className={`fleet-preflight ${preflight.status}`} data-testid="fleet-preflight">
                <div><strong>{preflightLabel(preflight.status)}</strong><span>{preflight.source?.source_id}</span></div>
                <div><span>{preflight.files?.length || 0} 文件</span><span>{formatBytes(preflight.total_bytes)}</span><span>{preflight.resolved_revision?.slice(0, 12)}</span></div>
              </div>
            )}
            <div className="fleet-row-actions fleet-form-actions">
              <button type="button" className="setting-btn secondary" onClick={resolvePull} disabled={Boolean(busy)}>{busy === 'resolve' ? '解析中…' : '解析检查'}</button>
              <button type="submit" className="setting-btn primary" disabled={Boolean(busy) || !preflightReady}>{busy === 'pull' ? '创建中…' : '确认拉取'}</button>
            </div>
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

      {tab === 'security' && (
        <div className="fleet-panel fleet-security" role="tabpanel">
          <section className="fleet-subsection">
            <div className="fleet-subsection-header">
              <h4>模型来源</h4>
              <button type="button" className="setting-btn secondary" onClick={resetSources} disabled={Boolean(busy)}>{busy === 'source-reset' ? '恢复中…' : '恢复默认'}</button>
            </div>
            <div className="fleet-source-list">
              {sources.map((source) => (
                <div className="fleet-source-row" key={source.source_id}>
                  <div className="fleet-source-main">
                    <div className="fleet-artifact-title">
                      <strong>{source.name}</strong>
                      <span className={`fleet-status ${source.enabled ? 'ready' : 'unchecked'}`}>{source.enabled ? '启用' : '停用'}</span>
                    </div>
                    <div className="fleet-artifact-meta">
                      <span>{source.source_id}</span><span>{source.provider}</span><span>优先级 {source.priority}</span>
                    </div>
                    <div className="fleet-artifact-detail">{source.endpoint} · {source.credential_ref || '无凭据'}</div>
                  </div>
                  <div className="fleet-row-actions">
                    <button type="button" className="setting-btn secondary" onClick={() => editSource(source)} disabled={Boolean(busy)}>编辑</button>
                    <button type="button" className="setting-btn danger-ghost" onClick={() => deleteSource(source)} disabled={Boolean(busy)}>{source.builtin ? '禁用' : '删除'}</button>
                  </div>
                </div>
              ))}
            </div>
            <form className="fleet-form" onSubmit={submitSource}>
              <div className="fleet-form-grid">
                <div><label htmlFor="fleet-source-id">来源 ID</label><input id="fleet-source-id" data-testid="fleet-source-id" value={sourceForm.sourceId} onChange={(event) => setSourceForm((value) => ({ ...value, sourceId: event.target.value }))} /></div>
                <div><label htmlFor="fleet-source-name">名称</label><input id="fleet-source-name" data-testid="fleet-source-name" value={sourceForm.name} onChange={(event) => setSourceForm((value) => ({ ...value, name: event.target.value }))} /></div>
                <div>
                  <label htmlFor="fleet-source-provider">Provider</label>
                  <select id="fleet-source-provider" value={sourceForm.provider} onChange={(event) => setSourceForm((value) => ({ ...value, provider: event.target.value }))}>
                    <option value="huggingface">Hugging Face</option><option value="modelscope">ModelScope</option>
                  </select>
                </div>
              </div>
              <label htmlFor="fleet-source-endpoint">Endpoint</label>
              <input id="fleet-source-endpoint" data-testid="fleet-source-endpoint" value={sourceForm.endpoint} onChange={(event) => setSourceForm((value) => ({ ...value, endpoint: event.target.value }))} />
              <div className="fleet-form-grid fleet-form-grid-compact">
                <div>
                  <label htmlFor="fleet-source-credential">凭据</label>
                  <select id="fleet-source-credential" value={sourceForm.credentialRef} onChange={(event) => setSourceForm((value) => ({ ...value, credentialRef: event.target.value }))}>
                    <option value="">无凭据</option>
                    {sourceForm.credentialRef && !credentials.some((item) => item.credential_ref === sourceForm.credentialRef) && <option value={sourceForm.credentialRef}>{sourceForm.credentialRef}（不可用）</option>}
                    {credentials.map((credential) => <option key={credential.credential_ref} value={credential.credential_ref}>{credential.credential_ref}</option>)}
                  </select>
                </div>
                <div><label htmlFor="fleet-source-priority">优先级</label><input id="fleet-source-priority" type="number" min="0" step="1" value={sourceForm.priority} onChange={(event) => setSourceForm((value) => ({ ...value, priority: event.target.value }))} /></div>
                <label className="fleet-checkbox" htmlFor="fleet-source-enabled"><input id="fleet-source-enabled" type="checkbox" checked={sourceForm.enabled} onChange={(event) => setSourceForm((value) => ({ ...value, enabled: event.target.checked }))} />启用</label>
              </div>
              <div className="fleet-row-actions fleet-form-actions">
                <button type="submit" className="setting-btn primary" disabled={Boolean(busy)}>{busy === 'source-save' ? '保存中…' : '保存来源'}</button>
                <button type="button" className="setting-btn secondary" onClick={() => setSourceForm({ ...EMPTY_SOURCE_FORM })} disabled={Boolean(busy)}>新建</button>
              </div>
            </form>
          </section>

          <section className="fleet-subsection">
            <h4>安全凭据</h4>
            {credentials.length ? (
              <div className="fleet-credential-list">
                {credentials.map((credential) => (
                  <div className="fleet-credential-row" key={credential.credential_ref}>
                    <div><strong>{credential.credential_ref}</strong><span>{credential.protection} · {formatTime(credential.updated_at)}</span></div>
                    <button type="button" className="setting-btn danger-ghost" onClick={() => deleteCredential(credential)} disabled={Boolean(busy)}>删除</button>
                  </div>
                ))}
              </div>
            ) : <div className="fleet-empty">没有已保存凭据</div>}
            <form className="fleet-form" onSubmit={submitCredential}>
              <div className="fleet-form-grid fleet-form-grid-compact">
                <div><label htmlFor="fleet-credential-id">凭据 ID</label><input id="fleet-credential-id" data-testid="fleet-credential-id" autoComplete="off" value={credentialForm.credentialId} onChange={(event) => setCredentialForm((value) => ({ ...value, credentialId: event.target.value }))} /></div>
                <div className="fleet-grid-span-2"><label htmlFor="fleet-credential-secret">密钥</label><input id="fleet-credential-secret" data-testid="fleet-credential-secret" type="password" autoComplete="new-password" value={credentialForm.secret} onChange={(event) => setCredentialForm((value) => ({ ...value, secret: event.target.value }))} /></div>
              </div>
              <button type="submit" className="setting-btn primary" disabled={Boolean(busy)}>{busy === 'credential-save' ? '写入中…' : '保存凭据'}</button>
            </form>
          </section>

          <section className="fleet-subsection">
            <h4>许可接受</h4>
            {acceptances.length ? (
              <div className="fleet-credential-list">
                {acceptances.map((acceptance) => (
                  <div className="fleet-credential-row" key={`${acceptance.repo_id}:${acceptance.license_id}`}>
                    <div><strong>{acceptance.repo_id}</strong><span>{acceptance.license_id} · {formatTime(acceptance.accepted_at)}</span></div>
                    <button type="button" className="setting-btn danger-ghost" onClick={() => revokeLicense(acceptance)} disabled={Boolean(busy)}>撤销</button>
                  </div>
                ))}
              </div>
            ) : <div className="fleet-empty">没有许可接受记录</div>}
            <form className="fleet-form" onSubmit={submitLicense}>
              <div className="fleet-form-grid fleet-form-grid-compact">
                <div className="fleet-grid-span-2"><label htmlFor="fleet-license-repo">仓库 ID</label><input id="fleet-license-repo" data-testid="fleet-license-repo" placeholder="org/repository" value={licenseForm.repoId} onChange={(event) => setLicenseForm((value) => ({ ...value, repoId: event.target.value }))} /></div>
                <div><label htmlFor="fleet-license-id">许可 ID</label><input id="fleet-license-id" data-testid="fleet-license-id" value={licenseForm.licenseId} onChange={(event) => setLicenseForm((value) => ({ ...value, licenseId: event.target.value }))} /></div>
              </div>
              <button type="submit" className="setting-btn primary" disabled={Boolean(busy)}>{busy === 'license-accept' ? '记录中…' : '接受许可'}</button>
            </form>
          </section>
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
