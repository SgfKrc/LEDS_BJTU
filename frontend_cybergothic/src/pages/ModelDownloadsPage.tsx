/**
 * 模型下载工作区（P0A/P1B）：预设、仓库搜索和后台下载任务。
 *
 * 搜索结果只消费后端的 provider-neutral 投影；安装仍复用同一个
 * /api/models/downloads job 管线，不在前端拼接下载 URL。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import {
  CircleAlert,
  Download,
  ExternalLink,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  Search,
  X,
} from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { StatusBadge } from '../components/StatusBadge';
import { pushToast } from '../components/Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import { useReveal } from '../motion/useReveal';
import {
  fixturesEnabled,
  modelDownloadsFixture,
  modelPresetsFixture,
  modelSearchFixture,
} from '../data/fixtures';
import { useResource } from '../data/useResource';
import * as api from '../data/api';
import type {
  ModelDownloadJob,
  ModelDownloadsResponse,
  ModelPreset,
  ModelPresetsResponse,
  ModelSearchResponse,
  ModelSearchResult,
} from '../data/types';
import { PageBackdrop } from '../visual/PageBackdrop';

type SearchSource = 'hf' | 'ms' | 'all';

function formatBytes(value: unknown): string {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

function kindLabel(kind: string): string {
  return kind === 'gguf' ? 'GGUF' : kind === 'safetensors' ? 'SAFETENSORS' : kind.toUpperCase();
}

function providerLabel(provider: string): string {
  return provider === 'ms' ? 'MODELSCOPE' : 'HUGGING FACE';
}

function jobState(job: ModelDownloadJob): { label: string; tone: 'ok' | 'warn' | 'danger' | 'info' | 'idle' } {
  switch (job.status) {
    case 'ready': return { label: 'READY', tone: 'ok' };
    case 'failed': return { label: 'FAILED', tone: 'danger' };
    case 'cancelled': return { label: 'CANCELLED', tone: 'idle' };
    case 'queued': return { label: 'QUEUED', tone: 'info' };
    case 'downloading': return { label: 'DOWNLOADING', tone: 'info' };
    case 'verifying': return { label: 'VERIFYING', tone: 'info' };
    case 'registering': return { label: 'REGISTERING', tone: 'info' };
    default: return { label: String(job.status).toUpperCase(), tone: 'idle' };
  }
}

const ACTIVE = new Set(['queued', 'downloading', 'verifying', 'registering']);

export function ModelDownloadsPage() {
  const usingFixtures = fixturesEnabled();
  const presets = useResource<ModelPresetsResponse>(
    (signal) => usingFixtures ? Promise.resolve(modelPresetsFixture) : api.fetchModelPresets(signal),
    { key: usingFixtures ? 'fixture' : 'live' },
  );
  const jobs = useResource<ModelDownloadsResponse>(
    (signal) => usingFixtures ? Promise.resolve(modelDownloadsFixture) : api.fetchModelDownloads(signal),
    { pollMs: usingFixtures ? 0 : 2000, key: usingFixtures ? 'fixture' : 'live' },
  );
  const [busy, setBusy] = useState('');
  const [fixtureJobs, setFixtureJobs] = useState<ModelDownloadJob[]>(modelDownloadsFixture.jobs);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchSource, setSearchSource] = useState<SearchSource>('all');
  const [searchProxy, setSearchProxy] = useState('');
  const [searchData, setSearchData] = useState<ModelSearchResponse | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const searchAbort = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    presets.refresh();
    jobs.refresh();
  }, [jobs.refresh, presets.refresh]);
  useRegisterRefresh(refresh);

  useEffect(() => () => searchAbort.current?.abort(), []);

  const presetList: ModelPreset[] = presets.data?.presets ?? [];
  const jobList = usingFixtures ? fixtureJobs : (jobs.data?.jobs ?? []);
  const searchResults = searchData?.results ?? [];

  useReveal([presetList.length, jobList.length, searchResults.length]);

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const query = searchQuery.trim();
    if (!query || searching) return;
    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    setSearching(true);
    setSearchError('');
    try {
      const result = usingFixtures
        ? {
          ...modelSearchFixture,
          query,
          source: searchSource,
          results: searchSource === 'all'
            ? modelSearchFixture.results
            : modelSearchFixture.results.filter((item) => item.source === searchSource),
        }
        : await api.searchModelRepositories({
          q: query,
          source: searchSource,
          limit: 20,
          proxy: searchProxy.trim(),
        }, controller.signal);
      if (!controller.signal.aborted) setSearchData(result);
    } catch (err) {
      if (!controller.signal.aborted) setSearchError(api.describeError(err));
    } finally {
      if (!controller.signal.aborted) setSearching(false);
    }
  };

  const queueFixtureJob = (source: string, modelId: string, useModelscope: boolean) => {
    const job: ModelDownloadJob = {
      job_id: `fixture-download-${Date.now()}`,
      status: 'queued',
      progress: 0,
      source,
      target: `models/${modelId.split('/').pop() || modelId}`,
      model_id: modelId,
      engine: useModelscope ? 'auto' : 'llama_cpp',
      quant: '',
    };
    setFixtureJobs((previous) => [job, ...previous]);
  };

  const handleDownload = async (preset: ModelPreset) => {
    if (busy || !preset.installable) return;
    setBusy(`dl:${preset.id}`);
    try {
      if (usingFixtures) {
        queueFixtureJob(preset.hf_repo || preset.id, preset.default_model_id || preset.id, false);
      } else {
        await api.createModelDownload({ preset_id: preset.id, use_modelscope: false });
        jobs.refresh();
      }
      pushToast(`已排队下载：${preset.display}`, 'info');
    } catch (err) {
      pushToast(`发起下载失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleSearchInstall = async (result: ModelSearchResult) => {
    if (busy || result.private || result.gated) return;
    setBusy(`search:${result.source}:${result.id}`);
    try {
      const modelId = result.id.split('/').pop() || result.id;
      if (usingFixtures) {
        queueFixtureJob(result.id, modelId, result.source === 'ms');
      } else {
        await api.createModelDownload({
          source: result.id,
          model_id: modelId,
          use_modelscope: result.source === 'ms',
          proxy: searchProxy.trim(),
        });
        jobs.refresh();
      }
      pushToast(`已排队：${result.display_name || result.id}`, 'info');
    } catch (err) {
      pushToast(`安装失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleCancel = async (job: ModelDownloadJob) => {
    if (busy) return;
    setBusy(`cancel:${job.job_id}`);
    try {
      if (usingFixtures) {
        setFixtureJobs((previous) => previous.map((item) => item.job_id === job.job_id ? { ...item, status: 'cancelled' } : item));
      } else {
        const result = await api.cancelModelDownload(job.job_id);
        pushToast(result.cancelled ? '已取消下载任务' : '任务已在执行中，无法取消', 'info');
        jobs.refresh();
      }
      if (usingFixtures) pushToast('已取消下载任务', 'info');
    } catch (err) {
      pushToast(`取消失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const attemptSummary = useMemo(() => {
    if (!searchData) return '';
    if (!searchData.fallback_used) return `${providerLabel(searchData.provider)} 直连返回 ${searchData.results.length} 条`;
    const failed = (searchData.attempts ?? []).filter((attempt) => attempt.status === 'failed');
    return `已使用回退链：${failed.map((attempt) => `${providerLabel(attempt.provider)} ${attempt.transport || 'direct'}`).join(' → ')} → ${providerLabel(searchData.provider)}`;
  }, [searchData]);

  return (
    <div className="page downloads-page">
      <PageBackdrop scene="models" className="downloads-page__bg" />
      <PageHeader
        tag="MODEL DOWNLOADS"
        title="模型下载"
        description="从预设或公开仓库搜索模型，统一进入本机下载、校验与登记队列。"
        actions={<CommandButton variant="ghost" icon={RefreshCw} busy={presets.refreshing || jobs.refreshing} onClick={refresh}>刷新</CommandButton>}
      />

      <section className="downloads-panel downloads-search" aria-labelledby="model-search-title" data-reveal>
        <SectionHead title="搜索模型仓库" hint="HF 直连失败会自动尝试代理，再以 ModelScope 兜底；结果只展示公开元数据。" id="model-search-title" />
        <form className="downloads-search__form" onSubmit={handleSearch}>
          <label className="field downloads-search__query">
            <span className="field__label">搜索词</span>
            <span className="field__row"><Search size={16} aria-hidden="true" /><input className="field__input" aria-label="搜索模型仓库" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="例如 Qwen3、Gemma、GGUF" maxLength={128} /></span>
          </label>
          <fieldset className="downloads-search__source">
            <legend className="field__label">搜索源</legend>
            <div className="downloads-source-tabs" role="group" aria-label="搜索源">
              {(['all', 'hf', 'ms'] as SearchSource[]).map((source) => <button type="button" className={`downloads-source-tab${searchSource === source ? ' is-active' : ''}`} aria-pressed={searchSource === source} key={source} onClick={() => setSearchSource(source)}>{source === 'all' ? '全部' : source === 'hf' ? 'Hugging Face' : 'ModelScope'}</button>)}
            </div>
          </fieldset>
          <label className="field downloads-search__proxy">
            <span className="field__label">回退代理（可选）</span>
            <input className="field__input" aria-label="回退代理" value={searchProxy} onChange={(event) => setSearchProxy(event.target.value)} placeholder="留空自动使用 127.0.0.1:7897" />
          </label>
          <CommandButton type="submit" icon={Search} busy={searching} disabled={!searchQuery.trim()}>搜索</CommandButton>
        </form>
        {searchError ? <div className="inline-error" role="alert"><CircleAlert size={15} /> {searchError}</div> : null}
        {searchData ? <div className="downloads-search__meta" role="status"><span>{attemptSummary}</span><span className="table__cell--muted">共 {searchData.total ?? searchResults.length} 条</span></div> : null}
        {searchResults.length > 0 ? (
          <div className="ttable__wrap downloads-results" aria-label="模型仓库搜索结果"><div className="ttable__scroll"><table className="ttable">
            <thead><tr><th>模型</th><th>来源</th><th>任务 / 大小</th><th>热度</th><th>许可</th><th className="ttable__action-col">操作</th></tr></thead>
            <tbody>{searchResults.map((result) => {
              const blocked = Boolean(result.private || result.gated);
              return <tr key={`${result.source}:${result.id}`}>
                <td data-label="模型" className="downloads-model-cell"><strong>{result.display_name || result.id}</strong><code>{result.id}</code>{result.description ? <small>{result.description}</small> : null}</td>
                <td data-label="来源"><StatusBadge tone={result.source === 'ms' ? 'info' : 'ok'} label={providerLabel(result.source)} size="sm" /></td>
                <td data-label="任务 / 大小" className="table__cell--muted"><span>{result.tasks?.slice(0, 2).join(' · ') || '—'}</span><small>{formatBytes(result.size_bytes)}</small></td>
                <td data-label="热度" className="table__cell--muted"><span>{result.downloads != null ? `${result.downloads.toLocaleString()} 下载` : '—'}</span><small>{result.likes != null ? `${result.likes.toLocaleString()} 喜欢` : ''}</small></td>
                <td data-label="许可" className="table__cell--muted">{result.license || '未声明'}{result.gated ? ' · 需授权' : ''}</td>
                <td data-label="操作" className="ttable__action-col downloads-result-actions"><CommandButton size="sm" icon={Download} disabled={blocked} busy={busy === `search:${result.source}:${result.id}`} onClick={() => void handleSearchInstall(result)}>{blocked ? '需授权' : '安装'}</CommandButton>{result.url ? <a className="downloads-result-link" href={result.url} target="_blank" rel="noreferrer" title="打开模型仓库" aria-label={`打开 ${result.id}`}><ExternalLink size={14} /></a> : null}</td>
              </tr>;
            })}</tbody>
          </table></div></div>
        ) : searchData && !searching ? <EmptyState compact title="没有匹配结果" description="换一个关键词或切换搜索源。" /> : null}
      </section>

      <section className="downloads-panel downloads-section" data-reveal>
      <SectionHead title="预设列表" hint="点击下载即开始后台任务，进度可在下方追踪。" />
      {presets.state === 'loading' && <SkeletonRows rows={3} />}
      {presets.error && <EmptyState kind="error" title="无法加载预设" description={presets.error} errorKind={presets.errorKind ?? undefined} errorStatus={presets.errorStatus} action={<CommandButton variant="ghost" onClick={presets.refresh}>重试</CommandButton>} />}
      {!presets.error && presetList.length === 0 && presets.state === 'ready' && <EmptyState title="暂无可下载预设" description="后端尚未配置预设列表。" />}
      {presetList.length > 0 ? <div className="ttable__wrap downloads-presets" aria-label="模型预设列表"><div className="ttable__scroll"><table className="ttable">
        <thead><tr><th>模型</th><th>类型</th><th>引擎</th><th>资源</th><th>状态</th><th className="ttable__action-col">操作</th></tr></thead>
        <tbody>{presetList.map((preset) => {
          const blocked = !preset.installable;
          return <tr key={preset.id}>
            <td data-label="模型" className="downloads-model-cell"><strong>{preset.display}</strong><code>{preset.id}</code>{preset.description ? <small>{preset.description}</small> : null}</td>
            <td data-label="类型">{kindLabel(preset.kind)}</td><td data-label="引擎">{preset.default_engine || 'auto'}</td>
            <td data-label="资源" className="table__cell--muted"><span><HardDrive size={13} /> {preset.resource_gate?.min_disk_gb ?? '—'} GB 磁盘</span>{preset.resource_gate?.min_vram_gb ? <small>{preset.resource_gate.min_vram_gb} GB VRAM</small> : null}</td>
            <td data-label="状态"><StatusBadge tone={blocked ? 'danger' : 'ok'} label={blocked ? 'BLOCKED' : 'INSTALLABLE'} size="sm" />{blocked ? <small>{Object.values(preset.blocked_reasons || {}).join(' · ')}</small> : null}</td>
            <td data-label="操作" className="ttable__action-col"><CommandButton size="sm" icon={Download} disabled={blocked} busy={busy === `dl:${preset.id}`} onClick={() => void handleDownload(preset)}>下载</CommandButton></td>
          </tr>;
        })}</tbody>
      </table></div></div> : null}
      </section>

      <section className="downloads-panel downloads-section" data-reveal>
      <SectionHead title="下载任务" hint="queued → downloading → verifying → registering → ready" />
      {jobs.state === 'loading' && <SkeletonRows rows={2} />}
      {jobs.error && <EmptyState kind="error" title="无法加载下载任务" description={jobs.error} errorKind={jobs.errorKind ?? undefined} errorStatus={jobs.errorStatus} action={<CommandButton variant="ghost" onClick={jobs.refresh}>重试</CommandButton>} />}
      {!jobs.error && jobList.length === 0 && jobs.state === 'ready' && <EmptyState title="暂无下载任务" description="从上方预设或搜索结果发起一个下载。" />}
      {jobList.length > 0 ? <div className="ttable__wrap downloads-jobs" aria-label="下载任务列表"><div className="ttable__scroll"><table className="ttable">
        <thead><tr><th>目标</th><th>状态</th><th>进度</th><th>错误</th><th className="ttable__action-col">操作</th></tr></thead>
        <tbody>{jobList.map((job) => {
          const state = jobState(job); const active = ACTIVE.has(job.status); const percent = Math.round((job.progress || 0) * 100);
          return <tr key={job.job_id}>
            <td data-label="目标" className="downloads-model-cell"><strong>{job.model_id || job.preset_id || job.target}</strong><code>{job.engine} · {job.quant || job.status}</code></td>
            <td data-label="状态"><StatusBadge tone={state.tone} label={state.label} size="sm" /></td>
            <td data-label="进度"><div className="downloads-progress" aria-label={`进度 ${percent}%`}><span style={{ width: `${percent}%` }} /></div><small className="table__cell--muted">{job.downloaded_bytes != null && job.total_bytes ? `${formatBytes(job.downloaded_bytes)} / ${formatBytes(job.total_bytes)}` : `${percent}%`}</small></td>
            <td data-label="错误" className="table__cell--muted">{job.error_code ? <><span>{job.error_code}</span>{job.error ? <small>{job.error}</small> : null}</> : '—'}</td>
            <td data-label="操作" className="ttable__action-col downloads-job-actions">{job.status === 'queued' ? <CommandButton variant="ghost" size="sm" icon={X} disabled={Boolean(busy)} busy={busy === `cancel:${job.job_id}`} onClick={() => void handleCancel(job)}>取消</CommandButton> : null}{job.status === 'failed' ? <span className="table__cell--muted"><CircleAlert size={13} /> {job.error_code}</span> : null}{job.status === 'ready' ? <span className="table__cell--muted"><Download size={13} /> 已就绪</span> : null}{active ? <LoaderCircle size={14} className="spin" aria-label="任务进行中" /> : null}</td>
          </tr>;
        })}</tbody>
      </table></div></div> : null}
      </section>
    </div>
  );
}
