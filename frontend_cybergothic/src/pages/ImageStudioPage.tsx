import { useCallback, useEffect, useRef, useState } from 'react';
import { Cpu, Download, Image as ImageIcon, LoaderCircle, PauseCircle, RefreshCw, Sparkles } from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { EmptyState } from '../components/EmptyState';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { StatusBadge } from '../components/StatusBadge';
import { pushToast } from '../components/Toast';
import { routeHref } from '../app/routes';
import { useRegisterRefresh } from '../app/refreshBus';
import { fixturesEnabled } from '../data/fixtures';
import * as api from '../data/api';
import {
  useDiffusionArtifacts,
  useDiffusionAssets,
  useDiffusionCapabilities,
} from '../data/hooks';
import type { DiffusionJob, DiffusionPreset } from '../data/types';
import { FoundryCanvas } from '../visual/FoundryCanvas';

interface ImageHistoryItem {
  jobId: string;
  state: string;
  prompt: string;
  negativePrompt: string;
  createdAt: number;
  blobId?: string;
  error?: string;
  parameters?: DiffusionJob['parameters'];
}

const HISTORY_KEY = 'qlh_cg_diffusion_history';
const TERMINAL_STATES = new Set(['completed', 'failed', 'cancelled']);

function readHistory(): ImageHistoryItem[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const data = JSON.parse(raw) as unknown;
    return Array.isArray(data) ? data.slice(0, 40) as ImageHistoryItem[] : [];
  } catch {
    return [];
  }
}

function jobLabel(state: string): string {
  return {
    queued: '排队中',
    running: '生成中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }[state] || state || '未知';
}

function jobTone(state: string): 'ok' | 'warn' | 'danger' | 'info' | 'idle' {
  if (state === 'completed') return 'ok';
  if (state === 'failed') return 'danger';
  if (state === 'cancelled') return 'idle';
  if (state === 'running' || state === 'queued') return 'info';
  return 'warn';
}

function formatTime(timestamp: number): string {
  if (!timestamp) return '刚刚';
  return new Date(timestamp * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function ImageStudioPage() {
  const capabilities = useDiffusionCapabilities();
  const artifacts = useDiffusionArtifacts();
  const assets = useDiffusionAssets();
  const usingFixtures = fixturesEnabled();
  const [history, setHistory] = useState<ImageHistoryItem[]>(readHistory);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [job, setJob] = useState<DiffusionJob | null>(null);
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted');
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [selectedArtifactId, setSelectedArtifactId] = useState('');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [steps, setSteps] = useState(28);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState(-1);
  const [busy, setBusy] = useState('');
  const mountedRef = useRef(true);

  const presets = capabilities.data?.presets ?? [];
  const artifactList = artifacts.data?.artifacts ?? [];
  const assetList = assets.data?.assets ?? [];
  const activeJob = job && !TERMINAL_STATES.has(job.state) ? job : null;
  const selectedHistory = history.find((item) => item.jobId === selectedJobId) ?? null;
  const loadedArtifactId = capabilities.data?.loaded_artifact?.artifact_id ?? '';
  const blobId = job?.blob?.blob_id || selectedHistory?.blobId || '';
  const blobUrl = blobId ? `/api/diffusion/blobs/${encodeURIComponent(blobId)}` : '';
  const canGenerate = Boolean(
    capabilities.data?.loaded && prompt.trim() && !activeJob && !busy,
  );

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 40)));
    } catch {
      // 本地存储不可用时，当前页仍可继续使用。
    }
  }, [history]);

  useEffect(() => {
    const active = capabilities.data?.active_job;
    if (active && !TERMINAL_STATES.has(active.state)) {
      setJob(active);
      setSelectedJobId(active.job_id);
    }
  }, [capabilities.data?.active_job]);

  useEffect(() => {
    if (selectedArtifactId || loadedArtifactId) return;
    const first = artifactList.find((item) => item.artifact?.artifact_kind === 'sd15_pipeline') ?? artifactList[0];
    if (first) setSelectedArtifactId(first.artifact_id);
  }, [artifactList, loadedArtifactId, selectedArtifactId]);

  const refresh = useCallback(() => {
    capabilities.refresh();
    artifacts.refresh();
    assets.refresh();
  }, [capabilities.refresh, artifacts.refresh, assets.refresh]);
  useRegisterRefresh(refresh);

  const updateHistory = useCallback((nextJob: DiffusionJob, requestPrompt: string, requestNegative: string) => {
    setHistory((previous) => {
      const next: ImageHistoryItem = {
        jobId: nextJob.job_id,
        state: nextJob.state,
        prompt: requestPrompt,
        negativePrompt: requestNegative,
        createdAt: nextJob.created_at ?? Date.now() / 1000,
        ...(nextJob.blob?.blob_id ? { blobId: nextJob.blob.blob_id } : {}),
        ...(nextJob.error ? { error: nextJob.error } : {}),
        ...(nextJob.parameters ? { parameters: nextJob.parameters } : {}),
      };
      const withoutCurrent = previous.filter((item) => item.jobId !== next.jobId);
      return [next, ...withoutCurrent].slice(0, 40);
    });
    setSelectedJobId(nextJob.job_id);
  }, []);

  const pollJob = useCallback(async (jobId: string, requestPrompt: string, requestNegative: string) => {
    while (mountedRef.current) {
      const snapshot = await api.fetchDiffusionJob(jobId);
      if (!mountedRef.current) return;
      setJob(snapshot);
      updateHistory(snapshot, requestPrompt, requestNegative);
      if (TERMINAL_STATES.has(snapshot.state)) return;
      await new Promise((resolve) => window.setTimeout(resolve, 900));
    }
  }, [updateHistory]);

  const applyPreset = (preset: DiffusionPreset) => {
    setSelectedPresetId(preset.preset_id);
    setPrompt(preset.prompt || '');
    setNegativePrompt(preset.negative_prompt || '');
    if (preset.width) setWidth(preset.width);
    if (preset.height) setHeight(preset.height);
    if (preset.steps) setSteps(preset.steps);
    if (preset.guidance_scale) setGuidanceScale(preset.guidance_scale);
  };

  const handleLoad = async () => {
    if (!selectedArtifactId || busy) return;
    setBusy('load');
    try {
      if (usingFixtures) {
        pushToast('演示数据模式下已模拟加载图像模型。', 'info');
      } else {
        await api.loadDiffusionArtifact(selectedArtifactId);
        await capabilities.refresh();
        pushToast('图像模型已加载。', 'ok');
      }
    } catch (err) {
      pushToast(`加载图像模型失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleUnload = async () => {
    if (busy || !loadedArtifactId) return;
    setBusy('unload');
    try {
      if (usingFixtures) {
        pushToast('演示数据模式下不执行真实卸载。', 'info');
      } else {
        await api.unloadDiffusionArtifact();
        await capabilities.refresh();
        pushToast('图像模型已卸载。', 'ok');
      }
    } catch (err) {
      pushToast(`卸载图像模型失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleGenerate = async () => {
    const text = prompt.trim();
    if (!text || activeJob || busy) return;
    setBusy('generate');
    const request = {
      ...(selectedPresetId ? { preset_id: selectedPresetId } : {}),
      prompt: text,
      negative_prompt: negativePrompt.trim(),
      seed: seed < 0 ? undefined : seed,
      width,
      height,
      steps,
      guidance_scale: guidanceScale,
    };
    try {
      if (usingFixtures) {
        const fixtureJob: DiffusionJob = {
          job_id: `fixture-sdjob-${Date.now().toString(36)}`,
          state: 'completed',
          created_at: Date.now() / 1000,
          parameters: request,
        };
        setJob(fixtureJob);
        updateHistory(fixtureJob, text, negativePrompt.trim());
        pushToast('演示数据已加入生图列表。', 'info');
      } else {
        const submitted = await api.generateDiffusionImage(request);
        setJob(submitted);
        updateHistory(submitted, text, negativePrompt.trim());
        void pollJob(submitted.job_id, text, negativePrompt.trim()).catch((err) => {
          if (mountedRef.current) pushToast(`任务状态读取失败：${api.describeError(err)}`, 'danger');
        });
      }
    } catch (err) {
      pushToast(`提交生图任务失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleCancel = async () => {
    if (!activeJob || busy) return;
    setBusy('cancel');
    try {
      if (usingFixtures) {
        const cancelled = { ...activeJob, state: 'cancelled' };
        setJob(cancelled);
        updateHistory(cancelled, prompt, negativePrompt);
      } else {
        const result = await api.cancelDiffusionJob(activeJob.job_id);
        if (result.job) setJob(result.job);
      }
      pushToast('生图任务已取消。', 'info');
    } catch (err) {
      pushToast(`取消任务失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const selectHistory = async (item: ImageHistoryItem) => {
    setSelectedJobId(item.jobId);
    setPrompt(item.prompt);
    setNegativePrompt(item.negativePrompt);
    if (item.parameters?.width) setWidth(item.parameters.width);
    if (item.parameters?.height) setHeight(item.parameters.height);
    if (usingFixtures || item.state === 'completed' || item.state === 'failed' || item.state === 'cancelled') {
      setJob({
        job_id: item.jobId,
        state: item.state,
        created_at: item.createdAt,
        ...(item.blobId ? { blob: { blob_id: item.blobId } } : {}),
        ...(item.error ? { error: item.error } : {}),
        ...(item.parameters ? { parameters: item.parameters } : {}),
      });
      return;
    }
    try {
      const latest = await api.fetchDiffusionJob(item.jobId);
      setJob(latest);
    } catch (err) {
      pushToast(`读取任务失败：${api.describeError(err)}`, 'danger');
    }
  };

  const capabilityState = capabilities.state === 'error'
    ? 'danger'
    : capabilities.data?.loaded
      ? 'ok'
      : 'warn';

  return (
    <div className="image-studio">
      <FoundryCanvas className="image-studio__bg" />
      <div className="image-studio__content">
        <PageHeader
          tag="IMAGE STUDIO"
          title="生图工坊"
          description="管理本地 SD 资产、提交生成任务，并在同一列表追踪结果。"
          actions={
            <>
              <CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={capabilities.refreshing} onClick={refresh}>
                刷新
              </CommandButton>
              <CommandButton variant="ghost" size="sm" icon={Cpu} href={routeHref('models')}>
                Models &amp; assets
              </CommandButton>
              <CommandButton variant="ghost" size="sm" href={routeHref('overview')}>
                返回概览
              </CommandButton>
            </>
          }
        />

        <div className="image-studio__layout">
          <aside className="image-studio__rail" aria-label="生图任务和资产">
            <section className="studio-panel studio-panel--rail">
              <SectionHead title="生图列表" hint={`${history.length} 个本地任务`} />
              {history.length === 0 ? (
                <EmptyState kind="empty" title="暂无生图任务" description="提交第一张图片后，任务会保留在这里。" compact />
              ) : (
                <ul className="image-jobs">
                  {history.map((item) => (
                    <li key={item.jobId} data-active={item.jobId === selectedJobId ? 'true' : undefined}>
                      <button type="button" className="image-job" onClick={() => void selectHistory(item)}>
                        <span className="image-job__head">
                          <StatusBadge label={jobLabel(item.state)} tone={jobTone(item.state)} size="sm" pulse={item.state === 'running'} />
                          <span className="image-job__time mono-label">{formatTime(item.createdAt)}</span>
                        </span>
                        <span className="image-job__prompt">{item.prompt || '未命名提示词'}</span>
                        <span className="image-job__id cell-mono">{item.jobId}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="studio-panel studio-panel--rail">
              <SectionHead title="资产目录" hint={`${assetList.length} 个资产`} />
              <ul className="image-assets">
                {assetList.map((asset) => (
                  <li key={asset.asset_id} data-installed={asset.installed ? 'true' : 'false'}>
                    <span>{asset.name || asset.asset_id}</span>
                    <StatusBadge label={asset.installed ? '已安装' : '待获取'} tone={asset.installed ? 'ok' : 'idle'} size="sm" />
                  </li>
                ))}
              </ul>
            </section>
          </aside>

          <main className="image-studio__main">
            <section className="studio-panel studio-panel--canvas">
              <div className="studio-canvas__head">
                <div>
                  <p className="mono-label">OUTPUT PLATE</p>
                  <h2>结果预览</h2>
                </div>
                <StatusBadge label={capabilities.data?.loaded ? '模型已就绪' : '模型未加载'} tone={capabilityState} pulse={Boolean(activeJob)} />
              </div>
              {blobUrl ? (
                <figure className="studio-preview">
                  <img src={blobUrl} alt="生图任务结果" />
                  <figcaption>
                    <span>{job?.parameters?.width || width} × {job?.parameters?.height || height}</span>
                    <a href={blobUrl} download={`qlh-image-${selectedJobId || 'result'}.png`}>
                      <Download size={14} aria-hidden="true" /> 下载 PNG
                    </a>
                  </figcaption>
                </figure>
              ) : (
                <div className="studio-preview studio-preview--empty">
                  <ImageIcon size={36} strokeWidth={1.2} aria-hidden="true" />
                  <strong>{activeJob ? `任务${jobLabel(activeJob.state)}` : '等待生成结果'}</strong>
                  <span>{capabilities.data?.last_error || '结果会显示在这块图像版片上。'}</span>
                </div>
              )}
              {activeJob ? (
                <div className="studio-progress" aria-live="polite">
                  <div className="studio-progress__head">
                    <span>{jobLabel(activeJob.state)}</span>
                    <span>{activeJob.progress?.step ?? 0} / {activeJob.progress?.total ?? steps}</span>
                  </div>
                  <div className="studio-progress__track">
                    <span style={{ width: `${Math.max(4, Math.min(100, activeJob.progress?.percent ?? 8))}%` }} />
                  </div>
                  <CommandButton variant="danger" size="sm" icon={PauseCircle} busy={busy === 'cancel'} onClick={() => void handleCancel()}>
                    取消任务
                  </CommandButton>
                </div>
              ) : null}
            </section>

            <section className="studio-panel studio-panel--form">
              <div className="studio-form__head">
                <div>
                  <p className="mono-label">TEXT TO IMAGE</p>
                  <h2>生成参数</h2>
                </div>
                <div className="studio-form__modes" role="group" aria-label="生成模式">
                  <button type="button" className="is-active" aria-pressed="true">文生图</button>
                  <button type="button" disabled aria-pressed="false">图生图</button>
                  <button type="button" disabled aria-pressed="false">局部重绘</button>
                </div>
              </div>
              <label className="studio-field studio-field--wide">
                <span>提示词</span>
                <textarea value={prompt} rows={4} maxLength={4000} placeholder="描述要生成的画面…" onChange={(event) => setPrompt(event.target.value)} />
              </label>
              <label className="studio-field studio-field--wide">
                <span>负面提示词</span>
                <input value={negativePrompt} maxLength={4000} onChange={(event) => setNegativePrompt(event.target.value)} />
              </label>
              <div className="studio-form__grid">
                <label className="studio-field"><span>预设</span><select value={selectedPresetId} onChange={(event) => {
                  setSelectedPresetId(event.target.value);
                  const preset = presets.find((item) => item.preset_id === event.target.value);
                  if (preset) applyPreset(preset);
                }}><option value="">自定义</option>{presets.map((preset) => <option key={preset.preset_id} value={preset.preset_id}>{preset.preset_id}</option>)}</select></label>
                <label className="studio-field"><span>模型资产</span><select value={selectedArtifactId || loadedArtifactId} onChange={(event) => setSelectedArtifactId(event.target.value)}><option value="">未选择</option>{artifactList.map((artifact) => <option key={artifact.artifact_id} value={artifact.artifact_id}>{artifact.name}</option>)}</select></label>
                <label className="studio-field"><span>宽度</span><input type="number" min={256} max={1024} step={64} value={width} onChange={(event) => setWidth(Number(event.target.value) || 512)} /></label>
                <label className="studio-field"><span>高度</span><input type="number" min={256} max={1024} step={64} value={height} onChange={(event) => setHeight(Number(event.target.value) || 512)} /></label>
                <label className="studio-field"><span>步数</span><input type="number" min={1} max={100} value={steps} onChange={(event) => setSteps(Number(event.target.value) || 28)} /></label>
                <label className="studio-field"><span>引导强度</span><input type="number" min={1} max={20} step={0.5} value={guidanceScale} onChange={(event) => setGuidanceScale(Number(event.target.value) || 7.5)} /></label>
                <label className="studio-field"><span>Seed</span><input type="number" min={-1} value={seed} onChange={(event) => setSeed(Number(event.target.value) || -1)} /></label>
              </div>
              <div className="studio-form__actions">
                {!capabilities.data?.loaded && selectedArtifactId ? <CommandButton variant="ghost" icon={LoaderCircle} busy={busy === 'load'} onClick={() => void handleLoad()}>加载模型</CommandButton> : null}
                {capabilities.data?.loaded ? <CommandButton variant="ghost" onClick={() => void handleUnload()} busy={busy === 'unload'}>卸载模型</CommandButton> : null}
                <CommandButton icon={Sparkles} busy={busy === 'generate'} disabled={!canGenerate} onClick={() => void handleGenerate()}>
                  {activeJob ? '生成中' : '开始生成'}
                </CommandButton>
              </div>
              {!capabilities.data?.loaded ? <p className="inline-note">请先加载可用的 SD 1.5 artifact；未安装依赖或模型时会在这里显示后端原因。</p> : null}
            </section>
          </main>

          <aside className="image-studio__details" aria-label="生图任务详情">
            <section className="studio-panel">
              <SectionHead title="任务详情" hint={selectedJobId || '尚未选择任务'} />
              {job ? (
                <dl className="studio-facts">
                  <div><dt>状态</dt><dd><StatusBadge label={jobLabel(job.state)} tone={jobTone(job.state)} size="sm" /></dd></div>
                  <div><dt>Job ID</dt><dd className="cell-mono">{job.job_id}</dd></div>
                  <div><dt>提示词</dt><dd>{job.parameters?.prompt || selectedHistory?.prompt || prompt || '—'}</dd></div>
                  <div><dt>Seed</dt><dd className="num-display">{job.parameters?.seed ?? seed}</dd></div>
                  <div><dt>尺寸</dt><dd className="num-display">{job.parameters?.width ?? width} × {job.parameters?.height ?? height}</dd></div>
                  {job.error ? <div><dt>错误</dt><dd className="studio-facts__error">{job.error}</dd></div> : null}
                </dl>
              ) : <EmptyState kind="empty" title="选择一个任务" description="列表中的任务详情会显示在这里。" compact />}
            </section>
            <section className="studio-panel">
              <SectionHead title="运行时" hint="来自 /api/diffusion/capabilities" />
              <dl className="studio-facts">
                <div><dt>引擎状态</dt><dd>{capabilities.data?.state || (capabilities.state === 'error' ? '不可用' : '读取中')}</dd></div>
                <div><dt>已注册资产</dt><dd className="num-display">{capabilities.data?.registered_artifacts ?? artifactList.length}</dd></div>
                <div><dt>依赖</dt><dd>{capabilities.data?.dependencies ? Object.values(capabilities.data.dependencies).filter(Boolean).length : '—'} 项可用</dd></div>
              </dl>
              {capabilities.state === 'error' ? <p className="inline-note">{capabilities.error}</p> : null}
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}
