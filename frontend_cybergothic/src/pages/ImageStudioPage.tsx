import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { Cpu, Download, FileSearch, Image as ImageIcon, LoaderCircle, Network, PauseCircle, RefreshCw, ShieldCheck, Sparkles, Upload } from 'lucide-react';
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
import type { DiffusionArtifactInspectResponse, DiffusionAssetActionResponse, DiffusionDistributedResponse, DiffusionJob, DiffusionPreset } from '../data/types';
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
  workflowId?: string;
  distributed?: boolean;
}

type StudioMode = 'txt2img' | 'img2img' | 'inpaint' | 'distributed';
type DistributedMode = 'single' | 'grid';

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
  const [mode, setMode] = useState<StudioMode>('txt2img');
  const [distributedMode, setDistributedMode] = useState<DistributedMode>('single');
  const [sourceBlobId, setSourceBlobId] = useState('');
  const [maskBlobId, setMaskBlobId] = useState('');
  const [strength, setStrength] = useState(0.75);
  const [gridSeeds, setGridSeeds] = useState('11, 22, 33, 44');
  const [distributedResult, setDistributedResult] = useState<DiffusionDistributedResponse | null>(null);
  const [selectedAssetId, setSelectedAssetId] = useState('');
  const [assetPath, setAssetPath] = useState('models/sd15-original-v1');
  const [assetName, setAssetName] = useState('');
  const [assetLicenseAccepted, setAssetLicenseAccepted] = useState(false);
  const [assetComputeHash, setAssetComputeHash] = useState(false);
  const [assetInspect, setAssetInspect] = useState<DiffusionArtifactInspectResponse | null>(null);
  const [assetStatus, setAssetStatus] = useState<DiffusionAssetActionResponse | null>(null);
  const [assetBusy, setAssetBusy] = useState('');
  const [busy, setBusy] = useState('');
  const mountedRef = useRef(true);

  const presets = capabilities.data?.presets ?? [];
  const artifactList = artifacts.data?.artifacts ?? [];
  const assetList = assets.data?.assets ?? [];
  const activeJob = job && !TERMINAL_STATES.has(job.state) ? job : null;
  const selectedHistory = history.find((item) => item.jobId === selectedJobId) ?? null;
  const loadedArtifactId = capabilities.data?.loaded_artifact?.artifact_id ?? '';
  const distributedWorkflow = (capabilities.data?.distributed_workflow || {}) as Record<string, unknown>;
  const distributedEnabled = Boolean(distributedWorkflow.enabled);
  const selectedAsset = assetList.find((asset) => asset.asset_id === selectedAssetId) ?? assetList[0];
  const blobId = job?.blob?.blob_id || selectedHistory?.blobId || '';
  const distributedWorkflowId = String(distributedResult?.workflow?.workflow_id || distributedResult?.workflow_id || '');
  const distributedImage = distributedResult?.result?.image;
  const blobUrl = distributedImage?.url || (distributedWorkflowId && distributedImage?.blob_id
    ? `/api/diffusion/distributed/workflows/${encodeURIComponent(distributedWorkflowId)}/blobs/${encodeURIComponent(distributedImage.blob_id)}`
    : blobId ? `/api/diffusion/blobs/${encodeURIComponent(blobId)}` : '');
  const canGenerate = Boolean(
    capabilities.data?.loaded && prompt.trim() && !activeJob && !busy
      && (mode === 'txt2img' || (mode === 'distributed' ? distributedEnabled : Boolean(sourceBlobId))),
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

  useEffect(() => {
    if (!selectedAssetId && assetList[0]) setSelectedAssetId(assetList[0].asset_id);
  }, [assetList, selectedAssetId]);

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
        ...(nextJob.workflow_id ? { workflowId: String(nextJob.workflow_id) } : {}),
        ...(nextJob.distributed ? { distributed: true } : {}),
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

  const handleEdit = async () => {
    if (!sourceBlobId.trim() || activeJob || busy) return;
    setBusy('edit');
    const editMode = mode === 'inpaint' ? 'inpaint' : 'img2img';
    const request = {
      mode: editMode as 'img2img' | 'inpaint',
      source_blob_id: sourceBlobId.trim(),
      ...(mode === 'inpaint' && maskBlobId.trim() ? { mask_blob_id: maskBlobId.trim() } : {}),
      prompt: prompt.trim(),
      negative_prompt: negativePrompt.trim(),
      seed: seed < 0 ? undefined : seed,
      width,
      height,
      steps,
      guidance_scale: guidanceScale,
      strength,
    };
    try {
      if (usingFixtures) {
        const fixtureJob: DiffusionJob = {
          job_id: `fixture-sdedit-${Date.now().toString(36)}`,
          state: 'completed',
          created_at: Date.now() / 1000,
          parameters: { ...request, prompt: prompt.trim() },
          blob: { blob_id: 'fixture-edited-image', content_type: 'image/png' },
        };
        setJob(fixtureJob);
        updateHistory(fixtureJob, prompt.trim(), negativePrompt.trim());
        pushToast('演示数据已完成图像编辑。', 'info');
      } else {
        const submitted = await api.editDiffusionImage(request);
        setJob(submitted);
        updateHistory(submitted, prompt.trim(), negativePrompt.trim());
        void pollJob(submitted.job_id, prompt.trim(), negativePrompt.trim()).catch((err) => {
          if (mountedRef.current) pushToast(`编辑任务状态读取失败：${api.describeError(err)}`, 'danger');
        });
      }
    } catch (err) {
      pushToast(`提交编辑任务失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleDistributed = async () => {
    if (!prompt.trim() || activeJob || busy || !distributedEnabled) return;
    const seeds = gridSeeds.split(',').map((value) => Number(value.trim())).filter((value) => Number.isInteger(value) && value >= 0);
    if (distributedMode === 'grid' && (seeds.length !== 4 || new Set(seeds).size !== 4)) {
      pushToast('四宫格必须填写四个不同的非负 seed。', 'danger');
      return;
    }
    setBusy('distributed');
    const request = {
      prompt: prompt.trim(),
      negative_prompt: negativePrompt.trim(),
      seed: seed < 0 ? undefined : seed,
      width,
      height,
      steps,
      guidance_scale: guidanceScale,
      ...(selectedArtifactId ? { artifact_id: selectedArtifactId } : {}),
    };
    try {
      let response: DiffusionDistributedResponse;
      if (usingFixtures) {
        const workflowId = `fixture-wf-${Date.now().toString(36)}`;
        response = {
          status: 'completed', distributed: distributedMode === 'grid', workflow_id: workflowId,
          workflow: { workflow_id: workflowId, state: 'completed', template: distributedMode === 'grid' ? 'image_grid_v1' : 'image_generate_v1' },
          result: distributedMode === 'grid'
            ? { images: seeds.map((item, index) => ({ blob_id: `fixture-grid-${index}`, url: `/api/diffusion/blobs/fixture-grid-${index}`, seed: item })) }
            : { image: { blob_id: 'fixture-distributed-image', url: '/api/diffusion/blobs/fixture-distributed-image' }, metrics: { elapsed_seconds: 0.42 } },
          provider_id: 'fixture_remote_diffusion', node_id: 'fixture-worker',
        };
      } else if (distributedMode === 'grid') {
        response = await api.generateDistributedDiffusionGrid({ ...request, seeds });
      } else {
        response = await api.generateDistributedDiffusion(request);
      }
      setDistributedResult(response);
      const firstImage = response.result?.image || response.result?.images?.[0];
      const workflowId = String(response.workflow?.workflow_id || response.workflow_id || `distributed-${Date.now().toString(36)}`);
      const completedJob: DiffusionJob = {
        job_id: workflowId,
        workflow_id: workflowId,
        state: 'completed',
        created_at: Date.now() / 1000,
        parameters: request,
        ...(firstImage?.blob_id ? { blob: { blob_id: firstImage.blob_id, content_type: 'image/png' } } : {}),
        distributed: true,
      };
      setJob(completedJob);
      updateHistory(completedJob, prompt.trim(), negativePrompt.trim());
      pushToast(distributedMode === 'grid' ? '分布式四宫格任务已完成。' : '分布式图像任务已完成。', usingFixtures ? 'info' : 'ok');
    } catch (err) {
      pushToast(`分布式任务失败：${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleSubmit = () => {
    if (mode === 'txt2img') return void handleGenerate();
    if (mode === 'distributed') return void handleDistributed();
    return void handleEdit();
  };

  const handleUploadSource = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || assetBusy) return;
    setAssetBusy('upload');
    try {
      if (usingFixtures) {
        setSourceBlobId('fixture-input-image');
        pushToast('演示数据已准备输入图像。', 'info');
      } else {
        const uploaded = await api.uploadDiffusionBlob(file);
        setSourceBlobId(uploaded.blob_id);
        pushToast(`输入图像已上传：${uploaded.blob_id}`, 'ok');
      }
    } catch (err) {
      pushToast(`上传输入图像失败：${api.describeError(err)}`, 'danger');
    } finally {
      setAssetBusy('');
      event.target.value = '';
    }
  };

  const handleInspectAsset = async () => {
    if (!assetPath.trim() || assetBusy) return;
    setAssetBusy('inspect');
    try {
      if (usingFixtures) {
        setAssetInspect({ artifact_id: 'fixture-inspected-artifact', name: assetName || 'Inspected SD asset', path: assetPath.trim(), artifact: { artifact_kind: 'sd15_pipeline', loadable: true, size_bytes: 3673657344 } });
        pushToast('演示数据已完成资产预检。', 'info');
      } else {
        setAssetInspect(await api.inspectDiffusionArtifact(assetPath.trim(), assetComputeHash));
        pushToast('资产预检完成。', 'ok');
      }
    } catch (err) {
      setAssetInspect(null);
      pushToast(`资产预检失败：${api.describeError(err)}`, 'danger');
    } finally {
      setAssetBusy('');
    }
  };

  const handleRegisterAsset = async () => {
    if (!assetPath.trim() || assetBusy) return;
    setAssetBusy('register');
    try {
      if (usingFixtures) {
        setAssetInspect({ artifact_id: assetName.trim() || 'fixture-registered-artifact', name: assetName.trim() || 'Registered SD asset', path: assetPath.trim(), artifact: { artifact_kind: 'sd15_pipeline', loadable: true } });
        pushToast('演示数据已登记资产。', 'info');
      } else {
        setAssetInspect(await api.registerDiffusionArtifact({ path: assetPath.trim(), ...(assetName.trim() ? { name: assetName.trim() } : {}), compute_hash: assetComputeHash }));
        await artifacts.refresh();
        pushToast('资产已登记。', 'ok');
      }
    } catch (err) {
      pushToast(`资产登记失败：${api.describeError(err)}`, 'danger');
    } finally {
      setAssetBusy('');
    }
  };

  const handleAssetStatus = async () => {
    if (!selectedAssetId || assetBusy) return;
    setAssetBusy('status');
    try {
      if (usingFixtures) {
        setAssetStatus({ asset_id: selectedAssetId, status: selectedAsset?.installed ? 'installed' : 'missing', present_bytes: selectedAsset?.present_bytes, total_bytes: selectedAsset?.total_bytes });
      } else {
        setAssetStatus(await api.fetchDiffusionAssetStatus(selectedAssetId));
      }
    } catch (err) {
      pushToast(`读取资产状态失败：${api.describeError(err)}`, 'danger');
    } finally {
      setAssetBusy('');
    }
  };

  const handleDownloadAsset = async () => {
    if (!selectedAssetId || assetBusy) return;
    if (!assetLicenseAccepted) {
      pushToast('下载前请确认资产许可证。', 'danger');
      return;
    }
    setAssetBusy('download');
    try {
      if (usingFixtures) {
        setAssetStatus({ asset_id: selectedAssetId, status: 'queued', job: { state: 'queued' } });
        pushToast('演示数据已排队下载资产。', 'info');
      } else {
        setAssetStatus(await api.downloadDiffusionAsset(selectedAssetId, { licenseAccepted: assetLicenseAccepted, useLocalProxyFallback: true }));
        pushToast('资产下载已排队。', 'ok');
      }
    } catch (err) {
      pushToast(`资产下载失败：${api.describeError(err)}`, 'danger');
    } finally {
      setAssetBusy('');
    }
  };

  const handleImportAsset = async () => {
    if (!selectedAssetId || !assetPath.trim() || assetBusy) return;
    if (!assetLicenseAccepted) {
      pushToast('导入前请确认资产许可证。', 'danger');
      return;
    }
    setAssetBusy('import');
    try {
      if (usingFixtures) {
        setAssetStatus({ asset_id: selectedAssetId, status: 'imported', valid: true, path: assetPath.trim() });
        pushToast('演示数据已登记本地资产。', 'info');
      } else {
        setAssetStatus(await api.importDiffusionAsset({ assetId: selectedAssetId, path: assetPath.trim(), licenseAccepted: assetLicenseAccepted }));
        await assets.refresh();
        pushToast('资产导入完成。', 'ok');
      }
    } catch (err) {
      pushToast(`资产导入失败：${api.describeError(err)}`, 'danger');
    } finally {
      setAssetBusy('');
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
                  <li key={asset.asset_id} data-installed={asset.installed ? 'true' : 'false'} data-selected={asset.asset_id === selectedAssetId ? 'true' : undefined}>
                    <button type="button" className="image-asset" onClick={() => setSelectedAssetId(asset.asset_id)}>
                      <span>{asset.name || asset.asset_id}</span>
                      <small>{asset.asset_id}</small>
                    </button>
                    <StatusBadge label={asset.installed ? '已安装' : '待获取'} tone={asset.installed ? 'ok' : 'idle'} size="sm" />
                  </li>
                ))}
              </ul>
              {selectedAsset ? <div className="asset-quick-actions">
                <span className="cell-mono">{selectedAsset.asset_id}</span>
                <CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={assetBusy === 'status'} onClick={() => void handleAssetStatus()}>状态</CommandButton>
              </div> : null}
            </section>

            <section className="studio-panel studio-panel--rail studio-panel--asset-tools" data-testid="image-asset-tools">
              <SectionHead title="资产工具" hint="本地主节点" />
              <label className="studio-field"><span>本地路径</span><input value={assetPath} onChange={(event) => setAssetPath(event.target.value)} placeholder="models/sd15-original-v1" /></label>
              <label className="studio-field"><span>登记名称</span><input value={assetName} onChange={(event) => setAssetName(event.target.value)} placeholder="可选" /></label>
              <label className="studio-check"><input type="checkbox" checked={assetComputeHash} onChange={(event) => setAssetComputeHash(event.target.checked)} /><span>计算 SHA256</span></label>
              <label className="studio-check"><input type="checkbox" checked={assetLicenseAccepted} onChange={(event) => setAssetLicenseAccepted(event.target.checked)} /><span>我确认许可证和来源</span></label>
              <div className="studio-inline-actions">
                <CommandButton variant="ghost" size="sm" icon={FileSearch} busy={assetBusy === 'inspect'} onClick={() => void handleInspectAsset()}>检查</CommandButton>
                <CommandButton variant="ghost" size="sm" icon={ShieldCheck} busy={assetBusy === 'register'} onClick={() => void handleRegisterAsset()}>登记</CommandButton>
                <CommandButton variant="ghost" size="sm" icon={Download} busy={assetBusy === 'download'} disabled={!selectedAssetId} onClick={() => void handleDownloadAsset()}>下载</CommandButton>
                <CommandButton variant="ghost" size="sm" icon={Upload} busy={assetBusy === 'import'} disabled={!selectedAssetId} onClick={() => void handleImportAsset()}>导入</CommandButton>
              </div>
              {assetInspect ? <p className="inline-note">预检：{assetInspect.artifact_id || 'unknown'} · {String(assetInspect.artifact?.artifact_kind || 'unknown')} · {assetInspect.path}</p> : null}
              {assetStatus ? <p className="inline-note">状态：{assetStatus.status || (assetStatus.valid ? 'valid' : 'reported')} {assetStatus.job ? `· ${String(assetStatus.job.state || 'queued')}` : ''}</p> : null}
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
                  <p className="mono-label">{mode === 'distributed' ? 'DISTRIBUTED IMAGE' : mode === 'txt2img' ? 'TEXT TO IMAGE' : mode === 'inpaint' ? 'MASKED EDIT' : 'IMAGE TO IMAGE'}</p>
                  <h2>{mode === 'distributed' ? '分布式任务' : mode === 'txt2img' ? '生成参数' : mode === 'inpaint' ? '局部重绘' : '图像编辑'}</h2>
                </div>
                <div className="studio-form__modes" role="group" aria-label="生成模式">
                  <button type="button" className={mode === 'txt2img' ? 'is-active' : ''} aria-pressed={mode === 'txt2img'} onClick={() => setMode('txt2img')}>文生图</button>
                  <button type="button" className={mode === 'img2img' ? 'is-active' : ''} aria-pressed={mode === 'img2img'} onClick={() => setMode('img2img')}>图生图</button>
                  <button type="button" className={mode === 'inpaint' ? 'is-active' : ''} aria-pressed={mode === 'inpaint'} onClick={() => setMode('inpaint')}>局部重绘</button>
                  <button type="button" className={mode === 'distributed' ? 'is-active' : ''} aria-pressed={mode === 'distributed'} onClick={() => setMode('distributed')}><Network size={13} aria-hidden="true" />分布式</button>
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
              {mode !== 'txt2img' && mode !== 'distributed' ? <div className="studio-source-fields">
                <label className="studio-field"><span>输入 Blob ID</span><input value={sourceBlobId} onChange={(event) => setSourceBlobId(event.target.value)} placeholder="上传后自动填入" /></label>
                <label className="studio-field"><span>上传输入图像</span><input type="file" accept="image/png,image/jpeg,image/webp" aria-label="上传输入图像" onChange={(event) => void handleUploadSource(event)} disabled={assetBusy !== ''} /></label>
                {mode === 'inpaint' ? <label className="studio-field"><span>蒙版 Blob ID</span><input value={maskBlobId} onChange={(event) => setMaskBlobId(event.target.value)} placeholder="可选，默认全图" /></label> : null}
                <label className="studio-field"><span>编辑强度</span><input type="number" min={0} max={1} step={0.05} value={strength} onChange={(event) => setStrength(Number(event.target.value) || 0.75)} /></label>
              </div> : null}
              {mode === 'distributed' ? <div className="studio-distributed-fields">
                <div className="studio-form__modes" role="group" aria-label="分布式模式"><button type="button" className={distributedMode === 'single' ? 'is-active' : ''} aria-pressed={distributedMode === 'single'} onClick={() => setDistributedMode('single')}>单图 Worker</button><button type="button" className={distributedMode === 'grid' ? 'is-active' : ''} aria-pressed={distributedMode === 'grid'} onClick={() => setDistributedMode('grid')}>四宫格 fan-out</button></div>
                {distributedMode === 'grid' ? <label className="studio-field"><span>四个 Seed（逗号分隔）</span><input value={gridSeeds} onChange={(event) => setGridSeeds(event.target.value)} placeholder="11, 22, 33, 44" /></label> : null}
                <p className="inline-note">{distributedEnabled ? '当前实验门已开启；结果会绑定 workflow 和远端 Blob。' : '分布式实验门未开启，页面不会提交请求。'}</p>
              </div> : null}
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
                <CommandButton icon={mode === 'distributed' ? Network : mode === 'txt2img' ? Sparkles : ImageIcon} busy={busy !== '' && ['generate', 'edit', 'distributed'].includes(busy)} disabled={!canGenerate} onClick={handleSubmit}>
                  {activeJob ? '处理中' : mode === 'distributed' ? '提交分布式' : mode === 'txt2img' ? '开始生成' : '开始编辑'}
                </CommandButton>
              </div>
              {!capabilities.data?.loaded ? <p className="inline-note">请先加载可用的 SD 1.5 artifact；未安装依赖或模型时会在这里显示后端原因。</p> : null}
              {mode !== 'txt2img' && mode !== 'distributed' && !sourceBlobId ? <p className="inline-note">图像编辑需要输入 Blob；可用上方文件选择器上传，或粘贴已有 Blob ID。</p> : null}
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
            {distributedResult ? <section className="studio-panel studio-panel--distributed-result" data-testid="distributed-result">
              <SectionHead title="分布式结果" hint={distributedWorkflowId || 'workflow'} />
              <dl className="studio-facts">
                <div><dt>执行模式</dt><dd>{String(distributedResult.execution_mode || (distributedResult.distributed ? 'multi-node' : 'worker'))}</dd></div>
                <div><dt>Provider</dt><dd className="cell-mono">{String(distributedResult.provider_id || (Array.isArray(distributedResult.provider_ids) ? distributedResult.provider_ids.join(', ') : '') || '—')}</dd></div>
                <div><dt>节点</dt><dd className="cell-mono">{String(distributedResult.node_id || (Array.isArray(distributedResult.node_ids) ? distributedResult.node_ids.join(', ') : '') || '—')}</dd></div>
              </dl>
              {distributedResult.result?.images?.length ? <div className="distributed-thumbnails">{distributedResult.result.images.map((image, index) => {
                const href = image.url || (distributedWorkflowId && image.blob_id ? `/api/diffusion/distributed/workflows/${encodeURIComponent(distributedWorkflowId)}/blobs/${encodeURIComponent(image.blob_id)}` : '');
                return href ? <a key={image.blob_id || index} href={href} download={`qlh-grid-${index}.png`}><img src={href} alt={`分布式结果 ${index + 1}`} /><span>#{index + 1}</span></a> : null;
              })}</div> : null}
            </section> : null}
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
