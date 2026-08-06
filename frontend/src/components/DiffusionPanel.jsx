import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  cancelDiffusionJob,
  deleteDiffusionBlob,
  downloadDiffusionAsset,
  editDiffusionImage,
  fetchDiffusionAssetCatalog,
  fetchDiffusionAssetStatus,
  fetchDiffusionArtifacts,
  fetchDiffusionBlob,
  fetchDiffusionCapabilities,
  fetchDiffusionJob,
  generateDiffusionImage,
  inspectDiffusionArtifact,
  importDiffusionAsset,
  loadDiffusionArtifact,
  registerDiffusionArtifact,
  unloadDiffusionArtifact,
  unloadModel,
  uploadDiffusionBlob,
} from '../api/client';
import {
  buildEditRequest,
  canUseLocalDiffusion,
  loadedArtifactId,
  normalizeDiffusionJob,
  presetToForm,
} from '../diffusionState';

const PROFILE_OPTIONS = [
  { id: 'balanced', label: '平衡', detail: 'FP16 · CPU offload' },
  { id: 'resident_fp16', label: '常驻', detail: 'FP16 · CUDA' },
  { id: 'qkv_fp16', label: 'QKV', detail: 'FP16 · 融合' },
  { id: 'unet_8bit', label: '8-bit', detail: 'U-Net Linear' },
  { id: 'unet_8bit_qkv', label: '8-bit + QKV', detail: '常驻 CUDA' },
];

const STATE_LABELS = {
  unloaded: '未加载',
  loading: '加载中',
  loaded: '已就绪',
  unloading: '卸载中',
  error: '异常',
  closed: '已关闭',
};

const JOB_STATE_LABELS = {
  queued: '排队中',
  running: '生成中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

const PRESET_LABELS = {
  sd15_original_v1: '原版 SD 1.5',
  sd15_retrovers_space_courier_v1: '90s Anime',
};

const REQUIRED_DEPENDENCIES = [
  'torch',
  'diffusers',
  'transformers',
  'accelerate',
  'safetensors',
  'PIL',
];

function getInitialModelPath() {
  try {
    return localStorage.getItem('qlh-sd-model-path') || 'models/sd15-original-v1';
  } catch (_) {
    return 'models/sd15-original-v1';
  }
}

function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function formatSeconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : '—';
}

export default function DiffusionPanel({
  myRole,
  onToast,
  onLlmUnloaded,
  onDiffusionStateChange,
}) {
  const [capabilities, setCapabilities] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [assetCatalog, setAssetCatalog] = useState([]);
  const [acceptedLicenses, setAcceptedLicenses] = useState({});
  const [selectedArtifactId, setSelectedArtifactId] = useState('');
  const [modelPath, setModelPath] = useState(getInitialModelPath);
  const [inspection, setInspection] = useState(null);
  const [profile, setProfile] = useState('balanced');
  const [form, setForm] = useState(() => presetToForm(null));
  const [editMode, setEditMode] = useState('txt2img');
  const [sourceBlob, setSourceBlob] = useState(null);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [action, setAction] = useState('');
  const [error, setError] = useState('');

  const pollTokenRef = useRef(0);
  const mountedRef = useRef(true);
  const resultRef = useRef(null);
  const sourceBlobRef = useRef(null);

  const isMaster = canUseLocalDiffusion(myRole);
  const roleResolved = Boolean(myRole);
  const loadedId = loadedArtifactId(capabilities);
  const selectedPreset = useMemo(
    () => capabilities?.presets?.find(item => item.preset_id === form.presetId) || null,
    [capabilities, form.presetId],
  );
  const selectedArtifact = useMemo(
    () => artifacts.find(item => item.artifact_id === selectedArtifactId) || null,
    [artifacts, selectedArtifactId],
  );
  const normalizedJob = job ? normalizeDiffusionJob(job) : null;
  const jobActive = Boolean(normalizedJob && !normalizedJob.terminal);
  const missingDependencies = capabilities?.missing_dependencies
    || REQUIRED_DEPENDENCIES.filter(name => capabilities?.dependencies?.[name] !== true);
  const dependenciesReady = Boolean(capabilities && missingDependencies.length === 0);

  const replaceResult = useCallback((next) => {
    const previous = resultRef.current;
    if (previous?.url) URL.revokeObjectURL(previous.url);
    resultRef.current = next;
    if (mountedRef.current) setResult(next);
  }, []);

  const replaceSourceBlob = useCallback((next) => {
    const previous = sourceBlobRef.current;
    if (previous?.url && !(next && previous.blobId === next.blobId)) {
      URL.revokeObjectURL(previous.url);
    }
    sourceBlobRef.current = next;
    if (mountedRef.current) setSourceBlob(next);
  }, []);

  const refresh = useCallback(async () => {
    if (!isMaster) return;
    const [nextCapabilities, artifactData, assetData] = await Promise.all([
      fetchDiffusionCapabilities(),
      fetchDiffusionArtifacts(),
      fetchDiffusionAssetCatalog(),
    ]);
    if (!mountedRef.current) return;
    const nextArtifacts = artifactData.artifacts || [];
    setCapabilities(nextCapabilities);
    setArtifacts(nextArtifacts);
    setAssetCatalog(assetData.assets || []);
    setSelectedArtifactId(previous => {
      if (previous && nextArtifacts.some(item => item.artifact_id === previous)) {
        return previous;
      }
      return nextCapabilities.loaded_artifact?.artifact_id
        || nextArtifacts[0]?.artifact_id
        || '';
    });
    setForm(previous => (
      previous.presetId
        ? previous
        : presetToForm(nextCapabilities.presets?.[0])
    ));
    onDiffusionStateChange?.(Boolean(nextCapabilities.loaded));
  }, [isMaster, onDiffusionStateChange]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      pollTokenRef.current += 1;
      if (resultRef.current?.url) URL.revokeObjectURL(resultRef.current.url);
      if (sourceBlobRef.current?.url) URL.revokeObjectURL(sourceBlobRef.current.url);
    };
  }, []);

  useEffect(() => {
    if (isMaster) {
      refresh().catch(err => setError(err.message));
      return;
    }
    pollTokenRef.current += 1;
    replaceResult(null);
    setCapabilities(null);
    setArtifacts([]);
    setAssetCatalog([]);
    setJob(null);
    setError('');
    onDiffusionStateChange?.(false);
  }, [isMaster, onDiffusionStateChange, refresh, replaceResult]);

  const handlePresetChange = (presetId) => {
    const preset = capabilities?.presets?.find(item => item.preset_id === presetId);
    setForm(presetToForm(preset));
    const catalogAsset = assetCatalog.find(item => item.preset_id === presetId);
    if (catalogAsset && artifacts.some(item => item.artifact_id === catalogAsset.artifact_id)) {
      setSelectedArtifactId(catalogAsset.artifact_id);
    }
  };

  const updateForm = (field, value) => {
    setForm(previous => ({ ...previous, [field]: value }));
  };

  const handleInspectAndRegister = async () => {
    const path = modelPath.trim();
    if (!path) return;
    setAction('register');
    setError('');
    try {
      const inspected = await inspectDiffusionArtifact(path, false);
      setInspection(inspected);
      if (inspected.artifact_kind !== 'sd15_pipeline' || !inspected.loadable) {
        throw new Error(inspected.warnings?.join('；') || '该路径不是完整的 SD 1.5 Diffusers 目录');
      }
      const registered = await registerDiffusionArtifact(path, {
        name: path.split(/[\\/]/).filter(Boolean).at(-1) || 'SD 1.5',
      });
      try { localStorage.setItem('qlh-sd-model-path', path); } catch (_) {}
      await refresh();
      setSelectedArtifactId(registered.artifact_id);
      onToast?.({ type: 'success', msg: `已登记图像模型: ${registered.name}` });
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `模型登记失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const updateCatalogStatus = useCallback((assetId, status) => {
    setAssetCatalog(previous => previous.map(item => (
      item.asset_id === assetId
        ? { ...item, installed: status.installed, present_bytes: status.present_bytes, job: status }
        : item
    )));
  }, []);

  const pollAssetDownload = useCallback(async (assetId) => {
    while (mountedRef.current) {
      const status = await fetchDiffusionAssetStatus(assetId);
      if (!mountedRef.current) return status;
      updateCatalogStatus(assetId, status);
      if (['completed', 'failed'].includes(status.state)) return status;
      await sleep(500);
    }
    return null;
  }, [updateCatalogStatus]);

  const handleAssetDownload = async (asset) => {
    const actionId = `download:${asset.asset_id}`;
    setAction(actionId);
    setError('');
    try {
      const started = await downloadDiffusionAsset(asset.asset_id, {
        licenseAccepted: acceptedLicenses[asset.asset_id] === true,
        useLocalProxyFallback: true,
      });
      updateCatalogStatus(asset.asset_id, started);
      const completed = ['completed', 'failed'].includes(started.state)
        ? started
        : await pollAssetDownload(asset.asset_id);
      if (completed?.state === 'failed') {
        throw new Error(completed.error || '图像模型下载失败');
      }
      await refresh();
      setSelectedArtifactId(asset.artifact_id);
      onToast?.({ type: 'success', msg: `图像模型已校验并登记: ${asset.name}` });
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `图像模型下载失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const handleAssetImport = async (asset) => {
    const path = modelPath.trim();
    if (!path) return;
    setAction(`import:${asset.asset_id}`);
    setError('');
    try {
      await importDiffusionAsset(
        asset.asset_id,
        path,
        acceptedLicenses[asset.asset_id] === true,
      );
      await refresh();
      setSelectedArtifactId(asset.artifact_id);
      onToast?.({ type: 'success', msg: `离线资产包校验通过: ${asset.name}` });
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `离线资产包导入失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const handleLoad = async () => {
    if (!selectedArtifactId) return;
    setAction('load');
    setError('');
    try {
      if (capabilities?.local_llm_loaded) {
        await unloadModel();
        onLlmUnloaded?.();
      }
      if (capabilities?.loaded && loadedId !== selectedArtifactId) {
        await unloadDiffusionArtifact();
      }
      const next = await loadDiffusionArtifact(selectedArtifactId, profile);
      setCapabilities(next);
      onDiffusionStateChange?.(true);
      onToast?.({ type: 'success', msg: `图像模型已加载: ${selectedArtifact?.name || selectedArtifactId}` });
      await refresh();
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `图像模型加载失败: ${err.message}` });
      await refresh().catch(() => {});
    } finally {
      setAction('');
    }
  };

  const handleUnload = async () => {
    setAction('unload');
    setError('');
    pollTokenRef.current += 1;
    try {
      const next = await unloadDiffusionArtifact();
      setCapabilities(next);
      setJob(previous => (
        previous && !normalizeDiffusionJob(previous).terminal
          ? { ...previous, state: 'cancelled', cancel_requested: true }
          : previous
      ));
      onDiffusionStateChange?.(false);
      onToast?.({ type: 'success', msg: '图像模型已卸载' });
      await refresh();
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `卸载失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const pollJob = useCallback(async (jobId, token, requestSnapshot) => {
    while (mountedRef.current && pollTokenRef.current === token) {
      const snapshot = normalizeDiffusionJob(await fetchDiffusionJob(jobId));
      if (!mountedRef.current || pollTokenRef.current !== token) return;
      setJob(snapshot);
      if (!snapshot.terminal) {
        await sleep(200);
        continue;
      }
      if (snapshot.state === 'completed' && snapshot.blob?.blob_id) {
        const fetched = await fetchDiffusionBlob(snapshot.blob.blob_id);
        if (!mountedRef.current || pollTokenRef.current !== token) return;
        const previous = resultRef.current;
        if (previous?.blobId && previous.blobId !== snapshot.blob.blob_id) {
          await deleteDiffusionBlob(previous.blobId).catch(() => {});
        }
        replaceResult({
          blobId: snapshot.blob.blob_id,
          url: URL.createObjectURL(fetched.blob),
          sizeBytes: fetched.blob.size,
          mode: requestSnapshot.mode === 'img2img' ? 'img2img' : 'txt2img',
          strength: requestSnapshot.mode === 'img2img' ? requestSnapshot.strength : null,
          sourceBlobId: requestSnapshot.mode === 'img2img' ? requestSnapshot.source_blob_id : null,
          seed: snapshot.parameters?.seed,
          width: snapshot.parameters?.width,
          height: snapshot.parameters?.height,
          steps: snapshot.parameters?.steps,
          elapsedSeconds: snapshot.metrics?.elapsed_seconds,
          artifactId: snapshot.artifact_id,
          prompt: requestSnapshot.prompt,
          negativePrompt: requestSnapshot.negative_prompt,
          guidanceScale: requestSnapshot.guidance_scale,
        });
        onToast?.({ type: 'success', msg: '图片生成完成' });
      } else if (snapshot.state === 'failed') {
        setError(snapshot.error || '图片生成失败');
        onToast?.({ type: 'error', msg: snapshot.error || '图片生成失败' });
      } else if (snapshot.state === 'cancelled') {
        onToast?.({ type: 'info', msg: '图片生成已取消' });
      }
      await refresh().catch(() => {});
      return;
    }
  }, [onToast, refresh, replaceResult]);

  const handleGenerate = async () => {
    if (!capabilities?.loaded || jobActive || !form.prompt.trim()) return;
    setAction('generate');
    setError('');
    try {
      const requestSnapshot = {
        mode: 'txt2img',
        preset_id: form.presetId || null,
        prompt: form.prompt.trim(),
        negative_prompt: form.negativePrompt.trim(),
        seed: Number(form.seed),
        width: Number(form.width),
        height: Number(form.height),
        steps: Number(form.steps),
        guidance_scale: Number(form.guidanceScale),
        scheduler: form.scheduler || null,
      };
      const submitted = await generateDiffusionImage(requestSnapshot);
      setJob(normalizeDiffusionJob(submitted));
      const token = pollTokenRef.current + 1;
      pollTokenRef.current = token;
      pollJob(submitted.job_id, token, requestSnapshot).catch(err => {
        if (mountedRef.current && pollTokenRef.current === token) {
          setError(err.message);
          onToast?.({ type: 'error', msg: `任务状态读取失败: ${err.message}` });
        }
      });
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `生成失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const handleSourceUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || !capabilities?.loaded || jobActive) return;
    if (!file.type.startsWith('image/')) {
      setError('源图必须是图片文件');
      onToast?.({ type: 'error', msg: '源图必须是图片文件' });
      return;
    }
    setAction('upload');
    setError('');
    try {
      const uploaded = await uploadDiffusionBlob(file, 'input_image');
      replaceSourceBlob({
        blobId: uploaded.blob_id,
        url: URL.createObjectURL(file),
        name: file.name,
        sizeBytes: file.size,
      });
      onToast?.({ type: 'success', msg: `源图已上传: ${file.name}` });
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `源图上传失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const handleRemoveSource = () => {
    replaceSourceBlob(null);
  };

  const handleEdit = async () => {
    if (!capabilities?.loaded || jobActive || !form.prompt.trim() || !sourceBlob) return;
    setAction('edit');
    setError('');
    try {
      const requestSnapshot = buildEditRequest(form, sourceBlob.blobId);
      const submitted = await editDiffusionImage(requestSnapshot);
      setJob(normalizeDiffusionJob(submitted));
      const token = pollTokenRef.current + 1;
      pollTokenRef.current = token;
      pollJob(submitted.job_id, token, requestSnapshot).catch(err => {
        if (mountedRef.current && pollTokenRef.current === token) {
          setError(err.message);
          onToast?.({ type: 'error', msg: `任务状态读取失败: ${err.message}` });
        }
      });
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `图生图失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const handleContinueEdit = async () => {
    const current = resultRef.current;
    if (!current?.blobId) return;
    setAction('continue-edit');
    try {
      let copiedUrl = current.url;
      try {
        const copied = await fetch(current.url).then(res => res.blob());
        copiedUrl = URL.createObjectURL(copied);
      } catch (_) {
        // fetch 对 blob: URL 失败时回退共享 URL；其结果被替换时预览可能失效
      }
      replaceSourceBlob({
        blobId: current.blobId,
        url: copiedUrl,
        name: `result-${current.blobId}.png`,
        sizeBytes: current.sizeBytes,
      });
      setEditMode('img2img');
      onToast?.({ type: 'info', msg: '已把当前结果作为源图，调整参数后可继续编辑' });
    } finally {
      setAction('');
    }
  };

  const handleCancel = async () => {
    if (!normalizedJob || normalizedJob.terminal) return;
    setAction('cancel');
    try {
      await cancelDiffusionJob(normalizedJob.job_id);
      setJob(previous => ({ ...previous, cancel_requested: true }));
    } catch (err) {
      setError(err.message);
      onToast?.({ type: 'error', msg: `取消失败: ${err.message}` });
    } finally {
      setAction('');
    }
  };

  const handleDeleteResult = async () => {
    const current = resultRef.current;
    if (!current) return;
    try {
      await deleteDiffusionBlob(current.blobId);
    } catch (err) {
      if (err.status !== 404) {
        onToast?.({ type: 'error', msg: `删除图片失败: ${err.message}` });
        return;
      }
    }
    if (sourceBlobRef.current?.blobId === current.blobId) {
      replaceSourceBlob(null);
    }
    replaceResult(null);
  };

  if (!roleResolved) {
    return <div className="diffusion-empty">正在读取节点角色...</div>;
  }

  if (!isMaster) {
    return (
      <div className="diffusion-empty">
        <strong>图像引擎仅在主节点本机开放</strong>
        <span>当前节点不持有完整 SD pipeline。</span>
      </div>
    );
  }

  return (
    <section className="diffusion-workspace">
      <header className="diffusion-header">
        <div>
          <h2>图像生成</h2>
          <div className="diffusion-header-meta">
            <span className="runtime-chip">Stable Diffusion 1.5</span>
            <span className="runtime-chip local">本地</span>
            <span className={`runtime-state state-${capabilities?.state || 'unknown'}`}>
              {STATE_LABELS[capabilities?.state] || '检测中'}
            </span>
          </div>
        </div>
        <div className="diffusion-header-actions">
          <button className="btn-ghost icon-action" onClick={() => refresh()} title="刷新图像引擎状态">
            ↻
          </button>
          {capabilities?.loaded && (
            <button className="btn-danger" onClick={handleUnload} disabled={Boolean(action)}>
              {action === 'unload' ? '卸载中...' : '卸载图像模型'}
            </button>
          )}
        </div>
      </header>

      <div className="diffusion-body">
        {error && (
          <div className="diffusion-alert error" role="alert">
            <span>{error}</span>
            <button onClick={() => setError('')} title="关闭">×</button>
          </div>
        )}

        <div className="diffusion-layout-grid">
          <div className="diffusion-controls">
            <section className="diffusion-section">
              <div className="diffusion-section-heading">
                <h3>模型</h3>
                <span>{artifacts.length} 个资产</span>
              </div>
              <div className="diffusion-asset-catalog">
                {assetCatalog.map(asset => {
                  const status = asset.job;
                  const active = status && ['queued', 'downloading', 'verifying'].includes(status.state);
                  const accepted = acceptedLicenses[asset.asset_id] === true;
                  return (
                    <div className="diffusion-asset-row" key={asset.asset_id}>
                      <div className="diffusion-asset-title">
                        <strong>{asset.name}</strong>
                        <span className={asset.installed ? 'asset-ready' : 'asset-missing'}>
                          {asset.installed ? '已就绪' : formatBytes(asset.download_bytes)}
                        </span>
                      </div>
                      <div className="diffusion-asset-meta">
                        <span>{asset.license_id}</span>
                        <span>{asset.revision.slice(0, 8)}</span>
                        <a href={asset.model_card_url} target="_blank" rel="noreferrer">模型卡</a>
                      </div>
                      {active && (
                        <div className="asset-download-progress" aria-label={`下载进度 ${status.progress_percent || 0}%`}>
                          <span style={{ width: `${status.progress_percent || 0}%` }} />
                        </div>
                      )}
                      {active && (
                        <div className="diffusion-asset-meta">
                          <span>{status.state === 'verifying' ? '校验中' : '下载中'}</span>
                          <span>{formatBytes(status.present_bytes)} / {formatBytes(status.download_bytes)}</span>
                        </div>
                      )}
                      <div className="diffusion-asset-actions">
                        <label className="asset-license-check">
                          <input
                            type="checkbox"
                            checked={accepted}
                            onChange={event => setAcceptedLicenses(previous => ({
                              ...previous,
                              [asset.asset_id]: event.target.checked,
                            }))}
                            disabled={Boolean(action)}
                          />
                          <span>接受许可</span>
                        </label>
                        {!asset.installed && (
                          <button
                            className="btn-ghost"
                            onClick={() => handleAssetDownload(asset)}
                            disabled={Boolean(action) || !accepted}
                          >
                            {action === `download:${asset.asset_id}` ? '处理中...' : '下载'}
                          </button>
                        )}
                        <button
                          className="btn-ghost"
                          onClick={() => handleAssetImport(asset)}
                          disabled={Boolean(action) || !accepted || !modelPath.trim()}
                        >
                          {action === `import:${asset.asset_id}` ? '校验中...' : '导入'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="path-register-row">
                <input
                  value={modelPath}
                  onChange={event => setModelPath(event.target.value)}
                  placeholder="本机 Diffusers 模型目录"
                  disabled={Boolean(action)}
                  aria-label="本机 Diffusers 模型目录"
                />
                <button className="btn-ghost" onClick={handleInspectAndRegister} disabled={Boolean(action) || !modelPath.trim()}>
                  {action === 'register' ? '检查中...' : '检查并登记'}
                </button>
              </div>
              {inspection && (
                <div className={`artifact-inspection ${inspection.loadable ? 'valid' : 'invalid'}`}>
                  <strong>{inspection.artifact_kind}</strong>
                  <span>{inspection.precision?.toUpperCase() || 'UNKNOWN'}</span>
                  <span>{formatBytes(inspection.size_bytes)}</span>
                </div>
              )}
              <div className="field-row two-columns">
                <label>
                  <span>已登记模型</span>
                  <select
                    value={selectedArtifactId}
                    onChange={event => setSelectedArtifactId(event.target.value)}
                    disabled={Boolean(action) || artifacts.length === 0}
                  >
                    {artifacts.length === 0 && <option value="">尚无可用资产</option>}
                    {artifacts.map(artifact => (
                      <option key={artifact.artifact_id} value={artifact.artifact_id}>
                        {artifact.name} · {artifact.artifact.precision?.toUpperCase() || 'UNKNOWN'}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>加载配置</span>
                  <select value={profile} onChange={event => setProfile(event.target.value)} disabled={Boolean(action)}>
                    {PROFILE_OPTIONS.map(option => (
                      <option key={option.id} value={option.id}>{option.label} · {option.detail}</option>
                    ))}
                  </select>
                </label>
              </div>
              {!dependenciesReady && capabilities && (
                <div className="diffusion-alert warning">
                  <span>
                    CUDA 运行环境缺少 {missingDependencies.join(', ') || '必要依赖'}，当前不能加载图像模型。
                    {capabilities.project_cuda_environment_available
                      ? ' 请用项目 CUDA 虚拟环境重新启动后端。'
                      : ' 项目 CUDA 虚拟环境尚未就绪。'}
                  </span>
                </div>
              )}
              <button
                className="btn-primary diffusion-load-button"
                onClick={handleLoad}
                disabled={Boolean(action) || !selectedArtifactId || !dependenciesReady || (capabilities?.loaded && loadedId === selectedArtifactId)}
              >
                {action === 'load'
                  ? '加载中...'
                  : capabilities?.local_llm_loaded
                    ? '卸载 LLM 并加载图像模型'
                    : capabilities?.loaded && loadedId !== selectedArtifactId
                      ? '切换图像模型'
                      : capabilities?.loaded
                        ? '图像模型已就绪'
                        : '加载图像模型'}
              </button>
            </section>

            <section className="diffusion-section">
              <div className="diffusion-section-heading">
                <h3>生成参数</h3>
                <span>{selectedPreset?.model_id || '自定义'}</span>
              </div>
              <div className="mode-segment" role="group" aria-label="图像生成模式">
                <button
                  type="button"
                  className={editMode === 'txt2img' ? 'active' : ''}
                  onClick={() => setEditMode('txt2img')}
                  disabled={Boolean(action) || jobActive}
                >
                  文生图
                </button>
                <button
                  type="button"
                  className={editMode === 'img2img' ? 'active' : ''}
                  onClick={() => setEditMode('img2img')}
                  disabled={Boolean(action) || jobActive}
                >
                  图生图
                </button>
              </div>
              {editMode === 'img2img' && (
                <div className="edit-source-box">
                  {sourceBlob ? (
                    <div className="edit-source-preview">
                      <img src={sourceBlob.url} alt="图生图源图预览" />
                      <div className="edit-source-meta">
                        <span>{sourceBlob.name}</span>
                        <span>{formatBytes(sourceBlob.sizeBytes)}</span>
                      </div>
                      <button
                        className="btn-ghost"
                        onClick={handleRemoveSource}
                        disabled={Boolean(action) || jobActive}
                      >
                        移除
                      </button>
                    </div>
                  ) : (
                    <label className="edit-source-upload">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleSourceUpload}
                        disabled={Boolean(action) || jobActive || !capabilities?.loaded}
                      />
                      <span>{action === 'upload' ? '上传中...' : '选择源图上传'}</span>
                    </label>
                  )}
                  <label className="stacked-field">
                    <span>Strength（重绘幅度）</span>
                    <input
                      type="number"
                      min="0.05"
                      max="1"
                      step="0.05"
                      value={form.strength}
                      onChange={event => updateForm('strength', event.target.value)}
                      disabled={Boolean(action) || jobActive}
                    />
                  </label>
                </div>
              )}
              <div className="preset-segment" role="group" aria-label="图像生成预设">
                {(capabilities?.presets || []).map(preset => (
                  <button
                    key={preset.preset_id}
                    className={form.presetId === preset.preset_id ? 'active' : ''}
                    onClick={() => handlePresetChange(preset.preset_id)}
                    type="button"
                  >
                    {PRESET_LABELS[preset.preset_id] || preset.preset_id}
                  </button>
                ))}
              </div>
              <label className="stacked-field">
                <span>Prompt</span>
                <textarea value={form.prompt} onChange={event => updateForm('prompt', event.target.value)} rows={4} maxLength={4000} />
              </label>
              <label className="stacked-field">
                <span>Negative prompt</span>
                <textarea value={form.negativePrompt} onChange={event => updateForm('negativePrompt', event.target.value)} rows={2} maxLength={4000} />
              </label>
              <div className="parameter-grid">
                <label><span>Seed</span><input type="number" value={form.seed} onChange={event => updateForm('seed', event.target.value)} /></label>
                <label><span>Steps</span><input type="number" min="1" max="100" value={form.steps} onChange={event => updateForm('steps', event.target.value)} /></label>
                <label><span>CFG</span><input type="number" min="0" max="30" step="0.5" value={form.guidanceScale} onChange={event => updateForm('guidanceScale', event.target.value)} /></label>
                <label><span>宽度</span><input type="number" min="256" max="1024" step="8" value={form.width} onChange={event => updateForm('width', event.target.value)} /></label>
                <label><span>高度</span><input type="number" min="256" max="1024" step="8" value={form.height} onChange={event => updateForm('height', event.target.value)} /></label>
              </div>
              <button
                className="btn-primary generate-button"
                onClick={jobActive ? handleCancel : (editMode === 'img2img' ? handleEdit : handleGenerate)}
                disabled={Boolean(action) || (!jobActive && (!capabilities?.loaded || !form.prompt.trim() || (editMode === 'img2img' && !sourceBlob)))}
              >
                {jobActive
                  ? action === 'cancel' || normalizedJob?.cancel_requested ? '正在取消...' : '停止生成'
                  : action === 'generate' || action === 'edit' ? '提交中...'
                    : editMode === 'img2img' ? '生成编辑图片' : '生成图片'}
              </button>
            </section>
          </div>

          <div className="diffusion-output">
            {normalizedJob && (
              <div className={`generation-status job-${normalizedJob.state}`}>
                <div className="generation-status-row">
                  <strong>{JOB_STATE_LABELS[normalizedJob.state] || normalizedJob.state}</strong>
                  <span>{normalizedJob.progress.step} / {normalizedJob.progress.total} steps</span>
                </div>
                <div className="generation-progress" aria-label={`生成进度 ${normalizedJob.progressPercent}%`}>
                  <span style={{ width: `${normalizedJob.progressPercent}%` }} />
                </div>
              </div>
            )}

            {result ? (
              <figure className="diffusion-result">
                <div className="diffusion-image-frame">
                  <img src={result.url} alt={`Stable Diffusion 生成结果，seed ${result.seed}`} />
                </div>
                <figcaption>
                  <div className="result-facts">
                    <span className={`result-mode mode-${result.mode}`}>
                      {result.mode === 'img2img' ? '图生图' : '文生图'}
                    </span>
                    <span>Seed {result.seed}</span>
                    <span>{result.width} × {result.height}</span>
                    <span>{result.steps} steps</span>
                    {result.strength != null && <span>Strength {result.strength}</span>}
                    <span>{formatSeconds(result.elapsedSeconds)}</span>
                    <span>{formatBytes(result.sizeBytes)}</span>
                    <span>本地</span>
                  </div>
                  <div className="result-actions">
                    <button
                      className="btn-ghost"
                      onClick={handleContinueEdit}
                      disabled={Boolean(action) || jobActive}
                      title="把当前结果作为源图进行图生图"
                    >
                      继续编辑
                    </button>
                    <a className="btn-primary result-download" href={result.url} download={`qlh-sd15-seed-${result.seed}.png`}>
                      ↓ 下载 PNG
                    </a>
                    <button className="btn-danger" onClick={handleDeleteResult}>删除</button>
                  </div>
                </figcaption>
              </figure>
            ) : (
              <div className="diffusion-preview-empty">
                <div className="preview-grid-mark" aria-hidden="true" />
                <strong>{capabilities?.loaded ? '等待生成' : '图像模型未加载'}</strong>
                <span>{capabilities?.loaded_artifact?.name || 'Stable Diffusion 1.5'}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
