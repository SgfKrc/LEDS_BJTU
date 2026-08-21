import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Check,
  CircleAlert,
  Cpu,
  Database,
  Gauge,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Unplug,
  Zap,
} from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { EmptyState } from '../components/EmptyState';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { StatusBadge } from '../components/StatusBadge';
import { pushToast } from '../components/Toast';
import { routeHref } from '../app/routes';
import { useRegisterRefresh } from '../app/refreshBus';
import * as api from '../data/api';
import { fixturesEnabled } from '../data/fixtures';
import {
  useAvailableModels,
  useCurrentModel,
  useLocalModelAssets,
  useModels,
} from '../data/hooks';
import type { CurrentModelResponse, GgufModelRecord, LocalModelAsset, ModelPreflightResponse, ModelSummary } from '../data/types';
import { ModelOrbitCanvas } from '../visual/ModelOrbitCanvas';

function formatBytes(value: unknown): string {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

function modelState(model: ModelSummary): { label: string; tone: 'ok' | 'warn' | 'danger' | 'info' | 'idle' } {
  if (model.is_available) return { label: 'READY', tone: 'ok' };
  if (model.is_experimental) return { label: 'EXPERIMENTAL', tone: 'warn' };
  return { label: 'MISSING', tone: 'idle' };
}

function runtimeState(runtime: CurrentModelResponse | null): { label: string; tone: 'ok' | 'warn' | 'danger' | 'info' | 'idle' } {
  if (!runtime) return { label: 'UNKNOWN', tone: 'idle' };
  if (runtime.loaded) return { label: 'LOADED', tone: 'ok' };
  if (runtime.pipeline_prepared) return { label: 'PREPARED', tone: 'info' };
  return { label: 'UNLOADED', tone: 'idle' };
}

export function ModelsPage() {
  const models = useModels();
  const available = useAvailableModels();
  const current = useCurrentModel();
  const assets = useLocalModelAssets();
  const usingFixtures = fixturesEnabled();
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [selectedEngine, setSelectedEngine] = useState('auto');
  const [selectedQuant, setSelectedQuant] = useState('');
  const [busy, setBusy] = useState('');
  const [preflight, setPreflight] = useState<ModelPreflightResponse | null>(null);
  const [runtimeOverride, setRuntimeOverride] = useState<CurrentModelResponse | null>(null);
  const [ggufModels, setGgufModels] = useState<GgufModelRecord[]>([]);

  const modelList = models.data?.models ?? [];
  const assetList = assets.data?.assets ?? [];
  const engineList = available.data?.available_engines ?? [];
  const runtime = runtimeOverride ?? current.data;
  const filteredModels = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return modelList;
    return modelList.filter((model) => `${model.name} ${model.model_id} ${model.description || ''}`.toLowerCase().includes(normalized));
  }, [modelList, query]);
  const selectedModel = modelList.find((model) => model.model_id === selectedId) ?? filteredModels[0] ?? null;
  const supportedEngines = selectedModel?.supported_engines ?? [];
  const effectiveEngine = selectedEngine === 'auto'
    ? selectedModel?.preferred_engine || available.data?.current_engine || 'auto'
    : selectedEngine;
  const quantOptions = useMemo(() => {
    const values = (selectedModel?.quant_types ?? []).map((value) => String(value));
    if (effectiveEngine === 'llama_cpp') return values.filter((value) => value.toLowerCase().includes('q') || value.toLowerCase() === 'gguf');
    return values.filter((value) => ['fp16', 'int8', 'int4'].includes(value.toLowerCase()));
  }, [effectiveEngine, selectedModel]);
  const effectiveQuant = selectedQuant || selectedModel?.default_quant_type || quantOptions[0] || 'int4';
  const currentState = runtimeState(runtime ?? null);
  const loadedModelId = runtime?.model_id || (runtime as (CurrentModelResponse & { active_model_id?: string | null }) | undefined)?.active_model_id || '';
  const selectedAsset = assetList.find((asset) => asset.model_id === selectedModel?.model_id) ?? null;

  const loadGguf = useCallback(async () => {
    if (usingFixtures) {
      setGgufModels([{ filename: 'qwen2.5-1.5b-q4_k_m.gguf', size_mb: 980, sha256: 'fixture' }]);
      return;
    }
    try { setGgufModels((await api.fetchGgufModels()).models ?? []); } catch { setGgufModels([]); }
  }, [usingFixtures]);

  useEffect(() => {
    if (!selectedId && (models.data?.active_model_id || modelList[0]?.model_id)) {
      setSelectedId(models.data?.active_model_id || modelList[0]?.model_id || '');
    }
  }, [modelList, models.data?.active_model_id, selectedId]);

  useEffect(() => {
    if (!selectedModel) return;
    setSelectedEngine(selectedModel.preferred_engine || 'auto');
    setSelectedQuant(selectedModel.default_quant_type || '');
    setPreflight(null);
  }, [selectedModel?.model_id]);

  useEffect(() => {
    if (!selectedModel || selectedEngine === 'auto' || supportedEngines.length === 0) return;
    if (!supportedEngines.includes(selectedEngine)) setSelectedEngine(selectedModel.preferred_engine || 'auto');
  }, [selectedEngine, selectedModel, supportedEngines]);

  const refresh = useCallback(() => {
    models.refresh();
    available.refresh();
    current.refresh();
    assets.refresh();
    void loadGguf();
  }, [assets.refresh, available.refresh, current.refresh, loadGguf, models.refresh]);

  useEffect(() => { void loadGguf(); }, [loadGguf]);
  useRegisterRefresh(refresh);

  const handleLoad = async () => {
    if (!selectedModel || !selectedModel.is_available || busy) return;
    setBusy('load');
    try {
      if (usingFixtures) {
        setRuntimeOverride({
          loaded: true,
          model_id: selectedModel.model_id,
          model_name: selectedModel.name,
          quant_type: effectiveQuant,
          engine: effectiveEngine,
        });
        pushToast(`Fixture runtime loaded: ${selectedModel.name}`, 'info');
      } else {
        const result = await api.loadModel(effectiveEngine, effectiveQuant, false, selectedModel.model_id);
        setRuntimeOverride({
          ...result,
          loaded: true,
          model_id: result.model_id || selectedModel.model_id,
          model_name: result.model_name || selectedModel.name,
          quant_type: result.quant_type || effectiveQuant,
          engine: result.engine || effectiveEngine,
        });
        await Promise.all([models.refresh(), available.refresh(), current.refresh()]);
        pushToast(`Loaded ${selectedModel.name}`, 'ok');
      }
    } catch (err) {
      pushToast(`Load failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleUnload = async () => {
    if (busy || !runtime?.loaded) return;
    setBusy('unload');
    try {
      if (usingFixtures) {
        setRuntimeOverride({ ...runtime, loaded: false, model_id: null, model_name: '' });
        pushToast('Fixture runtime unloaded', 'info');
      } else {
        const result = await api.unloadModel();
        setRuntimeOverride({ ...result, loaded: false, model_id: null });
        await current.refresh();
        pushToast('Model unloaded', 'ok');
      }
    } catch (err) {
      pushToast(`Unload failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handlePreflight = async (asset: LocalModelAsset) => {
    if (busy) return;
    setSelectedId(asset.model_id);
    setBusy(`preflight:${asset.model_id}`);
    try {
      if (usingFixtures) {
        setPreflight({
          model_id: asset.model_id,
          runtime_profile: asset.runtime_profile,
          read_only: true,
          starts_sidecar: false,
          gate_passed: true,
          status: 'ready',
          errors: [],
        });
        pushToast('Fixture preflight passed', 'info');
      } else {
        const result = await api.preflightLocalModelAsset(asset.model_id);
        setPreflight(result);
        pushToast(result.gate_passed ? 'Preflight passed' : 'Preflight returned warnings', result.gate_passed ? 'ok' : 'info');
      }
    } catch (err) {
      pushToast(`Preflight failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handlePreparePipeline = async () => {
    if (!selectedModel || busy || usingFixtures) {
      if (usingFixtures) pushToast('Pipeline preparation is disabled in fixture mode', 'info');
      return;
    }
    setBusy('prepare-pipeline');
    try {
      const result = await api.prepareModelPipeline(selectedModel.model_id, effectiveQuant || 'fp16');
      setRuntimeOverride((currentRuntime) => ({ ...(currentRuntime ?? { loaded: false }), ...result, pipeline_prepared: true, model_id: selectedModel.model_id }));
      await current.refresh();
      pushToast(`Pipeline prepared for ${selectedModel.name}`, 'ok');
    } catch (err) {
      pushToast(`Pipeline preparation failed: ${api.describeError(err)}`, 'danger');
    } finally { setBusy(''); }
  };

  const handleDownloadGguf = async (file: GgufModelRecord) => {
    if (busy) return;
    setBusy(`download:${file.filename}`);
    try {
      if (usingFixtures) { pushToast(`Fixture download queued: ${file.filename}`, 'info'); return; }
      const blob = await api.downloadGgufModel(file.filename);
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement('a'); anchor.href = href; anchor.download = file.filename; anchor.click(); URL.revokeObjectURL(href);
      pushToast(`Downloaded ${file.filename}`, 'ok');
    } catch (err) { pushToast(`GGUF download failed: ${api.describeError(err)}`, 'danger'); }
    finally { setBusy(''); }
  };

  const modelError = models.state === 'error' ? api.describeError(models.error) : '';

  return (
    <div className="models-page" data-testid="models-page">
      <ModelOrbitCanvas className="models-page__bg" />
      <div className="models-page__content">
        <PageHeader
          tag="MODEL LAB"
          title="Model workspace"
          description="Inspect local model packages, choose a runtime, and keep one explicit loading state."
          actions={
            <>
              <CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={models.refreshing} onClick={refresh}>Refresh</CommandButton>
              <CommandButton variant="ghost" size="sm" href={routeHref('image')} icon={Zap}>Image Studio</CommandButton>
            </>
          }
        />

        <div className="models-layout">
          <aside className="models-rail">
            <section className="model-panel model-panel--catalog">
              <SectionHead title="Model catalog" hint={`${modelList.length} registered`} />
              <label className="model-search">
                <Search size={14} aria-hidden="true" />
                <span className="sr-only">Filter models</span>
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter by name or id" />
              </label>
              {modelError ? <p className="model-inline-error"><CircleAlert size={14} />{modelError}</p> : null}
              {models.state === 'loading' && modelList.length === 0 ? (
                <EmptyState kind="loading" title="Loading catalog" compact />
              ) : filteredModels.length === 0 ? (
                <EmptyState title="No matching models" description="Change the filter or register a local model package." compact />
              ) : (
                <ul className="model-list">
                  {filteredModels.map((model) => {
                    const state = modelState(model);
                    return (
                      <li key={model.model_id} data-selected={model.model_id === selectedModel?.model_id ? 'true' : undefined}>
                        <button type="button" className="model-list__item" onClick={() => setSelectedId(model.model_id)}>
                          <span className="model-list__name">{model.name}</span>
                          <span className="model-list__meta"><span>{model.model_id}</span><StatusBadge label={state.label} tone={state.tone} size="sm" /></span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>

            <section className="model-panel model-panel--assets">
              <SectionHead title="Local assets" hint={`${assetList.length} discovered`} />
              {assetList.length === 0 ? (
                <EmptyState title="No local assets" description="The models directory is empty or unavailable." compact />
              ) : (
                <ul className="asset-list">
                  {assetList.map((asset) => (
                    <li key={asset.model_id} data-selected={asset.model_id === selectedModel?.model_id ? 'true' : undefined}>
                      <button type="button" className="asset-list__item" onClick={() => setSelectedId(asset.model_id)}>
                        <span><strong>{asset.name || asset.model_id}</strong><small>{asset.runtime_profile || 'manual runtime'}</small></span>
                        <span className="cell-mono">{formatBytes(asset.total_bytes)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>

          <main className="models-main">
            <section className="model-panel model-panel--runtime">
              <div className="model-panel__head">
                <div><p className="mono-label">ACTIVE RUNTIME</p><h2>Runtime control</h2></div>
                <StatusBadge label={currentState.label} tone={currentState.tone} pulse={Boolean(runtime?.loaded)} />
              </div>
              <div className="runtime-banner">
                <div className="runtime-banner__icon"><Cpu size={25} aria-hidden="true" /></div>
                <div><strong>{runtime?.loaded ? runtime.model_name || loadedModelId || 'Loaded model' : 'No model loaded'}</strong><span>{runtime?.loaded ? `${runtime.engine || 'auto'} / ${runtime.quant_type || 'default'}` : 'Chat and workflow execution is waiting for a model.'}</span></div>
              </div>
              <div className="runtime-controls">
                <div className="control-group">
                  <span className="control-label">Engine</span>
                  <div className="control-tabs" role="group" aria-label="Runtime engine">
                    <button type="button" className={selectedEngine === 'auto' ? 'is-active' : ''} onClick={() => setSelectedEngine('auto')}>AUTO</button>
                    {engineList.map((engine) => (
                      <button key={engine.id} type="button" className={selectedEngine === engine.id ? 'is-active' : ''} disabled={supportedEngines.length > 0 && !supportedEngines.includes(engine.id)} onClick={() => setSelectedEngine(engine.id)}>{engine.id}</button>
                    ))}
                  </div>
                </div>
                <div className="control-group">
                  <span className="control-label">Quantization</span>
                  <div className="control-tabs" role="group" aria-label="Quantization">
                    {(quantOptions.length ? quantOptions : [effectiveQuant]).map((quant) => (
                      <button key={quant} type="button" className={effectiveQuant === quant ? 'is-active' : ''} onClick={() => setSelectedQuant(quant)}>{quant.toUpperCase()}</button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="runtime-actions">
                <CommandButton icon={Gauge} busy={busy === 'load'} disabled={!selectedModel?.is_available || busy !== ''} onClick={() => void handleLoad()}>Load selected</CommandButton>
                <CommandButton variant="ghost" icon={Database} busy={busy === 'prepare-pipeline'} disabled={!selectedModel || busy !== ''} onClick={() => void handlePreparePipeline()}>Prepare pipeline</CommandButton>
                <CommandButton variant="danger" icon={Unplug} busy={busy === 'unload'} disabled={!runtime?.loaded || busy !== ''} onClick={() => void handleUnload()}>Unload runtime</CommandButton>
              </div>
              {!selectedModel?.is_available && selectedModel ? <p className="model-inline-note"><CircleAlert size={14} />{selectedModel.unavailable_reason || 'This model has no loadable local format.'}</p> : null}
            </section>

            <section className="model-panel model-panel--gguf">
              <SectionHead title="GGUF files" hint={`${ggufModels.length} downloadable`} actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} onClick={() => void loadGguf()}>Refresh</CommandButton>} />
              {ggufModels.length ? <ul className="gguf-list">{ggufModels.map((file) => <li key={file.filename}><div className="asset-list__item"><span><strong>{file.filename}</strong><small>{file.size_mb ? `${file.size_mb} MB` : formatBytes(file.size_bytes)} · {file.sha256 ? `${file.sha256.slice(0, 12)}…` : 'checksum pending'}</small></span><CommandButton variant="ghost" size="sm" busy={busy === `download:${file.filename}`} onClick={() => void handleDownloadGguf(file)}>Download</CommandButton></div></li>)}</ul> : <EmptyState title="No GGUF files advertised" description="The backend exposes local GGUF files here when they are available." compact />}
            </section>

            <section className="model-panel model-panel--engines">
              <SectionHead title="Available runtimes" hint={`${engineList.length} engine${engineList.length === 1 ? '' : 's'} detected`} />
              {engineList.length === 0 ? <EmptyState title="No runtime detected" description="The backend did not advertise a loadable engine." compact /> : (
                <div className="engine-grid">
                  {engineList.map((engine) => <article key={engine.id} className="engine-card"><div className="engine-card__head"><strong>{engine.name || engine.id}</strong><StatusBadge label={engine.id} tone="info" size="sm" /></div><p>{engine.description || 'Runtime description unavailable.'}</p><span className="cell-mono">{engine.requires_cuda ? 'CUDA required' : 'CPU compatible'}</span></article>)}
                </div>
              )}
            </section>
          </main>

          <aside className="models-details">
            <section className="model-panel model-panel--details">
              <SectionHead title="Selected model" hint={selectedModel?.model_id || 'none'} />
              {selectedModel ? (
                <>
                  <h2 className="detail-title">{selectedModel.name}</h2>
                  <p className="detail-description">{selectedModel.description || 'No description supplied by the registry.'}</p>
                  <dl className="model-facts">
                    <div><dt>Format</dt><dd>{(selectedModel.available_formats ?? []).join(' + ') || 'unknown'}</dd></div>
                    <div><dt>Context</dt><dd>{selectedModel.max_context ? `${selectedModel.max_context.toLocaleString()} tokens` : 'unknown'}</dd></div>
                    <div><dt>VRAM target</dt><dd>{selectedModel.recommended_vram_gb ? `${selectedModel.recommended_vram_gb} GB` : 'unknown'}</dd></div>
                    <div><dt>Location</dt><dd>{selectedModel.location || 'unknown'}</dd></div>
                    <div><dt>Expected path</dt><dd>{(selectedModel.expected_paths ?? []).join(', ') || 'not declared'}</dd></div>
                  </dl>
                </>
              ) : <EmptyState title="Select a model" compact />}
            </section>

            <section className="model-panel model-panel--preflight">
              <SectionHead title="Asset preflight" hint="read-only" />
              {selectedAsset ? <>
                <div className="preflight-target"><Database size={15} aria-hidden="true" /><span>{selectedAsset.name || selectedAsset.model_id}</span></div>
                <p className="detail-description">{selectedAsset.runtime_hint || 'Check package integrity and runtime eligibility without starting a sidecar.'}</p>
                <CommandButton variant="ghost" size="sm" icon={ShieldCheck} busy={busy.startsWith('preflight:')} onClick={() => void handlePreflight(selectedAsset)}>Run preflight</CommandButton>
                {preflight ? <div className={`preflight-result ${preflight.gate_passed ? 'is-ok' : 'is-warn'}`}><div><StatusBadge label={preflight.status || 'reported'} tone={preflight.gate_passed ? 'ok' : 'warn'} size="sm" /><span>{preflight.read_only ? 'No sidecar started' : 'Runtime action may be required'}</span></div>{preflight.errors?.length ? <ul>{preflight.errors.map((error, index) => <li key={`${error.code || 'error'}-${index}`}>{error.message || error.code || 'Unknown preflight error'}</li>)}</ul> : <p><Check size={14} /> Gate passed</p>}</div> : null}
              </> : <EmptyState title="No matching asset" description="Select a discovered local asset to inspect its runtime gate." compact />}
            </section>
          </aside>
        </div>
      </div>
      {busy === 'load' || busy === 'unload' ? <div className="models-busy" role="status"><LoaderCircle size={15} className="spin" /> Updating runtime</div> : null}
    </div>
  );
}
