import { useCallback, useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
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
  useMyRole,
  useModels,
} from '../data/hooks';
import type {
  CurrentModelResponse,
  DeploymentSimulationPlan,
  GgufModelRecord,
  LocalModelAsset,
  ModelDownloadManifest,
  ModelPipelineAssignmentResponse,
  ModelPreflightResponse,
  ModelSummary,
} from '../data/types';
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
  const role = useMyRole();
  const usingFixtures = fixturesEnabled();
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [selectedEngine, setSelectedEngine] = useState('auto');
  const [selectedQuant, setSelectedQuant] = useState('');
  const [busy, setBusy] = useState('');
  const [preflight, setPreflight] = useState<ModelPreflightResponse | null>(null);
  const [runtimeOverride, setRuntimeOverride] = useState<CurrentModelResponse | null>(null);
  const [ggufModels, setGgufModels] = useState<GgufModelRecord[]>([]);
  const [registryModels, setRegistryModels] = useState<ModelSummary[]>([]);
  const [manifest, setManifest] = useState<ModelDownloadManifest | null>(null);
  const [assignment, setAssignment] = useState<ModelPipelineAssignmentResponse | null>(null);
  const [simulationPlan, setSimulationPlan] = useState<DeploymentSimulationPlan | null>(null);
  const [governanceError, setGovernanceError] = useState('');
  const [registryForm, setRegistryForm] = useState({
    model_id: '', name: '', model_type: 'safetensors' as 'safetensors' | 'gguf' | 'both',
    model_path: '', gguf_path: '', description: '',
  });

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
  const canManageRegistry = usingFixtures || role.data?.is_master === true;
  const rawArtifactId = String(manifest?.sha256 || manifest?.artifact_sha256 || '');
  const artifactId = rawArtifactId && !rawArtifactId.toLowerCase().startsWith('sha256:') ? `sha256:${rawArtifactId}` : rawArtifactId;
  const runtimeProfile = String(preflight?.runtime_profile || selectedAsset?.runtime_profile || selectedModel?.preferred_engine || 'manual');
  const assignmentRows = (assignment?.assignments || assignment?.segments || []) as Array<Record<string, unknown>>;
  const simulationNodes = assignmentRows
    .map((row) => ({
      node_id: String(row.node_id || ''),
      artifact_ids: Array.isArray(row.artifact_ids) ? row.artifact_ids.map(String) : (artifactId ? [artifactId] : []),
      capabilities: Array.isArray(row.capabilities) ? row.capabilities.map(String) : [],
      runtime_fingerprint: String(row.runtime_fingerprint || ''),
      available: row.available !== false,
    }))
    .filter((node) => node.node_id && node.artifact_ids.length > 0 && /^sha256:[0-9a-f]{64}$/i.test(node.runtime_fingerprint));

  const fixtureDigest = `sha256:${'a'.repeat(64)}`;
  const useFixtureGovernance = (modelId: string) => {
    setManifest({
      model_id: modelId,
      sha256: fixtureDigest,
      total_layers: 24,
      count: 2,
      files: [
        { path: 'config.json', size_bytes: 8192, sha256: `sha256:${'b'.repeat(64)}` },
        { path: 'model-00001.safetensors', size_bytes: 1_900_000_000, sha256: `sha256:${'c'.repeat(64)}` },
      ],
    });
    setAssignment({
      model_id: modelId,
      assignments: [
        { node_id: 'master', artifact_ids: [fixtureDigest], capabilities: ['cpu', 'llama_cpp'], runtime_fingerprint: `sha256:${'d'.repeat(64)}`, available: true, layer_range: [0, 12] },
        { node_id: 'worker-tablet', artifact_ids: [fixtureDigest], capabilities: ['cpu'], runtime_fingerprint: `sha256:${'e'.repeat(64)}`, available: true, layer_range: [13, 23] },
      ],
    });
  };

  const loadGguf = useCallback(async () => {
    if (usingFixtures) {
      setGgufModels([{ filename: 'qwen2.5-1.5b-q4_k_m.gguf', size_mb: 980, sha256: 'fixture' }]);
      return;
    }
    try { setGgufModels((await api.fetchGgufModels()).models ?? []); } catch { setGgufModels([]); }
  }, [usingFixtures]);

  const loadRegistry = useCallback(async () => {
    if (usingFixtures) {
      setRegistryModels(modelList);
      return;
    }
    try {
      setRegistryModels((await api.fetchModelRegistry()).models ?? []);
    } catch (err) {
      setGovernanceError(`Registry unavailable: ${api.describeError(err)}`);
    }
  }, [modelList, usingFixtures]);

  const loadAssetGovernance = useCallback(async () => {
    if (!selectedModel || busy) return;
    setBusy('governance');
    setGovernanceError('');
    try {
      if (usingFixtures) {
        useFixtureGovernance(selectedModel.model_id);
        pushToast('Fixture asset contract loaded', 'info');
      } else {
        const [nextManifest, nextAssignment] = await Promise.all([
          api.fetchDownloadableModelManifest(selectedModel.model_id),
          api.fetchModelPipelineAssignment(selectedModel.model_id),
        ]);
        setManifest(nextManifest);
        setAssignment(nextAssignment);
        pushToast('Asset contract refreshed', 'ok');
      }
    } catch (err) {
      setGovernanceError(`Asset contract unavailable: ${api.describeError(err)}`);
    } finally {
      setBusy('');
    }
  }, [busy, selectedModel, usingFixtures]);

  const handleDownloadManagedFile = async (filePath: string) => {
    if (!selectedModel || busy) return;
    setBusy(`file:${filePath}`);
    try {
      if (usingFixtures) {
        pushToast(`Fixture file queued: ${filePath}`, 'info');
        return;
      }
      const blob = await api.downloadModelFile(selectedModel.model_id, filePath);
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = href;
      anchor.download = filePath.split('/').pop() || filePath;
      anchor.click();
      URL.revokeObjectURL(href);
      pushToast(`Downloaded ${filePath}`, 'ok');
    } catch (err) {
      pushToast(`Asset file download failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleRegisterModel = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canManageRegistry || busy || !registryForm.model_id.trim() || !registryForm.name.trim()) return;
    setBusy('register');
    try {
      const payload = {
        ...registryForm,
        model_id: registryForm.model_id.trim(),
        name: registryForm.name.trim(),
        model_path: registryForm.model_path.trim(),
        gguf_path: registryForm.gguf_path.trim(),
        description: registryForm.description.trim(),
      };
      if (usingFixtures) {
        setRegistryModels((currentRegistry) => [
          ...currentRegistry.filter((model) => model.model_id !== payload.model_id),
          { ...payload, is_builtin: false, is_experimental: true, location: 'local registry', is_available: false },
        ]);
      } else {
        await api.registerModel(payload);
        await loadRegistry();
      }
      setRegistryForm({ model_id: '', name: '', model_type: 'safetensors', model_path: '', gguf_path: '', description: '' });
      pushToast(`Registered ${payload.name}`, 'ok');
    } catch (err) {
      pushToast(`Register failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleUnregisterModel = async (model: ModelSummary) => {
    if (!canManageRegistry || busy || model.is_builtin) return;
    if (!usingFixtures && !window.confirm(`Unregister ${model.name}? Local files will not be deleted.`)) return;
    setBusy(`unregister:${model.model_id}`);
    try {
      if (usingFixtures) setRegistryModels((currentRegistry) => currentRegistry.filter((item) => item.model_id !== model.model_id));
      else { await api.unregisterModel(model.model_id); await loadRegistry(); }
      pushToast(`Unregistered ${model.name}`, 'ok');
    } catch (err) {
      pushToast(`Unregister failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleSimulation = async (action: 'create' | 'prepare' | 'activate' | 'rollback') => {
    if (!selectedModel || busy) return;
    if (action === 'create' && (!/^sha256:[0-9a-f]{64}$/i.test(artifactId) || simulationNodes.length === 0) && !usingFixtures) {
      setGovernanceError('Deployment simulation requires a sha256 artifact and an active, fingerprinted node assignment.');
      return;
    }
    if ((action === 'activate' || action === 'rollback') && !canManageRegistry) return;
    setBusy(`simulation:${action}`);
    setGovernanceError('');
    try {
      if (usingFixtures) {
        setSimulationPlan((currentPlan) => {
          const base = currentPlan ?? {
            plan_id: 'fixture-plan-01', artifact_id: fixtureDigest, runtime_profile: 'qwen3_sidecar',
            required_capabilities: ['cpu'], target_nodes: ['master', 'worker-tablet'], actual_nodes: [], status: 'planned', records: [],
          };
          const status = action === 'create' ? 'planned' : action === 'prepare' ? 'ready' : action === 'activate' ? 'active' : 'rolled_back';
          return { ...base, status, actual_nodes: action === 'prepare' || action === 'activate' ? ['master', 'worker-tablet'] : base.actual_nodes, updated_at: new Date().toISOString() };
        });
      } else {
        const result = action === 'create'
          ? await api.createDeploymentSimulation({ artifact_id: artifactId, runtime_profile: runtimeProfile, required_capabilities: [], nodes: simulationNodes })
          : action === 'prepare' && simulationPlan
            ? await api.prepareDeploymentSimulation(simulationPlan.plan_id, simulationNodes)
            : action === 'activate' && simulationPlan
              ? await api.activateDeploymentSimulation(simulationPlan.plan_id)
              : simulationPlan
                ? await api.rollbackDeploymentSimulation(simulationPlan.plan_id)
                : null;
        if (result?.plan) setSimulationPlan(result.plan);
      }
      pushToast(`Simulation ${action} complete`, 'ok');
    } catch (err) {
      setGovernanceError(`Simulation ${action} failed: ${api.describeError(err)}`);
    } finally {
      setBusy('');
    }
  };

  useEffect(() => {
    if (!selectedId && (models.data?.active_model_id || modelList[0]?.model_id)) {
      setSelectedId(models.data?.active_model_id || modelList[0]?.model_id || '');
    }
  }, [modelList, models.data?.active_model_id, selectedId]);

  useEffect(() => { void loadRegistry(); }, [loadRegistry]);

  useEffect(() => {
    if (!selectedModel) return;
    setManifest(null);
    setAssignment(null);
    setSimulationPlan(null);
    if (usingFixtures) useFixtureGovernance(selectedModel.model_id);
  }, [selectedModel?.model_id, usingFixtures]);

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

            <section className="model-panel model-panel--governance" data-testid="model-governance">
              <div className="model-panel__head">
                <div><p className="mono-label">ASSET GOVERNANCE</p><h2>Integrity and deployment contract</h2></div>
                <CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={busy === 'governance'} disabled={!selectedModel || busy !== ''} onClick={() => void loadAssetGovernance()}>Refresh contract</CommandButton>
              </div>
              {governanceError ? <p className="model-inline-error"><CircleAlert size={14} />{governanceError}</p> : null}
              <div className="governance-grid">
                <section className="governance-block">
                  <div className="governance-block__head"><strong>Download manifest</strong><StatusBadge label={manifest ? 'AVAILABLE' : 'NOT LOADED'} tone={manifest ? 'ok' : 'idle'} size="sm" /></div>
                  {manifest ? <>
                    <dl className="governance-facts">
                      <div><dt>Artifact</dt><dd className="cell-mono">{artifactId ? `${artifactId.slice(0, 20)}...` : 'not declared'}</dd></div>
                      <div><dt>Layers</dt><dd>{manifest.total_layers || 'unknown'}</dd></div>
                      <div><dt>Files</dt><dd>{manifest.files?.length ?? manifest.count ?? 0}</dd></div>
                    </dl>
                    <ul className="manifest-list">
                      {(manifest.files ?? []).map((file) => <li key={String(file.path)}><span><strong>{file.path}</strong><small>{formatBytes(file.size_bytes)} · {file.sha256 ? `${file.sha256.slice(0, 12)}...` : 'checksum pending'}</small></span><CommandButton variant="ghost" size="sm" busy={busy === `file:${file.path}`} disabled={!file.path || busy !== ''} onClick={() => void handleDownloadManagedFile(String(file.path))}>Download</CommandButton></li>)}
                    </ul>
                  </> : <EmptyState title="Manifest not loaded" description="Refresh the contract after an active model is selected." compact />}
                </section>

                <section className="governance-block">
                  <div className="governance-block__head"><strong>Pipeline assignment</strong><StatusBadge label={assignmentRows.length ? `${assignmentRows.length} NODES` : 'NO ACTIVE PLAN'} tone={assignmentRows.length ? 'info' : 'idle'} size="sm" /></div>
                  {assignmentRows.length ? <ul className="assignment-list">{assignmentRows.map((row, index) => <li key={`${String(row.node_id)}-${index}`}><span><strong>{String(row.node_id || 'unknown node')}</strong><small>{Array.isArray(row.layer_range) ? `layers ${row.layer_range.join('-')}` : 'layer range pending'}</small></span><span className="cell-mono">{row.available === false ? 'UNAVAILABLE' : 'READY'}</span></li>)}</ul> : <EmptyState title="No active assignment" description="The backend only exposes a generation-bound assignment during an active prepare transaction." compact />}
                  <div className="contract-note"><ShieldCheck size={14} /><span>Runtime profile: <strong>{runtimeProfile}</strong>. This view is read-only until a deployment simulation is created.</span></div>
                </section>
              </div>

              <div className="simulation-strip">
                <div><p className="mono-label">DEPLOYMENT SIMULATION</p><strong>{simulationPlan ? `Plan ${simulationPlan.plan_id}` : 'No simulation plan'}</strong><span>{simulationPlan ? `${simulationPlan.status || 'planned'} · ${simulationPlan.target_nodes?.length || 0} target nodes` : 'Validate artifact and node fingerprints before activation.'}</span></div>
                <div className="simulation-actions">
                  <CommandButton icon={Gauge} busy={busy === 'simulation:create'} disabled={!selectedModel || busy !== '' || (!usingFixtures && (!artifactId || simulationNodes.length === 0))} onClick={() => void handleSimulation('create')}>Simulate deployment</CommandButton>
                  <CommandButton variant="ghost" busy={busy === 'simulation:prepare'} disabled={!simulationPlan || busy !== ''} onClick={() => void handleSimulation('prepare')}>Prepare</CommandButton>
                  <CommandButton variant="ghost" busy={busy === 'simulation:activate'} disabled={!simulationPlan || simulationPlan.status !== 'ready' || !canManageRegistry || busy !== ''} onClick={() => void handleSimulation('activate')}>Activate</CommandButton>
                  <CommandButton variant="danger" busy={busy === 'simulation:rollback'} disabled={!simulationPlan || !['active', 'partial'].includes(simulationPlan.status || '') || !canManageRegistry || busy !== ''} onClick={() => void handleSimulation('rollback')}>Rollback</CommandButton>
                </div>
              </div>
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

            <section className="model-panel model-panel--registry">
              <SectionHead title="Local registry" hint={`${registryModels.length} entries`} />
              {registryModels.length ? <ul className="registry-list">{registryModels.map((model) => <li key={model.model_id}><span><strong>{model.name}</strong><small>{model.model_id} · {model.is_builtin ? 'built-in' : 'custom'}</small></span><CommandButton variant="ghost" size="sm" disabled={!canManageRegistry || Boolean(model.is_builtin) || busy !== ''} busy={busy === `unregister:${model.model_id}`} onClick={() => void handleUnregisterModel(model)}>Remove</CommandButton></li>)}</ul> : <EmptyState title="Registry unavailable" compact />}
              <details className="registry-editor">
                <summary>Register local package</summary>
                <form onSubmit={(event) => void handleRegisterModel(event)}>
                  <label>Model ID<input value={registryForm.model_id} onChange={(event) => setRegistryForm((form) => ({ ...form, model_id: event.target.value }))} placeholder="custom-model-id" required /></label>
                  <label>Name<input value={registryForm.name} onChange={(event) => setRegistryForm((form) => ({ ...form, name: event.target.value }))} placeholder="Display name" required /></label>
                  <label>Format<select value={registryForm.model_type} onChange={(event) => setRegistryForm((form) => ({ ...form, model_type: event.target.value as typeof form.model_type }))}><option value="safetensors">Safetensors</option><option value="gguf">GGUF</option><option value="both">Both</option></select></label>
                  <label>Safetensors path<input value={registryForm.model_path} onChange={(event) => setRegistryForm((form) => ({ ...form, model_path: event.target.value }))} placeholder="models/custom" /></label>
                  <label>GGUF path<input value={registryForm.gguf_path} onChange={(event) => setRegistryForm((form) => ({ ...form, gguf_path: event.target.value }))} placeholder="models/custom/model.gguf" /></label>
                  <label>Description<textarea value={registryForm.description} onChange={(event) => setRegistryForm((form) => ({ ...form, description: event.target.value }))} rows={2} /></label>
                  <CommandButton type="submit" icon={Database} busy={busy === 'register'} disabled={!canManageRegistry || busy !== ''}>Register</CommandButton>
                  {!canManageRegistry ? <p className="model-inline-note"><CircleAlert size={14} />Only the primary node can mutate its local registry.</p> : null}
                </form>
              </details>
            </section>
          </aside>
        </div>
      </div>
      {busy === 'load' || busy === 'unload' ? <div className="models-busy" role="status"><LoaderCircle size={15} className="spin" /> Updating runtime</div> : null}
    </div>
  );
}
