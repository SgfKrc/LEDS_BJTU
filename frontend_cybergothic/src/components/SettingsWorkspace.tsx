import { useCallback, useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Check,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Zap,
} from 'lucide-react';
import { CommandButton } from './CommandButton';
import { EmptyState } from './EmptyState';
import { SectionHead } from './PageHeader';
import { StatusBadge } from './StatusBadge';
import { pushToast } from './Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import * as api from '../data/api';
import {
  deviceProfileFixture,
  fixturesEnabled,
  ragAnnDecisionFixture,
  ragCapacityFixture,
  ragEmbeddingJobFixture,
  ragFixture,
  ragSearchFixture,
} from '../data/fixtures';
import { useDeviceProfile, useRagHealth, useRagSources } from '../data/hooks';
import type {
  DeviceProfileResponse,
  RagAnnDecisionResponse,
  RagCapacityResponse,
  RagEmbeddingJob,
  RagHealthResponse,
  RagSearchResult,
  RagSource,
} from '../data/types';

export type SettingsWorkspaceTab = 'device' | 'rag';

interface SettingsWorkspaceProps {
  /** Controlled mode lets SettingsPage own the navigation while preserving standalone use. */
  tab?: SettingsWorkspaceTab;
  onTabChange?: (tab: SettingsWorkspaceTab) => void;
  showNavigation?: boolean;
}

function count(value: unknown): string {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toLocaleString() : '0';
}

function formatBytes(value: unknown): string {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size >= 10 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

function scoreWidth(value: unknown, max: number): string {
  const parsed = Number(value || 0);
  return `${Math.max(0, Math.min(100, (parsed / max) * 100))}%`;
}

export function SettingsWorkspace({ tab, onTabChange, showNavigation = true }: SettingsWorkspaceProps) {
  const device = useDeviceProfile();
  const rag = useRagHealth();
  const sources = useRagSources();
  const usingFixtures = fixturesEnabled();
  const [localTab, setLocalTab] = useState<SettingsWorkspaceTab>('device');
  const [deviceBusy, setDeviceBusy] = useState('');
  const [deviceOverride, setDeviceOverride] = useState<DeviceProfileResponse | null>(null);
  const [ragOverride, setRagOverride] = useState<RagHealthResponse | null>(null);
  const [ragQuery, setRagQuery] = useState('');
  const [ragResults, setRagResults] = useState<RagSearchResult[]>([]);
  const [ragError, setRagError] = useState('');
  const [ragBusy, setRagBusy] = useState('');
  const [sourceOverride, setSourceOverride] = useState<RagSource[] | null>(null);
  const [ragCapacity, setRagCapacity] = useState<RagCapacityResponse | null>(null);
  const [ragAnnDecision, setRagAnnDecision] = useState<RagAnnDecisionResponse | null>(null);
  const [ragDimensions, setRagDimensions] = useState(768);
  const [ragScanBudget, setRagScanBudget] = useState(1024);
  const [embeddingJob, setEmbeddingJob] = useState<RagEmbeddingJob | null>(null);
  const [embeddingForm, setEmbeddingForm] = useState({
    model_id: 'nomic-embed-text:latest', model_sha256: '', source_id: '', batch_size: 16,
  });

  const profile = deviceOverride ?? device.data;
  const health = ragOverride ?? rag.data;
  const sourceList = sourceOverride ?? sources.data?.sources ?? [];
  const gpuList = profile?.gpus ?? (profile?.gpu ? [profile.gpu] : []);
  const selectedGpuIndex = profile?.selected_gpu_index ?? 0;
  const runtimeHealth = rag.state === 'error' ? 'danger' : health?.status === 'ok' ? 'ok' : 'warn';
  const activeTab = tab ?? localTab;

  const refreshRagGovernance = useCallback(async () => {
    setRagBusy('governance');
    setRagError('');
    try {
      if (usingFixtures) {
        setRagCapacity({ ...ragCapacityFixture, dimensions: ragDimensions });
        setRagAnnDecision({ ...ragAnnDecisionFixture, decision: { ...ragAnnDecisionFixture.decision, scan_budget: ragScanBudget } });
      } else {
        const [capacityResult, annResult] = await Promise.all([
          api.fetchRagCapacity(ragDimensions),
          api.fetchRagAnnDecision(ragScanBudget),
        ]);
        setRagCapacity(capacityResult);
        setRagAnnDecision(annResult);
      }
    } catch (err) {
      setRagError(`RAG governance unavailable: ${api.describeError(err)}`);
    } finally {
      setRagBusy('');
    }
  }, [ragDimensions, ragScanBudget, usingFixtures]);

  const refresh = useCallback(() => {
    device.refresh();
    rag.refresh();
    sources.refresh();
  }, [device.refresh, rag.refresh, sources.refresh]);
  useRegisterRefresh(refresh);

  useEffect(() => {
    if (activeTab !== 'rag') return;
    void refreshRagGovernance();
  }, [activeTab, refreshRagGovernance]);

  const activeGpu = gpuList[selectedGpuIndex] ?? gpuList[0];
  const selectTab = useCallback((next: SettingsWorkspaceTab) => {
    if (tab === undefined) setLocalTab(next);
    onTabChange?.(next);
  }, [onTabChange, tab]);
  const scoreRows = useMemo(() => [
    ['GPU', profile?.score_breakdown?.gpu ?? 0, 50],
    ['RAM', profile?.score_breakdown?.ram ?? 0, 30],
    ['CPU', profile?.score_breakdown?.cpu ?? 0, 20],
  ] as Array<[string, number, number]>, [profile]);

  const handleAutoConfigure = async () => {
    if (deviceBusy) return;
    setDeviceBusy('auto');
    try {
      if (usingFixtures) {
        pushToast('Fixture recommended configuration applied', 'info');
      } else {
        const result = await api.autoConfigureDevice();
        await device.refresh();
        pushToast(`Applied ${result.tier || 'recommended'} configuration`, 'ok');
      }
    } catch (err) {
      pushToast(`Device configuration failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setDeviceBusy('');
    }
  };

  const handleSelectGpu = async (index: number) => {
    if (deviceBusy || index === selectedGpuIndex) return;
    setDeviceBusy(`gpu:${index}`);
    try {
      if (usingFixtures) {
        setDeviceOverride({ ...(profile ?? deviceProfileFixture), selected_gpu_index: index, gpu: gpuList[index] });
        pushToast(`Fixture GPU selected: ${gpuList[index]?.name || index}`, 'info');
      } else {
        const result = await api.selectGpu(index);
        setDeviceOverride(result);
        await device.refresh();
        pushToast(`GPU selected: ${result.selected_gpu?.name || index}`, 'ok');
      }
    } catch (err) {
      pushToast(`GPU switch failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setDeviceBusy('');
    }
  };

  const handleRagSearch = async (event: FormEvent) => {
    event.preventDefault();
    const query = ragQuery.trim();
    if (!query || ragBusy) return;
    setRagBusy('search');
    setRagError('');
    try {
      if (usingFixtures) {
        setRagResults(ragSearchFixture.results.filter((item) => `${item.source_id} ${item.snippet}`.toLowerCase().includes(query.toLowerCase())));
      } else {
        const result = await api.searchRag(query, { mode: 'fts', access_scope: 'owner', limit: 20 });
        setRagResults(result.results ?? []);
      }
    } catch (err) {
      const message = api.describeError(err);
      setRagError(message);
      pushToast(`RAG search failed: ${message}`, 'danger');
    } finally {
      setRagBusy('');
    }
  };

  const handleCreateEmbeddingJob = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (ragBusy || !/^[0-9a-f]{64}$/i.test(embeddingForm.model_sha256.trim())) {
      setRagError('Embedding job requires a 64-character model SHA256 digest.');
      return;
    }
    setRagBusy('embedding:create');
    setRagError('');
    try {
      if (usingFixtures) {
        setEmbeddingJob({
          ...ragEmbeddingJobFixture,
          cursor: { ...ragEmbeddingJobFixture.cursor, batch_size: embeddingForm.batch_size, model_id: embeddingForm.model_id },
        });
      } else {
        setEmbeddingJob(await api.createRagEmbeddingJob({
          model_id: embeddingForm.model_id,
          model_sha256: embeddingForm.model_sha256.trim(),
          ...(embeddingForm.source_id.trim() ? { source_id: embeddingForm.source_id.trim() } : {}),
          batch_size: embeddingForm.batch_size,
        }));
      }
      pushToast('Embedding job created', usingFixtures ? 'info' : 'ok');
    } catch (err) {
      setRagError(`Embedding job creation failed: ${api.describeError(err)}`);
    } finally {
      setRagBusy('');
    }
  };

  const handleEmbeddingJobAction = async (action: 'refresh' | 'run' | 'cancel') => {
    if (!embeddingJob || ragBusy) return;
    setRagBusy(`embedding:${action}`);
    setRagError('');
    try {
      if (usingFixtures) {
        if (action === 'cancel') {
          setEmbeddingJob((job) => job ? { ...job, state: 'cancelled', lease_active: false } : job);
        } else if (action === 'run') {
          setEmbeddingJob((job) => job ? { ...job, state: 'running', lease_active: true, attempts: (job.attempts ?? 0) + 1, cursor: { ...job.cursor, indexed: Math.min(Number(job.cursor?.total ?? 0), Number(job.cursor?.indexed ?? 0) + Number(job.cursor?.batch_size ?? 16)), position: Math.min(Number(job.cursor?.total ?? 0), Number(job.cursor?.position ?? 0) + Number(job.cursor?.batch_size ?? 16)) } } : job);
        }
      } else if (action === 'refresh') {
        setEmbeddingJob(await api.fetchRagEmbeddingJob(embeddingJob.job_id));
      } else if (action === 'run') {
        setEmbeddingJob(await api.runRagEmbeddingJob(embeddingJob.job_id, { model_id: embeddingForm.model_id, expected_dimensions: ragDimensions, max_batches: 1 }));
      } else {
        setEmbeddingJob(await api.cancelRagEmbeddingJob(embeddingJob.job_id));
      }
      pushToast(`Embedding job ${action} complete`, usingFixtures ? 'info' : 'ok');
    } catch (err) {
      setRagError(`Embedding job ${action} failed: ${api.describeError(err)}`);
    } finally {
      setRagBusy('');
    }
  };

  const handleRagRebuild = async () => {
    if (ragBusy) return;
    setRagBusy('rebuild');
    setRagError('');
    try {
      if (usingFixtures) {
        setRagOverride({ ...(health ?? ragFixture), status: 'ok', fts_chunk_count: (health?.fts_chunk_count ?? ragFixture.fts_chunk_count) + 12 });
        pushToast('Fixture FTS index rebuilt', 'info');
      } else {
        await api.rebuildRagIndex();
        await rag.refresh();
        pushToast('RAG FTS index rebuilt', 'ok');
      }
    } catch (err) {
      const message = api.describeError(err);
      setRagError(message);
      pushToast(`RAG rebuild failed: ${message}`, 'danger');
    } finally {
      setRagBusy('');
    }
  };

  const handleDeleteSource = async (source: RagSource) => {
    if (ragBusy || !window.confirm(`Delete indexed source ${source.display_name || source.source_id}?`)) return;
    setRagBusy(`delete:${source.source_id}`);
    try {
      if (usingFixtures) {
        setSourceOverride(sourceList.filter((item) => item.source_id !== source.source_id));
      } else {
        await api.deleteRagSource(source.source_id);
        await sources.refresh();
        await rag.refresh();
      }
      pushToast('Indexed source removed', usingFixtures ? 'info' : 'ok');
    } catch (err) {
      pushToast(`Source removal failed: ${api.describeError(err)}`, 'danger');
    } finally {
      setRagBusy('');
    }
  };

  return (
    <section className={`settings-workspace${showNavigation ? '' : ' settings-workspace--embedded'}`} data-testid="settings-workspace">
      {showNavigation ? <aside className="settings-workspace__nav" aria-label="Runtime settings sections">
        <p className="mono-label">RUNTIME WORKSPACE</p>
        <button type="button" className={activeTab === 'device' ? 'is-active' : ''} onClick={() => selectTab('device')}>
          <Cpu size={16} aria-hidden="true" /><span>Device profile</span><StatusBadge label={profile ? 'READY' : 'WAIT'} tone={profile ? 'ok' : 'idle'} size="sm" />
        </button>
        <button type="button" className={activeTab === 'rag' ? 'is-active' : ''} onClick={() => selectTab('rag')}>
          <Database size={16} aria-hidden="true" /><span>Local RAG</span><StatusBadge label={health?.status?.toUpperCase() || 'WAIT'} tone={runtimeHealth} size="sm" />
        </button>
        <a href="#settings-preferences"><Sparkles size={16} aria-hidden="true" /><span>Console preferences</span></a>
      </aside> : null}

      <div className="settings-workspace__main">
        {activeTab === 'device' ? (
          <div className="settings-tool" data-testid="device-workspace">
            <div className="settings-tool__head">
              <SectionHead title="Device profile" hint="Hardware detection and runtime recommendations" actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={device.refreshing} onClick={refresh}>Refresh</CommandButton>} />
            </div>
            {device.state === 'loading' && !profile ? <EmptyState kind="loading" title="Detecting device" compact /> : profile ? <>
              <div className="device-summary">
                <div className="device-summary__tier"><span className="mono-label">DETECTED TIER</span><strong>{profile.tier_label || profile.tier || 'Unknown'}</strong><StatusBadge label={`${profile.score_total ?? 0}/100`} tone="info" size="sm" /></div>
                <div className="device-summary__gpu"><Gauge size={17} aria-hidden="true" /><span>{activeGpu?.name || profile.gpu?.name || 'No GPU detected'}</span><small>{activeGpu?.vram_total_gb ? `${activeGpu.vram_total_gb} GB VRAM` : 'Shared memory'}</small></div>
              </div>
              <div className="device-score-grid">
                {scoreRows.map(([label, value, max]) => <div className="device-score" key={label}><div><span>{label}</span><strong>{value}</strong></div><div className="device-score__track"><span style={{ width: scoreWidth(value, max) }} /></div></div>)}
              </div>
              <div className="device-metrics">
                <div><Cpu size={15} aria-hidden="true" /><span><strong>{profile.cpu?.physical_cores ?? '?'} cores</strong><small>{profile.cpu?.model_name || 'CPU unavailable'}</small></span></div>
                <div><HardDrive size={15} aria-hidden="true" /><span><strong>{profile.ram?.total_gb ?? '?'} GB RAM</strong><small>{profile.ram?.available_gb ?? '?'} GB available</small></span></div>
                <div><Zap size={15} aria-hidden="true" /><span><strong>{activeGpu?.cuda_available ? 'CUDA ready' : 'CPU mode'}</strong><small>{activeGpu?.vram_free_gb ? `${activeGpu.vram_free_gb} GB free` : 'No dedicated VRAM'}</small></span></div>
              </div>
              {gpuList.length > 1 ? <div className="gpu-picker"><span className="control-label">Inference GPU</span><div>{gpuList.map((gpu, index) => <button key={`${gpu.name || 'gpu'}-${index}`} type="button" className={index === selectedGpuIndex ? 'is-active' : ''} disabled={deviceBusy !== ''} onClick={() => void handleSelectGpu(index)}><span>{gpu.name || `GPU ${index}`}</span><small>{gpu.is_integrated ? 'Integrated' : gpu.cuda_available ? 'CUDA' : 'Dedicated'}</small>{index === selectedGpuIndex ? <Check size={14} aria-hidden="true" /> : null}</button>)}</div></div> : null}
              {profile.warnings?.length ? <div className="settings-notice settings-notice--warn"><ShieldAlert size={15} aria-hidden="true" /><span>{profile.warnings[0]}</span></div> : null}
              <div className="settings-tool__actions"><CommandButton icon={Sparkles} busy={deviceBusy === 'auto'} onClick={() => void handleAutoConfigure()}>Apply recommended configuration</CommandButton></div>
            </> : <EmptyState kind="error" title="Device profile unavailable" detail={device.error} errorKind={device.errorKind} errorStatus={device.errorStatus} compact action={<CommandButton variant="ghost" size="sm" onClick={refresh}>Retry</CommandButton>} />}
          </div>
        ) : (
          <div className="settings-tool" data-testid="rag-workspace">
            <div className="settings-tool__head"><SectionHead title="Local RAG" hint="Search and maintain the local knowledge index" actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={rag.refreshing} onClick={() => { rag.refresh(); setRagError(''); }}>Refresh</CommandButton>} /></div>
            <div className="rag-health-strip" role="status">
              <span><strong>{count(health?.source_count)}</strong><small>Sources</small></span><span><strong>{count(health?.document_count)}</strong><small>Documents</small></span><span><strong>{count(health?.chunk_count)}</strong><small>Chunks</small></span><span><strong>{count(health?.embedding_count)}</strong><small>Embeddings</small></span><span><strong>{health?.journal_mode?.toUpperCase() || '—'}</strong><small>Storage</small></span>
            </div>
            <div className="rag-actions"><form className="rag-search" onSubmit={handleRagSearch}><Search size={15} aria-hidden="true" /><input aria-label="RAG search" value={ragQuery} onChange={(event) => setRagQuery(event.target.value)} placeholder="Search local sources" maxLength={512} /><button type="submit" disabled={!ragQuery.trim() || ragBusy !== ''}>{ragBusy === 'search' ? 'Searching' : 'Search'}</button></form><CommandButton variant="ghost" size="sm" icon={Database} busy={ragBusy === 'rebuild'} onClick={() => void handleRagRebuild()}>Rebuild FTS index</CommandButton></div>
            {ragError ? <p className="settings-notice settings-notice--error" role="alert"><ShieldAlert size={15} />{ragError}</p> : null}
            <div className="rag-results" aria-live="polite">{ragResults.length ? ragResults.map((item, index) => <article key={item.chunk_id || `${item.source_id}-${index}`}><div><strong>{item.source_id || 'local source'}</strong><span>{item.relative_ref || 'reference unavailable'}</span></div><p>{item.snippet || 'No snippet returned.'}</p></article>) : <EmptyState title={ragQuery ? 'No matching sources' : 'Search the local index'} description={ragQuery ? 'Try a broader query or rebuild the FTS index.' : 'Results stay in this workspace and do not alter the source files.'} compact />}</div>
            <div className="rag-governance-grid">
              <section className="rag-governance-card" data-testid="rag-capacity">
                <div className="rag-governance-card__head"><div><p className="mono-label">CAPACITY</p><strong>SQLite vector budget</strong></div><StatusBadge label={ragCapacity ? 'READY' : 'WAIT'} tone={ragCapacity ? 'ok' : 'idle'} size="sm" /></div>
                <div className="rag-governance-controls"><label>Dimensions<input type="number" min="1" max="32768" value={ragDimensions} onChange={(event) => setRagDimensions(Number(event.target.value) || 1)} /></label><CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={ragBusy === 'governance'} onClick={() => void refreshRagGovernance()}>Inspect</CommandButton></div>
                {ragCapacity ? <dl className="rag-facts"><div><dt>Active chunks</dt><dd>{count(ragCapacity.active_chunk_count)}</dd></div><div><dt>Embeddings</dt><dd>{count(ragCapacity.embedding_count)}</dd></div><div><dt>Estimated vectors</dt><dd>{formatBytes(ragCapacity.estimated_vector_bytes)}</dd></div><div><dt>Total estimate</dt><dd>{formatBytes(ragCapacity.estimated_total_bytes)}</dd></div><div><dt>Stale</dt><dd>{count(ragCapacity.stale_embedding_count)}</dd></div><div><dt>Index ceiling</dt><dd>{count(ragCapacity.max_index_chunks)}</dd></div></dl> : <EmptyState title="Capacity not inspected" description="Estimate vector storage without materializing embeddings." compact />}
              </section>
              <section className="rag-governance-card" data-testid="rag-ann">
                <div className="rag-governance-card__head"><div><p className="mono-label">ANN GATE</p><strong>sqlite-vec decision</strong></div><StatusBadge label={ragAnnDecision?.decision?.decision || 'WAIT'} tone={ragAnnDecision?.decision?.decision === 'GO' ? 'warn' : ragAnnDecision ? 'info' : 'idle'} size="sm" /></div>
                <div className="rag-governance-controls"><label>Scan budget<input type="number" min="1" max="10000" value={ragScanBudget} onChange={(event) => setRagScanBudget(Number(event.target.value) || 1)} /></label><CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={ragBusy === 'governance'} onClick={() => void refreshRagGovernance()}>Evaluate</CommandButton></div>
                {ragAnnDecision?.decision ? <div className="rag-ann-result"><p><strong>{ragAnnDecision.decision.reason || 'Decision reported'}</strong></p><span>{ragAnnDecision.decision.extension || 'sqlite_vec'}: {ragAnnDecision.decision.extension_available ? 'available' : 'not installed'} · production approval: {ragAnnDecision.decision.production_approved ? 'yes' : 'no'}</span></div> : <EmptyState title="ANN gate not evaluated" description="The conservative decision never enables production automatically." compact />}
              </section>
            </div>
            <section className="rag-embedding-panel" data-testid="rag-embedding-jobs">
              <div className="rag-governance-card__head"><div><p className="mono-label">EMBEDDING DATA PLANE</p><strong>Resumable embedding job</strong></div>{embeddingJob ? <StatusBadge label={String(embeddingJob.state || 'unknown').toUpperCase()} tone={embeddingJob.state === 'completed' ? 'ok' : embeddingJob.state === 'failed' ? 'danger' : 'info'} size="sm" /> : <StatusBadge label="NO JOB" tone="idle" size="sm" />}</div>
              <form className="rag-embedding-form" onSubmit={(event) => void handleCreateEmbeddingJob(event)}>
                <label>Model<input value={embeddingForm.model_id} onChange={(event) => setEmbeddingForm((form) => ({ ...form, model_id: event.target.value }))} required /></label>
                <label>Model SHA256<input value={embeddingForm.model_sha256} onChange={(event) => setEmbeddingForm((form) => ({ ...form, model_sha256: event.target.value }))} placeholder="64 hexadecimal characters" minLength={64} maxLength={64} required /></label>
                <label>Source ID<input value={embeddingForm.source_id} onChange={(event) => setEmbeddingForm((form) => ({ ...form, source_id: event.target.value }))} placeholder="all active sources" /></label>
                <label>Batch size<input type="number" min="1" max="64" value={embeddingForm.batch_size} onChange={(event) => setEmbeddingForm((form) => ({ ...form, batch_size: Number(event.target.value) || 1 }))} /></label>
                <div className="rag-embedding-form__actions"><CommandButton icon={Database} type="submit" busy={ragBusy === 'embedding:create'} disabled={ragBusy !== ''}>Create job</CommandButton><CommandButton variant="ghost" size="sm" busy={ragBusy === 'embedding:run'} disabled={!embeddingJob || !['queued', 'paused', 'running'].includes(String(embeddingJob?.state)) || ragBusy !== ''} onClick={() => void handleEmbeddingJobAction('run')}>Run batch</CommandButton><CommandButton variant="ghost" size="sm" busy={ragBusy === 'embedding:refresh'} disabled={!embeddingJob || ragBusy !== ''} onClick={() => void handleEmbeddingJobAction('refresh')}>Refresh job</CommandButton><CommandButton variant="danger" size="sm" busy={ragBusy === 'embedding:cancel'} disabled={!embeddingJob || !['queued', 'paused', 'running'].includes(String(embeddingJob?.state)) || ragBusy !== ''} onClick={() => void handleEmbeddingJobAction('cancel')}>Cancel</CommandButton></div>
              </form>
              {embeddingJob ? <div className="rag-job-summary"><span><strong>{embeddingJob.job_id}</strong><small>{embeddingJob.cursor?.model_id || embeddingForm.model_id}</small></span><span>{count(embeddingJob.cursor?.indexed)} / {count(embeddingJob.cursor?.total)} indexed · attempts {count(embeddingJob.attempts)}</span></div> : <p className="rag-governance-note">The server never stores the model secret; the digest is only used to fence resumable work to one embedding identity.</p>}
            </section>
            <div className="rag-source-registry">
              <SectionHead title="Indexed sources" hint={`${sourceList.length} registered`} />
              {sources.state === 'error' ? <p className="settings-notice settings-notice--error" role="alert">{sources.error}</p> : sourceList.length ? <ul>{sourceList.map((source) => <li key={source.source_id}><div><strong>{source.display_name || source.source_id}</strong><span>{source.relative_ref || source.source_id} · {source.document_count ?? 0} docs · {source.chunk_count ?? 0} chunks</span></div><button type="button" className="settings-icon-button" aria-label={`Delete ${source.source_id}`} title="Delete indexed source" disabled={ragBusy !== ''} onClick={() => void handleDeleteSource(source)}>×</button></li>)}</ul> : <EmptyState title="No sources indexed" description="Import a source or rebuild the index to populate this registry." compact />}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
