import { useCallback, useMemo, useState } from 'react';
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
import { deviceProfileFixture, fixturesEnabled, ragFixture, ragSearchFixture } from '../data/fixtures';
import { useDeviceProfile, useRagHealth, useRagSources } from '../data/hooks';
import type { DeviceProfileResponse, RagHealthResponse, RagSearchResult, RagSource } from '../data/types';

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

  const profile = deviceOverride ?? device.data;
  const health = ragOverride ?? rag.data;
  const sourceList = sourceOverride ?? sources.data?.sources ?? [];
  const gpuList = profile?.gpus ?? (profile?.gpu ? [profile.gpu] : []);
  const selectedGpuIndex = profile?.selected_gpu_index ?? 0;
  const runtimeHealth = rag.state === 'error' ? 'danger' : health?.status === 'ok' ? 'ok' : 'warn';

  const refresh = useCallback(() => {
    device.refresh();
    rag.refresh();
    sources.refresh();
  }, [device.refresh, rag.refresh, sources.refresh]);
  useRegisterRefresh(refresh);

  const activeGpu = gpuList[selectedGpuIndex] ?? gpuList[0];
  const activeTab = tab ?? localTab;
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

  const handleRagSearch = async (event: React.FormEvent) => {
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
