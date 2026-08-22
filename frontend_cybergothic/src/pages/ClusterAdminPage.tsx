import { useCallback, useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Check,
  Copy,
  Database,
  ExternalLink,
  Layers3,
  Network,
  Plus,
  RefreshCw,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  UserRound,
  Wifi,
  XCircle,
} from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { EmptyState } from '../components/EmptyState';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { StatusBadge } from '../components/StatusBadge';
import { pushToast } from '../components/Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import * as api from '../data/api';
import { fixturesEnabled } from '../data/fixtures';
import {
  useClusterConfig,
  useClusterInvite,
  useClusterNodes,
  useClusterStatus,
  useMasterHealth,
  useMyRole,
} from '../data/hooks';
import type {
  ClusterLayersResponse,
  ClusterNode,
  ClusterProfile,
  ModelRuntimeContractsResponse,
  ModelRuntimeSidecarStatusResponse,
  PipelineCapacityResponse,
} from '../data/types';
import { ClusterConstellationCanvas } from '../visual/ClusterConstellationCanvas';

function relativeTime(timestamp?: number): string {
  if (!timestamp) return 'unknown';
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - timestamp));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

function nodeTone(node: ClusterNode): 'ok' | 'warn' | 'danger' | 'idle' {
  if (node.is_available) return 'ok';
  if (node.state === 'degraded') return 'warn';
  if (node.state === 'error') return 'danger';
  return 'idle';
}

function nodeLabel(node: ClusterNode): string {
  if (node.is_available) return 'ONLINE';
  if (node.state === 'degraded') return 'DEGRADED';
  return 'OFFLINE';
}

function recordText(value: unknown, fallback = ''): string {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : fallback;
}

function recordNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** exponent).toFixed(exponent > 1 ? 1 : 0)} ${units[exponent]}`;
}

export function ClusterAdminPage() {
  const role = useMyRole();
  const nodes = useClusterNodes();
  const status = useClusterStatus();
  const config = useClusterConfig();
  const canWrite = role.state === 'ready' && role.data?.is_master === true;
  const invite = useClusterInvite(30_000, canWrite);
  const health = useMasterHealth(15_000, role.state === 'ready' && !canWrite);
  const usingFixtures = fixturesEnabled();

  const [nodeOverride, setNodeOverride] = useState<ClusterNode[] | null>(null);
  const [maxNodes, setMaxNodes] = useState('');
  const [busy, setBusy] = useState('');
  const [registerForm, setRegisterForm] = useState({ node_id: '', hostname: '', address: '', network_type: 'tailscale', node_type: 'pc' });
  const [profiles, setProfiles] = useState<ClusterProfile[]>([]);
  const [profileForm, setProfileForm] = useState({ name: '', cluster_id: '', master_endpoint: '' });
  const [profileBusy, setProfileBusy] = useState('');
  const [currentProfile, setCurrentProfile] = useState<ClusterProfile | null>(null);
  const [endpoints, setEndpoints] = useState<Array<Record<string, unknown>>>([]);
  const [layers, setLayers] = useState<ClusterLayersResponse | null>(null);
  const [capacity, setCapacity] = useState<PipelineCapacityResponse | null>(null);
  const [sidecarStatus, setSidecarStatus] = useState<ModelRuntimeSidecarStatusResponse | null>(null);
  const [contracts, setContracts] = useState<ModelRuntimeContractsResponse | null>(null);
  const [operationsBusy, setOperationsBusy] = useState('');
  const nodeList = nodeOverride ?? nodes.data?.nodes ?? [];
  const effectiveMaxNodes = Number(maxNodes || config.data?.max_nodes || 0);

  const loadProfiles = useCallback(async () => {
    if (usingFixtures) {
      const profile = { profile_id: 'fixture-profile', name: 'Local master', cluster_id: 'fixture-cluster', master_endpoint: { scheme: 'http', host: '127.0.0.1', port: 8000 }, status: 'active' } satisfies ClusterProfile;
      setProfiles([profile]);
      setCurrentProfile(profile);
      return;
    }
    try { setProfiles((await api.fetchClusterProfiles()).profiles ?? []); } catch { setProfiles([]); }
  }, [usingFixtures]);

  const loadOperationalState = useCallback(async () => {
    if (usingFixtures) {
      setEndpoints([{ endpoint_id: 'fixture-endpoint', cluster_id: 'fixture-cluster', name: 'Local API', scheme: 'http', host: '127.0.0.1', port: 8000, status: 'advertised', last_verified_at: new Date().toISOString() }]);
      setLayers({ total: 24, strategy: 'dynamic', computed_at: Date.now() / 1000, assignments: [
        { node_id: 'master', role: 'master', start_layer: 0, end_layer: 12, layers_count: 12, has_embedding: true },
        { node_id: 'client_TABLET-2TLUCNU8', role: 'client', start_layer: 12, end_layer: 24, layers_count: 12, has_lm_head: true },
      ] });
      setCapacity({ model_id: 'qwen-1_8b', model_type: 'qwen', total_layers: 24, raw_model_bytes: 3673657344, candidate_node_count: 2, status: 'admitted', admitted: true, reason_code: '', reason: 'Fixture plan admitted', plan_id: 'fixture-plan', assignments: [{ node_id: 'master', start_layer: 0, end_layer: 12, layer_count: 12 }, { node_id: 'client_TABLET-2TLUCNU8', start_layer: 12, end_layer: 24, layer_count: 12 }], control_only_nodes: [], participating_node_count: 2, single_node_full_model_candidates: [], prepared_node_count: 2, ready_node_count: 2, worker_count: 1, computed_at: Date.now() / 1000 });
      setSidecarStatus({ schema_version: 1, role: canWrite ? 'master' : 'client', control_available: canWrite, production_admitted: false, profiles: {
        qwen3_sidecar: { display_name: 'Qwen3 Sidecar', runtime_environment: '.venv-qwen3-sidecar', preflight_supported: true, requires_task_contract: true, production_admitted: false, supported_actions: ['status', 'begin', 'release', 'cancel'], session: { active: false, state: { phase: 'idle' } } },
        gemma4_pipeline: { display_name: 'Gemma 4 Pipeline Sidecar', runtime_environment: '.venv-gemma4-pipeline', preflight_supported: false, requires_task_contract: true, production_admitted: false, supported_actions: ['status', 'begin', 'release', 'cancel'], session: { active: false, state: { phase: 'idle' } } },
      } });
      setContracts({ schema_version: 1, contracts: [] });
      return;
    }
    const results = await Promise.allSettled([
      api.fetchCurrentClusterProfile(),
      api.fetchClusterEndpoints(),
      api.fetchClusterLayers(),
      api.fetchPipelineCapacity(),
      api.fetchModelRuntimeSidecarStatus(),
      api.fetchModelRuntimeContracts(),
    ]);
    const [current, endpointData, layerData, capacityData, sidecarData, contractData] = results;
    if (current.status === 'fulfilled') setCurrentProfile(current.value.profile ?? null);
    if (endpointData.status === 'fulfilled') setEndpoints(endpointData.value.endpoints ?? []);
    if (layerData.status === 'fulfilled') setLayers(layerData.value);
    if (capacityData.status === 'fulfilled') setCapacity(capacityData.value);
    if (sidecarData.status === 'fulfilled') setSidecarStatus(sidecarData.value);
    if (contractData.status === 'fulfilled') setContracts(contractData.value);
  }, [canWrite, usingFixtures]);

  useEffect(() => {
    if (!maxNodes && config.data?.max_nodes) setMaxNodes(String(config.data.max_nodes));
  }, [config.data?.max_nodes, maxNodes]);

  const refresh = useCallback(() => {
    role.refresh();
    nodes.refresh();
    status.refresh();
    config.refresh();
    if (canWrite) invite.refresh();
    else health.refresh();
    void loadProfiles();
    void loadOperationalState();
  }, [canWrite, config, health, invite, loadOperationalState, loadProfiles, nodes, role, status]);

  useEffect(() => { void loadProfiles(); }, [loadProfiles]);
  useEffect(() => { void loadOperationalState(); }, [loadOperationalState]);

  useRegisterRefresh(refresh);

  const onlineNodes = nodeList.filter((node) => node.is_available).length;
  const offlineNodes = nodeList.filter((node) => !node.is_available);
  const statusData = status.data;
  const allReadErrors = [nodes, status, config].filter((resource) => resource.state === 'error');
  const statusCards = useMemo(() => [
    { label: 'RUN MODE', value: statusData?.run_mode || 'unknown', icon: Network },
    { label: 'READINESS', value: statusData?.nodes_ready ? 'ready' : 'waiting', icon: ShieldCheck },
    { label: 'ONLINE NODES', value: `${onlineNodes} / ${nodeList.length}`, icon: Server },
    { label: 'HEARTBEAT', value: config.data?.network?.heartbeat_interval_s ? `${config.data.network.heartbeat_interval_s}s` : 'unknown', icon: Wifi },
  ], [config.data?.network?.heartbeat_interval_s, nodeList.length, onlineNodes, statusData?.nodes_ready, statusData?.run_mode]);

  const handleMaxNodes = async () => {
    const value = Number(maxNodes);
    if (!Number.isInteger(value) || value < 1 || value > 64) {
      pushToast('Max nodes must be an integer from 1 to 64', 'danger');
      return;
    }
    setBusy('max-nodes');
    try {
      if (usingFixtures) {
        pushToast(`Fixture capacity updated to ${value}`, 'info');
      } else {
        await api.updateMaxNodes(value);
        await config.refresh();
        pushToast('Cluster capacity updated', 'ok');
      }
    } catch (error) {
      pushToast(`Capacity update failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleRegister = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!registerForm.node_id.trim()) {
      pushToast('Node ID is required', 'danger');
      return;
    }
    setBusy('register');
    try {
      if (usingFixtures) {
        const next: ClusterNode = {
          node_id: registerForm.node_id.trim(),
          role: 'client',
          node_type: registerForm.node_type,
          state: 'offline',
          address: registerForm.address.trim(),
          hostname: registerForm.hostname.trim() || registerForm.node_id.trim(),
          device_info: {},
          network_type: registerForm.network_type,
          connected_at: Date.now() / 1000,
          last_heartbeat: 0,
          avg_rtt_ms: 0,
          last_rtt_ms: 0,
          task_count: 0,
          error_count: 0,
          is_available: false,
        };
        setNodeOverride((current) => [...(current ?? nodeList), next]);
        pushToast('Fixture node reserved', 'info');
      } else {
        await api.registerClusterNode({ ...registerForm, node_id: registerForm.node_id.trim() });
        await nodes.refresh();
        pushToast('Node registered', 'ok');
      }
      setRegisterForm({ node_id: '', hostname: '', address: '', network_type: 'tailscale', node_type: 'pc' });
    } catch (error) {
      pushToast(`Node registration failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleDeregister = async (node: ClusterNode) => {
    if (node.role === 'master' || !window.confirm(`Deregister ${node.node_id}?`)) return;
    setBusy(`deregister:${node.node_id}`);
    try {
      if (usingFixtures) {
        setNodeOverride(nodeList.map((item) => item.node_id === node.node_id ? { ...item, state: 'offline', is_available: false } : item));
        pushToast('Fixture node deregistered', 'info');
      } else {
        await api.deregisterClusterNode(node.node_id);
        await nodes.refresh();
        pushToast('Node deregistered', 'ok');
      }
    } catch (error) {
      pushToast(`Deregister failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const handleDelete = async (node: ClusterNode) => {
    if (node.is_available || !window.confirm(`Delete offline node ${node.node_id}?`)) return;
    setBusy(`delete:${node.node_id}`);
    try {
      if (usingFixtures) {
        setNodeOverride(nodeList.filter((item) => item.node_id !== node.node_id));
        pushToast('Fixture node deleted', 'info');
      } else {
        await api.deleteClusterNode(node.node_id);
        await nodes.refresh();
        pushToast('Offline node deleted', 'ok');
      }
    } catch (error) {
      pushToast(`Delete failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const copyInvite = async () => {
    const value = `${invite.data?.master_host || '-'}:${invite.data?.master_port || '-'}`;
    try {
      await navigator.clipboard.writeText(value);
      pushToast('Master endpoint copied', 'ok');
    } catch {
      pushToast(value, 'info');
    }
  };

  const verifyProfile = async () => {
    if (!profileForm.master_endpoint.trim() || profileBusy) return;
    setProfileBusy('verify');
    try {
      const result = usingFixtures ? { status: 'ok', cluster_id: profileForm.cluster_id || 'fixture-cluster' } : await api.verifyClusterProfile(profileForm.master_endpoint.trim(), profileForm.name.trim());
      pushToast(result.status === 'ok' ? 'Master endpoint verified' : 'Endpoint is unreachable', result.status === 'ok' ? 'ok' : 'warn');
      if (result.status === 'ok' && !profileForm.cluster_id && typeof result.cluster_id === 'string') setProfileForm((current) => ({ ...current, cluster_id: result.cluster_id as string }));
    } catch (error) { pushToast(`Profile verification failed: ${api.describeError(error)}`, 'danger'); }
    finally { setProfileBusy(''); }
  };

  const createProfile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!profileForm.name.trim() || !profileForm.cluster_id.trim() || !profileForm.master_endpoint.trim() || profileBusy) return;
    setProfileBusy('create');
    try {
      if (usingFixtures) setProfiles((current) => [...current, { profile_id: `fixture-${Date.now()}`, name: profileForm.name.trim(), cluster_id: profileForm.cluster_id.trim(), master_endpoint: { scheme: 'http', host: profileForm.master_endpoint.trim(), port: 8000 }, status: 'saved' }]);
      else await api.createClusterProfile({ ...profileForm, node_role: canWrite ? 'master' : 'client' });
      await loadProfiles();
      setProfileForm({ name: '', cluster_id: '', master_endpoint: '' });
      pushToast('Cluster profile saved', usingFixtures ? 'info' : 'ok');
    } catch (error) { pushToast(`Profile save failed: ${api.describeError(error)}`, 'danger'); }
    finally { setProfileBusy(''); }
  };

  const removeProfile = async (profile: ClusterProfile) => {
    if (profileBusy || !window.confirm(`Delete cluster profile ${profile.name}?`)) return;
    setProfileBusy(`delete:${profile.profile_id}`);
    try {
      if (usingFixtures) setProfiles((current) => current.filter((item) => item.profile_id !== profile.profile_id));
      else await api.deleteClusterProfile(profile.profile_id);
      await loadProfiles();
      pushToast('Cluster profile deleted', usingFixtures ? 'info' : 'ok');
    } catch (error) { pushToast(`Profile deletion failed: ${api.describeError(error)}`, 'danger'); }
    finally { setProfileBusy(''); }
  };

  const activateProfile = async (profile: ClusterProfile) => {
    if (!canWrite || profileBusy || profile.profile_id === currentProfile?.profile_id) return;
    if (!window.confirm(`Activate cluster profile ${profile.name}?`)) return;
    setProfileBusy(`activate:${profile.profile_id}`);
    try {
      if (usingFixtures) setCurrentProfile({ ...profile, status: 'active' });
      else {
        const result = await api.activateClusterProfile(profile.profile_id);
        if (result.profile && typeof result.profile === 'object') setCurrentProfile(result.profile as ClusterProfile);
        else await loadOperationalState();
      }
      await loadProfiles();
      pushToast('Cluster profile activated', usingFixtures ? 'info' : 'ok');
    } catch (error) { pushToast(`Profile activation failed: ${api.describeError(error)}`, 'danger'); }
    finally { setProfileBusy(''); }
  };

  const verifyEndpoint = async (endpoint: Record<string, unknown>) => {
    const host = recordText(endpoint.host);
    if (!host || operationsBusy) return;
    setOperationsBusy(`endpoint:${host}`);
    try {
      const result = usingFixtures ? { reachable: true } : await api.verifyClusterEndpoint({ scheme: recordText(endpoint.scheme, 'http'), host, port: recordNumber(endpoint.port, 8000) });
      pushToast(result.reachable ? `${host} is reachable` : `${host} is unreachable`, result.reachable ? 'ok' : 'warn');
    } catch (error) { pushToast(`Endpoint check failed: ${api.describeError(error)}`, 'danger'); }
    finally { setOperationsBusy(''); }
  };

  const sidecarAction = async (profile: 'qwen3_sidecar' | 'gemma4_pipeline', action: 'release' | 'cancel') => {
    if (!canWrite || operationsBusy || !window.confirm(`${action === 'release' ? 'Release' : 'Cancel'} ${profile} session?`)) return;
    setOperationsBusy(`${action}:${profile}`);
    try {
      if (usingFixtures) {
        setSidecarStatus((current) => current ? { ...current, profiles: { ...current.profiles, [profile]: { ...current.profiles[profile], session: { active: false, state: { phase: action === 'release' ? 'released' : 'cancelled' } } } } } : current);
      } else if (action === 'release') await api.releaseModelRuntimeSidecar(profile);
      else await api.cancelModelRuntimeSidecar(profile);
      if (!usingFixtures) await loadOperationalState();
      pushToast(`${profile} ${action} requested`, usingFixtures ? 'info' : 'ok');
    } catch (error) { pushToast(`Sidecar ${action} failed: ${api.describeError(error)}`, 'danger'); }
    finally { setOperationsBusy(''); }
  };

  return (
    <div className="cluster-page" data-testid="cluster-admin-page">
      <ClusterConstellationCanvas className="cluster-page__bg" />
      <div className="cluster-page__content">
        <PageHeader
          tag="CLUSTER CONTROL"
          title="Cluster Admin"
          description="Observe node health and keep topology changes explicit, reviewable, and role-gated."
          actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={nodes.refreshing || status.refreshing} onClick={refresh}>Refresh</CommandButton>}
        />

        <div className="cluster-layout">
          <aside className="cluster-rail">
            <section className="cluster-panel cluster-identity">
              <div className="cluster-panel__eyebrow">LOCAL IDENTITY</div>
              <div className="cluster-identity__main"><UserRound size={18} aria-hidden="true" /><strong>{role.data?.node_id || 'detecting'}</strong></div>
              <StatusBadge label={role.state === 'loading' ? 'CHECKING' : canWrite ? 'MASTER / WRITE' : 'CLIENT / READ ONLY'} tone={canWrite ? 'ok' : 'info'} />
              <p>{canWrite ? 'Topology mutations are available on this node.' : 'This view is read-only until a master role is detected.'}</p>
            </section>

            <section className="cluster-panel cluster-profiles">
              <SectionHead title="Cluster profiles" hint={`${profiles.length} saved`} />
              {profiles.length ? <ul className="cluster-profile-list">{profiles.map((profile) => { const active = profile.profile_id === currentProfile?.profile_id || profile.status === 'active'; return <li key={profile.profile_id} className={active ? 'is-active' : ''}><div><strong>{profile.name} {active ? <span className="cluster-profile-state">ACTIVE</span> : null}</strong><span>{profile.cluster_id} · {profile.master_endpoint?.host || 'endpoint'}:{profile.master_endpoint?.port || 80}</span></div><div className="cluster-profile-actions">{canWrite && !active ? <button type="button" className="cluster-icon-button" title="Activate profile" aria-label={`Activate ${profile.name}`} disabled={profileBusy !== ''} onClick={() => void activateProfile(profile)}><ExternalLink size={14} /></button> : null}<button type="button" className="cluster-icon-button cluster-icon-button--danger" title="Delete profile" aria-label={`Delete ${profile.name}`} disabled={profileBusy !== '' || active} onClick={() => void removeProfile(profile)}><Trash2 size={14} /></button></div></li>; })}</ul> : <p className="cluster-readonly">No saved profiles.</p>}
              <form className="cluster-profile-form" onSubmit={(event) => void createProfile(event)}><input aria-label="Profile name" placeholder="Profile name" value={profileForm.name} onChange={(event) => setProfileForm((current) => ({ ...current, name: event.target.value }))} /><input aria-label="Cluster ID" placeholder="Cluster ID" value={profileForm.cluster_id} onChange={(event) => setProfileForm((current) => ({ ...current, cluster_id: event.target.value }))} /><input aria-label="Master endpoint" placeholder="http://100.x.x.x:8000" value={profileForm.master_endpoint} onChange={(event) => setProfileForm((current) => ({ ...current, master_endpoint: event.target.value }))} /><div><CommandButton type="button" variant="ghost" size="sm" busy={profileBusy === 'verify'} onClick={() => void verifyProfile()}>Verify</CommandButton><CommandButton type="submit" size="sm" busy={profileBusy === 'create'}>Save</CommandButton></div></form>
            </section>

            <nav className="cluster-nav" aria-label="Cluster admin sections">
              <a href="#cluster-overview">Overview <span>01</span></a>
              <a href="#cluster-nodes">Nodes <span>{nodeList.length.toString().padStart(2, '0')}</span></a>
              <a href="#cluster-plans">Plans <span>{capacity?.admitted ? 'OK' : 'WAIT'}</span></a>
              <a href="#cluster-access">Access <span>{canWrite ? 'RW' : 'RO'}</span></a>
            </nav>

            {canWrite ? (
              <section className="cluster-panel cluster-capacity">
                <SectionHead title="Capacity" hint="Maximum registered nodes" />
                <label className="cluster-field"><span>MAX NODES</span><input type="number" min="1" max="64" value={maxNodes} onChange={(event) => setMaxNodes(event.target.value)} /></label>
                <CommandButton size="sm" icon={Check} busy={busy === 'max-nodes'} onClick={handleMaxNodes}>Apply capacity</CommandButton>
              </section>
            ) : (
              <section className="cluster-panel cluster-health">
                <SectionHead title="Master link" hint="Client-side heartbeat" />
                <StatusBadge label={health.data?.master_online ? 'ONLINE' : 'UNREACHABLE'} tone={health.data?.master_online ? 'ok' : 'danger'} pulse={Boolean(health.data?.master_online)} />
                <p>{health.data?.master_host ? `${health.data.master_host}:${health.data.master_port || 8888}` : 'Waiting for master health.'}</p>
              </section>
            )}
          </aside>

          <main className="cluster-main">
            <section className="cluster-panel cluster-panel--hero" id="cluster-overview">
              <SectionHead title="Topology pulse" hint={statusData?.current_task ? `Current task ${String(statusData.current_task.task_id || 'active')}` : 'No active task reported'} actions={<span className="cluster-live"><span />LIVE</span>} />
              {allReadErrors.length ? <EmptyState kind="error" title="Some cluster data is unavailable" description="The remaining panels stay interactive where their data is available." detail={allReadErrors[0]?.error} errorKind={allReadErrors[0]?.errorKind} errorStatus={allReadErrors[0]?.errorStatus} compact /> : null}
              <div className="cluster-status-grid">
                {statusCards.map(({ label, value, icon: Icon }) => <div className="cluster-status-card" key={label}><Icon size={17} aria-hidden="true" /><span>{label}</span><strong>{value}</strong></div>)}
              </div>
            </section>

            <section className="cluster-panel" id="cluster-nodes">
              <SectionHead title="Registered nodes" hint={`${onlineNodes} online / ${offlineNodes.length} offline`} actions={<span className="mono-label">{nodes.updatedAt ? `UPDATED ${new Date(nodes.updatedAt).toLocaleTimeString()}` : ''}</span>} />
              {nodes.state === 'loading' && nodeList.length === 0 ? <EmptyState kind="loading" title="Loading node registry" compact /> : nodeList.length === 0 ? <EmptyState title="No registered nodes" description="Register a node from the access panel when this node is master." compact /> : (
                <div className="cluster-table-wrap"><table className="cluster-table"><thead><tr><th>Node</th><th>Role</th><th>Network</th><th>Last heartbeat</th><th>Status</th><th><span className="sr-only">Actions</span></th></tr></thead><tbody>
                  {nodeList.map((node) => <tr key={node.node_id}><td data-label="Node"><strong>{node.hostname || node.node_id}</strong><small>{node.node_id}</small></td><td data-label="Role"><span className="cell-mono">{node.role}</span></td><td data-label="Network"><span className="cell-mono">{node.network_type || 'unknown'}</span><small>{node.address || 'no address'}</small></td><td data-label="Last heartbeat"><span className="cell-mono">{relativeTime(node.last_heartbeat)}</span></td><td data-label="Status"><StatusBadge label={nodeLabel(node)} tone={nodeTone(node)} pulse={node.is_available} size="sm" /></td><td data-label="Actions" className="cluster-table__actions">{canWrite && node.role !== 'master' ? <><button type="button" className="cluster-icon-button" title="Deregister node" aria-label={`Deregister ${node.node_id}`} disabled={busy === `deregister:${node.node_id}`} onClick={() => void handleDeregister(node)}><XCircle size={15} /></button><button type="button" className="cluster-icon-button cluster-icon-button--danger" title="Delete offline node" aria-label={`Delete ${node.node_id}`} disabled={node.is_available || busy === `delete:${node.node_id}`} onClick={() => void handleDelete(node)}><Trash2 size={15} /></button></> : <span className="cluster-readonly">{canWrite ? 'protected' : 'read only'}</span>}</td></tr>)}
                </tbody></table></div>
              )}
            </section>

            <section className="cluster-ops-grid" id="cluster-plans">
              <section className="cluster-panel">
                <SectionHead title="Advertised endpoints" hint={`${endpoints.length} local records`} />
                {endpoints.length ? <div className="cluster-endpoint-list">{endpoints.map((endpoint, index) => { const endpointId = recordText(endpoint.endpoint_id, `${recordText(endpoint.host, 'endpoint')}-${index}`); const host = recordText(endpoint.host, 'unknown'); const reachable = recordText(endpoint.status).toLowerCase() === 'reachable' || recordText(endpoint.status).toLowerCase() === 'advertised'; return <div className="cluster-endpoint-row" key={endpointId}><div><strong>{recordText(endpoint.name, host)}</strong><span>{recordText(endpoint.scheme, 'http')}://{host}:{recordNumber(endpoint.port, 8000)}</span></div><div className="cluster-endpoint-row__actions"><StatusBadge label={reachable ? 'ADVERTISED' : recordText(endpoint.status, 'UNKNOWN').toUpperCase()} tone={reachable ? 'ok' : 'idle'} size="sm" /><button type="button" className="cluster-icon-button" title="Verify endpoint" aria-label={`Verify ${host}`} disabled={operationsBusy !== ''} onClick={() => void verifyEndpoint(endpoint)}><Wifi size={14} /></button></div></div>; })}</div> : <p className="cluster-readonly">No advertised endpoint records.</p>}
              </section>

              <section className="cluster-panel">
                <SectionHead title="Pipeline admission" hint="Metadata-only capacity plan" actions={<SlidersHorizontal size={16} aria-hidden="true" />} />
                {capacity ? <><div className="cluster-plan-status"><StatusBadge label={capacity.admitted ? 'ADMITTED' : 'NOT ADMITTED'} tone={capacity.admitted ? 'ok' : 'warn'} /><span className="cell-mono">{capacity.reason_code || capacity.status}</span></div><dl className="cluster-facts cluster-facts--grid"><div><dt>MODEL</dt><dd>{capacity.model_id || 'unknown'}</dd></div><div><dt>LAYERS</dt><dd>{capacity.total_layers || 0}</dd></div><div><dt>WEIGHTS</dt><dd>{formatBytes(capacity.raw_model_bytes)}</dd></div><div><dt>PARTICIPANTS</dt><dd>{capacity.participating_node_count} / {capacity.candidate_node_count}</dd></div></dl><p className="cluster-plan-reason">{capacity.reason || 'No capacity explanation reported.'}</p></> : <EmptyState kind="loading" title="Loading capacity plan" compact />}
              </section>

              <section className="cluster-panel cluster-panel--wide">
                <SectionHead title="Layer allocation" hint={layers ? `${layers.strategy || 'dynamic'} / ${layers.total || 0} layers` : 'Read-only runtime projection'} actions={<Layers3 size={16} aria-hidden="true" />} />
                {layers?.assignments?.length ? <div className="cluster-layer-list">{layers.assignments.map((assignment, index) => <div className="cluster-layer-row" key={`${recordText(assignment.node_id, 'node')}-${index}`}><div><strong>{recordText(assignment.node_id, 'unknown node')}</strong><span>{recordText(assignment.role, 'worker')}</span></div><div className="cluster-layer-range">Layer {recordNumber(assignment.start_layer)}-{recordNumber(assignment.end_layer, recordNumber(assignment.start_layer))}<small>{recordNumber(assignment.layers_count, recordNumber(assignment.layer_count))} layers{assignment.has_embedding ? ' / embedding' : ''}{assignment.has_lm_head ? ' / lm_head' : ''}</small></div></div>)}</div> : <p className="cluster-readonly">No layer assignment is available.</p>}
              </section>

              <section className="cluster-panel cluster-panel--wide">
                <SectionHead title="Sidecar contracts" hint={sidecarStatus?.control_available ? 'Master controls enabled' : 'Read-only status'} actions={<Database size={16} aria-hidden="true" />} />
                <div className="cluster-sidecar-list">{Object.entries(sidecarStatus?.profiles ?? {}).map(([profile, capability]) => { const session = capability.session ?? {}; const state = session.state; const phase = state && typeof state === 'object' ? recordText((state as Record<string, unknown>).phase, 'idle') : 'idle'; const active = session.active === true || !['idle', 'released', 'cancelled', 'unavailable'].includes(phase); const profileKey = profile === 'qwen3_sidecar' || profile === 'gemma4_pipeline' ? profile : null; return <div className="cluster-sidecar-row" key={profile}><div><strong>{recordText(capability.display_name, profile)}</strong><span>{recordText(capability.runtime_environment, 'runtime unknown')} · {phase}</span></div><div className="cluster-sidecar-row__actions"><StatusBadge label={active ? 'ACTIVE' : 'IDLE'} tone={active ? 'warn' : 'idle'} size="sm" />{canWrite && active && profileKey ? <><button type="button" className="cluster-icon-button" title="Release sidecar" aria-label={`Release ${profile}`} disabled={operationsBusy !== ''} onClick={() => void sidecarAction(profileKey, 'release')}><Check size={14} /></button><button type="button" className="cluster-icon-button cluster-icon-button--danger" title="Cancel sidecar" aria-label={`Cancel ${profile}`} disabled={operationsBusy !== ''} onClick={() => void sidecarAction(profileKey, 'cancel')}><XCircle size={14} /></button></> : null}</div></div>; })}</div><div className="cluster-contract-summary"><span>{contracts?.contracts?.length ?? 0} stored contracts</span><span>{sidecarStatus?.production_admitted ? 'production admitted' : 'experimental gate only'}</span></div>
              </section>
            </section>

            <section className="cluster-access-grid" id="cluster-access">
              {canWrite ? (
                <section className="cluster-panel">
                  <SectionHead title="Invite endpoint" hint="Share only with an approved peer" />
                  <div className="cluster-endpoint"><div><span className="mono-label">MASTER TCP</span><strong>{invite.data?.master_host || '-'}:{invite.data?.master_port || '-'}</strong></div><button type="button" className="cluster-icon-button" title="Copy master endpoint" aria-label="Copy master endpoint" onClick={() => void copyInvite()}><Copy size={15} /></button></div>
                  <div className="cluster-facts"><span><ShieldCheck size={14} />{invite.data?.identity_verified ? 'Identity verified' : 'Identity pending'}</span><span><Database size={14} />{invite.data?.node_count ?? nodeList.length} / {(invite.data?.max_nodes ?? effectiveMaxNodes) || '-'} slots</span></div>
                </section>
              ) : (
                <section className="cluster-panel">
                  <SectionHead title="Access boundary" hint="Write controls are hidden on clients" />
                  <div className="cluster-readonly-callout"><ShieldCheck size={20} /><div><strong>Read-only cluster view</strong><p>Ask the master node to register, remove, or resize topology. This client can still inspect health and heartbeat data.</p></div></div>
                </section>
              )}

              {canWrite ? (
                <section className="cluster-panel">
                  <SectionHead title="Reserve a node" hint="Create an offline slot before first connection" />
                  <form className="cluster-register" onSubmit={(event) => void handleRegister(event)}>
                    <div className="cluster-register__row"><label className="cluster-field"><span>NODE ID *</span><input value={registerForm.node_id} onChange={(event) => setRegisterForm((current) => ({ ...current, node_id: event.target.value }))} placeholder="client-lab-01" required /></label><label className="cluster-field"><span>HOSTNAME</span><input value={registerForm.hostname} onChange={(event) => setRegisterForm((current) => ({ ...current, hostname: event.target.value }))} placeholder="LAB-PC" /></label></div>
                    <div className="cluster-register__row"><label className="cluster-field"><span>ADDRESS</span><input value={registerForm.address} onChange={(event) => setRegisterForm((current) => ({ ...current, address: event.target.value }))} placeholder="100.x.x.x:8888" /></label><label className="cluster-field"><span>TYPE</span><select value={registerForm.node_type} onChange={(event) => setRegisterForm((current) => ({ ...current, node_type: event.target.value }))}><option value="pc">PC</option><option value="android">Android</option></select></label></div>
                    <CommandButton type="submit" size="sm" icon={Plus} busy={busy === 'register'}>Reserve node</CommandButton>
                  </form>
                </section>
              ) : null}
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}
