import { useCallback, useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Check,
  Copy,
  Database,
  Network,
  Plus,
  RefreshCw,
  Server,
  ShieldCheck,
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
import type { ClusterNode, ClusterProfile } from '../data/types';
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
  const nodeList = nodeOverride ?? nodes.data?.nodes ?? [];
  const effectiveMaxNodes = Number(maxNodes || config.data?.max_nodes || 0);

  const loadProfiles = useCallback(async () => {
    if (usingFixtures) {
      setProfiles([{ profile_id: 'fixture-profile', name: 'Local master', cluster_id: 'fixture-cluster', master_endpoint: { scheme: 'http', host: '127.0.0.1', port: 8000 }, status: 'active' }]);
      return;
    }
    try { setProfiles((await api.fetchClusterProfiles()).profiles ?? []); } catch { setProfiles([]); }
  }, [usingFixtures]);

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
  }, [canWrite, config, health, invite, loadProfiles, nodes, role, status]);

  useEffect(() => { void loadProfiles(); }, [loadProfiles]);

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
              {profiles.length ? <ul className="cluster-profile-list">{profiles.map((profile) => <li key={profile.profile_id}><div><strong>{profile.name}</strong><span>{profile.cluster_id} · {profile.master_endpoint?.host || 'endpoint'}:{profile.master_endpoint?.port || 80}</span></div><button type="button" className="cluster-icon-button cluster-icon-button--danger" title="Delete profile" aria-label={`Delete ${profile.name}`} disabled={profileBusy !== ''} onClick={() => void removeProfile(profile)}><Trash2 size={14} /></button></li>)}</ul> : <p className="cluster-readonly">No saved profiles.</p>}
              <form className="cluster-profile-form" onSubmit={(event) => void createProfile(event)}><input aria-label="Profile name" placeholder="Profile name" value={profileForm.name} onChange={(event) => setProfileForm((current) => ({ ...current, name: event.target.value }))} /><input aria-label="Cluster ID" placeholder="Cluster ID" value={profileForm.cluster_id} onChange={(event) => setProfileForm((current) => ({ ...current, cluster_id: event.target.value }))} /><input aria-label="Master endpoint" placeholder="http://100.x.x.x:8000" value={profileForm.master_endpoint} onChange={(event) => setProfileForm((current) => ({ ...current, master_endpoint: event.target.value }))} /><div><CommandButton type="button" variant="ghost" size="sm" busy={profileBusy === 'verify'} onClick={() => void verifyProfile()}>Verify</CommandButton><CommandButton type="submit" size="sm" busy={profileBusy === 'create'}>Save</CommandButton></div></form>
            </section>

            <nav className="cluster-nav" aria-label="Cluster admin sections">
              <a href="#cluster-overview">Overview <span>01</span></a>
              <a href="#cluster-nodes">Nodes <span>{nodeList.length.toString().padStart(2, '0')}</span></a>
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
