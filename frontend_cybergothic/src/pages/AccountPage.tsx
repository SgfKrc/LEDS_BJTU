import { useCallback, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Check,
  Copy,
  KeyRound,
  LogIn,
  LogOut,
  Network,
  Plus,
  RefreshCw,
  ShieldCheck,
  Trash2,
  UserRound,
  Users,
  Wifi,
  XCircle,
} from 'lucide-react';
import { AuthProvisioningPanel } from '../components/AuthProvisioningPanel';
import { CommandButton } from '../components/CommandButton';
import { EmptyState } from '../components/EmptyState';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { StatusBadge } from '../components/StatusBadge';
import { pushToast } from '../components/Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import * as api from '../data/api';
import { fixturesEnabled } from '../data/fixtures';
import {
  useAuthCapability,
  useAuthSession,
  useAuthSessions,
  useLocalTailscaleStatus,
  useManagedUsers,
  useTailscaleBindings,
} from '../data/hooks';
import type { AuthMutationResponse, AuthProvisioning, AuthSessionRecord, AuthSessionResponse, AuthUser, TailscaleBinding } from '../data/types';
import { IronGateCanvas } from '../visual/IronGateCanvas';

function dateLabel(value?: string): string {
  if (!value) return 'unknown';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function roleTone(role?: string): 'ok' | 'info' | 'warn' {
  if (role === 'owner') return 'ok';
  if (role === 'admin') return 'warn';
  return 'info';
}

function statusTone(status?: string): 'ok' | 'warn' | 'danger' | 'idle' {
  if (status === 'active') return 'ok';
  if (status === 'suspended') return 'warn';
  if (status === 'revoked') return 'danger';
  return 'idle';
}

type ProvisioningTarget = 'bootstrap' | 'member';

interface PendingProvisioning {
  material: AuthProvisioning;
  username: string;
  target: ProvisioningTarget;
}

function provisioningFrom(response: AuthMutationResponse): AuthProvisioning {
  const value = response.provisioning;
  if (!value
    || typeof value.user_id !== 'string'
    || typeof value.authenticator_id !== 'string'
    || typeof value.secret !== 'string'
    || typeof value.qr_payload !== 'string'
    || typeof value.otpauth_uri !== 'string') {
    throw new Error('The primary node returned an incomplete Auth App provisioning response.');
  }
  return value as AuthProvisioning;
}

function fixtureProvisioning(userId: string): AuthProvisioning {
  const secret = 'JBSWY3DPEHPK3PXP';
  const uri = `otpauth://totp/QLH:${encodeURIComponent(userId)}?secret=${secret}&issuer=QLH&algorithm=SHA1&digits=6&period=30`;
  return { user_id: userId, authenticator_id: `totp_${userId}`, secret, qr_payload: uri, otpauth_uri: uri };
}

const fixtureRecoveryCodes = ['A1C2-E3F4', 'B5D6-G7H8', 'J9K1-L2M3', 'N4P5-Q6R7', 'S8T9-V1W2', 'X3Y4-Z5A6', 'C7D8-E9F1', 'G2H3-J4K5', 'L6M7-N8P9', 'Q1R2-S3T4'];

export function AccountPage() {
  const capability = useAuthCapability();
  const authRequired = capability.data?.required === true;
  const [bootstrapFinished, setBootstrapFinished] = useState(false);
  const [showBootstrap, setShowBootstrap] = useState(false);
  const bootstrapRequired = (capability.data?.bootstrap_available === true || showBootstrap) && !bootstrapFinished;
  const [loggedOut, setLoggedOut] = useState(false);
  const [sessionOverride, setSessionOverride] = useState<AuthSessionResponse | null>(null);
  const session = useAuthSession(authRequired && !loggedOut && !bootstrapRequired);
  const currentSession = sessionOverride ?? session.data;
  const authenticated = authRequired
    ? Boolean(currentSession?.session_id && currentSession.user?.user_id) && !loggedOut
    : false;
  const currentUser = currentSession?.user ?? null;
  const manager = currentUser?.role === 'owner' || currentUser?.role === 'admin';
  const sessions = useAuthSessions(authenticated);
  const users = useManagedUsers(authenticated && manager);
  const bindings = useTailscaleBindings(authenticated);
  const localTailscale = useLocalTailscaleStatus(authenticated);
  const usingFixtures = fixturesEnabled();

  const [activeWorkspace, setActiveWorkspace] = useState<'security' | 'network' | 'users'>('security');
  const [loginForm, setLoginForm] = useState({ username: '', code: '', recoveryCode: '', mode: 'totp' });
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [sessionOverrideList, setSessionOverrideList] = useState<AuthSessionRecord[] | null>(null);
  const [userOverrideList, setUserOverrideList] = useState<AuthUser[] | null>(null);
  const [bindingOverrideList, setBindingOverrideList] = useState<TailscaleBinding[] | null>(null);
  const [newUser, setNewUser] = useState({ username: '', display_name: '', role: 'member' });
  const [tailscaleForm, setTailscaleForm] = useState({ tailnet_id: '', tailscale_user_id: '', node_id: '' });
  const [bootstrapForm, setBootstrapForm] = useState({ username: '', display_name: '' });
  const [provisioning, setProvisioning] = useState<PendingProvisioning | null>(null);
  const [visibleRecoveryCodes, setVisibleRecoveryCodes] = useState<string[]>([]);
  const [recoveryLabel, setRecoveryLabel] = useState('');
  const [recoveryRotationCode, setRecoveryRotationCode] = useState('');

  const sessionList = sessionOverrideList ?? sessions.data?.sessions ?? [];
  const userList = userOverrideList ?? users.data?.users ?? [];
  const bindingList = bindingOverrideList ?? bindings.data?.bindings ?? [];
  const activeBinding = useMemo(() => bindingList.find((binding) => binding.state === 'active') ?? null, [bindingList]);
  const pendingBinding = useMemo(() => bindingList.find((binding) => binding.state === 'pending') ?? null, [bindingList]);

  const refresh = useCallback(() => {
    capability.refresh();
    session.refresh();
    sessions.refresh();
    users.refresh();
    bindings.refresh();
    localTailscale.refresh();
  }, [bindings, capability, localTailscale, session, sessions, users]);
  useRegisterRefresh(refresh);

  const authError = capability.state === 'error' ? capability.error : !bootstrapRequired && session.state === 'error' ? session.error : '';

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy('login');
    setError('');
    try {
      if (usingFixtures) {
        setSessionOverride({ session_id: 'auth_session_fixture_login', expires_at: '2026-08-22T12:00:00.000Z', user: { user_id: 'user_fixture_login', username: loginForm.username.trim() || 'operator', display_name: 'Fixture Operator', role: 'owner', status: 'active', totp_state: 'enabled', active_session_count: 1, aggregate_version: 1 } });
      } else {
        const result = await api.loginAuth(loginForm.username, loginForm.mode === 'recovery' ? '' : loginForm.code, loginForm.mode === 'recovery' ? loginForm.recoveryCode : '');
        setSessionOverride(result);
      }
      setLoggedOut(false);
      setLoginForm({ username: '', code: '', recoveryCode: '', mode: 'totp' });
      pushToast('Authentication successful', 'ok');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleBootstrap = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!bootstrapForm.username.trim()) return;
    setBusy('bootstrap');
    setError('');
    try {
      const material = usingFixtures
        ? fixtureProvisioning('user_fixture_owner')
        : provisioningFrom(await api.bootstrapAuth(bootstrapForm));
      setProvisioning({ material, username: bootstrapForm.username.trim(), target: 'bootstrap' });
      setVisibleRecoveryCodes([]);
      setRecoveryLabel('');
    } catch (nextError) {
      setError(api.describeError(nextError));
      if (nextError instanceof api.ApiError && nextError.status === 409) {
        setBootstrapFinished(true);
        setShowBootstrap(false);
        capability.refresh();
      }
    } finally {
      setBusy('');
    }
  };

  const handleVerifyProvisioning = async (code: string) => {
    if (!provisioning) return;
    setBusy('verify-provisioning');
    setError('');
    try {
      const result = usingFixtures
        ? { recovery_codes: fixtureRecoveryCodes }
        : await api.verifyAuthTotp({
          user_id: provisioning.material.user_id,
          authenticator_id: provisioning.material.authenticator_id,
          code,
        });
      const recoveryCodes = result.recovery_codes ?? [];
      setVisibleRecoveryCodes(recoveryCodes);
      setRecoveryLabel(provisioning.username);
      if (provisioning.target === 'bootstrap') {
        setLoginForm((current) => ({ ...current, username: provisioning.username, code: '', recoveryCode: '', mode: 'totp' }));
      } else if (usingFixtures) {
        setUserOverrideList(userList.map((user) => user.user_id === provisioning.material.user_id ? { ...user, totp_state: 'active' } : user));
      } else {
        await users.refresh();
      }
      setProvisioning(null);
      pushToast('Auth App activated. Recovery codes are ready.', 'ok');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleRotateRecoveryCodes = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!recoveryRotationCode.trim()) return;
    setBusy('rotate-recovery');
    setError('');
    try {
      const result = usingFixtures ? { recovery_codes: fixtureRecoveryCodes } : await api.rotateRecoveryCodes(recoveryRotationCode);
      setVisibleRecoveryCodes(result.recovery_codes ?? []);
      setRecoveryLabel(currentUser?.username || 'current user');
      setRecoveryRotationCode('');
      pushToast('Recovery codes rotated', 'ok');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleProvisionUser = async (user: AuthUser) => {
    setBusy(`provision:${user.user_id}`);
    setError('');
    try {
      const material = usingFixtures
        ? fixtureProvisioning(user.user_id)
        : provisioningFrom(await api.provisionUserTotp(user.user_id));
      setProvisioning({ material, username: user.username, target: 'member' });
      setVisibleRecoveryCodes([]);
      setRecoveryLabel('');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleLogout = async () => {
    setBusy('logout');
    try {
      if (!usingFixtures) await api.logoutAuth();
      setSessionOverride(null);
      setLoggedOut(true);
      pushToast('Session signed out', 'info');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleRevokeSession = async (record: AuthSessionRecord) => {
    setBusy(`session:${record.session_id}`);
    try {
      if (usingFixtures) {
        setSessionOverrideList(sessionList.filter((item) => item.session_id !== record.session_id));
      } else {
        await api.revokeAuthSession(record.session_id);
        await sessions.refresh();
      }
      pushToast('Session revoked', 'ok');
      if (record.current) await handleLogout();
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleCreateUser = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!newUser.username.trim()) return;
    setBusy('create-user');
    try {
      if (usingFixtures) {
        const next: AuthUser = { user_id: `user_${Date.now()}`, username: newUser.username.trim(), display_name: newUser.display_name.trim(), role: newUser.role, status: 'active', totp_state: 'pending', active_session_count: 0, aggregate_version: 1 };
        setUserOverrideList((current) => [...(current ?? userList), next]);
      } else {
        await api.createManagedUser(newUser);
        await users.refresh();
      }
      setNewUser({ username: '', display_name: '', role: 'member' });
      pushToast('User created', 'ok');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleToggleUser = async (user: AuthUser) => {
    const nextStatus = user.status === 'active' ? 'suspended' : 'active';
    setBusy(`user:${user.user_id}`);
    try {
      if (usingFixtures) setUserOverrideList(userList.map((item) => item.user_id === user.user_id ? { ...item, status: nextStatus } : item));
      else {
        await api.updateManagedUser(user.user_id, { expected_version: user.aggregate_version, status: nextStatus });
        await users.refresh();
      }
      pushToast(`User ${nextStatus}`, 'ok');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleInspectTailscale = async () => {
    if (usingFixtures) {
      const candidate = localTailscale.data?.local_status?.candidate;
      if (candidate) setTailscaleForm({ tailnet_id: candidate.tailnet_id || '', tailscale_user_id: candidate.tailscale_user_id || '', node_id: candidate.node_id || '' });
      pushToast('Local Tailscale identity loaded', 'info');
      return;
    }
    setBusy('inspect-tailscale');
    try {
      const result = await api.fetchLocalTailscaleStatus();
      const candidate = result.local_status?.candidate;
      if (!candidate) throw new Error('No Tailscale identity detected');
      setTailscaleForm({ tailnet_id: candidate.tailnet_id || '', tailscale_user_id: candidate.tailscale_user_id || '', node_id: candidate.node_id || '' });
      pushToast('Local Tailscale identity loaded', 'info');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handlePrepareBinding = async () => {
    setBusy('prepare-tailscale');
    try {
      if (usingFixtures) {
        setBindingOverrideList((current) => [{ binding_id: 'binding_fixture_pending', state: 'pending', authorization_method: 'local_status' }, ...(current ?? bindingList)]);
      } else {
        await api.prepareTailscaleBinding('local_status');
        await bindings.refresh();
      }
      pushToast('Tailscale binding request prepared', 'info');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleConfirmBinding = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!pendingBinding) return;
    setBusy('confirm-tailscale');
    try {
      if (usingFixtures) setBindingOverrideList(bindingList.map((binding) => binding.binding_id === pendingBinding.binding_id ? { ...binding, ...tailscaleForm, state: 'active', confirmed_at: new Date().toISOString() } : binding));
      else {
        await api.confirmTailscaleBinding(pendingBinding.binding_id, tailscaleForm);
        await bindings.refresh();
      }
      pushToast('Tailscale binding confirmed', 'ok');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const handleRevokeBinding = async (binding: TailscaleBinding) => {
    setBusy(`revoke-binding:${binding.binding_id}`);
    try {
      if (usingFixtures) setBindingOverrideList(bindingList.map((item) => item.binding_id === binding.binding_id ? { ...item, state: 'revoked' } : item));
      else {
        await api.revokeTailscaleBinding(binding.binding_id);
        await bindings.refresh();
      }
      pushToast('Tailscale binding revoked', 'info');
    } catch (nextError) {
      setError(api.describeError(nextError));
    } finally {
      setBusy('');
    }
  };

  const copyBinding = async () => {
    const value = activeBinding ? `${activeBinding.tailnet_id} / ${activeBinding.tailscale_user_id}` : '';
    try {
      await navigator.clipboard.writeText(value);
      pushToast('Binding identity copied', 'ok');
    } catch {
      pushToast(value || 'No active binding', 'info');
    }
  };

  return (
    <div className="account-page" data-testid="account-page">
      <IronGateCanvas className="account-page__bg" />
      <div className="account-page__content">
        <PageHeader tag="IRON GATE" title="Account & Security" description="Keep identity, sessions, local users, and tailnet bindings visible at one control surface." actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={capability.refreshing || session.refreshing} onClick={refresh}>Refresh</CommandButton>} />

        {capability.state === 'loading' ? <EmptyState kind="loading" title="Checking authentication capability" /> : capability.state === 'error' ? <EmptyState kind="error" title="Authentication capability unavailable" detail={authError} errorKind={capability.errorKind} errorStatus={capability.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={capability.refresh}>Retry</CommandButton>} /> : !authRequired ? (
          <section className="account-panel account-disabled"><ShieldCheck size={24} /><div><h2>Authentication is disabled</h2><p>This node is running in local mode. Account controls will appear when the Auth service is enabled.</p></div></section>
        ) : !authenticated && bootstrapRequired ? (
          visibleRecoveryCodes.length ? (
            <section className="account-login account-panel account-recovery" aria-labelledby="account-recovery-title">
              <div className="account-login__mark"><KeyRound size={26} /></div>
              <div className="account-login__body"><span className="mono-label">ONE-TIME RECOVERY MATERIAL</span><h2 id="account-recovery-title">Recovery codes for {recoveryLabel}</h2><p>Store these offline before continuing.</p><div className="account-recovery-grid">{visibleRecoveryCodes.map((code) => <code key={code}>{code}</code>)}</div><CommandButton icon={LogIn} onClick={() => { setVisibleRecoveryCodes([]); setBootstrapFinished(true); capability.refresh(); }}>Continue to sign in</CommandButton></div>
            </section>
          ) : provisioning ? (
            <AuthProvisioningPanel provisioning={provisioning.material} username={provisioning.username} busy={busy === 'verify-provisioning'} onVerify={(code) => void handleVerifyProvisioning(code)} onDismiss={() => setProvisioning(null)} />
          ) : (
            <section className="account-login account-panel" aria-labelledby="account-bootstrap-title">
              <div className="account-login__mark"><KeyRound size={26} /></div>
              <div className="account-login__body"><span className="mono-label">LOCAL PRIMARY NODE</span><h2 id="account-bootstrap-title">Initialize the owner account</h2><p>{error || 'Create the local owner before the first sign in.'}</p>
                <form className="account-form" onSubmit={(event) => void handleBootstrap(event)}>
                  <label className="account-field"><span>USERNAME</span><input value={bootstrapForm.username} autoComplete="username" onChange={(event) => setBootstrapForm((current) => ({ ...current, username: event.target.value }))} required autoFocus /></label>
                  <label className="account-field"><span>DISPLAY NAME</span><input value={bootstrapForm.display_name} onChange={(event) => setBootstrapForm((current) => ({ ...current, display_name: event.target.value }))} /></label>
                  <CommandButton type="submit" icon={KeyRound} busy={busy === 'bootstrap'}>Create Auth App setup</CommandButton>
                </form>
              </div>
            </section>
          )
        ) : !authenticated ? (
            <section className="account-login account-panel" aria-labelledby="account-login-title">
            <div className="account-login__mark"><KeyRound size={26} /></div>
            <div className="account-login__body"><span className="mono-label">LOCAL PRIMARY NODE</span><h2 id="account-login-title">Sign in to continue</h2><p>{authError || error || 'Use your Auth App code or a recovery code.'}</p>
              <form className="account-form" onSubmit={(event) => void handleLogin(event)}>
                <label className="account-field"><span>USERNAME</span><input value={loginForm.username} autoComplete="username" onChange={(event) => setLoginForm((current) => ({ ...current, username: event.target.value }))} required /></label>
                <div className="account-mode" role="group" aria-label="Authentication method"><button type="button" className={loginForm.mode === 'totp' ? 'is-active' : ''} onClick={() => setLoginForm((current) => ({ ...current, mode: 'totp' }))}>Auth App</button><button type="button" className={loginForm.mode === 'recovery' ? 'is-active' : ''} onClick={() => setLoginForm((current) => ({ ...current, mode: 'recovery' }))}>Recovery code</button></div>
                <label className="account-field"><span>{loginForm.mode === 'totp' ? 'AUTH APP CODE' : 'RECOVERY CODE'}</span><input value={loginForm.mode === 'totp' ? loginForm.code : loginForm.recoveryCode} inputMode="numeric" autoComplete="one-time-code" onChange={(event) => setLoginForm((current) => loginForm.mode === 'totp' ? { ...current, code: event.target.value } : { ...current, recoveryCode: event.target.value })} required /></label>
                <CommandButton type="submit" icon={LogIn} busy={busy === 'login'}>Sign in</CommandButton>
              </form>
              {capability.data?.bootstrap_available !== false ? <button type="button" className="account-link-button" onClick={() => { setShowBootstrap(true); setError(''); }}>Initialize owner account</button> : null}
            </div>
          </section>
        ) : (
          <div className="account-layout">
            <aside className="account-rail">
              <section className="account-panel account-profile"><div className="account-avatar"><UserRound size={20} /></div><div><strong>{currentUser?.display_name || currentUser?.username}</strong><span>{currentUser?.username}</span></div><StatusBadge label={(currentUser?.role || 'member').toUpperCase()} tone={roleTone(currentUser?.role)} /></section>
              <nav className="account-nav" aria-label="Account sections"><button type="button" className={activeWorkspace === 'security' ? 'is-active' : ''} onClick={() => setActiveWorkspace('security')}><ShieldCheck size={15} />Security <span>{sessionList.length}</span></button><button type="button" className={activeWorkspace === 'network' ? 'is-active' : ''} onClick={() => setActiveWorkspace('network')}><Network size={15} />Tailscale <span>{activeBinding ? 'ON' : 'OFF'}</span></button>{manager ? <button type="button" className={activeWorkspace === 'users' ? 'is-active' : ''} onClick={() => setActiveWorkspace('users')}><Users size={15} />Users <span>{userList.length}</span></button> : null}</nav>
              <CommandButton variant="danger" icon={LogOut} busy={busy === 'logout'} onClick={() => void handleLogout()}>Sign out</CommandButton>
            </aside>

            <main className="account-main">
              {error ? <div className="account-alert" role="alert">{error}</div> : null}
              {provisioning ? <AuthProvisioningPanel provisioning={provisioning.material} username={provisioning.username} busy={busy === 'verify-provisioning'} onVerify={(code) => void handleVerifyProvisioning(code)} onDismiss={() => setProvisioning(null)} /> : null}
              {visibleRecoveryCodes.length ? <section className="account-panel account-recovery" aria-labelledby="account-recovery-codes-title"><SectionHead title={`Recovery codes${recoveryLabel ? ` · ${recoveryLabel}` : ''}`} hint="Shown once" actions={<button type="button" className="account-icon-button" title="Dismiss recovery codes" aria-label="Dismiss recovery codes" onClick={() => setVisibleRecoveryCodes([])}><XCircle size={15} /></button>} /><div className="account-recovery-grid">{visibleRecoveryCodes.map((code) => <code key={code}>{code}</code>)}</div><p className="account-note">Keep these offline. They are cleared when this panel is dismissed.</p></section> : null}
              {activeWorkspace === 'security' ? <>
                <section className="account-panel"><SectionHead title="Active sessions" hint={`${sessionList.filter((item) => item.active).length} active`} actions={<span className="mono-label">TOKEN LIFECYCLE</span>} />{sessions.state === 'error' ? <EmptyState kind="error" title="Session list unavailable" detail={sessions.error} errorKind={sessions.errorKind} errorStatus={sessions.errorStatus} compact /> : <div className="account-session-list">{sessionList.map((record) => <article className={`account-session-row${record.current ? ' is-current' : ''}`} key={record.session_id}><div><strong>{record.current ? 'Current browser' : 'Other session'}</strong><span>{record.session_id}</span><small>Last seen {dateLabel(record.last_seen_at)} · Expires {dateLabel(record.expires_at)}</small></div><div className="account-session-actions"><StatusBadge label={record.active ? 'ACTIVE' : 'ENDED'} tone={record.active ? 'ok' : 'idle'} size="sm" />{record.active ? <button type="button" className="account-icon-button" title="Revoke session" aria-label={`Revoke ${record.session_id}`} disabled={Boolean(busy)} onClick={() => void handleRevokeSession(record)}><XCircle size={15} /></button> : null}</div></article>)}</div>}</section>
                <section className="account-panel"><SectionHead title="Recovery posture" hint="Recovery codes are shown only after rotation" /><div className="account-security-facts"><div><KeyRound size={16} /><strong>Auth App enabled</strong><span>{currentUser?.totp_state || 'unknown'}</span></div><div><Wifi size={16} /><strong>Tailnet binding</strong><span>{activeBinding ? 'active' : 'not bound'}</span></div></div><form className="account-form account-recovery-form" onSubmit={(event) => void handleRotateRecoveryCodes(event)}><label className="account-field"><span>CURRENT AUTH APP CODE</span><input value={recoveryRotationCode} inputMode="numeric" autoComplete="one-time-code" onChange={(event) => setRecoveryRotationCode(event.target.value)} required /></label><CommandButton type="submit" icon={RefreshCw} busy={busy === 'rotate-recovery'}>Rotate recovery codes</CommandButton></form><p className="account-note">For high-risk actions, keep a recovery code offline and revoke sessions from devices you no longer control.</p></section>
              </> : null}

              {activeWorkspace === 'network' ? <section className="account-panel"><SectionHead title="Tailscale binding" hint="Bind stable identity only; QLH never stores a Tailscale password or token" actions={<StatusBadge label={activeBinding ? 'BOUND' : 'NOT BOUND'} tone={activeBinding ? 'ok' : 'idle'} />} />{bindings.state === 'error' ? <EmptyState kind="error" title="Tailscale bindings unavailable" detail={bindings.error} errorKind={bindings.errorKind} errorStatus={bindings.errorStatus} compact /> : <><div className="account-binding-current">{activeBinding ? <><div><strong>{activeBinding.tailnet_id}</strong><span>{activeBinding.tailscale_user_id} · {activeBinding.node_id || 'no node id'}</span></div><button type="button" className="account-icon-button" title="Copy binding identity" aria-label="Copy binding identity" onClick={() => void copyBinding()}><Copy size={15} /></button></> : <p>No active binding. Inspect local status, then prepare a binding request.</p>}</div><div className="account-tailscale-actions"><CommandButton variant="ghost" size="sm" icon={Wifi} busy={busy === 'inspect-tailscale'} onClick={() => void handleInspectTailscale()}>Inspect local status</CommandButton><CommandButton size="sm" icon={Plus} busy={busy === 'prepare-tailscale'} onClick={() => void handlePrepareBinding()} disabled={Boolean(pendingBinding)}>Prepare binding</CommandButton></div>{localTailscale.data?.local_status ? <div className="account-detected" role="status"><strong>{localTailscale.data.local_status.candidate?.tailnet_display_name || 'Local Tailscale'}</strong><span>{localTailscale.data.local_status.candidate?.hostname || 'identity detected'} · {localTailscale.data.local_status.state}</span></div> : null}{pendingBinding ? <form className="account-form account-binding-form" onSubmit={(event) => void handleConfirmBinding(event)}><span className="mono-label">PENDING CONFIRMATION</span><label className="account-field"><span>TAILNET ID</span><input value={tailscaleForm.tailnet_id} onChange={(event) => setTailscaleForm((current) => ({ ...current, tailnet_id: event.target.value }))} required /></label><label className="account-field"><span>TAILSCALE USER ID</span><input value={tailscaleForm.tailscale_user_id} onChange={(event) => setTailscaleForm((current) => ({ ...current, tailscale_user_id: event.target.value }))} required /></label><label className="account-field"><span>NODE ID</span><input value={tailscaleForm.node_id} onChange={(event) => setTailscaleForm((current) => ({ ...current, node_id: event.target.value }))} /></label><CommandButton type="submit" icon={Check} busy={busy === 'confirm-tailscale'}>Confirm binding</CommandButton></form> : null}{activeBinding ? <button type="button" className="account-danger-link" onClick={() => void handleRevokeBinding(activeBinding)} disabled={Boolean(busy)}><Trash2 size={14} />Revoke current binding</button> : null}</>}</section> : null}

              {activeWorkspace === 'users' && manager ? <section className="account-panel"><SectionHead title="Managed users" hint="Owner and admin roles are enforced by the backend" actions={<span className="mono-label">{userList.length} ACCOUNTS</span>} />{users.state === 'error' ? <EmptyState kind="error" title="User directory unavailable" detail={users.error} errorKind={users.errorKind} errorStatus={users.errorStatus} compact /> : <><form className="account-create-user" onSubmit={(event) => void handleCreateUser(event)}><label className="account-field"><span>USERNAME</span><input value={newUser.username} onChange={(event) => setNewUser((current) => ({ ...current, username: event.target.value }))} required /></label><label className="account-field"><span>DISPLAY NAME</span><input value={newUser.display_name} onChange={(event) => setNewUser((current) => ({ ...current, display_name: event.target.value }))} /></label><label className="account-field"><span>ROLE</span><select value={newUser.role} onChange={(event) => setNewUser((current) => ({ ...current, role: event.target.value }))}><option value="member">member</option>{currentUser?.role === 'owner' ? <option value="admin">admin</option> : null}</select></label><CommandButton type="submit" icon={Plus} busy={busy === 'create-user'}>Create user</CommandButton></form><div className="account-user-list">{userList.map((user) => <article className="account-user-row" key={user.user_id}><div><strong>{user.display_name || user.username}</strong><span>{user.username} · {user.role}</span></div><div className="account-user-meta"><StatusBadge label={(user.status || 'unknown').toUpperCase()} tone={statusTone(user.status)} size="sm" /><StatusBadge label={(user.totp_state || 'none').toUpperCase()} tone={user.totp_state === 'active' || user.totp_state === 'enabled' ? 'ok' : 'idle'} size="sm" /><span>{user.active_session_count ?? 0} sessions</span>{user.status === 'active' && user.totp_state !== 'active' && user.totp_state !== 'enabled' ? <button type="button" className="account-icon-button" title="Provision Auth App" aria-label={`Provision Auth App for ${user.username}`} disabled={Boolean(busy)} onClick={() => void handleProvisionUser(user)}><KeyRound size={15} /></button> : null}<button type="button" className="account-icon-button" title={user.status === 'active' ? 'Suspend user' : 'Activate user'} aria-label={`${user.status === 'active' ? 'Suspend' : 'Activate'} ${user.username}`} disabled={user.user_id === currentUser?.user_id || Boolean(busy)} onClick={() => void handleToggleUser(user)}>{user.status === 'active' ? <XCircle size={15} /> : <Check size={15} />}</button></div></article>)}</div></>}</section> : null}
            </main>
          </div>
        )}
      </div>
    </div>
  );
}
