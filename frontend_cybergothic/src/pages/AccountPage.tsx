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
import type { AuthSessionRecord, AuthSessionResponse, AuthUser, TailscaleBinding } from '../data/types';
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

export function AccountPage() {
  const capability = useAuthCapability();
  const authRequired = capability.data?.required === true;
  const [loggedOut, setLoggedOut] = useState(false);
  const [sessionOverride, setSessionOverride] = useState<AuthSessionResponse | null>(null);
  const session = useAuthSession(authRequired && !loggedOut);
  const currentSession = sessionOverride ?? session.data;
  const authenticated = authRequired ? Boolean(currentSession) && !loggedOut : false;
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

  const authError = capability.state === 'error' ? capability.error : session.state === 'error' ? session.error : '';

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
              {activeWorkspace === 'security' ? <>
                <section className="account-panel"><SectionHead title="Active sessions" hint={`${sessionList.filter((item) => item.active).length} active`} actions={<span className="mono-label">TOKEN LIFECYCLE</span>} />{sessions.state === 'error' ? <EmptyState kind="error" title="Session list unavailable" detail={sessions.error} errorKind={sessions.errorKind} errorStatus={sessions.errorStatus} compact /> : <div className="account-session-list">{sessionList.map((record) => <article className={`account-session-row${record.current ? ' is-current' : ''}`} key={record.session_id}><div><strong>{record.current ? 'Current browser' : 'Other session'}</strong><span>{record.session_id}</span><small>Last seen {dateLabel(record.last_seen_at)} · Expires {dateLabel(record.expires_at)}</small></div><div className="account-session-actions"><StatusBadge label={record.active ? 'ACTIVE' : 'ENDED'} tone={record.active ? 'ok' : 'idle'} size="sm" />{record.active ? <button type="button" className="account-icon-button" title="Revoke session" aria-label={`Revoke ${record.session_id}`} disabled={Boolean(busy)} onClick={() => void handleRevokeSession(record)}><XCircle size={15} /></button> : null}</div></article>)}</div>}</section>
                <section className="account-panel"><SectionHead title="Recovery posture" hint="Recovery codes are shown only after rotation" /><div className="account-security-facts"><div><KeyRound size={16} /><strong>Auth App enabled</strong><span>{currentUser?.totp_state || 'unknown'}</span></div><div><Wifi size={16} /><strong>Tailnet binding</strong><span>{activeBinding ? 'active' : 'not bound'}</span></div></div><p className="account-note">For high-risk actions, keep a recovery code offline and revoke sessions from devices you no longer control.</p></section>
              </> : null}

              {activeWorkspace === 'network' ? <section className="account-panel"><SectionHead title="Tailscale binding" hint="Bind stable identity only; QLH never stores a Tailscale password or token" actions={<StatusBadge label={activeBinding ? 'BOUND' : 'NOT BOUND'} tone={activeBinding ? 'ok' : 'idle'} />} />{bindings.state === 'error' ? <EmptyState kind="error" title="Tailscale bindings unavailable" detail={bindings.error} errorKind={bindings.errorKind} errorStatus={bindings.errorStatus} compact /> : <><div className="account-binding-current">{activeBinding ? <><div><strong>{activeBinding.tailnet_id}</strong><span>{activeBinding.tailscale_user_id} · {activeBinding.node_id || 'no node id'}</span></div><button type="button" className="account-icon-button" title="Copy binding identity" aria-label="Copy binding identity" onClick={() => void copyBinding()}><Copy size={15} /></button></> : <p>No active binding. Inspect local status, then prepare a binding request.</p>}</div><div className="account-tailscale-actions"><CommandButton variant="ghost" size="sm" icon={Wifi} busy={busy === 'inspect-tailscale'} onClick={() => void handleInspectTailscale()}>Inspect local status</CommandButton><CommandButton size="sm" icon={Plus} busy={busy === 'prepare-tailscale'} onClick={() => void handlePrepareBinding()} disabled={Boolean(pendingBinding)}>Prepare binding</CommandButton></div>{localTailscale.data?.local_status ? <div className="account-detected" role="status"><strong>{localTailscale.data.local_status.candidate?.tailnet_display_name || 'Local Tailscale'}</strong><span>{localTailscale.data.local_status.candidate?.hostname || 'identity detected'} · {localTailscale.data.local_status.state}</span></div> : null}{pendingBinding ? <form className="account-form account-binding-form" onSubmit={(event) => void handleConfirmBinding(event)}><span className="mono-label">PENDING CONFIRMATION</span><label className="account-field"><span>TAILNET ID</span><input value={tailscaleForm.tailnet_id} onChange={(event) => setTailscaleForm((current) => ({ ...current, tailnet_id: event.target.value }))} required /></label><label className="account-field"><span>TAILSCALE USER ID</span><input value={tailscaleForm.tailscale_user_id} onChange={(event) => setTailscaleForm((current) => ({ ...current, tailscale_user_id: event.target.value }))} required /></label><label className="account-field"><span>NODE ID</span><input value={tailscaleForm.node_id} onChange={(event) => setTailscaleForm((current) => ({ ...current, node_id: event.target.value }))} /></label><CommandButton type="submit" icon={Check} busy={busy === 'confirm-tailscale'}>Confirm binding</CommandButton></form> : null}{activeBinding ? <button type="button" className="account-danger-link" onClick={() => void handleRevokeBinding(activeBinding)} disabled={Boolean(busy)}><Trash2 size={14} />Revoke current binding</button> : null}</>}</section> : null}

              {activeWorkspace === 'users' && manager ? <section className="account-panel"><SectionHead title="Managed users" hint="Owner and admin roles are enforced by the backend" actions={<span className="mono-label">{userList.length} ACCOUNTS</span>} />{users.state === 'error' ? <EmptyState kind="error" title="User directory unavailable" detail={users.error} errorKind={users.errorKind} errorStatus={users.errorStatus} compact /> : <><form className="account-create-user" onSubmit={(event) => void handleCreateUser(event)}><label className="account-field"><span>USERNAME</span><input value={newUser.username} onChange={(event) => setNewUser((current) => ({ ...current, username: event.target.value }))} required /></label><label className="account-field"><span>DISPLAY NAME</span><input value={newUser.display_name} onChange={(event) => setNewUser((current) => ({ ...current, display_name: event.target.value }))} /></label><label className="account-field"><span>ROLE</span><select value={newUser.role} onChange={(event) => setNewUser((current) => ({ ...current, role: event.target.value }))}><option value="member">member</option>{currentUser?.role === 'owner' ? <option value="admin">admin</option> : null}</select></label><CommandButton type="submit" icon={Plus} busy={busy === 'create-user'}>Create user</CommandButton></form><div className="account-user-list">{userList.map((user) => <article className="account-user-row" key={user.user_id}><div><strong>{user.display_name || user.username}</strong><span>{user.username} · {user.role}</span></div><div className="account-user-meta"><StatusBadge label={(user.status || 'unknown').toUpperCase()} tone={statusTone(user.status)} size="sm" /><span>{user.active_session_count ?? 0} sessions</span><button type="button" className="account-icon-button" title={user.status === 'active' ? 'Suspend user' : 'Activate user'} aria-label={`${user.status === 'active' ? 'Suspend' : 'Activate'} ${user.username}`} disabled={user.user_id === currentUser?.user_id || Boolean(busy)} onClick={() => void handleToggleUser(user)}>{user.status === 'active' ? <XCircle size={15} /> : <Check size={15} />}</button></div></article>)}</div></>}</section> : null}
            </main>
          </div>
        )}
      </div>
    </div>
  );
}
