import { useCallback, useEffect, useMemo, useState } from 'react';
import QRCode from 'qrcode';
import {
  copyText as copyAuthText,
} from './auth-utils';
import {
  createManagedUser,
  fetchAuthSessions,
  fetchManagedUsers,
  provisionUserTotp,
  revokeAuthSession,
  revokeManagedUser,
  rotateAuthRecoveryCodes,
  updateManagedUser,
  verifyAuthProvisioning,
} from '../api/client';
import TailscaleBindingPanel from './TailscaleBindingPanel';

function messageOf(error) {
  return error?.detail || error?.message || '操作失败';
}

function roleLabel(role) {
  return { owner: 'owner', admin: 'admin', member: 'member' }[role] || role;
}

function statusLabel(status) {
  return { active: '启用', suspended: '暂停', revoked: '撤销' }[status] || status;
}

function formatDate(value) {
  if (!value) return '-';
  try { return new Date(value).toLocaleString(); } catch (_) { return value; }
}

const ACCOUNT_WORKSPACES = Object.freeze([
  { id: 'security', label: '安全与会话' },
  { id: 'network', label: '组网绑定' },
  { id: 'users', label: '用户管理', managerOnly: true },
]);

export default function UserManagementPanel({ authSession, onLogout, onToast, onClose }) {
  const currentUser = authSession?.user || {};
  const isManager = currentUser.role === 'owner' || currentUser.role === 'admin';
  const [users, setUsers] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [recoveryCode, setRecoveryCode] = useState('');
  const [recoveryCodes, setRecoveryCodes] = useState([]);
  const [newUser, setNewUser] = useState({ username: '', displayName: '', role: 'member' });
  const [editing, setEditing] = useState(null);
  const [provisioning, setProvisioning] = useState(null);
  const [provisioningMode, setProvisioningMode] = useState('qr');
  const [provisionCode, setProvisionCode] = useState('');
  const [provisionQr, setProvisionQr] = useState('');
  const [provisionRecovery, setProvisionRecovery] = useState([]);
  const [activeWorkspace, setActiveWorkspace] = useState('security');

  const notifyError = useCallback((nextError) => {
    setError(messageOf(nextError));
    onToast?.({ type: 'error', msg: messageOf(nextError) });
  }, [onToast]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [sessionData, userData] = await Promise.all([
        fetchAuthSessions(),
        isManager ? fetchManagedUsers() : Promise.resolve({ users: [] }),
      ]);
      setSessions(sessionData.sessions || []);
      setUsers(userData.users || []);
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setLoading(false);
    }
  }, [isManager, notifyError]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    if (!provisioning?.qr_payload) {
      setProvisionQr('');
      return undefined;
    }
    QRCode.toDataURL(provisioning.qr_payload, {
      width: 224,
      margin: 1,
      errorCorrectionLevel: 'M',
      color: { dark: '#111827', light: '#ffffff' },
    }).then((dataUrl) => {
      if (!cancelled) setProvisionQr(dataUrl);
    }).catch((nextError) => {
      if (!cancelled) notifyError(nextError);
    });
    return () => { cancelled = true; };
  }, [provisioning, notifyError]);

  const activeSessions = useMemo(() => sessions.filter((entry) => entry.active), [sessions]);

  const handleRotateRecovery = async (event) => {
    event.preventDefault();
    setBusy(true);
    try {
      const result = await rotateAuthRecoveryCodes(recoveryCode);
      setRecoveryCodes(result.recovery_codes || []);
      setRecoveryCode('');
      onToast?.({ type: 'success', msg: '恢复码已轮换' });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const handleRevokeSession = async (sessionId) => {
    setBusy(true);
    try {
      await revokeAuthSession(sessionId);
      onToast?.({ type: 'success', msg: '会话已撤销' });
      if (sessionId === authSession?.session_id) {
        await onLogout?.();
      } else {
        await refresh();
      }
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const handleCreate = async (event) => {
    event.preventDefault();
    setBusy(true);
    try {
      const result = await createManagedUser(newUser);
      setNewUser({ username: '', displayName: '', role: 'member' });
      onToast?.({ type: 'success', msg: `用户 ${result.user?.username || ''} 已创建` });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const handleUpdate = async (event) => {
    event.preventDefault();
    setBusy(true);
    try {
      await updateManagedUser(editing.user_id, editing);
      setEditing(null);
      onToast?.({ type: 'success', msg: '用户资料已更新' });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const handleToggleStatus = async (user) => {
    const nextStatus = user.status === 'active' ? 'suspended' : 'active';
    setBusy(true);
    try {
      await updateManagedUser(user.user_id, {
        expectedVersion: user.aggregate_version,
        status: nextStatus,
      });
      onToast?.({ type: 'success', msg: `${user.username} 已${nextStatus === 'active' ? '启用' : '暂停'}` });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const startProvisioning = async (user) => {
    setBusy(true);
    setError('');
    try {
      const result = await provisionUserTotp(user.user_id);
      setProvisioning({ ...result.provisioning, username: user.username });
      setProvisioningMode('qr');
      setProvisionCode('');
      setProvisionRecovery([]);
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const verifyProvisioning = async (event) => {
    event.preventDefault();
    setBusy(true);
    try {
      const result = await verifyAuthProvisioning({
        userId: provisioning.user_id,
        authenticatorId: provisioning.authenticator_id,
        code: provisionCode,
      });
      setProvisionRecovery(result.recovery_codes || []);
      setProvisionCode('');
      onToast?.({ type: 'success', msg: 'Auth App 已启用' });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const copy = async (value) => {
    try {
      await copyAuthText(value);
      onToast?.({ type: 'success', msg: '已复制' });
    } catch (nextError) {
      notifyError(nextError);
    }
  };

  return (
    <section className="account-workspace" aria-labelledby="account-title">
      <header className="workspace-header account-header">
        <div>
          <span className="workspace-kicker">LOCAL IDENTITY</span>
          <h2 id="account-title">账户与安全</h2>
          <p>{currentUser.display_name || currentUser.username} · {roleLabel(currentUser.role)}</p>
        </div>
        <div className="account-header-actions">
          <button type="button" className="quiet-btn" onClick={onClose}>返回工作区</button>
          <button type="button" className="quiet-btn" onClick={refresh} disabled={loading}>刷新</button>
          <button type="button" className="danger-btn" onClick={onLogout} disabled={busy}>退出登录</button>
        </div>
      </header>

      {error && <div className="account-alert" role="alert">{error}</div>}

      <div className="account-workspace-tabs" role="tablist" aria-label="账户与安全工作区">
        {ACCOUNT_WORKSPACES
          .filter((workspace) => !workspace.managerOnly || isManager)
          .map(({ id, label }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={activeWorkspace === id}
              className={activeWorkspace === id ? 'active' : ''}
              onClick={() => setActiveWorkspace(id)}
            >
              {label}
            </button>
          ))}
      </div>

      {activeWorkspace === 'security' && (
      <div className="account-grid">
        <section className="account-section">
          <div className="section-heading"><h3>当前会话</h3><span>{activeSessions.length} 个活跃</span></div>
          <div className="session-table">
            {loading && <div className="account-empty">加载中...</div>}
            {!loading && sessions.length === 0 && <div className="account-empty">暂无会话</div>}
            {!loading && sessions.map((session) => (
              <div className={`session-row${session.current ? ' current' : ''}`} key={session.session_id}>
                <div>
                  <strong>{session.current ? '本次登录' : '其他登录'}</strong>
                  <span>最近活动 {formatDate(session.last_seen_at)}</span>
                  <small>创建于 {formatDate(session.created_at)}</small>
                </div>
                <div className="session-row-actions">
                  <span className={`state-dot ${session.active ? 'active' : 'inactive'}`}>{session.active ? '活跃' : '已结束'}</span>
                  {session.active && <button type="button" className="danger-link" onClick={() => handleRevokeSession(session.session_id)} disabled={busy}>撤销</button>}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="account-section">
          <div className="section-heading"><h3>恢复码</h3><span>仅显示本次生成结果</span></div>
          <form className="account-form" onSubmit={handleRotateRecovery}>
            <label>当前 Auth App 验证码<input value={recoveryCode} onChange={(event) => setRecoveryCode(event.target.value)} inputMode="numeric" autoComplete="one-time-code" required /></label>
            <button type="submit" className="primary-btn" disabled={busy}>轮换恢复码</button>
          </form>
          {recoveryCodes.length > 0 && (
            <div className="recovery-output">
              <div className="recovery-output-head"><strong>请立即保存</strong><button type="button" className="quiet-btn" onClick={() => copy(recoveryCodes.join('\n'))}>复制全部</button></div>
              <div className="recovery-code-grid">{recoveryCodes.map((code) => <code key={code}>{code}</code>)}</div>
            </div>
          )}
        </section>
      </div>
      )}

      {activeWorkspace === 'network' && <TailscaleBindingPanel onToast={onToast} />}

      {activeWorkspace === 'users' && isManager && (
        <section className="account-section manager-section">
          <div className="section-heading"><h3>本地主节点用户</h3><span>{users.length} 个账户</span></div>
          <form className="create-user-form" onSubmit={handleCreate}>
            <input aria-label="用户名" placeholder="用户名" value={newUser.username} onChange={(event) => setNewUser((prev) => ({ ...prev, username: event.target.value }))} required />
            <input aria-label="显示名称" placeholder="显示名称" value={newUser.displayName} onChange={(event) => setNewUser((prev) => ({ ...prev, displayName: event.target.value }))} />
            <select aria-label="角色" value={newUser.role} onChange={(event) => setNewUser((prev) => ({ ...prev, role: event.target.value }))}>
              <option value="member">member</option>
              {currentUser.role === 'owner' && <option value="admin">admin</option>}
            </select>
            <button type="submit" className="primary-btn" disabled={busy}>创建用户</button>
          </form>
          <div className="user-list">
            {users.map((user) => (
              <div className="user-row" key={user.user_id}>
                <div className="user-row-main"><strong>{user.display_name || user.username}</strong><span>{user.username} · {roleLabel(user.role)}</span></div>
                <div className="user-row-meta"><span className={`user-status ${user.status}`}>{statusLabel(user.status)}</span><span>Auth {user.totp_state}</span><span>{user.active_session_count} 会话</span></div>
                <div className="user-row-actions">
                  <button type="button" className="quiet-btn" onClick={() => setEditing({ ...user, displayName: user.display_name, expectedVersion: user.aggregate_version })}>编辑</button>
                  <button type="button" className="quiet-btn" onClick={() => startProvisioning(user)} disabled={busy || user.status !== 'active' || (currentUser.role === 'admin' && user.role === 'owner')}>配置 Auth App</button>
                  {user.user_id !== currentUser.user_id && user.status !== 'revoked' && <button type="button" className="danger-link" onClick={() => handleToggleStatus(user)} disabled={busy}>{user.status === 'active' ? '暂停' : '启用'}</button>}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {editing && (
        <div className="account-dialog-backdrop" role="presentation">
          <form className="account-dialog" onSubmit={handleUpdate}>
            <div className="section-heading"><h3>编辑用户</h3><button type="button" className="icon-close" onClick={() => setEditing(null)} aria-label="关闭">×</button></div>
            <label>显示名称<input value={editing.displayName || ''} onChange={(event) => setEditing((prev) => ({ ...prev, displayName: event.target.value }))} /></label>
            <label>角色<select value={editing.role} onChange={(event) => setEditing((prev) => ({ ...prev, role: event.target.value }))} disabled={currentUser.role !== 'owner'}><option value="member">member</option><option value="admin">admin</option><option value="owner">owner</option></select></label>
            <div className="dialog-actions"><button type="button" className="quiet-btn" onClick={() => setEditing(null)}>取消</button><button type="submit" className="primary-btn" disabled={busy}>保存</button></div>
          </form>
        </div>
      )}

      {provisioning && (
        <div className="account-dialog-backdrop" role="presentation">
          <form className="account-dialog provisioning-dialog" onSubmit={verifyProvisioning}>
            <div className="section-heading"><h3>配置 Auth App · {provisioning.username}</h3><button type="button" className="icon-close" onClick={() => setProvisioning(null)} aria-label="关闭">×</button></div>
            <div className="auth-segmented" role="group" aria-label="密钥下发方式"><button type="button" className={provisioningMode === 'qr' ? 'active' : ''} onClick={() => setProvisioningMode('qr')}>二维码</button><button type="button" className={provisioningMode === 'string' ? 'active' : ''} onClick={() => setProvisioningMode('string')}>字符串</button></div>
            {provisioningMode === 'qr' ? <div className="account-qr-box">{provisionQr ? <img src={provisionQr} alt="Auth App 配置二维码" /> : <span>生成中...</span>}</div> : <div className="secret-box"><code>{provisioning.secret}</code><button type="button" className="quiet-btn" onClick={() => copy(provisioning.secret)}>复制</button></div>}
            <button type="button" className="quiet-btn full-btn" onClick={() => copy(provisioning.otpauth_uri)}>复制 otpauth 字符串</button>
            <label>输入 Auth App 验证码<input value={provisionCode} onChange={(event) => setProvisionCode(event.target.value)} inputMode="numeric" autoComplete="one-time-code" required /></label>
            <div className="dialog-actions"><button type="button" className="quiet-btn" onClick={() => setProvisioning(null)}>稍后验证</button><button type="submit" className="primary-btn" disabled={busy}>确认启用</button></div>
            {provisionRecovery.length > 0 && <div className="recovery-output"><div className="recovery-output-head"><strong>恢复码</strong><button type="button" className="quiet-btn" onClick={() => copy(provisionRecovery.join('\n'))}>复制全部</button></div><div className="recovery-code-grid">{provisionRecovery.map((code) => <code key={code}>{code}</code>)}</div></div>}
          </form>
        </div>
      )}
    </section>
  );
}
