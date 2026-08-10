import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  confirmTailscaleBinding,
  fetchLocalTailscaleStatus,
  fetchTailscaleBindings,
  prepareTailscaleBinding,
  revokeTailscaleBinding,
} from '../api/client';

function messageOf(error) {
  return error?.detail || error?.message || 'Tailscale 操作失败';
}

function stateLabel(state) {
  return {
    active: '已绑定',
    pending: '待确认',
    revoked: '已撤销',
    expired: '已过期',
  }[state] || state;
}

function methodLabel(method) {
  return method === 'tailscale_cli' ? '官方 CLI' : '本机状态';
}

function formatDate(value) {
  if (!value) return '-';
  try { return new Date(value).toLocaleString(); } catch (_) { return value; }
}

function localStatusMessage(status) {
  return {
    cli_not_found: '未检测到 Tailscale CLI，请先安装官方客户端。',
    not_running: 'Tailscale 服务尚未运行。',
    not_logged_in: '本机 Tailscale 尚未登录。',
    incomplete_identity: '本机状态缺少稳定身份字段，请改用官方 CLI 手动确认。',
    invalid_response: 'Tailscale CLI 返回了无法识别的状态。',
    unavailable: '暂时无法读取本机 Tailscale 状态。',
  }[status?.state] || '暂时无法读取本机 Tailscale 状态。';
}

export default function TailscaleBindingPanel({ userId = '', onToast }) {
  const [bindings, setBindings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState('');
  const [authorizationMethod, setAuthorizationMethod] = useState('local_status');
  const [pending, setPending] = useState(null);
  const [tailnetId, setTailnetId] = useState('');
  const [tailscaleUserId, setTailscaleUserId] = useState('');
  const [nodeId, setNodeId] = useState('');
  const [localStatus, setLocalStatus] = useState(null);

  const activeBinding = useMemo(() => bindings.find((entry) => entry.state === 'active') || null, [bindings]);
  const historicalBindings = useMemo(() => bindings.filter((entry) => entry.state !== 'active'), [bindings]);

  const notifyError = useCallback((nextError) => {
    const message = messageOf(nextError);
    setError(message);
    onToast?.({ type: 'error', msg: message });
  }, [onToast]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const result = await fetchTailscaleBindings(userId);
      const nextBindings = result.bindings || [];
      setBindings(nextBindings);
      setPending(nextBindings.find((entry) => entry.state === 'pending') || null);
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setLoading(false);
    }
  }, [notifyError, userId]);

  useEffect(() => { refresh(); }, [refresh]);

  const selectAuthorizationMethod = (method) => {
    setAuthorizationMethod(method);
    if (method === 'tailscale_cli') {
      setLocalStatus(null);
      setTailnetId('');
      setTailscaleUserId('');
      setNodeId('');
    }
  };

  const handleInspectLocalStatus = async () => {
    setInspecting(true);
    setError('');
    try {
      const result = await fetchLocalTailscaleStatus();
      const status = result.local_status || null;
      setLocalStatus(status);
      if (!status?.available || !status.candidate) {
        const message = localStatusMessage(status);
        setError(message);
        onToast?.({ type: 'error', msg: message });
        return;
      }
      setTailnetId(status.candidate.tailnet_id || '');
      setTailscaleUserId(status.candidate.tailscale_user_id || '');
      setNodeId(status.candidate.node_id || '');
      onToast?.({ type: 'success', msg: '已读取本机 Tailscale 身份候选' });
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setInspecting(false);
    }
  };

  const handlePrepare = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const result = await prepareTailscaleBinding({ userId, authorizationMethod });
      setPending(result.binding || null);
      onToast?.({ type: 'success', msg: '已创建待确认的 Tailscale 换网请求' });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const handleConfirm = async (event) => {
    event.preventDefault();
    if (!pending) return;
    setBusy(true);
    setError('');
    try {
      await confirmTailscaleBinding(pending.binding_id, { tailnetId, tailscaleUserId, nodeId });
      setPending(null);
      setTailnetId('');
      setTailscaleUserId('');
      setNodeId('');
      setLocalStatus(null);
      onToast?.({ type: 'success', msg: 'Tailscale 绑定已切换' });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  const handleRevoke = async (bindingId) => {
    setBusy(true);
    setError('');
    try {
      await revokeTailscaleBinding(bindingId);
      onToast?.({ type: 'success', msg: 'Tailscale 绑定已撤销' });
      await refresh();
    } catch (nextError) {
      notifyError(nextError);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="account-section tailscale-section" aria-labelledby="tailscale-binding-title">
      <div className="section-heading">
        <div>
          <h3 id="tailscale-binding-title">Tailscale 组网绑定</h3>
          <p className="section-note">绑定只保存 tailnet、用户和节点稳定 ID，资产仍由本地主节点管理。</p>
        </div>
        <button type="button" className="quiet-btn" onClick={refresh} disabled={loading || busy}>刷新</button>
      </div>

      {error && <div className="account-alert" role="alert">{error}</div>}

      <div className="binding-current">
        <div className="binding-status-line">
          <span className={`user-status ${activeBinding ? 'active' : 'suspended'}`}>{activeBinding ? '当前已绑定' : '尚未绑定'}</span>
          {activeBinding && <span>最近确认 {formatDate(activeBinding.confirmed_at)}</span>}
        </div>
        {activeBinding ? (
          <dl className="binding-details">
            <div><dt>tailnet ID</dt><dd>{activeBinding.tailnet_id}</dd></div>
            <div><dt>Tailscale 用户 ID</dt><dd>{activeBinding.tailscale_user_id}</dd></div>
            <div><dt>节点 ID</dt><dd>{activeBinding.node_id || '未提供'}</dd></div>
            <div><dt>授权方式</dt><dd>{methodLabel(activeBinding.authorization_method)}</dd></div>
          </dl>
        ) : (
          <p className="binding-empty">首次使用时，可先在 Tailscale 官方客户端注册或登录，再回到这里完成绑定。</p>
        )}
      </div>

      <form className="binding-prepare" onSubmit={handlePrepare}>
        <div className="binding-form-title">
          <strong>{activeBinding ? '切换到另一组网' : '绑定现有组网'}</strong>
          <span>{pending ? '已有待确认请求' : '新授权会先保持 pending，确认后才会撤销旧绑定'}</span>
        </div>
        <div className="auth-segmented binding-methods" role="group" aria-label="Tailscale 授权方式">
          <button type="button" className={authorizationMethod === 'local_status' ? 'active' : ''} onClick={() => selectAuthorizationMethod('local_status')} disabled={Boolean(pending) || busy}>本机状态</button>
          <button type="button" className={authorizationMethod === 'tailscale_cli' ? 'active' : ''} onClick={() => selectAuthorizationMethod('tailscale_cli')} disabled={Boolean(pending) || busy}>官方 CLI</button>
        </div>
        <p className="binding-method-note">
          {authorizationMethod === 'tailscale_cli'
            ? '在本机完成 tailscale up 或官方登录后，将 CLI 显示的稳定 ID 填入确认表单。QLH 不收集 Tailscale 密码或 token。'
            : '确认本机 Tailscale 客户端已经登录目标组网，再从客户端或 tailscale status 获取稳定 ID。'}
        </p>
        {localStatus?.available && localStatus.candidate && (
          <div className="binding-detected" role="status">
            <div><strong>{localStatus.candidate.tailnet_display_name || localStatus.candidate.tailnet_id}</strong><span>{localStatus.candidate.hostname || localStatus.candidate.node_id}</span></div>
            <small>{localStatus.candidate.addresses?.join(' · ') || '未返回 Tailscale 地址'} · 需要确认</small>
          </div>
        )}
        <div className="binding-prepare-actions">
          {authorizationMethod === 'local_status' && <button type="button" className="quiet-btn" onClick={handleInspectLocalStatus} disabled={busy || inspecting || Boolean(pending)}>{inspecting ? '读取中...' : '读取本机状态'}</button>}
          <button type="submit" className="primary-btn" disabled={busy || inspecting || Boolean(pending)}>{activeBinding ? '发起换网' : '发起绑定'}</button>
        </div>
      </form>

      {pending && (
        <form className="binding-confirm" onSubmit={handleConfirm}>
          <div className="binding-form-title">
            <strong>确认新绑定</strong>
            <span>{methodLabel(pending.authorization_method)} · {pending.binding_id}</span>
          </div>
          <div className="binding-input-grid">
            <label>tailnet ID<input value={tailnetId} onChange={(event) => setTailnetId(event.target.value)} placeholder="例如 tailnet-example" required /></label>
            <label>Tailscale 用户 ID<input value={tailscaleUserId} onChange={(event) => setTailscaleUserId(event.target.value)} placeholder="稳定用户 ID" required /></label>
            <label>节点 ID<input value={nodeId} onChange={(event) => setNodeId(event.target.value)} placeholder="可选" /></label>
          </div>
          <p className="binding-warning">确认后，旧 active 绑定会在同一事务中变为 revoked；身份冲突会拒绝并保留旧绑定。</p>
          <div className="dialog-actions"><button type="button" className="danger-link" onClick={() => handleRevoke(pending.binding_id)} disabled={busy}>取消此次请求</button><button type="submit" className="primary-btn" disabled={busy}>确认绑定</button></div>
        </form>
      )}

      {activeBinding && <button type="button" className="danger-link binding-revoke" onClick={() => handleRevoke(activeBinding.binding_id)} disabled={busy}>撤销当前绑定</button>}

      {historicalBindings.length > 0 && (
        <details className="binding-history">
          <summary>查看历史绑定（{historicalBindings.length}）</summary>
          <div className="binding-history-list">
            {historicalBindings.map((binding) => (
              <div className="binding-history-row" key={binding.binding_id}>
                <span>{binding.tailnet_id || '待确认'} · {stateLabel(binding.state)}</span>
                <small>{formatDate(binding.updated_at)}</small>
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
