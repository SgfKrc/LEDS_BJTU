import { useEffect, useState } from 'react';
import { ShieldCheck } from 'lucide-react';
import QRCode from 'qrcode';
import {
  bootstrapAuthOwner,
  fetchAuthCapability,
  fetchAuthSession,
  getAuthSessionToken,
  loginAuth,
  logoutAuth,
  setAuthSessionToken,
  verifyAuthProvisioning,
} from '../api/client';

function errorMessage(error) {
  return error?.detail || error?.message || '认证操作失败';
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const node = document.createElement('textarea');
  node.value = value;
  node.style.position = 'fixed';
  node.style.opacity = '0';
  document.body.appendChild(node);
  node.select();
  document.execCommand('copy');
  node.remove();
}

export default function AuthGate({ children }) {
  const [phase, setPhase] = useState('loading');
  const [session, setSession] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [code, setCode] = useState('');
  const [loginMode, setLoginMode] = useState('totp');
  const [provisioningMode, setProvisioningMode] = useState('qr');
  const [provisioning, setProvisioning] = useState(null);
  const [qrDataUrl, setQrDataUrl] = useState('');
  const [recoveryCodes, setRecoveryCodes] = useState([]);
  const [copied, setCopied] = useState('');

  useEffect(() => {
    let cancelled = false;
    fetchAuthCapability()
      .then(async (capability) => {
        if (cancelled) return;
        if (capability?.required !== true) {
          throw new Error('主节点返回了无效的认证能力声明');
        }
        if (!getAuthSessionToken()) {
          setPhase('login');
          return;
        }
        try {
          const current = await fetchAuthSession();
          if (cancelled) return;
          setSession(current);
          setPhase('authenticated');
        } catch {
          if (cancelled) return;
          setAuthSessionToken('');
          setPhase('login');
        }
      })
      .catch((nextError) => {
        if (cancelled) return;
        if (nextError?.status === 404 && nextError?.detail === 'Not Found') {
          setPhase('legacy');
          return;
        }
        setError('无法确认主节点认证状态');
        setPhase('unavailable');
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setQrDataUrl('');
    if (!provisioning?.qr_payload) return undefined;
    QRCode.toDataURL(provisioning.qr_payload, {
      width: 232,
      margin: 1,
      errorCorrectionLevel: 'M',
      color: { dark: '#111827', light: '#ffffff' },
    }).then((url) => {
      if (!cancelled) setQrDataUrl(url);
    }).catch(() => {
      if (!cancelled) setError('二维码生成失败，请改用字符串配置');
    });
    return () => { cancelled = true; };
  }, [provisioning]);

  const handleCopy = async (value, label) => {
    try {
      await copyText(value);
      setCopied(label);
      setTimeout(() => setCopied(''), 1800);
    } catch {
      setError('复制失败');
    }
  };

  const submitLogin = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const result = await loginAuth({
        username,
        ...(loginMode === 'recovery' ? { recoveryCode: code } : { code }),
      });
      setSession({
        session_id: result.session_id,
        expires_at: result.expires_at,
        user: result.user,
      });
      setCode('');
      setPhase('authenticated');
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusy(false);
    }
  };

  const submitBootstrap = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const result = await bootstrapAuthOwner({ username, displayName });
      setProvisioning(result.provisioning);
      setCode('');
      setPhase('provisioning');
    } catch (nextError) {
      setError(errorMessage(nextError));
      if (nextError?.status === 409) setPhase('login');
    } finally {
      setBusy(false);
    }
  };

  const submitProvisioning = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const verified = await verifyAuthProvisioning({
        userId: provisioning.user_id,
        authenticatorId: provisioning.authenticator_id,
        code,
      });
      const loggedIn = await loginAuth({ username, code });
      setSession({
        session_id: loggedIn.session_id,
        expires_at: loggedIn.expires_at,
        user: loggedIn.user,
      });
      setRecoveryCodes(verified.recovery_codes || []);
      setCode('');
      setProvisioning(null);
      setQrDataUrl('');
      setPhase('recovery');
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    setBusy(true);
    try {
      await logoutAuth();
    } catch {
      // The local token is cleared in logoutAuth even when the node is offline.
    } finally {
      setSession(null);
      setCode('');
      setRecoveryCodes([]);
      setPhase('login');
      setBusy(false);
    }
  };

  if (phase === 'authenticated') {
    return typeof children === 'function'
      ? children({ session, onLogout: handleLogout })
      : children;
  }

  if (phase === 'legacy') {
    return typeof children === 'function'
      ? children({ session: null, onLogout: null })
      : children;
  }

  if (phase === 'loading') {
    return (
      <main className="auth-shell">
        <div className="auth-loading" role="status">正在连接主节点...</div>
      </main>
    );
  }

  if (phase === 'unavailable') {
    return (
      <main className="auth-shell">
        <section className="auth-panel" aria-labelledby="auth-title">
          <header className="auth-brand">
            <div className="auth-brand-mark" aria-hidden="true"><ShieldCheck size={23} strokeWidth={1.7} /></div>
            <div>
              <h1 id="auth-title">QLH</h1>
              <p>主节点认证</p>
            </div>
          </header>
          <div className="auth-form">
            <div className="auth-heading">
              <h2>认证服务不可用</h2>
              <span>连接已阻止</span>
            </div>
            <div className="auth-error" role="alert">{error}</div>
            <button className="auth-primary" type="button" onClick={() => window.location.reload()}>
              重新连接
            </button>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="auth-shell">
      <section className="auth-panel" aria-labelledby="auth-title">
        <header className="auth-brand">
          <div className="auth-brand-mark" aria-hidden="true"><ShieldCheck size={23} strokeWidth={1.7} /></div>
          <div>
            <h1 id="auth-title">QLH</h1>
            <p>主节点认证</p>
          </div>
        </header>

        {phase === 'login' && (
          <form className="auth-form" onSubmit={submitLogin}>
            <div className="auth-heading">
              <h2>登录</h2>
              <span>本地主节点</span>
            </div>
            <label>
              用户名
              <input
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                required
                autoFocus
              />
            </label>
            <div className="auth-segmented" role="group" aria-label="认证方式">
              <button
                type="button"
                className={loginMode === 'totp' ? 'active' : ''}
                onClick={() => { setLoginMode('totp'); setCode(''); setError(''); }}
              >
                Auth App
              </button>
              <button
                type="button"
                className={loginMode === 'recovery' ? 'active' : ''}
                onClick={() => { setLoginMode('recovery'); setCode(''); setError(''); }}
              >
                恢复码
              </button>
            </div>
            <label>
              {loginMode === 'totp' ? '验证码' : '恢复码'}
              <input
                autoComplete="one-time-code"
                inputMode={loginMode === 'totp' ? 'numeric' : 'text'}
                value={code}
                onChange={(event) => setCode(event.target.value)}
                placeholder={loginMode === 'totp' ? '6 位数字' : 'XXXX-XXXX-XXXX'}
                required
              />
            </label>
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="auth-primary" type="submit" disabled={busy}>
              {busy ? '正在登录...' : '登录'}
            </button>
            <button
              className="auth-link"
              type="button"
              onClick={() => { setPhase('bootstrap'); setError(''); setCode(''); }}
            >
              初始化主节点
            </button>
          </form>
        )}

        {phase === 'bootstrap' && (
          <form className="auth-form" onSubmit={submitBootstrap}>
            <div className="auth-heading">
              <h2>创建 owner</h2>
              <span>首次初始化</span>
            </div>
            <label>
              用户名
              <input
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                required
                autoFocus
              />
            </label>
            <label>
              显示名称
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                placeholder="可选"
              />
            </label>
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="auth-primary" type="submit" disabled={busy}>
              {busy ? '正在创建...' : '创建并配置 Auth App'}
            </button>
            <button className="auth-link" type="button" onClick={() => { setPhase('login'); setError(''); }}>
              返回登录
            </button>
          </form>
        )}

        {phase === 'provisioning' && provisioning && (
          <form className="auth-form auth-provisioning" onSubmit={submitProvisioning}>
            <div className="auth-heading">
              <h2>配置 Auth App</h2>
              <span>仅显示一次</span>
            </div>
            <div className="auth-segmented" role="group" aria-label="密钥下发方式">
              <button
                type="button"
                className={provisioningMode === 'qr' ? 'active' : ''}
                onClick={() => setProvisioningMode('qr')}
              >
                二维码
              </button>
              <button
                type="button"
                className={provisioningMode === 'string' ? 'active' : ''}
                onClick={() => setProvisioningMode('string')}
              >
                字符串
              </button>
            </div>
            {provisioningMode === 'qr' ? (
              <div className="auth-qr-box">
                {qrDataUrl
                  ? <img src={qrDataUrl} alt="Auth App 配置二维码" />
                  : <span>正在生成二维码...</span>}
              </div>
            ) : (
              <div className="auth-secret-stack">
                <div className="auth-secret-row">
                  <span>Base32</span>
                  <code>{provisioning.secret}</code>
                  <button type="button" onClick={() => handleCopy(provisioning.secret, 'secret')}>
                    {copied === 'secret' ? '已复制' : '复制'}
                  </button>
                </div>
                <div className="auth-secret-row auth-uri-row">
                  <span>URI</span>
                  <code>{provisioning.otpauth_uri}</code>
                  <button type="button" onClick={() => handleCopy(provisioning.otpauth_uri, 'uri')}>
                    {copied === 'uri' ? '已复制' : '复制'}
                  </button>
                </div>
              </div>
            )}
            <label>
              验证码
              <input
                autoComplete="one-time-code"
                inputMode="numeric"
                value={code}
                onChange={(event) => setCode(event.target.value)}
                placeholder="6 位数字"
                required
                autoFocus
              />
            </label>
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="auth-primary" type="submit" disabled={busy}>
              {busy ? '正在确认...' : '确认并生成恢复码'}
            </button>
          </form>
        )}

        {phase === 'recovery' && (
          <div className="auth-form">
            <div className="auth-heading">
              <h2>恢复码</h2>
              <span>仅显示一次</span>
            </div>
            <div className="auth-recovery-grid">
              {recoveryCodes.map((recoveryCode) => <code key={recoveryCode}>{recoveryCode}</code>)}
            </div>
            <button
              className="auth-secondary"
              type="button"
              onClick={() => handleCopy(recoveryCodes.join('\n'), 'recovery')}
            >
              {copied === 'recovery' ? '已复制' : '复制全部'}
            </button>
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="auth-primary" type="button" onClick={() => setPhase('authenticated')}>
              进入主节点
            </button>
          </div>
        )}
      </section>
    </main>
  );
}
