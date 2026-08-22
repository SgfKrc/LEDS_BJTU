import { useEffect, useState } from 'react';
import QRCode from 'qrcode';
import { Check, Copy, KeyRound, QrCode, X } from 'lucide-react';
import { CommandButton } from './CommandButton';
import type { AuthProvisioning } from '../data/types';

interface AuthProvisioningPanelProps {
  provisioning: AuthProvisioning;
  username: string;
  busy?: boolean;
  onVerify: (code: string) => void;
  onDismiss: () => void;
}

async function copyText(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const field = document.createElement('textarea');
  field.value = value;
  field.style.position = 'fixed';
  field.style.opacity = '0';
  document.body.appendChild(field);
  field.select();
  document.execCommand('copy');
  field.remove();
}

export function AuthProvisioningPanel({
  provisioning,
  username,
  busy = false,
  onVerify,
  onDismiss,
}: AuthProvisioningPanelProps) {
  const [mode, setMode] = useState<'qr' | 'string'>('qr');
  const [qrDataUrl, setQrDataUrl] = useState('');
  const [code, setCode] = useState('');
  const [copied, setCopied] = useState('');
  const [qrError, setQrError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setQrDataUrl('');
    setQrError('');
    QRCode.toDataURL(provisioning.qr_payload, {
      width: 224,
      margin: 1,
      errorCorrectionLevel: 'M',
      color: { dark: '#101014', light: '#ffffff' },
    }).then((dataUrl) => {
      if (!cancelled) setQrDataUrl(dataUrl);
    }).catch(() => {
      if (!cancelled) setQrError('QR generation unavailable. Use the string view.');
    });
    return () => { cancelled = true; };
  }, [provisioning.qr_payload]);

  const handleCopy = async (value: string, label: string) => {
    try {
      await copyText(value);
      setCopied(label);
      window.setTimeout(() => setCopied(''), 1600);
    } catch {
      setCopied('copy failed');
    }
  };

  return (
    <section className="account-panel account-provisioning" aria-labelledby="auth-provisioning-title">
      <SectionMark />
      <div className="account-provisioning__heading">
        <div>
          <span className="mono-label">ONE-TIME PROVISIONING</span>
          <h2 id="auth-provisioning-title">Auth App for {username}</h2>
        </div>
        <button type="button" className="account-icon-button" title="Discard provisioning material" aria-label="Discard provisioning material" onClick={onDismiss} disabled={busy}><X size={15} /></button>
      </div>
      <div className="account-mode" role="group" aria-label="Provisioning format">
        <button type="button" className={mode === 'qr' ? 'is-active' : ''} onClick={() => setMode('qr')}><QrCode size={14} />QR</button>
        <button type="button" className={mode === 'string' ? 'is-active' : ''} onClick={() => setMode('string')}><KeyRound size={14} />String</button>
      </div>
      {mode === 'qr' ? (
        <div className="account-provisioning__qr">
          {qrDataUrl ? <img src={qrDataUrl} alt="Auth App provisioning QR" /> : <span>{qrError || 'Generating QR...'}</span>}
        </div>
      ) : (
        <div className="account-secret-stack">
          <div className="account-secret-row"><span>BASE32</span><code>{provisioning.secret}</code><button type="button" onClick={() => void handleCopy(provisioning.secret, 'secret')}><Copy size={13} />{copied === 'secret' ? 'Copied' : 'Copy'}</button></div>
          <div className="account-secret-row"><span>URI</span><code>{provisioning.otpauth_uri}</code><button type="button" onClick={() => void handleCopy(provisioning.otpauth_uri, 'uri')}><Copy size={13} />{copied === 'uri' ? 'Copied' : 'Copy'}</button></div>
        </div>
      )}
      <form className="account-form account-provisioning__verify" onSubmit={(event) => { event.preventDefault(); onVerify(code); }}>
        <label className="account-field"><span>AUTH APP CODE</span><input value={code} inputMode="numeric" autoComplete="one-time-code" onChange={(event) => setCode(event.target.value)} required autoFocus /></label>
        <CommandButton type="submit" icon={Check} busy={busy}>Verify and activate</CommandButton>
      </form>
    </section>
  );
}

function SectionMark() {
  return <div className="account-provisioning__mark" aria-hidden="true"><KeyRound size={18} /></div>;
}
