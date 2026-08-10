import { Injectable } from '@nestjs/common';
import { execFile } from 'child_process';
import { existsSync } from 'fs';
import { isIP } from 'net';
import * as path from 'path';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const MAX_STATUS_BYTES = 4 * 1024 * 1024;
const STATUS_CACHE_MS = 5_000;

export type TailscaleLocalStatusState =
  | 'ready'
  | 'cli_not_found'
  | 'not_running'
  | 'not_logged_in'
  | 'incomplete_identity'
  | 'invalid_response'
  | 'unavailable';

export interface TailscaleLocalIdentityCandidate {
  tailnet_id: string;
  tailnet_id_source: 'magic_dns_suffix' | 'tailnet_name';
  tailnet_display_name: string | null;
  tailscale_user_id: string;
  node_id: string;
  hostname: string | null;
  dns_name: string | null;
  addresses: string[];
}

export interface TailscaleLocalStatusView {
  available: boolean;
  state: TailscaleLocalStatusState;
  reason_code: string | null;
  source: 'tailscale_status_json';
  observed_at: string;
  requires_confirmation: true;
  candidate: TailscaleLocalIdentityCandidate | null;
}

interface StatusRecord {
  BackendState?: unknown;
  MagicDNSSuffix?: unknown;
  CurrentTailnet?: unknown;
  Self?: unknown;
}

function cleanString(value: unknown, maxLength = 256): string | null {
  if (typeof value !== 'string') return null;
  const result = value.trim();
  if (!result || result.length > maxLength || /[\0\r\n]/.test(result)) return null;
  return result;
}

function stableId(value: unknown): string | null {
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return String(value);
  return cleanString(value);
}

function isTailscaleAddress(value: string): boolean {
  if (isIP(value) === 6) return value.toLowerCase().startsWith('fd7a:115c:a1e0:');
  if (isIP(value) !== 4) return false;
  const octets = value.split('.').map(Number);
  return octets[0] === 100 && octets[1] >= 64 && octets[1] <= 127;
}

function emptyStatus(state: TailscaleLocalStatusState, reasonCode: string): TailscaleLocalStatusView {
  return {
    available: false,
    state,
    reason_code: reasonCode,
    source: 'tailscale_status_json',
    observed_at: new Date().toISOString(),
    requires_confirmation: true,
    candidate: null,
  };
}

export function parseTailscaleStatusJson(raw: string): TailscaleLocalStatusView {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyStatus('invalid_response', 'status_json_invalid');
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return emptyStatus('invalid_response', 'status_json_invalid');
  }

  const status = parsed as StatusRecord;
  const backendState = cleanString(status.BackendState, 64);
  if (backendState !== 'Running') return emptyStatus('not_running', 'tailscale_not_running');
  if (!status.Self || typeof status.Self !== 'object' || Array.isArray(status.Self)) {
    return emptyStatus('not_logged_in', 'tailscale_not_logged_in');
  }

  const self = status.Self as Record<string, unknown>;
  const currentTailnet = status.CurrentTailnet && typeof status.CurrentTailnet === 'object'
    && !Array.isArray(status.CurrentTailnet)
    ? status.CurrentTailnet as Record<string, unknown>
    : {};
  const magicDnsSuffix = cleanString(currentTailnet.MagicDNSSuffix)
    || cleanString(status.MagicDNSSuffix);
  const tailnetName = cleanString(currentTailnet.Name);
  const tailnetId = magicDnsSuffix || tailnetName;
  const tailscaleUserId = stableId(self.UserID);
  const nodeId = stableId(self.ID);
  if (!tailnetId || !tailscaleUserId || !nodeId) {
    return emptyStatus('incomplete_identity', 'tailscale_identity_incomplete');
  }

  const addresses = Array.isArray(self.TailscaleIPs)
    ? self.TailscaleIPs
      .map((entry) => cleanString(entry, 64))
      .filter((entry): entry is string => Boolean(entry && isTailscaleAddress(entry)))
      .slice(0, 4)
    : [];
  return {
    available: true,
    state: 'ready',
    reason_code: null,
    source: 'tailscale_status_json',
    observed_at: new Date().toISOString(),
    requires_confirmation: true,
    candidate: {
      tailnet_id: tailnetId,
      tailnet_id_source: magicDnsSuffix ? 'magic_dns_suffix' : 'tailnet_name',
      tailnet_display_name: tailnetName,
      tailscale_user_id: tailscaleUserId,
      node_id: nodeId,
      hostname: cleanString(self.HostName),
      dns_name: cleanString(self.DNSName),
      addresses,
    },
  };
}

function timeoutMs(): number {
  const configured = Number(process.env.QLH_TAILSCALE_STATUS_TIMEOUT_MS || 10_000);
  if (!Number.isFinite(configured)) return 10_000;
  return Math.min(15_000, Math.max(500, Math.trunc(configured)));
}

function executable(): string {
  const configured = cleanString(process.env.QLH_TAILSCALE_CLI, 1024);
  if (configured) return configured;
  if (process.platform === 'win32') {
    const candidates = [
      process.env.ProgramFiles
        ? path.join(process.env.ProgramFiles, 'Tailscale', 'tailscale.exe') : null,
      process.env['ProgramFiles(x86)']
        ? path.join(process.env['ProgramFiles(x86)'], 'Tailscale', 'tailscale.exe') : null,
    ].filter((entry): entry is string => Boolean(entry));
    const installed = candidates.find((entry) => existsSync(entry));
    return installed || 'tailscale.exe';
  }
  return 'tailscale';
}

@Injectable()
export class TailscaleLocalStatusService {
  private cached: { expiresAt: number; value: TailscaleLocalStatusView } | null = null;
  private inFlight: Promise<TailscaleLocalStatusView> | null = null;

  async inspect(): Promise<TailscaleLocalStatusView> {
    if (this.cached && this.cached.expiresAt > Date.now()) return this.cached.value;
    if (this.inFlight) return this.inFlight;
    this.inFlight = this.inspectOnce();
    try {
      const value = await this.inFlight;
      this.cached = { expiresAt: Date.now() + STATUS_CACHE_MS, value };
      return value;
    } finally {
      this.inFlight = null;
    }
  }

  private async inspectOnce(): Promise<TailscaleLocalStatusView> {
    try {
      const result = await execFileAsync(executable(), ['status', '--json'], {
        encoding: 'utf8',
        windowsHide: true,
        timeout: timeoutMs(),
        maxBuffer: MAX_STATUS_BYTES,
      });
      return parseTailscaleStatusJson(String(result.stdout || ''));
    } catch (error) {
      const code = (error as NodeJS.ErrnoException)?.code;
      if (code === 'ENOENT') return emptyStatus('cli_not_found', 'tailscale_cli_not_found');
      const timedOut = code === 'ETIMEDOUT' || (error as { killed?: boolean })?.killed === true;
      return emptyStatus('unavailable', timedOut ? 'tailscale_status_timeout' : 'tailscale_status_unavailable');
    }
  }
}
