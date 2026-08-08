import { Injectable, Optional } from '@nestjs/common';
import { spawn } from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export interface CredentialProtector {
  readonly name: string;
  protect(secret: string): Promise<string>;
  unprotect(ciphertext: string): Promise<string>;
}

export interface CredentialStoreOptions {
  rootDir?: string;
  protector?: CredentialProtector;
  env?: NodeJS.ProcessEnv;
}

export interface CredentialStatus {
  credential_ref: string;
  exists: boolean;
  protection: string;
  updated_at: string | null;
}

interface CredentialRecord {
  schema_version: 1;
  credential_ref: string;
  protection: string;
  ciphertext: string;
  updated_at: string;
}

export function normalizeCredentialRef(input: string): string {
  const value = input.trim();
  if (!/^os:[a-z0-9][a-z0-9._/-]{0,126}$/i.test(value)
      || value.includes('..') || value.includes('//') || value.endsWith('/')) {
    throw new Error('credential_ref is invalid');
  }
  return value;
}

export function credentialRefForId(input: string): string {
  const id = input.trim();
  if (!/^[a-z0-9][a-z0-9._-]{0,63}$/i.test(id)) {
    throw new Error('credential id is invalid');
  }
  return `os:qlh/${id}`;
}

export class WindowsDpapiProtector implements CredentialProtector {
  readonly name = 'windows-dpapi-current-user';

  async protect(secret: string): Promise<string> {
    return this.runPowerShell([
      'Add-Type -AssemblyName System.Security;',
      '$plain = [Console]::In.ReadToEnd();',
      '$bytes = [Text.Encoding]::UTF8.GetBytes($plain);',
      '$cipher = [Security.Cryptography.ProtectedData]::Protect(',
      '  $bytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser);',
      '[Console]::Out.Write([Convert]::ToBase64String($cipher));',
    ].join(' '), secret);
  }

  async unprotect(ciphertext: string): Promise<string> {
    return this.runPowerShell([
      'Add-Type -AssemblyName System.Security;',
      '$encoded = [Console]::In.ReadToEnd();',
      '$cipher = [Convert]::FromBase64String($encoded);',
      '$plain = [Security.Cryptography.ProtectedData]::Unprotect(',
      '  $cipher, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser);',
      '[Console]::Out.Write([Text.Encoding]::UTF8.GetString($plain));',
    ].join(' '), ciphertext);
  }

  private runPowerShell(script: string, stdin: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const executable = process.env.SystemRoot
        ? path.join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
        : 'powershell.exe';
      const child = spawn(executable, [
        '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script,
      ], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
      const stdout: Buffer[] = [];
      let outputBytes = 0;
      child.stdout.on('data', (chunk: Buffer) => {
        outputBytes += chunk.length;
        if (outputBytes <= 512 * 1024) stdout.push(chunk);
      });
      child.stderr.resume();
      child.on('error', () => reject(new Error('OS credential protection is unavailable')));
      child.on('close', (code) => {
        if (code !== 0 || outputBytes > 512 * 1024) {
          reject(new Error('OS credential protection failed'));
          return;
        }
        resolve(Buffer.concat(stdout).toString('utf-8'));
      });
      child.stdin.on('error', () => undefined);
      child.stdin.end(stdin, 'utf-8');
    });
  }
}

class UnsupportedCredentialProtector implements CredentialProtector {
  readonly name = 'unavailable';

  async protect(): Promise<string> {
    throw new Error('OS credential store is unavailable on this platform');
  }

  async unprotect(): Promise<string> {
    throw new Error('OS credential store is unavailable on this platform');
  }
}

function defaultCredentialRoot(env: NodeJS.ProcessEnv): string {
  const override = env.QLH_CREDENTIAL_STORE_DIR?.trim();
  if (override) return path.resolve(override);
  if (process.platform === 'win32') {
    const base = env.LOCALAPPDATA || env.APPDATA;
    if (base) return path.join(base, 'QLH', 'credentials');
  }
  const xdg = env.XDG_DATA_HOME?.trim();
  if (xdg) return path.join(xdg, 'qlh', 'credentials');
  return path.join(os.homedir(), '.local', 'share', 'qlh', 'credentials');
}

@Injectable()
export class ModelCredentialStore {
  readonly root: string;
  private readonly protector: CredentialProtector;

  constructor(@Optional() options: CredentialStoreOptions = {}) {
    const env = options.env ?? process.env;
    this.root = options.rootDir
      ? path.resolve(options.rootDir)
      : defaultCredentialRoot(env);
    this.protector = options.protector ?? (
      process.platform === 'win32'
        ? new WindowsDpapiProtector()
        : new UnsupportedCredentialProtector()
    );
  }

  async set(credentialRef: string, secret: string): Promise<CredentialStatus> {
    const ref = normalizeCredentialRef(credentialRef);
    if (!secret || Buffer.byteLength(secret, 'utf-8') > 64 * 1024) {
      throw new Error('credential secret must contain 1-65536 UTF-8 bytes');
    }
    const updatedAt = new Date().toISOString();
    const record: CredentialRecord = {
      schema_version: 1,
      credential_ref: ref,
      protection: this.protector.name,
      ciphertext: await this.protector.protect(secret),
      updated_at: updatedAt,
    };
    const target = this.recordPath(ref);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const temp = `${target}.tmp-${process.pid}-${crypto.randomUUID().slice(0, 8)}`;
    fs.writeFileSync(temp, `${JSON.stringify(record)}\n`, { encoding: 'utf-8', mode: 0o600 });
    fs.renameSync(temp, target);
    return this.toStatus(record);
  }

  async get(credentialRef: string | null | undefined): Promise<string | null> {
    if (!credentialRef) return null;
    const ref = normalizeCredentialRef(credentialRef);
    const record = this.readRecord(ref);
    if (!record) return null;
    if (record.protection !== this.protector.name) {
      throw new Error('credential protection provider mismatch');
    }
    return this.protector.unprotect(record.ciphertext);
  }

  status(credentialRef: string): CredentialStatus {
    const ref = normalizeCredentialRef(credentialRef);
    const record = this.readRecord(ref);
    if (!record) {
      return {
        credential_ref: ref,
        exists: false,
        protection: this.protector.name,
        updated_at: null,
      };
    }
    return this.toStatus(record);
  }

  delete(credentialRef: string): boolean {
    const ref = normalizeCredentialRef(credentialRef);
    const target = this.recordPath(ref);
    if (!fs.existsSync(target)) return false;
    fs.rmSync(target, { force: true });
    return true;
  }

  private recordPath(credentialRef: string): string {
    const digest = crypto.createHash('sha256').update(credentialRef).digest('hex');
    return path.join(this.root, `${digest}.json`);
  }

  private readRecord(credentialRef: string): CredentialRecord | null {
    const target = this.recordPath(credentialRef);
    if (!fs.existsSync(target)) return null;
    try {
      const record = JSON.parse(fs.readFileSync(target, 'utf-8')) as CredentialRecord;
      if (record.schema_version !== 1 || record.credential_ref !== credentialRef
          || !record.ciphertext || !record.updated_at) {
        throw new Error('invalid credential record');
      }
      return record;
    } catch {
      throw new Error('credential record is invalid');
    }
  }

  private toStatus(record: CredentialRecord): CredentialStatus {
    return {
      credential_ref: record.credential_ref,
      exists: true,
      protection: record.protection,
      updated_at: record.updated_at,
    };
  }
}
