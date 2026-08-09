import { Injectable } from '@nestjs/common';
import {
  createHash,
  createHmac,
  randomBytes,
  randomUUID,
  scryptSync,
  timingSafeEqual,
} from 'crypto';
import {
  AuthAssetRepository,
  LocalUserRecord,
  LocalUserRole,
  TailscaleAuthorizationMethod,
  TailscaleBindingRecord,
  UserAuthenticatorRecord,
} from './auth-asset-repository';
import { ModelCredentialStore } from './model-credential-store';
import { SqliteStore } from './sqlite-store';

const TOTP_ISSUER = 'QLH';
const TOTP_WINDOW = 1;
const LOGIN_WINDOW_MS = 15 * 60 * 1000;
const LOGIN_LOCK_MS = 5 * 60 * 1000;
const LOGIN_MAX_FAILURES = 5;
const SESSION_TTL_MS = 12 * 60 * 60 * 1000;
const RECOVERY_CODE_COUNT = 10;
const BASE32_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';

export class AuthServiceError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'AuthServiceError';
  }
}

export interface ProvisioningPayload {
  user_id: string;
  authenticator_id: string;
  secret: string;
  otpauth_uri: string;
  qr_payload: string;
  algorithm: 'SHA1' | 'SHA256' | 'SHA512';
  digits: 6 | 8;
  period_seconds: number;
}

export interface AuthenticatedSession {
  session_id: string;
  user: LocalUserRecord;
  expires_at: string;
}

export interface LoginResult extends AuthenticatedSession {
  access_token: string;
  token_type: 'Bearer';
}

export interface TailscaleBindingView {
  binding_id: string;
  user_id: string;
  tailnet_id: string | null;
  tailscale_user_id: string | null;
  node_id: string | null;
  state: TailscaleBindingRecord['state'];
  authorization_method: TailscaleAuthorizationMethod;
  aggregate_version: number;
  prepared_at: string;
  updated_at: string;
  confirmed_at: string | null;
  revoked_at: string | null;
  last_verified_at: string | null;
}

function nowDate(): Date {
  return new Date();
}

function iso(date = nowDate()): string {
  return date.toISOString();
}

function normalizeSecret(secret: string): string {
  const normalized = secret.replace(/[\s-]/g, '').toUpperCase();
  if (!normalized || !/^[A-Z2-7]+$/.test(normalized) || normalized.length < 16) {
    throw new AuthServiceError(422, 'TOTP secret is invalid');
  }
  return normalized;
}

function encodeBase32(bytes: Buffer): string {
  let bits = 0;
  let value = 0;
  let output = '';
  for (const byte of bytes) {
    value = (value << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      output += BASE32_ALPHABET[(value >> bits) & 31];
    }
  }
  if (bits > 0) output += BASE32_ALPHABET[(value << (5 - bits)) & 31];
  return output;
}

function decodeBase32(secret: string): Buffer {
  const normalized = normalizeSecret(secret);
  let bits = 0;
  let value = 0;
  const output: number[] = [];
  for (const char of normalized) {
    value = (value << 5) | BASE32_ALPHABET.indexOf(char);
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      output.push((value >> bits) & 0xff);
    }
  }
  return Buffer.from(output);
}

function normalizeOtpCode(value: unknown, digits: number): string {
  const code = String(value ?? '').trim();
  if (!new RegExp(`^\\d{${digits}}$`).test(code)) {
    throw new AuthServiceError(401, '用户名或验证码错误');
  }
  return code;
}

function totp(secret: string, timestampMs: number, algorithm: string, digits: number, period: number, offset: number): string {
  const counter = Math.floor(timestampMs / 1000 / period) + offset;
  if (counter < 0) return '';
  const counterBytes = Buffer.alloc(8);
  counterBytes.writeBigUInt64BE(BigInt(counter));
  const digest = createHmac(algorithm.toLowerCase(), decodeBase32(secret))
    .update(counterBytes)
    .digest();
  const index = digest[digest.length - 1] & 0x0f;
  const binary = ((digest[index] & 0x7f) << 24)
    | ((digest[index + 1] & 0xff) << 16)
    | ((digest[index + 2] & 0xff) << 8)
    | (digest[index + 3] & 0xff);
  return String(binary % (10 ** digits)).padStart(digits, '0');
}

function verifyTotp(secret: string, code: string, authenticator: UserAuthenticatorRecord, timestampMs = Date.now()): boolean {
  const normalizedCode = normalizeOtpCode(code, authenticator.digits);
  for (let offset = -TOTP_WINDOW; offset <= TOTP_WINDOW; offset += 1) {
    const expected = totp(
      secret,
      timestampMs,
      authenticator.algorithm,
      authenticator.digits,
      authenticator.period_seconds,
      offset,
    );
    if (timingSafeEqual(Buffer.from(expected), Buffer.from(normalizedCode))) return true;
  }
  return false;
}

function hashToken(token: string): string {
  return createHash('sha256').update(token, 'utf8').digest('hex');
}

function recoveryCode(): string {
  return encodeBase32(randomBytes(8)).slice(0, 12).match(/.{1,4}/g)!.join('-');
}

function encodeScryptHash(code: string): string {
  const salt = randomBytes(16);
  const digest = scryptSync(code, salt, 32, { N: 16_384, r: 8, p: 1 });
  return `$scrypt$n=16384$r=8$p=1$${salt.toString('base64')}$${digest.toString('base64')}`;
}

function verifyScryptHash(code: string, encoded: string): boolean {
  const match = /^\$scrypt\$n=(\d+)\$r=(\d+)\$p=(\d+)\$([^$]+)\$([^$]+)$/.exec(encoded);
  if (!match) return false;
  const [, nRaw, rRaw, pRaw, saltRaw, digestRaw] = match;
  const n = Number(nRaw);
  const r = Number(rRaw);
  const p = Number(pRaw);
  if (!Number.isSafeInteger(n) || n < 16_384 || n > 1_048_576 || r < 1 || r > 32 || p < 1 || p > 16) {
    return false;
  }
  try {
    const expected = Buffer.from(digestRaw, 'base64');
    const actual = scryptSync(code, Buffer.from(saltRaw, 'base64'), expected.length, { N: n, r, p });
    return expected.length > 0 && timingSafeEqual(actual, expected);
  } catch {
    return false;
  }
}

function safeUser(user: LocalUserRecord): LocalUserRecord {
  return { ...user };
}

@Injectable()
export class AuthService {
  constructor(
    private readonly store: SqliteStore,
    private readonly assets: AuthAssetRepository,
    private readonly credentials: ModelCredentialStore,
  ) {}

  async bootstrapOwner(input: { username: string; display_name?: string }): Promise<ProvisioningPayload> {
    const owner = this.assets.listUsers().find((entry) => entry.role === 'owner' && entry.status !== 'revoked');
    if (owner) throw new AuthServiceError(409, '本地主节点 owner 已存在');
    const user = this.assets.createUser({
      username: input.username,
      display_name: input.display_name,
      role: 'owner',
    });
    try {
      return await this.createProvisioning(user);
    } catch (error) {
      try {
        this.assets.updateUser(user.user_id, user.aggregate_version, { status: 'revoked' });
      } catch {
        // Keep the original credential-store failure as the actionable error.
      }
      throw error;
    }
  }

  async createProvisioningForUser(userId: string): Promise<ProvisioningPayload> {
    const user = this.requireActiveUser(userId);
    return this.createProvisioning(user);
  }

  async verifyProvisioning(input: { user_id: string; authenticator_id: string; code: string }): Promise<{ user: LocalUserRecord; recovery_codes: string[] }> {
    const authenticator = this.assets.getAuthenticator(input.authenticator_id);
    if (!authenticator || authenticator.user_id !== input.user_id || authenticator.state !== 'pending') {
      throw new AuthServiceError(404, '待确认的认证器不存在');
    }
    const secret = await this.credentials.get(authenticator.secret_ref);
    if (!secret) throw new AuthServiceError(503, 'OS credential store 中缺少 TOTP 密钥');
    let valid = false;
    try {
      valid = verifyTotp(secret, input.code, authenticator);
    } catch (error) {
      if (error instanceof AuthServiceError) throw error;
    }
    if (!valid) {
      this.assets.appendAudit({
        user_id: input.user_id,
        event_type: 'totp_verify_failed',
        outcome: 'denied',
        reason_code: 'totp_mismatch',
        subject_id: authenticator.authenticator_id,
        details: { source: 'local_api' },
      });
      throw new AuthServiceError(401, '验证码错误');
    }
    this.assets.activateTotpAuthenticator(authenticator.authenticator_id);
    const recoveryCodes = this.replaceRecoveryCodes(input.user_id);
    this.assets.appendAudit({
      user_id: input.user_id,
      event_type: 'totp_verified',
      outcome: 'success',
      subject_id: authenticator.authenticator_id,
      details: { source: 'local_api' },
    });
    return { user: this.requireActiveUser(input.user_id), recovery_codes: recoveryCodes };
  }

  async login(input: { username: string; code?: string; recovery_code?: string }): Promise<LoginResult> {
    const user = this.assets.getUserByUsername(input.username);
    if (!user || user.status !== 'active') {
      this.assets.appendAudit({ event_type: 'login_failed', outcome: 'denied', reason_code: 'invalid_credentials', details: { source: 'local_api' } });
      throw new AuthServiceError(401, '用户名或验证码错误');
    }
    this.ensureLoginAllowed(user.user_id);
    let valid = false;
    let method = 'totp';
    if (input.recovery_code) {
      valid = this.consumeRecoveryCode(user.user_id, input.recovery_code);
      method = 'recovery_code';
    } else if (input.code) {
      valid = await this.verifyActiveTotp(user, input.code);
    }
    if (!valid) {
      this.recordLoginFailure(user.user_id);
      this.assets.appendAudit({
        user_id: user.user_id,
        event_type: 'login_failed',
        outcome: 'denied',
        reason_code: method === 'recovery_code' ? 'recovery_code_invalid' : 'totp_mismatch',
        details: { source: 'local_api' },
      });
      throw new AuthServiceError(401, '用户名或验证码错误');
    }
    this.clearLoginFailures(user.user_id);
    this.assets.appendAudit({
      user_id: user.user_id,
      event_type: 'login_succeeded',
      outcome: 'success',
      details: { method, source: 'local_api' },
    });
    return this.issueSession(user);
  }

  authenticateToken(token: string | null | undefined): AuthenticatedSession {
    const normalized = String(token ?? '').trim();
    if (!normalized || normalized.length > 512) throw new AuthServiceError(401, '本地主节点会话无效');
    const now = iso();
    const row = this.store.prepare(
      `SELECT s.session_id, s.user_id, s.expires_at, u.username, u.username_normalized,
              u.display_name, u.role, u.status, u.aggregate_version, u.created_at,
              u.updated_at, u.revoked_at
       FROM auth_sessions s JOIN local_users u ON u.user_id = s.user_id
       WHERE s.token_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ?
         AND u.status = 'active'`,
    ).get(hashToken(normalized), now) as (LocalUserRecord & { session_id: string; expires_at: string }) | undefined;
    if (!row) throw new AuthServiceError(401, '本地主节点会话无效');
    this.store.prepare('UPDATE auth_sessions SET last_seen_at = ? WHERE session_id = ?').run(now, row.session_id);
    return {
      session_id: row.session_id,
      expires_at: row.expires_at,
      user: {
        user_id: row.user_id,
        username: row.username,
        username_normalized: row.username_normalized,
        display_name: row.display_name,
        role: row.role,
        status: row.status,
        aggregate_version: row.aggregate_version,
        created_at: row.created_at,
        updated_at: row.updated_at,
        revoked_at: row.revoked_at,
      },
    };
  }

  logout(token: string | null | undefined): void {
    const normalized = String(token ?? '').trim();
    if (!normalized) return;
    const now = iso();
    let current: AuthenticatedSession | null = null;
    try {
      current = this.authenticateToken(normalized);
    } catch {
      // Logout is idempotent and must not disclose whether an expired token existed.
    }
    this.store.prepare('UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL')
      .run(now, hashToken(normalized));
    if (current) {
      this.assets.appendAudit({
        user_id: current.user.user_id,
        actor_user_id: current.user.user_id,
        event_type: 'logout',
        outcome: 'success',
        details: { source: 'local_api' },
      });
    }
  }

  async rotateRecoveryCodes(session: AuthenticatedSession, code: string): Promise<string[]> {
    const valid = await this.verifyActiveTotp(session.user, code);
    if (!valid) {
      this.assets.appendAudit({
        user_id: session.user.user_id,
        actor_user_id: session.user.user_id,
        event_type: 'recovery_codes_rotate_failed',
        outcome: 'denied',
        reason_code: 'totp_mismatch',
        details: { source: 'local_api' },
      });
      throw new AuthServiceError(401, '验证码错误');
    }
    const codes = this.replaceRecoveryCodes(session.user.user_id);
    this.assets.appendAudit({
      user_id: session.user.user_id,
      actor_user_id: session.user.user_id,
      event_type: 'recovery_codes_rotated',
      outcome: 'success',
      details: { count: codes.length },
    });
    return codes;
  }

  listUsers(session: AuthenticatedSession): LocalUserRecord[] {
    this.requireManager(session);
    return this.assets.listUsers().map(safeUser);
  }

  createUser(session: AuthenticatedSession, input: { username: string; display_name?: string; role?: LocalUserRole }): LocalUserRecord {
    this.requireManager(session, input.role);
    const user = this.assets.createUser(input);
    this.assets.appendAudit({
      user_id: user.user_id,
      actor_user_id: session.user.user_id,
      event_type: 'user_created_by_manager',
      outcome: 'success',
      subject_id: user.user_id,
      details: { role: user.role },
    });
    return safeUser(user);
  }

  updateUser(session: AuthenticatedSession, userId: string, expectedVersion: number, patch: { display_name?: string; role?: LocalUserRole; status?: 'active' | 'suspended' | 'revoked' }): LocalUserRecord {
    const target = this.assets.getUser(userId);
    if (!target) throw new AuthServiceError(404, '本地用户不存在');
    this.requireManager(session, patch.role);
    if (session.user.role !== 'owner' && target.role === 'owner') {
      throw new AuthServiceError(403, '只有 owner 可以管理 owner');
    }
    const updated = this.assets.updateUser(userId, expectedVersion, patch);
    if (updated.status !== 'active') this.revokeUserSessions(userId);
    this.assets.appendAudit({
      user_id: userId,
      actor_user_id: session.user.user_id,
      event_type: 'user_updated_by_manager',
      outcome: 'success',
      subject_id: userId,
      details: { status: updated.status, role: updated.role },
    });
    return safeUser(updated);
  }

  listTailscaleBindings(session: AuthenticatedSession, userId = session.user.user_id): TailscaleBindingView[] {
    this.requireBindingAccess(session, userId);
    return this.assets.listTailscaleBindings(userId).map(safeTailscaleBinding);
  }

  prepareTailscaleBinding(
    session: AuthenticatedSession,
    input: {
      user_id?: string;
      authorization_method?: TailscaleAuthorizationMethod;
      credential_ref?: string | null;
    },
  ): TailscaleBindingView {
    const userId = input.user_id ?? session.user.user_id;
    this.requireBindingAccess(session, userId);
    const method = input.authorization_method ?? 'local_status';
    if (method === 'oauth_app') {
      throw new AuthServiceError(501, 'Tailscale OAuth Apps 当前未启用，请使用本机 CLI 或 local status 授权');
    }
    try {
      const binding = this.assets.prepareTailscaleBinding({
        user_id: userId,
        authorization_method: method,
        credential_ref: input.credential_ref ?? null,
      });
      this.assets.appendAudit({
        user_id: userId,
        actor_user_id: session.user.user_id,
        event_type: 'tailscale_binding_prepare_requested',
        outcome: 'success',
        subject_id: binding.binding_id,
        details: { authorization_method: method },
      });
      return safeTailscaleBinding(binding);
    } catch (error) {
      if (/UNIQUE constraint|constraint failed|already exists/i.test(error instanceof Error ? error.message : String(error))) {
        throw new AuthServiceError(409, '该用户已有待确认的 Tailscale 换网请求');
      }
      throw error;
    }
  }

  confirmTailscaleBinding(
    session: AuthenticatedSession,
    bindingId: string,
    input: { tailnet_id: string; tailscale_user_id: string; node_id?: string | null },
  ): TailscaleBindingView {
    const pending = this.assets.getTailscaleBinding(bindingId);
    if (!pending) throw new AuthServiceError(404, 'Tailscale 绑定不存在');
    this.requireBindingAccess(session, pending.user_id);
    if (pending.state !== 'pending') throw new AuthServiceError(409, 'Tailscale 绑定不处于待确认状态');
    try {
      const binding = this.assets.confirmTailscaleBinding(bindingId, input);
      this.assets.appendAudit({
        user_id: pending.user_id,
        actor_user_id: session.user.user_id,
        event_type: 'tailscale_binding_confirm_requested',
        outcome: 'success',
        subject_id: bindingId,
        details: { tailnet_id: input.tailnet_id.trim(), tailscale_user_id: input.tailscale_user_id.trim(), node_id: input.node_id?.trim() ?? null },
      });
      return safeTailscaleBinding(binding);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (/already bound to another local user/i.test(message)) {
        throw new AuthServiceError(409, 'Tailscale 身份已绑定到其他本地用户');
      }
      if (/pending Tailscale binding not found|activation conflict/i.test(message)) {
        throw new AuthServiceError(409, 'Tailscale 绑定状态已变化，请重新发起换网');
      }
      throw error;
    }
  }

  revokeTailscaleBinding(session: AuthenticatedSession, bindingId: string): TailscaleBindingView {
    const current = this.assets.getTailscaleBinding(bindingId);
    if (!current) throw new AuthServiceError(404, 'Tailscale 绑定不存在');
    this.requireBindingAccess(session, current.user_id);
    if (current.state === 'revoked' || current.state === 'expired') return safeTailscaleBinding(current);
    return safeTailscaleBinding(this.assets.revokeTailscaleBinding(bindingId));
  }

  private async createProvisioning(user: LocalUserRecord): Promise<ProvisioningPayload> {
    const secret = encodeBase32(randomBytes(20));
    const credentialRef = `os:qlh/auth/${user.user_id}/totp-${randomUUID()}`;
    await this.credentials.set(credentialRef, secret);
    try {
      const authenticator = this.assets.createTotpAuthenticator({
        user_id: user.user_id,
        secret_ref: credentialRef,
      });
      const uri = `otpauth://totp/${encodeURIComponent(`${TOTP_ISSUER}:${user.username}`)}`
        + `?secret=${secret}&issuer=${encodeURIComponent(TOTP_ISSUER)}`
        + `&algorithm=${authenticator.algorithm}&digits=${authenticator.digits}`
        + `&period=${authenticator.period_seconds}`;
      return {
        user_id: user.user_id,
        authenticator_id: authenticator.authenticator_id,
        secret,
        otpauth_uri: uri,
        qr_payload: uri,
        algorithm: authenticator.algorithm,
        digits: authenticator.digits,
        period_seconds: authenticator.period_seconds,
      };
    } catch (error) {
      this.credentials.delete(credentialRef);
      throw error;
    }
  }

  private async verifyActiveTotp(user: LocalUserRecord, code: string): Promise<boolean> {
    const authenticator = this.assets.listAuthenticators(user.user_id)
      .find((entry) => entry.state === 'active');
    if (!authenticator) return false;
    const secret = await this.credentials.get(authenticator.secret_ref);
    if (!secret) throw new AuthServiceError(503, 'OS credential store 中缺少 TOTP 密钥');
    try {
      return verifyTotp(secret, code, authenticator);
    } catch (error) {
      if (error instanceof AuthServiceError && error.status === 401) return false;
      throw error;
    }
  }

  private consumeRecoveryCode(userId: string, code: string): boolean {
    const normalized = String(code ?? '').trim().toUpperCase();
    if (!/^[A-Z2-7]{4}(?:-[A-Z2-7]{4}){2}$/.test(normalized)) return false;
    const candidate = this.assets.listRecoveryCodes(userId).find((entry) => (
      entry.state === 'active' && entry.hash_scheme === 'scrypt' && verifyScryptHash(normalized, entry.code_hash)
    ));
    return candidate ? this.assets.consumeRecoveryCodeHash(userId, candidate.code_hash) : false;
  }

  private replaceRecoveryCodes(userId: string): string[] {
    const codes = Array.from({ length: RECOVERY_CODE_COUNT }, () => recoveryCode());
    this.assets.replaceRecoveryCodeHashes(userId, codes.map((code) => ({
      hash_scheme: 'scrypt' as const,
      code_hash: encodeScryptHash(code),
    })));
    return codes;
  }

  private issueSession(user: LocalUserRecord): LoginResult {
    const token = randomBytes(32).toString('base64url');
    const sessionId = `sess_${randomBytes(16).toString('hex')}`;
    const created = nowDate();
    const expires = new Date(created.getTime() + SESSION_TTL_MS);
    this.store.prepare(
      `INSERT INTO auth_sessions
         (session_id, user_id, token_hash, created_at, expires_at, last_seen_at, revoked_at)
       VALUES (?, ?, ?, ?, ?, ?, NULL)`,
    ).run(sessionId, user.user_id, hashToken(token), iso(created), iso(expires), iso(created));
    return {
      access_token: token,
      token_type: 'Bearer',
      session_id: sessionId,
      expires_at: iso(expires),
      user: safeUser(user),
    };
  }

  private ensureLoginAllowed(userId: string): void {
    const row = this.store.prepare(
      'SELECT locked_until FROM auth_login_limits WHERE user_id = ?',
    ).get(userId) as { locked_until: string | null } | undefined;
    if (row?.locked_until && row.locked_until > iso()) {
      throw new AuthServiceError(429, '登录失败次数过多，请稍后重试');
    }
  }

  private recordLoginFailure(userId: string): void {
    const now = nowDate();
    const current = this.store.prepare(
      'SELECT failure_count, first_failure_at FROM auth_login_limits WHERE user_id = ?',
    ).get(userId) as { failure_count: number; first_failure_at: string | null } | undefined;
    let count = Number(current?.failure_count ?? 0);
    const first = current?.first_failure_at ? new Date(current.first_failure_at) : null;
    if (!first || now.getTime() - first.getTime() > LOGIN_WINDOW_MS) count = 0;
    count += 1;
    const firstAt = count === 1 ? iso(now) : (current?.first_failure_at ?? iso(now));
    const lockedUntil = count >= LOGIN_MAX_FAILURES
      ? iso(new Date(now.getTime() + LOGIN_LOCK_MS)) : null;
    this.store.prepare(
      `INSERT INTO auth_login_limits(user_id, failure_count, first_failure_at, locked_until, updated_at)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(user_id) DO UPDATE SET failure_count=excluded.failure_count,
         first_failure_at=excluded.first_failure_at, locked_until=excluded.locked_until,
         updated_at=excluded.updated_at`,
    ).run(userId, count, firstAt, lockedUntil, iso(now));
  }

  private clearLoginFailures(userId: string): void {
    this.store.prepare('DELETE FROM auth_login_limits WHERE user_id = ?').run(userId);
  }

  private revokeUserSessions(userId: string): void {
    this.store.prepare(
      `UPDATE auth_sessions SET revoked_at = ?
       WHERE user_id = ? AND revoked_at IS NULL`,
    ).run(iso(), userId);
  }

  private requireActiveUser(userId: string): LocalUserRecord {
    const user = this.assets.getUser(userId);
    if (!user || user.status !== 'active') throw new AuthServiceError(404, '本地用户不存在或已停用');
    return user;
  }

  private requireManager(session: AuthenticatedSession, requestedRole?: LocalUserRole): void {
    if (session.user.role !== 'owner' && session.user.role !== 'admin') {
      throw new AuthServiceError(403, '需要 owner 或 admin 权限');
    }
    if (requestedRole && requestedRole !== 'member' && session.user.role !== 'owner') {
      throw new AuthServiceError(403, '只有 owner 可以授予 admin 或 owner 角色');
    }
  }

  private requireBindingAccess(session: AuthenticatedSession, userId: string): void {
    const target = this.assets.getUser(userId);
    if (!target || target.status !== 'active') throw new AuthServiceError(404, '本地用户不存在或未激活');
    if (session.user.user_id === userId) return;
    if (session.user.role === 'owner') return;
    if (session.user.role === 'admin' && target.role !== 'owner') return;
    throw new AuthServiceError(403, '没有管理该用户 Tailscale 绑定的权限');
  }
}

export { encodeBase32, totp };

function safeTailscaleBinding(binding: TailscaleBindingRecord): TailscaleBindingView {
  return {
    binding_id: binding.binding_id,
    user_id: binding.user_id,
    tailnet_id: binding.tailnet_id,
    tailscale_user_id: binding.tailscale_user_id,
    node_id: binding.node_id,
    state: binding.state,
    authorization_method: binding.authorization_method,
    aggregate_version: binding.aggregate_version,
    prepared_at: binding.prepared_at,
    updated_at: binding.updated_at,
    confirmed_at: binding.confirmed_at,
    revoked_at: binding.revoked_at,
    last_verified_at: binding.last_verified_at,
  };
}
