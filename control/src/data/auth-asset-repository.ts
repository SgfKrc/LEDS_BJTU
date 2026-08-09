import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { normalizeCredentialRef } from './model-credential-store';
import { SqliteStore } from './sqlite-store';

export type LocalUserRole = 'owner' | 'admin' | 'member';
export type LocalUserStatus = 'active' | 'suspended' | 'revoked';
export type AuthenticatorState = 'pending' | 'active' | 'revoked';
export type RecoveryCodeState = 'active' | 'consumed' | 'revoked';
export type TailscaleBindingState = 'pending' | 'active' | 'revoked' | 'expired';
export type TailscaleAuthorizationMethod = 'tailscale_cli' | 'local_status' | 'oauth_app';
export type AuthAuditOutcome = 'success' | 'failure' | 'denied';

export interface LocalUserRecord {
  user_id: string;
  username: string;
  username_normalized: string;
  display_name: string;
  role: LocalUserRole;
  status: LocalUserStatus;
  aggregate_version: number;
  created_at: string;
  updated_at: string;
  revoked_at: string | null;
}

export interface UserAuthenticatorRecord {
  authenticator_id: string;
  user_id: string;
  kind: 'totp';
  state: AuthenticatorState;
  secret_ref: string;
  algorithm: 'SHA1' | 'SHA256' | 'SHA512';
  digits: 6 | 8;
  period_seconds: number;
  created_at: string;
  updated_at: string;
  confirmed_at: string | null;
  revoked_at: string | null;
}

export interface RecoveryCodeRecord {
  recovery_code_id: string;
  user_id: string;
  batch_id: string;
  hash_scheme: 'scrypt' | 'argon2id';
  code_hash: string;
  state: RecoveryCodeState;
  created_at: string;
  consumed_at: string | null;
  revoked_at: string | null;
}

export interface TailscaleBindingRecord {
  binding_id: string;
  user_id: string;
  tailnet_id: string | null;
  tailscale_user_id: string | null;
  node_id: string | null;
  state: TailscaleBindingState;
  authorization_method: TailscaleAuthorizationMethod;
  credential_ref: string | null;
  aggregate_version: number;
  prepared_at: string;
  updated_at: string;
  confirmed_at: string | null;
  revoked_at: string | null;
  last_verified_at: string | null;
}

export interface AuthAuditEventRecord {
  event_id: string;
  user_id: string | null;
  actor_user_id: string | null;
  event_type: string;
  outcome: AuthAuditOutcome;
  reason_code: string | null;
  subject_id: string | null;
  details: Record<string, unknown>;
  created_at: string;
}

const FORBIDDEN_AUDIT_KEYS = new Set([
  'secret', 'totp_secret', 'password', 'token', 'access_token', 'refresh_token',
  'recovery_code', 'otp', 'verification_code', 'otpauth_uri', 'qr', 'ciphertext',
]);

function nowIso(): string {
  return new Date().toISOString();
}

function normalizedUsername(username: string): string {
  const value = username.trim();
  if (!/^[a-z0-9][a-z0-9._-]{0,63}$/i.test(value)) {
    throw new Error('username must contain 1-64 letters, digits, dot, underscore, or hyphen');
  }
  return value.toLowerCase();
}

function requiredIdentifier(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized || normalized.length > 256 || /[\0\r\n]/.test(normalized)) {
    throw new Error(`${field} is invalid`);
  }
  return normalized;
}

function validateRecoveryHash(scheme: 'scrypt' | 'argon2id', codeHash: string): string {
  const value = codeHash.trim();
  const prefix = scheme === 'scrypt' ? '$scrypt$' : '$argon2id$';
  if (value.length < 48 || value.length > 1024 || !value.startsWith(prefix)) {
    throw new Error(`recovery code hash must be an encoded ${scheme} hash`);
  }
  return value;
}

function normalizeAuditKey(key: string): string {
  return key.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function validateAuditValue(value: unknown, key?: string): void {
  if (key && FORBIDDEN_AUDIT_KEYS.has(normalizeAuditKey(key))) {
    throw new Error(`auth audit details cannot contain ${key}`);
  }
  if (typeof value === 'string'
      && (/otpauth:\/\//i.test(value) || /\bbearer\s+/i.test(value) || /\btskey-/i.test(value))) {
    throw new Error('auth audit details contain secret material');
  }
  if (Array.isArray(value)) {
    value.forEach((entry) => validateAuditValue(entry));
  } else if (value && typeof value === 'object') {
    Object.entries(value as Record<string, unknown>)
      .forEach(([entryKey, entry]) => validateAuditValue(entry, entryKey));
  }
}

@Injectable()
export class AuthAssetRepository {
  constructor(private readonly store: SqliteStore) {}

  createUser(input: {
    username: string;
    display_name?: string;
    role?: LocalUserRole;
    status?: LocalUserStatus;
  }): LocalUserRecord {
    const username = input.username.trim();
    const usernameNormalized = normalizedUsername(username);
    const displayName = (input.display_name ?? '').trim();
    if (displayName.length > 128) throw new Error('display_name is too long');
    const role = input.role ?? 'member';
    const status = input.status ?? 'active';
    const userId = `usr_${randomUUID()}`;
    const now = nowIso();
    this.store.transaction(() => {
      this.store.prepare(
        `INSERT INTO local_users
           (user_id, username, username_normalized, display_name, role, status,
            aggregate_version, created_at, updated_at, revoked_at)
         VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)`,
      ).run(
        userId, username, usernameNormalized, displayName, role, status,
        now, now, status === 'revoked' ? now : null,
      );
      this.insertAudit({
        user_id: userId,
        event_type: 'user_created',
        outcome: 'success',
        subject_id: userId,
        details: { role, status },
      }, now);
    });
    return this.getUser(userId) as LocalUserRecord;
  }

  getUser(userId: string): LocalUserRecord | null {
    const row = this.store.prepare(
      `SELECT user_id, username, username_normalized, display_name, role, status,
              aggregate_version, created_at, updated_at, revoked_at
       FROM local_users WHERE user_id = ?`,
    ).get(userId) as unknown as LocalUserRecord | undefined;
    return row ?? null;
  }

  getUserByUsername(username: string): LocalUserRecord | null {
    const row = this.store.prepare(
      `SELECT user_id, username, username_normalized, display_name, role, status,
              aggregate_version, created_at, updated_at, revoked_at
       FROM local_users WHERE username_normalized = ?`,
    ).get(normalizedUsername(username)) as unknown as LocalUserRecord | undefined;
    return row ?? null;
  }

  listUsers(): LocalUserRecord[] {
    return this.store.prepare(
      `SELECT user_id, username, username_normalized, display_name, role, status,
              aggregate_version, created_at, updated_at, revoked_at
       FROM local_users ORDER BY created_at, user_id`,
    ).all() as unknown as LocalUserRecord[];
  }

  updateUser(
    userId: string,
    expectedVersion: number,
    patch: { display_name?: string; role?: LocalUserRole; status?: LocalUserStatus },
  ): LocalUserRecord {
    const current = this.requireUser(userId);
    const displayName = patch.display_name === undefined
      ? current.display_name : patch.display_name.trim();
    if (displayName.length > 128) throw new Error('display_name is too long');
    const role = patch.role ?? current.role;
    const status = patch.status ?? current.status;
    if (current.role === 'owner' && current.status === 'active'
        && (role !== 'owner' || status !== 'active')) {
      const otherOwner = this.store.prepare(
        `SELECT COUNT(*) AS count FROM local_users
         WHERE user_id <> ? AND role = 'owner' AND status = 'active'`,
      ).get(userId) as { count: number };
      if (Number(otherOwner.count) === 0) {
        throw new Error('cannot deactivate the only active owner');
      }
    }
    const now = nowIso();
    const result = this.store.prepare(
      `UPDATE local_users SET display_name = ?, role = ?, status = ?,
         aggregate_version = aggregate_version + 1, updated_at = ?, revoked_at = ?
       WHERE user_id = ? AND aggregate_version = ?`,
    ).run(
      displayName, role, status, now, status === 'revoked' ? now : null,
      userId, expectedVersion,
    );
    if (Number(result.changes) !== 1) throw new Error('local user version conflict');
    return this.getUser(userId) as LocalUserRecord;
  }

  createTotpAuthenticator(input: {
    user_id: string;
    secret_ref: string;
    algorithm?: 'SHA1' | 'SHA256' | 'SHA512';
    digits?: 6 | 8;
    period_seconds?: number;
  }): UserAuthenticatorRecord {
    this.requireUser(input.user_id);
    const secretRef = normalizeCredentialRef(input.secret_ref);
    const algorithm = input.algorithm ?? 'SHA1';
    const digits = input.digits ?? 6;
    const period = input.period_seconds ?? 30;
    if (!Number.isInteger(period) || period < 15 || period > 120) {
      throw new Error('TOTP period_seconds must be between 15 and 120');
    }
    const authenticatorId = `totp_${randomUUID()}`;
    const now = nowIso();
    this.store.transaction(() => {
      this.store.prepare(
        `INSERT INTO user_authenticators
           (authenticator_id, user_id, kind, state, secret_ref, algorithm, digits,
            period_seconds, created_at, updated_at, confirmed_at, revoked_at)
         VALUES (?, ?, 'totp', 'pending', ?, ?, ?, ?, ?, ?, NULL, NULL)`,
      ).run(authenticatorId, input.user_id, secretRef, algorithm, digits, period, now, now);
      this.insertAudit({
        user_id: input.user_id,
        event_type: 'totp_reference_prepared',
        outcome: 'success',
        subject_id: authenticatorId,
        details: { algorithm, digits, period_seconds: period },
      }, now);
    });
    return this.getAuthenticator(authenticatorId) as UserAuthenticatorRecord;
  }

  activateTotpAuthenticator(authenticatorId: string): UserAuthenticatorRecord {
    const now = nowIso();
    this.store.transaction(() => {
      const pending = this.getAuthenticator(authenticatorId);
      if (!pending || pending.state !== 'pending') {
        throw new Error('pending TOTP authenticator not found');
      }
      this.store.prepare(
        `UPDATE user_authenticators SET state = 'revoked', updated_at = ?, revoked_at = ?
         WHERE user_id = ? AND kind = 'totp' AND state = 'active'`,
      ).run(now, now, pending.user_id);
      const result = this.store.prepare(
        `UPDATE user_authenticators SET state = 'active', updated_at = ?,
           confirmed_at = ?, revoked_at = NULL
         WHERE authenticator_id = ? AND state = 'pending'`,
      ).run(now, now, authenticatorId);
      if (Number(result.changes) !== 1) throw new Error('TOTP activation conflict');
      this.insertAudit({
        user_id: pending.user_id,
        event_type: 'totp_activated',
        outcome: 'success',
        subject_id: authenticatorId,
        details: {},
      }, now);
    });
    return this.getAuthenticator(authenticatorId) as UserAuthenticatorRecord;
  }

  revokeTotpAuthenticator(authenticatorId: string): UserAuthenticatorRecord {
    const current = this.getAuthenticator(authenticatorId);
    if (!current) throw new Error('TOTP authenticator not found');
    const now = nowIso();
    this.store.transaction(() => {
      this.store.prepare(
        `UPDATE user_authenticators SET state = 'revoked', updated_at = ?, revoked_at = ?
         WHERE authenticator_id = ?`,
      ).run(now, now, authenticatorId);
      this.insertAudit({
        user_id: current.user_id,
        event_type: 'totp_revoked',
        outcome: 'success',
        subject_id: authenticatorId,
        details: {},
      }, now);
    });
    return this.getAuthenticator(authenticatorId) as UserAuthenticatorRecord;
  }

  getAuthenticator(authenticatorId: string): UserAuthenticatorRecord | null {
    const row = this.store.prepare(
      `SELECT authenticator_id, user_id, kind, state, secret_ref, algorithm, digits,
              period_seconds, created_at, updated_at, confirmed_at, revoked_at
       FROM user_authenticators WHERE authenticator_id = ?`,
    ).get(authenticatorId) as unknown as UserAuthenticatorRecord | undefined;
    return row ?? null;
  }

  listAuthenticators(userId: string): UserAuthenticatorRecord[] {
    return this.store.prepare(
      `SELECT authenticator_id, user_id, kind, state, secret_ref, algorithm, digits,
              period_seconds, created_at, updated_at, confirmed_at, revoked_at
       FROM user_authenticators WHERE user_id = ? ORDER BY created_at, authenticator_id`,
    ).all(userId) as unknown as UserAuthenticatorRecord[];
  }

  replaceRecoveryCodeHashes(
    userId: string,
    hashes: Array<{ hash_scheme: 'scrypt' | 'argon2id'; code_hash: string }>,
  ): RecoveryCodeRecord[] {
    this.requireUser(userId);
    if (hashes.length < 1 || hashes.length > 32) {
      throw new Error('recovery code batch must contain 1-32 hashes');
    }
    const normalized = hashes.map((entry) => ({
      hash_scheme: entry.hash_scheme,
      code_hash: validateRecoveryHash(entry.hash_scheme, entry.code_hash),
    }));
    if (new Set(normalized.map((entry) => entry.code_hash)).size !== normalized.length) {
      throw new Error('recovery code hashes must be unique');
    }
    const batchId = `rcb_${randomUUID()}`;
    const now = nowIso();
    this.store.transaction(() => {
      this.store.prepare(
        `UPDATE auth_recovery_codes SET state = 'revoked', revoked_at = ?
         WHERE user_id = ? AND state = 'active'`,
      ).run(now, userId);
      for (const entry of normalized) {
        this.store.prepare(
          `INSERT INTO auth_recovery_codes
             (recovery_code_id, user_id, batch_id, hash_scheme, code_hash, state,
              created_at, consumed_at, revoked_at)
           VALUES (?, ?, ?, ?, ?, 'active', ?, NULL, NULL)`,
        ).run(
          `rc_${randomUUID()}`, userId, batchId,
          entry.hash_scheme, entry.code_hash, now,
        );
      }
      this.insertAudit({
        user_id: userId,
        event_type: 'recovery_codes_replaced',
        outcome: 'success',
        subject_id: batchId,
        details: { count: normalized.length },
      }, now);
    });
    return this.listRecoveryCodes(userId).filter((entry) => entry.batch_id === batchId);
  }

  consumeRecoveryCodeHash(userId: string, codeHash: string): boolean {
    const now = nowIso();
    return this.store.transaction(() => {
      const result = this.store.prepare(
        `UPDATE auth_recovery_codes SET state = 'consumed', consumed_at = ?
         WHERE user_id = ? AND code_hash = ? AND state = 'active'`,
      ).run(now, userId, codeHash);
      const consumed = Number(result.changes) === 1;
      this.insertAudit({
        user_id: userId,
        event_type: 'recovery_code_consumed',
        outcome: consumed ? 'success' : 'denied',
        reason_code: consumed ? null : 'recovery_code_unavailable',
        details: {},
      }, now);
      return consumed;
    });
  }

  listRecoveryCodes(userId: string): RecoveryCodeRecord[] {
    return this.store.prepare(
      `SELECT recovery_code_id, user_id, batch_id, hash_scheme, code_hash, state,
              created_at, consumed_at, revoked_at
       FROM auth_recovery_codes WHERE user_id = ? ORDER BY created_at, recovery_code_id`,
    ).all(userId) as unknown as RecoveryCodeRecord[];
  }

  prepareTailscaleBinding(input: {
    user_id: string;
    authorization_method: TailscaleAuthorizationMethod;
    credential_ref?: string | null;
  }): TailscaleBindingRecord {
    this.requireUser(input.user_id);
    const credentialRef = input.credential_ref
      ? normalizeCredentialRef(input.credential_ref) : null;
    const bindingId = `tsb_${randomUUID()}`;
    const now = nowIso();
    this.store.transaction(() => {
      this.store.prepare(
        `INSERT INTO tailscale_bindings
           (binding_id, user_id, tailnet_id, tailscale_user_id, node_id, state,
            authorization_method, credential_ref, aggregate_version, prepared_at,
            updated_at, confirmed_at, revoked_at, last_verified_at)
         VALUES (?, ?, NULL, NULL, NULL, 'pending', ?, ?, 1, ?, ?, NULL, NULL, NULL)`,
      ).run(bindingId, input.user_id, input.authorization_method, credentialRef, now, now);
      this.insertAudit({
        user_id: input.user_id,
        event_type: 'tailscale_binding_prepared',
        outcome: 'success',
        subject_id: bindingId,
        details: { authorization_method: input.authorization_method },
      }, now);
    });
    return this.getTailscaleBinding(bindingId) as TailscaleBindingRecord;
  }

  confirmTailscaleBinding(
    bindingId: string,
    input: { tailnet_id: string; tailscale_user_id: string; node_id?: string | null },
  ): TailscaleBindingRecord {
    const tailnetId = requiredIdentifier(input.tailnet_id, 'tailnet_id');
    const tailscaleUserId = requiredIdentifier(input.tailscale_user_id, 'tailscale_user_id');
    const nodeId = input.node_id ? requiredIdentifier(input.node_id, 'node_id') : null;
    const now = nowIso();
    this.store.transaction(() => {
      const pending = this.getTailscaleBinding(bindingId);
      if (!pending || pending.state !== 'pending') {
        throw new Error('pending Tailscale binding not found');
      }
      const conflict = this.store.prepare(
        `SELECT binding_id, user_id FROM tailscale_bindings
         WHERE tailnet_id = ? AND tailscale_user_id = ? AND state = 'active'`,
      ).get(tailnetId, tailscaleUserId) as { binding_id: string; user_id: string } | undefined;
      if (conflict && conflict.user_id !== pending.user_id) {
        throw new Error('Tailscale identity is already bound to another local user');
      }
      this.store.prepare(
        `UPDATE tailscale_bindings SET state = 'revoked',
           aggregate_version = aggregate_version + 1, updated_at = ?, revoked_at = ?
         WHERE user_id = ? AND state = 'active'`,
      ).run(now, now, pending.user_id);
      const result = this.store.prepare(
        `UPDATE tailscale_bindings SET tailnet_id = ?, tailscale_user_id = ?, node_id = ?,
           state = 'active', aggregate_version = aggregate_version + 1, updated_at = ?,
           confirmed_at = ?, revoked_at = NULL, last_verified_at = ?
         WHERE binding_id = ? AND state = 'pending'`,
      ).run(tailnetId, tailscaleUserId, nodeId, now, now, now, bindingId);
      if (Number(result.changes) !== 1) throw new Error('Tailscale binding activation conflict');
      this.insertAudit({
        user_id: pending.user_id,
        event_type: 'tailscale_binding_activated',
        outcome: 'success',
        subject_id: bindingId,
        details: { tailnet_id: tailnetId, tailscale_user_id: tailscaleUserId, node_id: nodeId },
      }, now);
    });
    return this.getTailscaleBinding(bindingId) as TailscaleBindingRecord;
  }

  revokeTailscaleBinding(bindingId: string): TailscaleBindingRecord {
    const current = this.getTailscaleBinding(bindingId);
    if (!current) throw new Error('Tailscale binding not found');
    const now = nowIso();
    this.store.transaction(() => {
      this.store.prepare(
        `UPDATE tailscale_bindings SET state = 'revoked',
           aggregate_version = aggregate_version + 1, updated_at = ?, revoked_at = ?
         WHERE binding_id = ?`,
      ).run(now, now, bindingId);
      this.insertAudit({
        user_id: current.user_id,
        event_type: 'tailscale_binding_revoked',
        outcome: 'success',
        subject_id: bindingId,
        details: {},
      }, now);
    });
    return this.getTailscaleBinding(bindingId) as TailscaleBindingRecord;
  }

  getTailscaleBinding(bindingId: string): TailscaleBindingRecord | null {
    const row = this.store.prepare(
      `SELECT binding_id, user_id, tailnet_id, tailscale_user_id, node_id, state,
              authorization_method, credential_ref, aggregate_version, prepared_at,
              updated_at, confirmed_at, revoked_at, last_verified_at
       FROM tailscale_bindings WHERE binding_id = ?`,
    ).get(bindingId) as unknown as TailscaleBindingRecord | undefined;
    return row ?? null;
  }

  listTailscaleBindings(userId: string): TailscaleBindingRecord[] {
    return this.store.prepare(
      `SELECT binding_id, user_id, tailnet_id, tailscale_user_id, node_id, state,
              authorization_method, credential_ref, aggregate_version, prepared_at,
              updated_at, confirmed_at, revoked_at, last_verified_at
       FROM tailscale_bindings WHERE user_id = ? ORDER BY prepared_at, binding_id`,
    ).all(userId) as unknown as TailscaleBindingRecord[];
  }

  appendAudit(input: {
    user_id?: string | null;
    actor_user_id?: string | null;
    event_type: string;
    outcome: AuthAuditOutcome;
    reason_code?: string | null;
    subject_id?: string | null;
    details?: Record<string, unknown>;
  }): AuthAuditEventRecord {
    const eventId = this.insertAudit(input, nowIso());
    return this.getAuditEvent(eventId) as AuthAuditEventRecord;
  }

  getAuditEvent(eventId: string): AuthAuditEventRecord | null {
    const row = this.store.prepare(
      `SELECT event_id, user_id, actor_user_id, event_type, outcome, reason_code,
              subject_id, details, created_at
       FROM auth_audit_events WHERE event_id = ?`,
    ).get(eventId) as Record<string, unknown> | undefined;
    return row ? this.toAuditRecord(row) : null;
  }

  listAuditEvents(userId?: string): AuthAuditEventRecord[] {
    const rows = userId
      ? this.store.prepare(
        `SELECT event_id, user_id, actor_user_id, event_type, outcome, reason_code,
                subject_id, details, created_at
         FROM auth_audit_events WHERE user_id = ? ORDER BY created_at, event_id`,
      ).all(userId)
      : this.store.prepare(
        `SELECT event_id, user_id, actor_user_id, event_type, outcome, reason_code,
                subject_id, details, created_at
         FROM auth_audit_events ORDER BY created_at, event_id`,
      ).all();
    return (rows as Array<Record<string, unknown>>).map((row) => this.toAuditRecord(row));
  }

  private requireUser(userId: string): LocalUserRecord {
    const user = this.getUser(userId);
    if (!user) throw new Error('local user not found');
    return user;
  }

  private insertAudit(input: {
    user_id?: string | null;
    actor_user_id?: string | null;
    event_type: string;
    outcome: AuthAuditOutcome;
    reason_code?: string | null;
    subject_id?: string | null;
    details?: Record<string, unknown>;
  }, createdAt: string): string {
    const eventType = input.event_type.trim();
    if (!/^[a-z0-9][a-z0-9._-]{0,63}$/i.test(eventType)) {
      throw new Error('auth audit event_type is invalid');
    }
    const details = input.details ?? {};
    validateAuditValue(details);
    const payload = JSON.stringify(details);
    if (Buffer.byteLength(payload, 'utf8') > 16 * 1024) {
      throw new Error('auth audit details are too large');
    }
    const eventId = `aae_${randomUUID()}`;
    this.store.prepare(
      `INSERT INTO auth_audit_events
         (event_id, user_id, actor_user_id, event_type, outcome, reason_code,
          subject_id, details, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      eventId, input.user_id ?? null, input.actor_user_id ?? null, eventType,
      input.outcome, input.reason_code ?? null, input.subject_id ?? null,
      payload, createdAt,
    );
    return eventId;
  }

  private toAuditRecord(row: Record<string, unknown>): AuthAuditEventRecord {
    return {
      event_id: String(row.event_id),
      user_id: row.user_id === null ? null : String(row.user_id),
      actor_user_id: row.actor_user_id === null ? null : String(row.actor_user_id),
      event_type: String(row.event_type),
      outcome: String(row.outcome) as AuthAuditOutcome,
      reason_code: row.reason_code === null ? null : String(row.reason_code),
      subject_id: row.subject_id === null ? null : String(row.subject_id),
      details: JSON.parse(String(row.details)) as Record<string, unknown>,
      created_at: String(row.created_at),
    };
  }
}
