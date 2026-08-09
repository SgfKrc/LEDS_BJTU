/**
 * M1 主节点本地事务数据库 — SQLite 基础层（M0 调研结论：node:sqlite 内置驱动）。
 *
 * 职责（一键模型部署计划 §16 M1）：
 *  - 打开本地 SQLite（WAL、foreign_keys、busy_timeout）；
 *  - 版本化迁移器（PRAGMA user_version，事务包裹，幂等可重复执行）；
 *  - 本地健康（SELECT 1 + 可写性探测）与 backup；
 *  - control-svc 是唯一写者，其他服务走 API。
 *
 * 路径：QLH_SQLITE_PATH 覆盖，默认 <cwd>/qlh-control.sqlite3。
 */
import { DatabaseSync, backup as sqliteBackup } from 'node:sqlite';
import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
  scryptSync,
} from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const BACKUP_MAGIC = Buffer.from('QLHSQLB1', 'ascii');
const BACKUP_VERSION = 1;
const BACKUP_KDF_N = 16_384;
const BACKUP_KDF_R = 8;
const BACKUP_KDF_P = 1;
const MAX_BACKUP_HEADER_BYTES = 64 * 1024;

interface EncryptedBackupHeader {
  version: number;
  cipher: 'aes-256-gcm';
  kdf: 'scrypt';
  kdf_n: number;
  kdf_r: number;
  kdf_p: number;
  salt: string;
  iv: string;
  auth_tag: string;
  ciphertext_sha256: string;
  created_at: string;
}

export interface EncryptedBackupInfo {
  version: number;
  schema_version: number;
  plaintext_bytes: number;
  ciphertext_bytes: number;
  created_at: string;
  reference_integrity: BackupReferenceIntegrity;
}

export interface BackupReferenceIntegrity {
  ok: boolean;
  manifest_index: boolean;
  checked_manifests: number;
  checked_references: number;
  checked_blobs: number;
  external_blobs_checked: boolean;
  errors: string[];
}

export interface RestoreResult extends EncryptedBackupInfo {
  restored_path: string;
  previous_path: string | null;
}

export interface Migration {
  version: number;
  up: (db: DatabaseSync) => void;
}

export interface LocalStorageHealth {
  status: 'ok' | 'unavailable';
  backend: 'sqlite';
  writable: boolean;
  path: string;
  schema_version: number;
}

export function resolveSqlitePath(): string {
  const override = process.env.QLH_SQLITE_PATH?.trim();
  if (override) return path.resolve(override);
  const configuredStateDir = process.env.QLH_STATE_DIR?.trim();
  let stateDir: string;
  if (configuredStateDir) {
    stateDir = path.resolve(configuredStateDir);
  } else if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA?.trim()
      || path.join(os.homedir(), 'AppData', 'Local');
    stateDir = path.join(localAppData, 'QLH-Edge-Inference', 'state');
  } else {
    const xdgStateHome = process.env.XDG_STATE_HOME?.trim()
      || path.join(os.homedir(), '.local', 'state');
    stateDir = path.join(xdgStateHome, 'qlh-edge-inference');
  }

  const statePath = path.resolve(stateDir, 'qlh-control.sqlite3');
  const legacyPath = path.resolve(process.cwd(), 'qlh-control.sqlite3');
  if (fs.existsSync(legacyPath) && !fs.existsSync(statePath)) return legacyPath;
  return statePath;
}

/** v1：模型/集群/outbox 目标表（对齐 schemas/ 与 migration-map.json）。 */
function migrateV1(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS cluster_settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS model_registry (
      model_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      model_path TEXT NOT NULL,
      gguf_path TEXT,
      quantization TEXT,
      sha256 TEXT,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS cluster_endpoints (
      endpoint_id TEXT PRIMARY KEY,
      cluster_id TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      scheme TEXT NOT NULL,
      host TEXT NOT NULL,
      port INTEGER NOT NULL,
      status TEXT NOT NULL,
      last_verified_at TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS cluster_profiles (
      profile_id TEXT PRIMARY KEY,
      cluster_id TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      master_endpoint TEXT NOT NULL,
      status TEXT NOT NULL,
      key_ref TEXT NOT NULL,
      node_role TEXT NOT NULL,
      last_verified_at TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS catalog_models (
      model_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS pull_jobs (
      job_id TEXT PRIMARY KEY,
      idempotency_key TEXT NOT NULL UNIQUE,
      state TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS artifacts (
      artifact_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS deployments (
      deployment_id TEXT PRIMARY KEY,
      artifact_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      status TEXT NOT NULL,
      epoch INTEGER NOT NULL DEFAULT 0,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS outbox (
      event_id TEXT PRIMARY KEY,
      aggregate TEXT NOT NULL,
      aggregate_version INTEGER NOT NULL,
      event_type TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL,
      projected_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_outbox_pending
      ON outbox(created_at) WHERE projected_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_deployments_artifact
      ON deployments(artifact_id);
  `);
}

/** v2: local runtime trial-load results, separate from immutable manifests. */
function migrateV2(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS artifact_runtime_checks (
      artifact_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      runtime_profile TEXT NOT NULL,
      status TEXT NOT NULL,
      payload TEXT NOT NULL,
      checked_at TEXT NOT NULL,
      PRIMARY KEY (artifact_id, node_id, runtime_profile)
    );
    CREATE INDEX IF NOT EXISTS idx_artifact_runtime_checks_status
      ON artifact_runtime_checks(status, checked_at);
  `);
}

/** v3: control-svc core domains previously stored as JSON files. */
function migrateV3(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      session_id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      message_count INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_updated
      ON sessions(updated_at DESC);
    CREATE TABLE IF NOT EXISTS session_messages (
      message_id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at TEXT,
      metrics TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_session_messages_session
      ON session_messages(session_id, message_id);
    CREATE TABLE IF NOT EXISTS workflow_journal (
      workflow_id TEXT PRIMARY KEY,
      updated_at REAL NOT NULL,
      payload TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_workflow_journal_updated
      ON workflow_journal(updated_at DESC);
    CREATE TABLE IF NOT EXISTS review_tickets (
      ticket_id TEXT PRIMARY KEY,
      status TEXT NOT NULL,
      created_at REAL NOT NULL,
      payload TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_review_tickets_created
      ON review_tickets(created_at DESC);
  `);
}

/** v4: content-addressed artifact manifest index and recovery references. */
function migrateV4(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS artifact_manifests (
      manifest_key TEXT PRIMARY KEY,
      artifact_id TEXT NOT NULL,
      namespace TEXT NOT NULL,
      name TEXT NOT NULL,
      tag TEXT NOT NULL,
      manifest_path TEXT NOT NULL,
      manifest_sha256 TEXT NOT NULL,
      file_count INTEGER NOT NULL,
      total_bytes INTEGER NOT NULL,
      blob_digests TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_manifests_reference
      ON artifact_manifests(namespace, name, tag);
    CREATE INDEX IF NOT EXISTS idx_artifact_manifests_artifact
      ON artifact_manifests(artifact_id);
  `);
}

/** v5: main-node-owned local identities, authenticators, recovery, and tailnet bindings. */
function migrateV5(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS local_users (
      user_id TEXT PRIMARY KEY,
      username TEXT NOT NULL CHECK(length(trim(username)) BETWEEN 1 AND 64),
      username_normalized TEXT NOT NULL CHECK(length(username_normalized) BETWEEN 1 AND 64),
      display_name TEXT NOT NULL DEFAULT '' CHECK(length(display_name) <= 128),
      role TEXT NOT NULL CHECK(role IN ('owner', 'admin', 'member')),
      status TEXT NOT NULL CHECK(status IN ('active', 'suspended', 'revoked')),
      aggregate_version INTEGER NOT NULL DEFAULT 1 CHECK(aggregate_version >= 1),
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      revoked_at TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_local_users_username
      ON local_users(username_normalized);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_local_users_owner
      ON local_users(role) WHERE role = 'owner' AND status <> 'revoked';

    CREATE TABLE IF NOT EXISTS user_authenticators (
      authenticator_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES local_users(user_id) ON DELETE CASCADE,
      kind TEXT NOT NULL CHECK(kind = 'totp'),
      state TEXT NOT NULL CHECK(state IN ('pending', 'active', 'revoked')),
      secret_ref TEXT NOT NULL CHECK(secret_ref LIKE 'os:%'),
      algorithm TEXT NOT NULL CHECK(algorithm IN ('SHA1', 'SHA256', 'SHA512')),
      digits INTEGER NOT NULL CHECK(digits IN (6, 8)),
      period_seconds INTEGER NOT NULL CHECK(period_seconds BETWEEN 15 AND 120),
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      confirmed_at TEXT,
      revoked_at TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_user_authenticators_pending
      ON user_authenticators(user_id) WHERE kind = 'totp' AND state = 'pending';
    CREATE UNIQUE INDEX IF NOT EXISTS idx_user_authenticators_active
      ON user_authenticators(user_id) WHERE kind = 'totp' AND state = 'active';

    CREATE TABLE IF NOT EXISTS auth_recovery_codes (
      recovery_code_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES local_users(user_id) ON DELETE CASCADE,
      batch_id TEXT NOT NULL,
      hash_scheme TEXT NOT NULL CHECK(hash_scheme IN ('scrypt', 'argon2id')),
      code_hash TEXT NOT NULL CHECK(length(code_hash) >= 48),
      state TEXT NOT NULL CHECK(state IN ('active', 'consumed', 'revoked')),
      created_at TEXT NOT NULL,
      consumed_at TEXT,
      revoked_at TEXT,
      UNIQUE(user_id, code_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_auth_recovery_codes_active
      ON auth_recovery_codes(user_id, batch_id) WHERE state = 'active';

    CREATE TABLE IF NOT EXISTS tailscale_bindings (
      binding_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES local_users(user_id) ON DELETE CASCADE,
      tailnet_id TEXT,
      tailscale_user_id TEXT,
      node_id TEXT,
      state TEXT NOT NULL CHECK(state IN ('pending', 'active', 'revoked', 'expired')),
      authorization_method TEXT NOT NULL
        CHECK(authorization_method IN ('tailscale_cli', 'local_status', 'oauth_app')),
      credential_ref TEXT CHECK(credential_ref IS NULL OR credential_ref LIKE 'os:%'),
      aggregate_version INTEGER NOT NULL DEFAULT 1 CHECK(aggregate_version >= 1),
      prepared_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      confirmed_at TEXT,
      revoked_at TEXT,
      last_verified_at TEXT,
      CHECK(state <> 'active' OR (
        tailnet_id IS NOT NULL AND length(trim(tailnet_id)) > 0
        AND tailscale_user_id IS NOT NULL AND length(trim(tailscale_user_id)) > 0
        AND confirmed_at IS NOT NULL
      ))
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tailscale_bindings_pending_user
      ON tailscale_bindings(user_id) WHERE state = 'pending';
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tailscale_bindings_active_user
      ON tailscale_bindings(user_id) WHERE state = 'active';
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tailscale_bindings_active_external
      ON tailscale_bindings(tailnet_id, tailscale_user_id) WHERE state = 'active';

    CREATE TABLE IF NOT EXISTS auth_audit_events (
      event_id TEXT PRIMARY KEY,
      user_id TEXT REFERENCES local_users(user_id) ON DELETE SET NULL,
      actor_user_id TEXT REFERENCES local_users(user_id) ON DELETE SET NULL,
      event_type TEXT NOT NULL CHECK(length(trim(event_type)) BETWEEN 1 AND 64),
      outcome TEXT NOT NULL CHECK(outcome IN ('success', 'failure', 'denied')),
      reason_code TEXT,
      subject_id TEXT,
      details TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_auth_audit_events_created
      ON auth_audit_events(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_auth_audit_events_user
      ON auth_audit_events(user_id, created_at DESC);
  `);
}

/** v6: local record for permanently retiring the legacy PostgreSQL runtime. */
function migrateV6(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS storage_retirement (
      retirement_id INTEGER PRIMARY KEY CHECK(retirement_id = 1),
      status TEXT NOT NULL CHECK(status IN ('prepared', 'retired')),
      manifest_version INTEGER NOT NULL CHECK(manifest_version >= 1),
      backup_sha256 TEXT NOT NULL CHECK(length(backup_sha256) = 64),
      manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
      prepared_at TEXT NOT NULL,
      retired_at TEXT,
      removed_outbox_events INTEGER NOT NULL DEFAULT 0 CHECK(removed_outbox_events >= 0),
      details TEXT NOT NULL DEFAULT '{}'
    );
  `);
}

/** v7: local authentication sessions and persistent login throttling state. */
function migrateV7(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS auth_sessions (
      session_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES local_users(user_id) ON DELETE CASCADE,
      token_hash TEXT NOT NULL UNIQUE CHECK(length(token_hash) = 64),
      created_at TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      last_seen_at TEXT NOT NULL,
      revoked_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
      ON auth_sessions(user_id, expires_at DESC);
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_active
      ON auth_sessions(token_hash) WHERE revoked_at IS NULL;

    CREATE TABLE IF NOT EXISTS auth_login_limits (
      user_id TEXT PRIMARY KEY REFERENCES local_users(user_id) ON DELETE CASCADE,
      failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
      first_failure_at TEXT,
      locked_until TEXT,
      updated_at TEXT NOT NULL
    );
  `);
}

export const MIGRATIONS: Migration[] = [
  { version: 1, up: migrateV1 },
  { version: 2, up: migrateV2 },
  { version: 3, up: migrateV3 },
  { version: 4, up: migrateV4 },
  { version: 5, up: migrateV5 },
  { version: 6, up: migrateV6 },
  { version: 7, up: migrateV7 },
];

export class SqliteStore {
  private db: DatabaseSync | null = null;
  readonly filePath: string;

  constructor(filePath?: string) {
    this.filePath = filePath ?? resolveSqlitePath();
  }

  get isOpen(): boolean {
    return this.db !== null;
  }

  /** 打开 + WAL + 迁移（幂等；重复 open 安全）。 */
  open(): void {
    if (this.db) return;
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    const db = new DatabaseSync(this.filePath);
    try {
      db.exec('PRAGMA journal_mode=WAL;');
      db.exec('PRAGMA foreign_keys=ON;');
      db.exec('PRAGMA busy_timeout=5000;');
      this.db = db;
      this.runMigrations();
    } catch (err) {
      db.close();
      this.db = null;
      throw err;
    }
  }

  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  /** 事务包裹执行迁移；user_version 记录已应用版本（幂等可重复执行）。 */
  runMigrations(): void {
    const db = this.requireDb();
    const row = db.prepare('PRAGMA user_version').get() as { user_version: number };
    let current = Number(row.user_version ?? 0);
    for (const migration of MIGRATIONS) {
      if (migration.version <= current) continue;
      db.exec('BEGIN');
      try {
        migration.up(db);
        db.exec(`PRAGMA user_version = ${migration.version}`);
        db.exec('COMMIT');
        current = migration.version;
      } catch (err) {
        db.exec('ROLLBACK');
        throw err;
      }
    }
  }

  get schemaVersion(): number {
    const db = this.requireDb();
    const row = db.prepare('PRAGMA user_version').get() as { user_version: number };
    return Number(row.user_version ?? 0);
  }

  /** 本地健康：SELECT 1 + 可写性探测（只读故障要明确报错，禁止假成功）。 */
  health(): LocalStorageHealth {
    const db = this.requireDb();
    let writable = false;
    try {
      db.prepare('SELECT 1').get();
      db.exec('BEGIN');
      try {
        db.exec('INSERT INTO cluster_settings(key, value, updated_at) '
          + "VALUES('__health_probe__', '1', 'epoch') "
          + "ON CONFLICT(key) DO UPDATE SET value='1';");
        db.exec('ROLLBACK');
        writable = true;
      } catch (probeErr) {
        try {
          db.exec('ROLLBACK');
        } catch {
          // 连接层失败时回滚也失败，交由上层报只读故障
        }
      }
    } catch (err) {
      writable = false;
    }
    return {
      status: writable ? 'ok' : 'unavailable',
      backend: 'sqlite',
      writable,
      path: this.filePath,
      schema_version: this.schemaVersion,
    };
  }

  /** 单事务执行：业务行 + outbox 事件同事务（BEGIN/COMMIT/ROLLBACK）。 */
  transaction<T>(fn: () => T): T {
    const db = this.requireDb();
    db.exec('BEGIN');
    try {
      const result = fn();
      db.exec('COMMIT');
      return result;
    } catch (err) {
      try {
        db.exec('ROLLBACK');
      } catch {
        // 回滚失败时保持原异常
      }
      throw err;
    }
  }

  /** SQLite backup API（在线备份到目标文件；模块级 backup(source, path)）。 */
  async backupTo(targetPath: string): Promise<void> {
    const db = this.requireDb();
    fs.mkdirSync(path.dirname(targetPath), { recursive: true });
    // @types/node 22.10 未声明模块级 backup；Node ≥22.5 runtime 提供
    await (sqliteBackup as unknown as (source: DatabaseSync, dest: string) => Promise<void>)(db, targetPath);
  }

  /**
   * 用户主动导出的加密备份。明文 SQLite 只存在于临时文件，
   * 目标包使用 scrypt + AES-256-GCM，不能被开发组或远端服务解密。
   */
  async exportEncryptedBackup(
    targetPath: string,
    passphrase: string,
    blobExists?: (digest: string) => boolean,
  ): Promise<EncryptedBackupInfo> {
    this.requirePassphrase(passphrase);
    const absoluteTarget = path.resolve(targetPath);
    fs.mkdirSync(path.dirname(absoluteTarget), { recursive: true });
    const tempPlain = `${absoluteTarget}.${randomBytes(8).toString('hex')}.plain.sqlite3`;
    try {
      await this.backupTo(tempPlain);
      const plaintext = fs.readFileSync(tempPlain);
      const encrypted = this.encryptBackup(plaintext, passphrase);
      this.writeAtomic(absoluteTarget, encrypted.envelope);
      const validation = this.validateBackupFile(tempPlain, blobExists);
      return {
        version: encrypted.header.version,
        schema_version: validation.schemaVersion,
        plaintext_bytes: plaintext.length,
        ciphertext_bytes: encrypted.ciphertext.length,
        created_at: encrypted.header.created_at,
        reference_integrity: validation.referenceIntegrity,
      };
    } finally {
      this.removeSqliteFiles(tempPlain);
    }
  }

  /** 解密并执行 quick_check；不修改当前主节点。 */
  verifyEncryptedBackup(
    backupPath: string,
    passphrase: string,
    blobExists?: (digest: string) => boolean,
  ): EncryptedBackupInfo {
    this.requirePassphrase(passphrase);
    const absoluteBackup = path.resolve(backupPath);
    const decoded = this.decryptBackup(fs.readFileSync(absoluteBackup), passphrase);
    const tempPlain = `${absoluteBackup}.${randomBytes(8).toString('hex')}.verify.sqlite3`;
    try {
      fs.writeFileSync(tempPlain, decoded.plaintext, { mode: 0o600 });
      const validation = this.validateBackupFile(tempPlain, blobExists);
      return {
        version: decoded.header.version,
        schema_version: validation.schemaVersion,
        plaintext_bytes: decoded.plaintext.length,
        ciphertext_bytes: decoded.ciphertext.length,
        created_at: decoded.header.created_at,
        reference_integrity: validation.referenceIntegrity,
      };
    } finally {
      this.removeSqliteFiles(tempPlain);
    }
  }

  /**
   * 从用户自己的加密包恢复主节点。当前库先移动为可回滚的 `.pre-restore-*`
   * 文件，恢复失败会自动还原；成功后保留旧文件，便于用户自行清理。
   */
  restoreEncryptedBackup(
    backupPath: string,
    passphrase: string,
    blobExists?: (digest: string) => boolean,
  ): RestoreResult {
    this.requirePassphrase(passphrase);
    const absoluteBackup = path.resolve(backupPath);
    const decoded = this.decryptBackup(fs.readFileSync(absoluteBackup), passphrase);
    const tempPlain = `${this.filePath}.${randomBytes(8).toString('hex')}.restore.sqlite3`;
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    try {
      fs.writeFileSync(tempPlain, decoded.plaintext, { mode: 0o600 });
      const validation = this.validateBackupFile(tempPlain, blobExists);
      const hadCurrent = fs.existsSync(this.filePath);
      const previousPath = hadCurrent
        ? `${this.filePath}.pre-restore-${Date.now()}.sqlite3`
        : null;
      this.close();
      if (hadCurrent && previousPath) {
        this.moveSqliteFiles(this.filePath, previousPath);
      }
      fs.renameSync(tempPlain, this.filePath);
      try {
        this.open();
      } catch (err) {
        this.close();
        this.removeSqliteFiles(this.filePath);
        if (previousPath && fs.existsSync(previousPath)) {
          this.moveSqliteFiles(previousPath, this.filePath);
          this.open();
        }
        throw err;
      }
      return {
        version: decoded.header.version,
        schema_version: validation.schemaVersion,
        plaintext_bytes: decoded.plaintext.length,
        ciphertext_bytes: decoded.ciphertext.length,
        created_at: decoded.header.created_at,
        reference_integrity: validation.referenceIntegrity,
        restored_path: this.filePath,
        previous_path: previousPath,
      };
    } finally {
      this.removeSqliteFiles(tempPlain);
    }
  }

  exec(sql: string): void {
    this.requireDb().exec(sql);
  }

  prepare(sql: string) {
    return this.requireDb().prepare(sql);
  }

  private requireDb(): DatabaseSync {
    if (!this.db) {
      this.open();
    }
    return this.db as DatabaseSync;
  }

  private requirePassphrase(passphrase: string): void {
    if (typeof passphrase !== 'string' || passphrase.length < 12) {
      throw new Error('备份口令至少需要 12 个字符');
    }
  }

  private deriveBackupKey(passphrase: string, salt: Buffer): Buffer {
    return scryptSync(passphrase, salt, 32, {
      N: BACKUP_KDF_N,
      r: BACKUP_KDF_R,
      p: BACKUP_KDF_P,
      maxmem: 64 * 1024 * 1024,
    });
  }

  private encryptBackup(plaintext: Buffer, passphrase: string): {
    header: EncryptedBackupHeader;
    ciphertext: Buffer;
    envelope: Buffer;
  } {
    const salt = randomBytes(16);
    const iv = randomBytes(12);
    const cipher = createCipheriv('aes-256-gcm', this.deriveBackupKey(passphrase, salt), iv);
    const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
    const header: EncryptedBackupHeader = {
      version: BACKUP_VERSION,
      cipher: 'aes-256-gcm',
      kdf: 'scrypt',
      kdf_n: BACKUP_KDF_N,
      kdf_r: BACKUP_KDF_R,
      kdf_p: BACKUP_KDF_P,
      salt: salt.toString('base64'),
      iv: iv.toString('base64'),
      auth_tag: cipher.getAuthTag().toString('base64'),
      ciphertext_sha256: createHash('sha256').update(ciphertext).digest('hex'),
      created_at: new Date().toISOString(),
    };
    const headerBytes = Buffer.from(JSON.stringify(header), 'utf8');
    const length = Buffer.allocUnsafe(4);
    length.writeUInt32BE(headerBytes.length, 0);
    return {
      header,
      ciphertext,
      envelope: Buffer.concat([BACKUP_MAGIC, length, headerBytes, ciphertext]),
    };
  }

  private decryptBackup(envelope: Buffer, passphrase: string): {
    header: EncryptedBackupHeader;
    ciphertext: Buffer;
    plaintext: Buffer;
  } {
    if (envelope.length < BACKUP_MAGIC.length + 4
      || !envelope.subarray(0, BACKUP_MAGIC.length).equals(BACKUP_MAGIC)) {
      throw new Error('不是有效的 QLH SQLite 加密备份');
    }
    const headerLength = envelope.readUInt32BE(BACKUP_MAGIC.length);
    if (headerLength <= 0 || headerLength > MAX_BACKUP_HEADER_BYTES) {
      throw new Error('SQLite 加密备份头无效');
    }
    const headerStart = BACKUP_MAGIC.length + 4;
    const headerEnd = headerStart + headerLength;
    if (headerEnd >= envelope.length) throw new Error('SQLite 加密备份内容不完整');
    let header: EncryptedBackupHeader;
    try {
      header = JSON.parse(envelope.subarray(headerStart, headerEnd).toString('utf8')) as EncryptedBackupHeader;
    } catch {
      throw new Error('SQLite 加密备份头无法解析');
    }
    if (header.version !== BACKUP_VERSION || header.cipher !== 'aes-256-gcm' || header.kdf !== 'scrypt') {
      throw new Error('不支持的 SQLite 加密备份格式');
    }
    const ciphertext = envelope.subarray(headerEnd);
    const digest = createHash('sha256').update(ciphertext).digest('hex');
    if (digest !== header.ciphertext_sha256) throw new Error('SQLite 加密备份摘要校验失败');
    try {
      const decipher = createDecipheriv(
        'aes-256-gcm',
        this.deriveBackupKey(passphrase, Buffer.from(header.salt, 'base64')),
        Buffer.from(header.iv, 'base64'),
      );
      decipher.setAuthTag(Buffer.from(header.auth_tag, 'base64'));
      return {
        header,
        ciphertext,
        plaintext: Buffer.concat([decipher.update(ciphertext), decipher.final()]),
      };
    } catch {
      throw new Error('SQLite 加密备份口令错误或内容已损坏');
    }
  }

  private validateBackupFile(
    filePath: string,
    blobExists?: (digest: string) => boolean,
  ): {
    schemaVersion: number;
    referenceIntegrity: BackupReferenceIntegrity;
  } {
    const db = new DatabaseSync(filePath, { readOnly: true });
    try {
      const row = db.prepare('PRAGMA quick_check').get() as { quick_check: string };
      if (String(row.quick_check).toLowerCase() !== 'ok') {
        throw new Error('SQLite 备份 quick_check 未通过');
      }
      const version = Number((db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version ?? 0);
      if (version > MIGRATIONS[MIGRATIONS.length - 1].version) {
        throw new Error(`SQLite 备份 schema_version ${version} 高于当前支持版本`);
      }
      const referenceIntegrity = this.validateBackupReferences(db, blobExists);
      if (!referenceIntegrity.ok) {
        throw new Error(`SQLite 备份引用完整性检查失败: ${referenceIntegrity.errors.join('; ')}`);
      }
      return { schemaVersion: version, referenceIntegrity };
    } finally {
      db.close();
    }
  }

  private validateBackupReferences(
    db: DatabaseSync,
    blobExists?: (digest: string) => boolean,
  ): BackupReferenceIntegrity {
    const table = db.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_manifests'",
    ).get() as { name: string } | undefined;
    if (!table) {
      return {
        ok: true,
        manifest_index: false,
        checked_manifests: 0,
        checked_references: 0,
        checked_blobs: 0,
        external_blobs_checked: Boolean(blobExists),
        errors: [],
      };
    }
    const errors: string[] = [];
    const rows = db.prepare(
      `SELECT manifest_key, artifact_id, manifest_sha256, blob_digests, payload
       FROM artifact_manifests`,
    ).all() as Array<{
      manifest_key: string;
      artifact_id: string;
      manifest_sha256: string;
      blob_digests: string;
      payload: string;
    }>;
    const knownArtifacts = new Set<string>();
    let checkedBlobs = 0;
    for (const row of rows) {
      knownArtifacts.add(row.artifact_id);
      const payloadDigest = createHash('sha256').update(row.payload).digest('hex');
      if (payloadDigest !== row.manifest_sha256) {
        errors.push(`${row.manifest_key}: manifest payload digest mismatch`);
      }
      try {
        const payload = JSON.parse(row.payload) as Record<string, unknown>;
        if (String(payload.artifact_id ?? '') !== row.artifact_id) {
          errors.push(`${row.manifest_key}: indexed artifact_id differs from payload`);
        }
        const digests = JSON.parse(row.blob_digests) as unknown;
        if (!Array.isArray(digests)
          || digests.some((digest) => !/^[0-9a-f]{64}$/.test(String(digest)))) {
          errors.push(`${row.manifest_key}: blob digest index is invalid`);
        } else {
          checkedBlobs += digests.length;
          if (blobExists) {
            for (const digest of digests) {
              if (!blobExists(String(digest))) {
                errors.push(`${row.manifest_key}: external blob is missing: ${digest}`);
              }
            }
          }
        }
      } catch {
        errors.push(`${row.manifest_key}: manifest index JSON is invalid`);
      }
    }
    const references = db.prepare(
      `SELECT DISTINCT artifact_id FROM deployments
       UNION SELECT DISTINCT artifact_id FROM artifact_runtime_checks`,
    ).all() as Array<{ artifact_id: string }>;
    for (const reference of references) {
      if (reference.artifact_id && !knownArtifacts.has(reference.artifact_id)) {
        errors.push(`missing manifest for referenced artifact: ${reference.artifact_id}`);
      }
    }
    return {
      ok: errors.length === 0,
      manifest_index: true,
      checked_manifests: rows.length,
      checked_references: references.length,
      checked_blobs: checkedBlobs,
      external_blobs_checked: Boolean(blobExists),
      errors,
    };
  }

  private writeAtomic(targetPath: string, data: Buffer): void {
    const temp = `${targetPath}.${randomBytes(8).toString('hex')}.tmp`;
    try {
      fs.writeFileSync(temp, data, { mode: 0o600 });
      fs.rmSync(targetPath, { force: true });
      fs.renameSync(temp, targetPath);
      try {
        fs.chmodSync(targetPath, 0o600);
      } catch {
        // Windows ACLs are managed by the user profile; chmod is best effort.
      }
    } finally {
      fs.rmSync(temp, { force: true });
    }
  }

  private removeSqliteFiles(filePath: string): void {
    for (const suffix of ['', '-wal', '-shm']) {
      fs.rmSync(`${filePath}${suffix}`, { force: true });
    }
  }

  private moveSqliteFiles(source: string, target: string): void {
    for (const suffix of ['', '-wal', '-shm']) {
      const from = `${source}${suffix}`;
      const to = `${target}${suffix}`;
      if (fs.existsSync(from)) {
        fs.rmSync(to, { force: true });
        fs.renameSync(from, to);
      }
    }
  }
}
