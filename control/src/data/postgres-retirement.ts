/**
 * M1.3 one-time retirement flow for the legacy PostgreSQL compatibility path.
 *
 * The flow never reads from PostgreSQL and never needs a remote password:
 * encrypted local backup -> verified restore drill -> versioned manifest ->
 * local compatibility cleanup. SQLite remains the only fact source.
 */
import { createHash, randomBytes } from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { ArtifactStore } from './artifact-store';
import { EncryptedBackupInfo, SqliteStore } from './sqlite-store';

export const POSTGRES_RETIREMENT_FORMAT = 'qlh.postgres-retirement';
export const POSTGRES_RETIREMENT_VERSION = 1;

export const LEGACY_POSTGRES_ENV_KEYS = [
  'DATABASE_URL',
  'QLH_DB_ENABLED',
  'QLH_DB_HOST',
  'QLH_DB_PORT',
  'QLH_DB_NAME',
  'QLH_DB_USER',
  'QLH_DB_PASSWORD',
  'QLH_DB_SSLMODE',
  'QLH_DB_MIN_CONN',
  'QLH_DB_MAX_CONN',
  'QLH_DB_CONNECT_TIMEOUT',
  'QLH_DB_AUTO_CREATE',
  'QLH_DB_RETRY_SECONDS',
] as const;

const LEGACY_SQLITE_CONFIG_KEYS = new Set([
  'database_url',
  'postgresql_url',
  'legacy_postgres.enabled',
  'legacy_postgres.host',
  'legacy_postgres.port',
  'legacy_postgres.name',
  'legacy_postgres.user',
  'legacy_postgres.password_ref',
  'legacy_postgres.sslmode',
]);

const INVENTORY_TABLES = [
  'cluster_settings',
  'model_registry',
  'cluster_endpoints',
  'cluster_profiles',
  'catalog_models',
  'pull_jobs',
  'artifacts',
  'artifact_manifests',
  'artifact_runtime_checks',
  'deployments',
  'sessions',
  'session_messages',
  'workflow_journal',
  'review_tickets',
  'local_users',
  'user_authenticators',
  'auth_recovery_codes',
  'tailscale_bindings',
  'auth_audit_events',
  'auth_sessions',
  'auth_login_limits',
] as const;

export interface PostgresRetirementManifest {
  format: typeof POSTGRES_RETIREMENT_FORMAT;
  version: typeof POSTGRES_RETIREMENT_VERSION;
  created_at: string;
  sqlite_schema_version: number;
  backup: {
    filename: string;
    sha256: string;
    bytes: number;
    format_version: number;
    created_at: string;
    reference_integrity: EncryptedBackupInfo['reference_integrity'];
  };
  restore_drill: {
    passed: true;
    schema_version: number;
  };
  assets: Record<string, number>;
  legacy_postgresql: {
    direction: 'local_to_user_export_only';
    remote_writeback: false;
    runtime_projector_required: false;
    env_keys_detected: string[];
    sqlite_config_keys_detected: string[];
    outbox_events: number;
    pending_outbox_events: number;
  };
}

export interface RetirementState {
  status: 'prepared' | 'retired';
  manifest_version: number;
  backup_sha256: string;
  manifest_sha256: string;
  prepared_at: string;
  retired_at: string | null;
  removed_outbox_events: number;
  details: Record<string, unknown>;
}

export interface PrepareRetirementOptions {
  backupPath: string;
  manifestPath: string;
  passphrase: string;
  envFile?: string;
}

export interface RetirePostgresOptions extends PrepareRetirementOptions {}

export interface RetirementResult {
  status: 'prepared' | 'retired' | 'verified';
  backup_path: string;
  manifest_path: string;
  backup_sha256: string;
  manifest_sha256: string;
  schema_version: number;
  removed_outbox_events: number;
  removed_sqlite_config_keys: string[];
  removed_env_keys: string[];
  env_backup_path: string | null;
  warnings: string[];
}

export class PostgresRetirementService {
  constructor(
    private readonly store: SqliteStore,
    private readonly artifacts: ArtifactStore,
  ) {}

  async prepare(options: PrepareRetirementOptions): Promise<RetirementResult> {
    this.store.open();
    const existing = this.getState();
    if (existing?.status === 'retired') {
      throw new Error('legacy PostgreSQL runtime 已退场，不能重新进入 prepare');
    }
    const backupPath = path.resolve(options.backupPath);
    const manifestPath = path.resolve(options.manifestPath);
    const blobExists = (digest: string): boolean => this.artifacts.blobExists(digest);
    const backupInfo = await this.store.exportEncryptedBackup(
      backupPath,
      options.passphrase,
      blobExists,
    );
    this.store.verifyEncryptedBackup(backupPath, options.passphrase, blobExists);
    const restoreSchema = this.restoreDrill(backupPath, options.passphrase);
    const backupSha = sha256File(backupPath);
    const outbox = this.outboxStats();
    const manifest: PostgresRetirementManifest = {
      format: POSTGRES_RETIREMENT_FORMAT,
      version: POSTGRES_RETIREMENT_VERSION,
      created_at: new Date().toISOString(),
      sqlite_schema_version: this.store.schemaVersion,
      backup: {
        filename: path.basename(backupPath),
        sha256: backupSha,
        bytes: fs.statSync(backupPath).size,
        format_version: backupInfo.version,
        created_at: backupInfo.created_at,
        reference_integrity: backupInfo.reference_integrity,
      },
      restore_drill: {
        passed: true,
        schema_version: restoreSchema,
      },
      assets: this.assetInventory(),
      legacy_postgresql: {
        direction: 'local_to_user_export_only',
        remote_writeback: false,
        runtime_projector_required: false,
        env_keys_detected: this.detectLegacyEnvKeys(options.envFile),
        sqlite_config_keys_detected: this.legacySqliteConfigKeys(),
        outbox_events: outbox.total,
        pending_outbox_events: outbox.pending,
      },
    };
    writeJsonAtomic(manifestPath, manifest);
    const manifestSha = sha256File(manifestPath);
    this.upsertState({
      status: 'prepared',
      manifest_version: POSTGRES_RETIREMENT_VERSION,
      backup_sha256: backupSha,
      manifest_sha256: manifestSha,
      prepared_at: manifest.created_at,
      retired_at: null,
      removed_outbox_events: 0,
      details: {
        backup_path: backupPath,
        manifest_path: manifestPath,
        restore_drill: true,
      },
    });
    return {
      status: 'prepared',
      backup_path: backupPath,
      manifest_path: manifestPath,
      backup_sha256: backupSha,
      manifest_sha256: manifestSha,
      schema_version: this.store.schemaVersion,
      removed_outbox_events: 0,
      removed_sqlite_config_keys: [],
      removed_env_keys: [],
      env_backup_path: null,
      warnings: [],
    };
  }

  retire(options: RetirePostgresOptions): RetirementResult {
    this.store.open();
    const verified = this.verifyPreparedArtifacts(options);
    const state = this.getState();
    if (!state || state.status !== 'prepared') {
      throw new Error('必须先执行 prepare 并完成备份恢复演练');
    }
    if (state.backup_sha256 !== verified.backupSha
      || state.manifest_sha256 !== verified.manifestSha) {
      throw new Error('prepare 状态与当前备份/manifest 不一致');
    }

    const removedConfigKeys = this.legacySqliteConfigKeys();
    let removedOutbox = 0;
    const retiredAt = new Date().toISOString();
    this.store.transaction(() => {
      if (this.tableExists('outbox')) {
        removedOutbox = this.outboxStats().total;
        this.store.exec('DROP INDEX IF EXISTS idx_outbox_pending');
        this.store.exec('DROP TABLE outbox');
      }
      for (const key of removedConfigKeys) {
        this.store.prepare('DELETE FROM cluster_settings WHERE key = ?').run(key);
      }
      this.upsertState({
        status: 'retired',
        manifest_version: POSTGRES_RETIREMENT_VERSION,
        backup_sha256: verified.backupSha,
        manifest_sha256: verified.manifestSha,
        prepared_at: state.prepared_at,
        retired_at: retiredAt,
        removed_outbox_events: removedOutbox,
        details: {
          backup_path: path.resolve(options.backupPath),
          manifest_path: path.resolve(options.manifestPath),
          runtime_projector_removed: true,
        },
      });
    });

    const envCleanup = this.sanitizeEnvFile(options.envFile);
    const warnings = [...envCleanup.warnings];
    this.updateStateDetails({
      env_file: options.envFile ? path.resolve(options.envFile) : null,
      env_cleanup_complete: warnings.length === 0,
      removed_env_keys: envCleanup.removedKeys,
      env_backup_path: envCleanup.backupPath,
    });

    return {
      status: 'retired',
      backup_path: path.resolve(options.backupPath),
      manifest_path: path.resolve(options.manifestPath),
      backup_sha256: verified.backupSha,
      manifest_sha256: verified.manifestSha,
      schema_version: this.store.schemaVersion,
      removed_outbox_events: removedOutbox,
      removed_sqlite_config_keys: removedConfigKeys,
      removed_env_keys: envCleanup.removedKeys,
      env_backup_path: envCleanup.backupPath,
      warnings,
    };
  }

  verify(options: RetirePostgresOptions): RetirementResult {
    this.store.open();
    const verified = this.verifyPreparedArtifacts(options);
    const state = this.getState();
    if (!state || state.status !== 'retired') {
      throw new Error('legacy PostgreSQL runtime 尚未标记为 retired');
    }
    if (state.backup_sha256 !== verified.backupSha
      || state.manifest_sha256 !== verified.manifestSha) {
      throw new Error('retired 状态与当前备份/manifest 不一致');
    }
    if (this.tableExists('outbox')) {
      throw new Error('legacy outbox 表仍存在');
    }
    const remainingConfig = this.legacySqliteConfigKeys();
    if (remainingConfig.length > 0) {
      throw new Error(`legacy PostgreSQL SQLite 配置仍存在: ${remainingConfig.join(', ')}`);
    }
    const envKeys = this.detectLegacyEnvKeys(options.envFile, false);
    if (options.envFile && envKeys.length > 0) {
      throw new Error(`legacy PostgreSQL 环境配置仍存在: ${envKeys.join(', ')}`);
    }
    return {
      status: 'verified',
      backup_path: path.resolve(options.backupPath),
      manifest_path: path.resolve(options.manifestPath),
      backup_sha256: verified.backupSha,
      manifest_sha256: verified.manifestSha,
      schema_version: this.store.schemaVersion,
      removed_outbox_events: state.removed_outbox_events,
      removed_sqlite_config_keys: [],
      removed_env_keys: [],
      env_backup_path: typeof state.details.env_backup_path === 'string'
        ? state.details.env_backup_path : null,
      warnings: [],
    };
  }

  getState(): RetirementState | null {
    if (!this.tableExists('storage_retirement')) return null;
    const row = this.store.prepare(
      `SELECT status, manifest_version, backup_sha256, manifest_sha256,
              prepared_at, retired_at, removed_outbox_events, details
       FROM storage_retirement WHERE retirement_id = 1`,
    ).get() as Omit<RetirementState, 'details'> & { details: string } | undefined;
    if (!row) return null;
    let details: Record<string, unknown> = {};
    try {
      const parsed = JSON.parse(row.details) as unknown;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        details = parsed as Record<string, unknown>;
      }
    } catch {
      details = {};
    }
    return { ...row, details };
  }

  private verifyPreparedArtifacts(options: RetirePostgresOptions): {
    manifest: PostgresRetirementManifest;
    backupSha: string;
    manifestSha: string;
  } {
    const backupPath = path.resolve(options.backupPath);
    const manifestPath = path.resolve(options.manifestPath);
    const manifest = readManifest(manifestPath);
    const backupSha = sha256File(backupPath);
    const manifestSha = sha256File(manifestPath);
    if (backupSha !== manifest.backup.sha256) {
      throw new Error('退场备份摘要与 manifest 不一致');
    }
    const blobExists = (digest: string): boolean => this.artifacts.blobExists(digest);
    this.store.verifyEncryptedBackup(backupPath, options.passphrase, blobExists);
    this.restoreDrill(backupPath, options.passphrase);
    return { manifest, backupSha, manifestSha };
  }

  private restoreDrill(backupPath: string, passphrase: string): number {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'qlh-pg-retire-'));
    const drillStore = new SqliteStore(path.join(tempDir, 'restore.sqlite3'));
    try {
      const result = drillStore.restoreEncryptedBackup(
        backupPath,
        passphrase,
        (digest) => this.artifacts.blobExists(digest),
      );
      const health = drillStore.health();
      if (!health.writable) throw new Error('退场备份恢复演练不可写');
      return result.schema_version;
    } finally {
      drillStore.close();
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  }

  private upsertState(state: RetirementState): void {
    this.store.prepare(
      `INSERT INTO storage_retirement
         (retirement_id, status, manifest_version, backup_sha256,
          manifest_sha256, prepared_at, retired_at, removed_outbox_events, details)
       VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(retirement_id) DO UPDATE SET
         status = excluded.status,
         manifest_version = excluded.manifest_version,
         backup_sha256 = excluded.backup_sha256,
         manifest_sha256 = excluded.manifest_sha256,
         prepared_at = excluded.prepared_at,
         retired_at = excluded.retired_at,
         removed_outbox_events = excluded.removed_outbox_events,
         details = excluded.details`,
    ).run(
      state.status,
      state.manifest_version,
      state.backup_sha256,
      state.manifest_sha256,
      state.prepared_at,
      state.retired_at,
      state.removed_outbox_events,
      JSON.stringify(state.details),
    );
  }

  private updateStateDetails(extra: Record<string, unknown>): void {
    const state = this.getState();
    if (!state) return;
    this.upsertState({ ...state, details: { ...state.details, ...extra } });
  }

  private tableExists(name: string): boolean {
    const row = this.store.prepare(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
    ).get(name) as { name: string } | undefined;
    return Boolean(row);
  }

  private outboxStats(): { total: number; pending: number } {
    if (!this.tableExists('outbox')) return { total: 0, pending: 0 };
    const row = this.store.prepare(
      `SELECT COUNT(*) AS total,
              SUM(CASE WHEN projected_at IS NULL THEN 1 ELSE 0 END) AS pending
       FROM outbox`,
    ).get() as { total: number; pending: number | null };
    return { total: Number(row.total), pending: Number(row.pending ?? 0) };
  }

  private assetInventory(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const table of INVENTORY_TABLES) {
      if (!this.tableExists(table)) {
        result[table] = 0;
        continue;
      }
      const row = this.store.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get() as {
        count: number;
      };
      result[table] = Number(row.count);
    }
    return result;
  }

  private legacySqliteConfigKeys(): string[] {
    if (!this.tableExists('cluster_settings')) return [];
    const rows = this.store.prepare('SELECT key FROM cluster_settings').all() as Array<{
      key: string;
    }>;
    return rows.map((row) => row.key).filter((key) => (
      LEGACY_SQLITE_CONFIG_KEYS.has(key)
      || key.startsWith('legacy_postgres.')
      || key.startsWith('postgresql.')
    )).sort();
  }

  private detectLegacyEnvKeys(envFile?: string, includeProcess = true): string[] {
    const found = new Set<string>();
    if (includeProcess) {
      for (const key of LEGACY_POSTGRES_ENV_KEYS) {
        if (process.env[key] !== undefined) found.add(key);
      }
    }
    if (envFile && fs.existsSync(envFile)) {
      for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
        const match = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/);
        if (match && (LEGACY_POSTGRES_ENV_KEYS as readonly string[]).includes(match[1])) {
          found.add(match[1]);
        }
      }
    }
    return [...found].sort();
  }

  private sanitizeEnvFile(envFile?: string): {
    removedKeys: string[];
    backupPath: string | null;
    warnings: string[];
  } {
    if (!envFile) return { removedKeys: [], backupPath: null, warnings: [] };
    const absolute = path.resolve(envFile);
    if (!fs.existsSync(absolute)) {
      return {
        removedKeys: [],
        backupPath: null,
        warnings: [`env file 不存在，跳过清理: ${absolute}`],
      };
    }
    try {
      const original = fs.readFileSync(absolute, 'utf8');
      const removed = new Set<string>();
      const kept = original.split(/\r?\n/).filter((line) => {
        const match = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/);
        if (!match || !(LEGACY_POSTGRES_ENV_KEYS as readonly string[]).includes(match[1])) {
          return true;
        }
        removed.add(match[1]);
        return false;
      });
      if (removed.size === 0) {
        return { removedKeys: [], backupPath: null, warnings: [] };
      }
      const backupPath = `${absolute}.pre-postgres-retirement`;
      fs.copyFileSync(absolute, backupPath);
      const newline = original.includes('\r\n') ? '\r\n' : '\n';
      writeTextAtomic(absolute, kept.join(newline));
      return { removedKeys: [...removed].sort(), backupPath, warnings: [] };
    } catch (err) {
      return {
        removedKeys: [],
        backupPath: null,
        warnings: [
          `env file 清理失败，但 local_only 运行不受影响: ${err instanceof Error ? err.message : String(err)}`,
        ],
      };
    }
  }
}

function readManifest(manifestPath: string): PostgresRetirementManifest {
  const parsed = JSON.parse(fs.readFileSync(path.resolve(manifestPath), 'utf8')) as
    Partial<PostgresRetirementManifest>;
  if (parsed.format !== POSTGRES_RETIREMENT_FORMAT
    || parsed.version !== POSTGRES_RETIREMENT_VERSION
    || !parsed.backup
    || typeof parsed.backup.sha256 !== 'string') {
    throw new Error('不支持的 PostgreSQL 退场 manifest');
  }
  return parsed as PostgresRetirementManifest;
}

function sha256File(filePath: string): string {
  return createHash('sha256').update(fs.readFileSync(path.resolve(filePath))).digest('hex');
}

function writeJsonAtomic(targetPath: string, value: unknown): void {
  writeTextAtomic(path.resolve(targetPath), `${JSON.stringify(value, null, 2)}\n`);
}

function writeTextAtomic(targetPath: string, value: string): void {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  const tempPath = `${targetPath}.${randomBytes(8).toString('hex')}.tmp`;
  try {
    fs.writeFileSync(tempPath, value, { encoding: 'utf8', mode: 0o600 });
    fs.rmSync(targetPath, { force: true });
    fs.renameSync(tempPath, targetPath);
  } finally {
    fs.rmSync(tempPath, { force: true });
  }
}
