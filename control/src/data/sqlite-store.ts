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
import * as fs from 'fs';
import * as path from 'path';

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
  return path.resolve(process.cwd(), 'qlh-control.sqlite3');
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

export const MIGRATIONS: Migration[] = [
  { version: 1, up: migrateV1 },
  { version: 2, up: migrateV2 },
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
      status: 'ok',
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
}
