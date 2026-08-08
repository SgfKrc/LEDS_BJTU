/**
 * M1 本地事务数据库测试：SQLite 迁移/WAL/backup/health、outbox、
 * storage-health 双健康、postgres projector 退避与幂等。
 */
import { mkdtempSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { DatabaseSync } from 'node:sqlite';
import { MIGRATIONS, SqliteStore } from '../src/data/sqlite-store';
import { OutboxService } from '../src/data/outbox.service';
import { StorageHealthService } from '../src/data/storage-health';
import { PostgresProjector } from '../src/data/postgres-projector';

function tempStore(): SqliteStore {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-m1-'));
  return new SqliteStore(join(dir, 'control.sqlite3'));
}

class FakeConfigDao {
  enabled = false;
  host = '127.0.0.1';
  port = 1;
  db = 'qlh';
  pingOk = false;
  dbEnabled() { return this.enabled; }
  getConnectionInfo() { return { host: this.host, port: this.port, db: this.db }; }
  async ping() {
    return this.pingOk ? { ok: true } : { ok: false, error: 'fake down' };
  }
}

describe('SqliteStore (M1)', () => {
  it('opens with WAL and migrates to schema v2', () => {
    const store = tempStore();
    store.open();
    expect(store.schemaVersion).toBe(2);
    const db = new DatabaseSync(store.filePath);
    const mode = db.prepare('PRAGMA journal_mode').get() as { journal_mode: string };
    db.close();
    expect(mode.journal_mode).toBe('wal');
    store.close();
  });

  it('reopen is idempotent and does not re-run migrations', () => {
    const store = tempStore();
    store.open();
    expect(store.schemaVersion).toBe(2);
    store.close();
    // 重新打开：迁移器跳过已应用版本
    store.open();
    expect(store.schemaVersion).toBe(2);
    store.close();
    store.open();
    store.close();
  });

  it('reports health ok and writable', () => {
    const store = tempStore();
    store.open();
    const health = store.health();
    expect(health.status).toBe('ok');
    expect(health.backend).toBe('sqlite');
    expect(health.writable).toBe(true);
    expect(health.schema_version).toBe(2);
    store.close();
  });

  it('upgrades an existing v1 database to v2 without losing data', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-m1-upgrade-'));
    const dbPath = join(dir, 'control.sqlite3');
    const legacy = new DatabaseSync(dbPath);
    MIGRATIONS[0].up(legacy);
    legacy.exec('PRAGMA user_version = 1');
    legacy.prepare(
      'INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)',
    ).run('preserved', 'yes', '2026-08-08T00:00:00Z');
    legacy.close();

    const store = new SqliteStore(dbPath);
    store.open();
    expect(store.schemaVersion).toBe(2);
    const preserved = store.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get('preserved') as { value: string };
    expect(preserved.value).toBe('yes');
    const table = store.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_runtime_checks'",
    ).get() as { name: string };
    expect(table.name).toBe('artifact_runtime_checks');
    store.close();
  });

  it('creates a valid online backup file', async () => {
    const store = tempStore();
    store.open();
    const backupPath = join(store.filePath, '..', 'backup.sqlite3');
    await store.backupTo(backupPath);
    expect(existsSync(backupPath)).toBe(true);
    const backup = new DatabaseSync(backupPath);
    const row = backup.prepare('SELECT COUNT(*) AS c FROM outbox').get() as { c: number };
    backup.close();
    expect(row.c).toBe(0);
    store.close();
  });

  it('health probe does not leave rows behind', () => {
    const store = tempStore();
    store.open();
    store.health();
    const db = new DatabaseSync(store.filePath);
    const row = db.prepare(
      "SELECT COUNT(*) AS c FROM cluster_settings WHERE key='__health_probe__'",
    ).get() as { c: number };
    db.close();
    expect(row.c).toBe(0);
    store.close();
  });
});

describe('OutboxService (M1)', () => {
  it('assigns monotonic aggregate versions and tracks pending', () => {
    const store = tempStore();
    store.open();
    const outbox = new OutboxService(store);
    const e1 = outbox.enqueue('model_registry', 'created', { model_id: 'm1' });
    const e2 = outbox.enqueue('model_registry', 'created', { model_id: 'm2' });
    expect(e2.aggregate_version).toBe(e1.aggregate_version + 1);
    expect(outbox.pendingCount()).toBe(2);
    expect(outbox.pending().map((e) => e.event_id)).toEqual([e1.event_id, e2.event_id]);

    outbox.markProjected(e1.event_id);
    expect(outbox.pendingCount()).toBe(1);
    expect(outbox.pending()[0].event_id).toBe(e2.event_id);
    store.close();
  });

  it('explicit aggregate version is respected', () => {
    const store = tempStore();
    store.open();
    const outbox = new OutboxService(store);
    const e = outbox.enqueue('cluster_settings', 'updated', { key: 'k' }, 7);
    expect(e.aggregate_version).toBe(7);
    expect(outbox.nextVersion('cluster_settings')).toBe(8);
    store.close();
  });

  it('oldest pending age is reported in seconds', () => {
    const store = tempStore();
    store.open();
    const outbox = new OutboxService(store);
    outbox.enqueue('pull_jobs', 'started', {});
    const age = outbox.oldestPendingAgeSeconds();
    expect(age).not.toBeNull();
    expect(Number(age)).toBeGreaterThanOrEqual(0);
    store.close();
  });
});

describe('StorageHealthService (M1)', () => {
  it('local_only when remote not configured', async () => {
    const store = tempStore();
    store.open();
    const config = new FakeConfigDao();
    config.enabled = false;
    const health = await new StorageHealthService(store, config as any, new OutboxService(store)).snapshot();
    expect(health.local.status).toBe('ok');
    expect(health.remote.status).toBe('not_configured');
    expect(health.effective_mode).toBe('local_only');
    expect(health.projection.pending_events).toBe(0);
    store.close();
  });

  it('local_primary when remote ok and no backlog', async () => {
    const store = tempStore();
    store.open();
    const config = new FakeConfigDao();
    config.enabled = true;
    config.pingOk = true;
    const health = await new StorageHealthService(store, config as any, new OutboxService(store)).snapshot();
    expect(health.remote.status).toBe('ok');
    expect(health.effective_mode).toBe('local_primary');
    store.close();
  });

  it('local_primary_pending when outbox has backlog', async () => {
    const store = tempStore();
    store.open();
    const outbox = new OutboxService(store);
    outbox.enqueue('model_registry', 'created', {});
    const config = new FakeConfigDao();
    config.enabled = true;
    config.pingOk = true;
    const health = await new StorageHealthService(store, config as any, outbox).snapshot();
    expect(health.projection.pending_events).toBe(1);
    expect(health.effective_mode).toBe('local_primary_pending');
    store.close();
  });

  it('unavailable remote keeps local primary semantics', async () => {
    const store = tempStore();
    store.open();
    const config = new FakeConfigDao();
    config.enabled = true;
    config.pingOk = false;
    const health = await new StorageHealthService(store, config as any, new OutboxService(store)).snapshot();
    expect(health.remote.status).toBe('unavailable');
    expect(health.effective_mode).toBe('local_only');
    store.close();
  });
});

describe('PostgresProjector (M1)', () => {
  it('skips cleanly when postgres not configured', async () => {
    const store = tempStore();
    store.open();
    const outbox = new OutboxService(store);
    outbox.enqueue('model_registry', 'created', {});
    const config = new FakeConfigDao();
    config.enabled = false;
    const projector = new PostgresProjector(config as any, outbox, {
      baseIntervalMs: 1000,
      maxIntervalMs: 4000,
    });
    const result = await projector.runOnce();
    expect(result.error).toBe('not_configured');
    expect(result.skipped).toBe(1);
    expect(outbox.pendingCount()).toBe(1); // 未投影，保留积压
    store.close();
  });

  it('backs off exponentially on connection failure', async () => {
    const store = tempStore();
    store.open();
    const config = new FakeConfigDao();
    config.enabled = true;
    config.host = '127.0.0.1';
    config.port = 1; // 必失败端口
    const projector = new PostgresProjector(config as any, new OutboxService(store), {
      baseIntervalMs: 1000,
      maxIntervalMs: 4000,
    });
    const first = await projector.runOnce();
    expect(first.error).toBeTruthy();
    expect(projector.intervalMs).toBe(2000);
    const second = await projector.runOnce();
    expect(second.error).toBeTruthy();
    expect(projector.intervalMs).toBe(4000);
    const third = await projector.runOnce();
    expect(projector.intervalMs).toBe(4000); // 上限封顶
    store.close();
  });
});
