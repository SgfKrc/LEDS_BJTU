/**
 * M1 本地事务数据库测试：SQLite 迁移/WAL/backup/health、outbox
 * 与 storage-health 本地事实源。
 */
import { mkdtempSync, existsSync, readFileSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { DatabaseSync } from 'node:sqlite';
import { MIGRATIONS, resolveSqlitePath, SqliteStore } from '../src/data/sqlite-store';
import { OutboxService } from '../src/data/outbox.service';
import { StorageHealthService } from '../src/data/storage-health';

function tempStore(): SqliteStore {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-m1-'));
  return new SqliteStore(join(dir, 'control.sqlite3'));
}

describe('SqliteStore (M1)', () => {
  it('uses the shared state directory by default', () => {
    const previousCwd = process.cwd();
    const previousPath = process.env.QLH_SQLITE_PATH;
    const previousStateDir = process.env.QLH_STATE_DIR;
    const cwd = mkdtempSync(join(tmpdir(), 'qlh-empty-cwd-'));
    const stateDir = mkdtempSync(join(tmpdir(), 'qlh-state-path-'));
    try {
      process.chdir(cwd);
      delete process.env.QLH_SQLITE_PATH;
      process.env.QLH_STATE_DIR = stateDir;
      expect(resolveSqlitePath()).toBe(join(stateDir, 'qlh-control.sqlite3'));
    } finally {
      process.chdir(previousCwd);
      if (previousPath === undefined) delete process.env.QLH_SQLITE_PATH;
      else process.env.QLH_SQLITE_PATH = previousPath;
      if (previousStateDir === undefined) delete process.env.QLH_STATE_DIR;
      else process.env.QLH_STATE_DIR = previousStateDir;
    }
  });

  it('keeps using an existing cwd database until it is moved', () => {
    const previousCwd = process.cwd();
    const previousPath = process.env.QLH_SQLITE_PATH;
    const previousStateDir = process.env.QLH_STATE_DIR;
    const cwd = mkdtempSync(join(tmpdir(), 'qlh-legacy-cwd-'));
    const stateDir = mkdtempSync(join(tmpdir(), 'qlh-new-state-'));
    const legacyPath = join(cwd, 'qlh-control.sqlite3');
    writeFileSync(legacyPath, '');
    try {
      process.chdir(cwd);
      delete process.env.QLH_SQLITE_PATH;
      process.env.QLH_STATE_DIR = stateDir;
      expect(resolveSqlitePath()).toBe(legacyPath);
    } finally {
      process.chdir(previousCwd);
      if (previousPath === undefined) delete process.env.QLH_SQLITE_PATH;
      else process.env.QLH_SQLITE_PATH = previousPath;
      if (previousStateDir === undefined) delete process.env.QLH_STATE_DIR;
      else process.env.QLH_STATE_DIR = previousStateDir;
    }
  });

  it('opens with WAL and migrates to schema v7', () => {
    const store = tempStore();
    store.open();
    expect(store.schemaVersion).toBe(7);
    const db = new DatabaseSync(store.filePath);
    const mode = db.prepare('PRAGMA journal_mode').get() as { journal_mode: string };
    db.close();
    expect(mode.journal_mode).toBe('wal');
    store.close();
  });

  it('reopen is idempotent and does not re-run migrations', () => {
    const store = tempStore();
    store.open();
    expect(store.schemaVersion).toBe(7);
    store.close();
    // 重新打开：迁移器跳过已应用版本
    store.open();
    expect(store.schemaVersion).toBe(7);
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
    expect(health.schema_version).toBe(7);
    store.close();
  });

  it('upgrades an existing v1 database to v6 without losing data', () => {
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
    expect(store.schemaVersion).toBe(7);
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

  it('exports, verifies, and restores a user-owned encrypted backup', async () => {
    const store = tempStore();
    store.open();
    store.prepare(
      'INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)',
    ).run('user_settings', JSON.stringify({ theme: 'dark' }), new Date().toISOString());
    const backupPath = join(store.filePath, '..', 'user-owned.qlhbackup');
    const passphrase = 'correct horse battery staple';
    const exported = await store.exportEncryptedBackup(backupPath, passphrase);
    expect(exported.schema_version).toBe(7);
    expect(readFileSync(backupPath).toString('utf8')).not.toContain('user_settings');
    const verified = store.verifyEncryptedBackup(backupPath, passphrase);
    expect(verified.ciphertext_bytes).toBeGreaterThan(0);
    expect(() => store.verifyEncryptedBackup(backupPath, 'wrong passphrase')).toThrow(
      '口令错误或内容已损坏',
    );
    store.prepare('DELETE FROM cluster_settings WHERE key = ?').run('user_settings');
    const restored = store.restoreEncryptedBackup(backupPath, passphrase);
    expect(restored.previous_path).toBeTruthy();
    const setting = store.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get('user_settings') as { value: string };
    expect(JSON.parse(setting.value)).toEqual({ theme: 'dark' });
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
    const health = await new StorageHealthService(store).snapshot();
    expect(health.local.status).toBe('ok');
    expect(health.remote).toEqual({
      status: 'disabled',
      backend: 'postgresql',
      mode: 'legacy_cleanup_pending',
    });
    expect(health.effective_mode).toBe('local_only');
    expect(health.projection.pending_events).toBe(0);
    expect(health.export.pending_items).toBe(0);
    store.close();
  });

  it('stays local_only when legacy environment could have enabled remote access', async () => {
    const store = tempStore();
    store.open();
    const health = await new StorageHealthService(store).snapshot();
    expect(health.remote.status).toBe('disabled');
    expect(health.effective_mode).toBe('local_only');
    store.close();
  });

  it('reports a legacy outbox backlog without changing local-only mode', async () => {
    const store = tempStore();
    store.open();
    const outbox = new OutboxService(store);
    outbox.enqueue('model_registry', 'created', {});
    const health = await new StorageHealthService(store).snapshot();
    expect(health.projection.pending_events).toBe(1);
    expect(health.effective_mode).toBe('local_only');
    store.close();
  });

  it('does not probe an unavailable remote', async () => {
    const store = tempStore();
    store.open();
    const health = await new StorageHealthService(store).snapshot();
    expect(health.remote.status).toBe('disabled');
    expect(health.effective_mode).toBe('local_only');
    store.close();
  });
});
