import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Test } from '@nestjs/testing';
import { AppModule } from '../src/app';
import { ArtifactStore } from '../src/data/artifact-store';
import { OutboxService } from '../src/data/outbox.service';
import {
  POSTGRES_RETIREMENT_FORMAT,
  POSTGRES_RETIREMENT_VERSION,
  PostgresRetirementService,
} from '../src/data/postgres-retirement';
import { SqliteStore } from '../src/data/sqlite-store';

describe('M1.3 legacy PostgreSQL retirement', () => {
  let tmpBase: string;
  let store: SqliteStore;
  let artifacts: ArtifactStore;
  let service: PostgresRetirementService;
  const passphrase = 'correct horse battery staple';

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'qlh-pg-retirement-'));
    store = new SqliteStore(join(tmpBase, 'control.sqlite3'));
    store.open();
    artifacts = new ArtifactStore(join(tmpBase, 'model-store'));
    service = new PostgresRetirementService(store, artifacts);
    store.prepare(
      'INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)',
    ).run('user_settings', JSON.stringify({ theme: 'dark' }), '2026-08-09T00:00:00Z');
    store.prepare(
      'INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)',
    ).run('postgresql_url', 'postgres://alice:db-secret@example.invalid/qlh', '2026-08-09T00:00:00Z');
    store.prepare(
      'INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)',
    ).run('legacy_postgres.password_ref', 'secret://postgres/main', '2026-08-09T00:00:00Z');
    new OutboxService(store).enqueue('model_registry', 'created', { model_id: 'm1' });
  });

  afterEach(() => {
    store.close();
    rmSync(tmpBase, { recursive: true, force: true });
  });

  function options() {
    return {
      backupPath: join(tmpBase, 'retirement.qlhbackup'),
      manifestPath: join(tmpBase, 'retirement.json'),
      envFile: join(tmpBase, '.env'),
      passphrase,
    };
  }

  it('backs up, restore-drills, removes compatibility state, and remains restorable', async () => {
    const opts = options();
    writeFileSync(
      opts.envFile,
      'QLH_CONTROL_PORT=8030\nQLH_DB_ENABLED=1\nQLH_DB_PASSWORD=env-secret\nDATABASE_URL=postgres://env-secret@example.invalid/qlh\n',
    );

    const prepared = await service.prepare(opts);
    expect(prepared.status).toBe('prepared');
    const manifestText = readFileSync(opts.manifestPath, 'utf8');
    const manifest = JSON.parse(manifestText);
    expect(manifest).toMatchObject({
      format: POSTGRES_RETIREMENT_FORMAT,
      version: POSTGRES_RETIREMENT_VERSION,
      sqlite_schema_version: 7,
      restore_drill: { passed: true, schema_version: 7 },
      legacy_postgresql: {
        remote_writeback: false,
        runtime_projector_required: false,
        outbox_events: 1,
        pending_outbox_events: 1,
      },
    });
    expect(manifest.legacy_postgresql.env_keys_detected).toEqual(
      expect.arrayContaining(['DATABASE_URL', 'QLH_DB_ENABLED', 'QLH_DB_PASSWORD']),
    );
    expect(manifestText).not.toContain('db-secret');
    expect(manifestText).not.toContain('env-secret');
    expect(service.getState()?.status).toBe('prepared');

    const retired = service.retire(opts);
    expect(retired).toMatchObject({
      status: 'retired',
      removed_outbox_events: 1,
      removed_sqlite_config_keys: ['legacy_postgres.password_ref', 'postgresql_url'],
      removed_env_keys: ['DATABASE_URL', 'QLH_DB_ENABLED', 'QLH_DB_PASSWORD'],
    });
    expect(store.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='outbox'",
    ).get()).toBeUndefined();
    expect(store.prepare(
      "SELECT value FROM cluster_settings WHERE key='postgresql_url'",
    ).get()).toBeUndefined();
    const local = store.prepare(
      "SELECT value FROM cluster_settings WHERE key='user_settings'",
    ).get() as { value: string };
    expect(JSON.parse(local.value)).toEqual({ theme: 'dark' });
    expect(readFileSync(opts.envFile, 'utf8')).toBe('QLH_CONTROL_PORT=8030\n');
    expect(existsSync(`${opts.envFile}.pre-postgres-retirement`)).toBe(true);
    expect(service.verify(opts).status).toBe('verified');

    const restored = new SqliteStore(join(tmpBase, 'restored.sqlite3'));
    try {
      restored.restoreEncryptedBackup(opts.backupPath, passphrase);
      const restoredSettings = restored.prepare(
        "SELECT value FROM cluster_settings WHERE key='user_settings'",
      ).get() as { value: string };
      expect(JSON.parse(restoredSettings.value)).toEqual({ theme: 'dark' });
      expect(restored.prepare('SELECT COUNT(*) AS c FROM outbox').get()).toEqual({ c: 1 });
    } finally {
      restored.close();
    }
  });

  it('does not clean anything when backup verification fails', async () => {
    const opts = options();
    writeFileSync(opts.envFile, 'QLH_DB_ENABLED=1\nKEEP_ME=yes\n');
    await service.prepare(opts);

    expect(() => service.retire({ ...opts, passphrase: 'wrong passphrase value' }))
      .toThrow('口令错误或内容已损坏');
    expect(service.getState()?.status).toBe('prepared');
    expect(store.prepare('SELECT COUNT(*) AS c FROM outbox').get()).toEqual({ c: 1 });
    expect(store.prepare(
      "SELECT value FROM cluster_settings WHERE key='postgresql_url'",
    ).get()).toBeTruthy();
    expect(readFileSync(opts.envFile, 'utf8')).toContain('QLH_DB_ENABLED=1');
  });

  it('rejects a manifest changed after retirement', async () => {
    const opts = options();
    writeFileSync(opts.envFile, 'KEEP_ME=yes\n');
    await service.prepare(opts);
    service.retire(opts);
    const manifest = JSON.parse(readFileSync(opts.manifestPath, 'utf8'));
    manifest.assets.cluster_settings = 999;
    writeFileSync(opts.manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    expect(() => service.verify(opts)).toThrow(
      'retired 状态与当前备份/manifest 不一致',
    );
  });

  it('keeps PostgreSQL services outside the production Nest graph', async () => {
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore)
      .useValue(store)
      .compile();
    expect(() => moduleRef.get(OutboxService, { strict: false })).toThrow();
    await moduleRef.close();

    const packageJson = JSON.parse(readFileSync(join(__dirname, '..', 'package.json'), 'utf8'));
    expect(packageJson.dependencies.pg).toBeUndefined();
    expect(packageJson.devDependencies.pg).toBeUndefined();
    expect(packageJson.devDependencies['@types/pg']).toBeUndefined();
  });
});
