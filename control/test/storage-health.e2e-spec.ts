/** M1.3 storage health contract: local SQLite plus retirement state. */
import { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { AppModule } from '../src/app';
import { OutboxService } from '../src/data/outbox.service';
import { SqliteStore } from '../src/data/sqlite-store';

describe('storage/health local-only contract', () => {
  let app: NestFastifyApplication | null = null;
  let store: SqliteStore;
  let tmpBase: string;

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'control-m1-'));
    store = new SqliteStore(join(tmpBase, 'qlh-control.sqlite3'));
    store.open();
  });

  afterEach(async () => {
    if (app) await app.close();
    store.close();
    rmSync(tmpBase, { recursive: true, force: true });
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore)
      .useValue(store)
      .compile();
    const fastify = new (require('@nestjs/platform-fastify').FastifyAdapter)();
    const testApp = moduleRef.createNestApplication(fastify) as NestFastifyApplication;
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  it('reports local-only storage without probing a remote database', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/storage/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      local: {
        status: 'ok', backend: 'sqlite', writable: true,
        path: store.filePath, schema_version: 7,
      },
      remote: {
        status: 'disabled', backend: 'postgresql', mode: 'legacy_cleanup_pending',
      },
      projection: { pending_events: 0, oldest_event_age_seconds: null },
      export: { pending_items: 0, oldest_item_age_seconds: null },
      effective_mode: 'local_only',
      retirement: { status: 'not_prepared', prepared_at: null, retired_at: null },
    });
  });

  it('shows legacy outbox rows for cleanup diagnostics only', async () => {
    new OutboxService(store).enqueue('model_registry', 'created', {});
    app = await createTestApp();
    const body = (await app.inject({ method: 'GET', url: '/storage/health' })).json();
    expect(body.projection.pending_events).toBe(1);
    expect(body.effective_mode).toBe('local_only');
    expect(body.remote.status).toBe('disabled');
  });

  it('reports retired after the one-time cleanup removed outbox', async () => {
    store.exec('DROP INDEX IF EXISTS idx_outbox_pending; DROP TABLE outbox;');
    store.prepare(
      `INSERT INTO storage_retirement
         (retirement_id, status, manifest_version, backup_sha256, manifest_sha256,
          prepared_at, retired_at, removed_outbox_events, details)
       VALUES (1, 'retired', 1, ?, ?, ?, ?, 0, '{}')`,
    ).run('a'.repeat(64), 'b'.repeat(64), '2026-08-09T00:00:00Z', '2026-08-09T00:01:00Z');
    app = await createTestApp();
    const body = (await app.inject({ method: 'GET', url: '/storage/health' })).json();
    expect(body.remote).toEqual({
      status: 'retired', backend: 'postgresql', mode: 'retired',
    });
    expect(body.retirement.status).toBe('retired');
    expect(body.projection.pending_events).toBe(0);
  });
});
