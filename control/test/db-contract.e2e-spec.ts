/** M1.3 compatibility contract for GET /db/health. */
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { AppModule } from '../src/app';
import { SqliteStore } from '../src/data/sqlite-store';

describe('control-svc db/health local SQLite contract', () => {
  let app: NestFastifyApplication | null = null;
  let store: SqliteStore;
  let tmpBase: string;

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'control-db-'));
    store = new SqliteStore(join(tmpBase, 'control.sqlite3'));
    store.open();
  });

  afterEach(async () => {
    if (app) await app.close();
    store.close();
    rmSync(tmpBase, { recursive: true, force: true });
    delete process.env.QLH_DB_ENABLED;
    delete process.env.QLH_DB_HOST;
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

  it('reports the user-owned SQLite database as healthy', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/db/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      status: 'ok',
      backend: 'sqlite',
      mode: 'local_only',
      path: store.filePath,
      schema_version: 7,
      legacy_remote: 'disabled',
    });
  });

  it('does not contact PostgreSQL even when legacy environment values remain', async () => {
    process.env.QLH_DB_ENABLED = '1';
    process.env.QLH_DB_HOST = '203.0.113.254';
    app = await createTestApp();
    const started = Date.now();
    const res = await app.inject({ method: 'GET', url: '/db/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json().backend).toBe('sqlite');
    expect(Date.now() - started).toBeLessThan(1000);
  });

  it('reports a completed legacy retirement without changing local health', async () => {
    store.prepare(
      `INSERT INTO storage_retirement
         (retirement_id, status, manifest_version, backup_sha256, manifest_sha256,
          prepared_at, retired_at, removed_outbox_events, details)
       VALUES (1, 'retired', 1, ?, ?, ?, ?, 0, '{}')`,
    ).run('a'.repeat(64), 'b'.repeat(64), '2026-08-09T00:00:00Z', '2026-08-09T00:01:00Z');
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/db/health' });
    expect(res.json()).toMatchObject({ status: 'ok', legacy_remote: 'retired' });
  });
});
