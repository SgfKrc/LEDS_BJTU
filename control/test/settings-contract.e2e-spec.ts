/** Settings are stored only in the main-node SQLite database. */
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { AppModule, createApp } from '../src/app';
import { JsonDetailFilter } from '../src/common/json-detail.filter';
import { SqliteStore } from '../src/data/sqlite-store';

describe('control-svc settings local-only contract', () => {
  let app: NestFastifyApplication | null = null;
  let store: SqliteStore;
  let tmpBase: string;

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'control-settings-'));
    store = new SqliteStore(join(tmpBase, 'settings.sqlite3'));
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
    testApp.useGlobalFilters(new JsonDetailFilter());
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  it('returns an empty local document on first use', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/user/settings' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ settings: {}, source: 'local' });
  });

  it('writes and reads settings from SQLite', async () => {
    app = await createTestApp();
    const put = await app.inject({
      method: 'PUT',
      url: '/user/settings',
      payload: { settings: { theme: 'dark', saveHistory: false } },
    });
    expect(put.statusCode).toBe(200);
    expect(put.json().synced_fields.sort()).toEqual(['saveHistory', 'theme']);
    const get = await app.inject({ method: 'GET', url: '/user/settings' });
    expect(get.json()).toEqual({
      settings: { theme: 'dark', saveHistory: false },
      source: 'local',
    });
    const saveHistory = store.prepare(
      "SELECT value FROM cluster_settings WHERE key='save_history'",
    ).get() as { value: string };
    expect(saveHistory.value).toBe('false');
  });

  it('ignores legacy PostgreSQL environment configuration', async () => {
    process.env.QLH_DB_ENABLED = '1';
    process.env.QLH_DB_HOST = '203.0.113.254';
    app = await createTestApp();
    const started = Date.now();
    const res = await app.inject({
      method: 'PUT',
      url: '/user/settings',
      payload: { settings: { maxNewTokens: 512 } },
    });
    expect(res.statusCode).toBe(200);
    expect(Date.now() - started).toBeLessThan(1000);
    expect(store.prepare('SELECT COUNT(*) AS c FROM outbox').get()).toEqual({ c: 0 });
  });

  it('GET /health remains available', async () => {
    app = await createApp();
    const res = await app.inject({ method: 'GET', url: '/health' });
    expect(res.json()).toMatchObject({ status: 'ok', service: 'control-svc' });
  });

  it('unknown routes keep the JSON detail contract', async () => {
    app = await createApp();
    const res = await app.inject({ method: 'GET', url: '/nope' });
    expect(res.statusCode).toBe(404);
    expect(res.json()).toHaveProperty('detail');
  });
});
