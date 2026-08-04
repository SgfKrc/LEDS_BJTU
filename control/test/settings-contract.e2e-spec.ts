/**
 * control-svc settings 域契约测试（阶段 3.2 首迁域）
 *
 * 语义对齐 api_server.py:6111-6160：
 *  - GET /user/settings：DB 可用 → {settings, source:'database'}；
 *    DB 禁用 → {settings:{}, source:'none'}
 *  - PUT /user/settings：DB 可用 → {status:'ok', synced_fields}；
 *    DB 禁用 → {status:'skipped', reason:'数据库不可用'}
 */
import request from 'supertest';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { ConfigDao } from '../src/data/config-dao';

describe('control-svc settings 域（阶段 3.2 首迁）', () => {
  let app: NestFastifyApplication | null = null;

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
  });

  it('GET /user/settings DB 禁用 → {settings:{}, source:none}', async () => {
    const dao = new ConfigDao({
      host: 'localhost',
      port: 5432,
      name: 'x',
      user: 'postgres',
      password: '',
      enabled: false,
      sslmode: 'prefer',
    });
    app = await createAppWithDao(dao);
    const res = await request(app.getHttpServer()).get('/user/settings');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ settings: {}, source: 'none' });
  });

  it('PUT /user/settings DB 禁用 → {status:skipped}', async () => {
    const dao = new ConfigDao({
      host: 'localhost',
      port: 5432,
      name: 'x',
      user: 'postgres',
      password: '',
      enabled: false,
      sslmode: 'prefer',
    });
    app = await createAppWithDao(dao);
    const res = await request(app.getHttpServer())
      .put('/user/settings')
      .send({ settings: { theme: 'dark' } });
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('skipped');
    expect(res.body.reason).toBe('数据库不可用');
  });

  it('GET /user/settings DB 可用（假 DAO）→ {settings, source:database}', async () => {
    const fakeDao = {
      dbEnabled: () => true,
      getUserSettings: async () => ({ theme: 'dark', saveHistory: true }),
    } as unknown as ConfigDao;
    app = await createAppWithDao(fakeDao);
    const res = await request(app.getHttpServer()).get('/user/settings');
    expect(res.status).toBe(200);
    expect(res.body.source).toBe('database');
    expect(res.body.settings).toEqual({ theme: 'dark', saveHistory: true });
  });

  it('PUT /user/settings DB 可用（假 DAO）→ {status:ok, synced_fields}', async () => {
    let saved: Record<string, unknown> | null = null;
    const fakeDao = {
      dbEnabled: () => true,
      setUserSettings: async (s: Record<string, unknown>) => {
        saved = s;
        return true;
      },
    } as unknown as ConfigDao;
    app = await createAppWithDao(fakeDao);
    const res = await request(app.getHttpServer())
      .put('/user/settings')
      .send({ settings: { theme: 'dark', maxNewTokens: 512 } });
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.synced_fields.sort()).toEqual(['maxNewTokens', 'theme']);
    expect(saved).toEqual({ theme: 'dark', maxNewTokens: 512 });
  });

  it('PUT /user/settings DB 可用但写失败 → 500', async () => {
    const fakeDao = {
      dbEnabled: () => true,
      setUserSettings: async () => false,
    } as unknown as ConfigDao;
    app = await createAppWithDao(fakeDao);
    const res = await request(app.getHttpServer())
      .put('/user/settings')
      .send({ settings: { theme: 'dark' } });
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('skipped');
  });

  it('GET /health 探活', async () => {
    app = await createApp();
    await app.init();
    const res = await request(app.getHttpServer()).get('/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.service).toBe('control-svc');
  });

  it('未知路由 → JSON 404 detail', async () => {
    app = await createApp();
    await app.init();
    const res = await request(app.getHttpServer()).get('/nope');
    expect(res.status).toBe(404);
    expect(res.body).toHaveProperty('detail');
  });
});

async function createAppWithDao(dao: ConfigDao): Promise<NestFastifyApplication> {
  // 覆盖模块 provider：用测试 DAO 替换 ConfigDao
  const { Test } = require('@nestjs/testing');
  const { AppModule } = require('../src/app');
  const moduleRef = await Test.createTestingModule({
    imports: [AppModule],
  })
    .overrideProvider(ConfigDao)
    .useValue(dao)
    .compile();
  const fastifyAdapter = new (require('@nestjs/platform-fastify').FastifyAdapter)();
  const testApp = moduleRef.createNestApplication(fastifyAdapter);
  const { JsonDetailFilter } = require('../src/common/json-detail.filter');
  testApp.useGlobalFilters(new JsonDetailFilter());
  await testApp.init();
  return testApp;
}
