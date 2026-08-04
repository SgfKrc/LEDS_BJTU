/**
 * control-svc settings 域契约测试（阶段 3.2 首迁域）
 *
 * 语义对齐 api_server.py:6111-6160：
 *  - GET /user/settings：DB 可用 → {settings, source:'database'}；
 *    DB 禁用 → {settings:{}, source:'none'}
 *  - PUT /user/settings：DB 可用 → {status:'ok', synced_fields}；
 *    DB 禁用 → {status:'skipped', reason:'数据库不可用'}
 *
 * 注意：必须用 Fastify 原生 app.inject()（light-my-request），不能用
 * supertest + app.getHttpServer() —— NestJS Fastify adapter 的
 * getHttpServer() 返回 fastify 实例而非 Node http.Server，supertest
 * 会挂起至超时，且路由 preParsing 上下文未构建时抛 TypeError。
 * 请求前需 fastify.ready()（幂等），见 src/app.ts createApp() 注释。
 */
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
    const res = await app.inject({ method: 'GET', url: '/user/settings' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ settings: {}, source: 'none' });
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
    const res = await app.inject({
      method: 'PUT',
      url: '/user/settings',
      payload: { settings: { theme: 'dark' } },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('skipped');
    expect(res.json().reason).toBe('数据库不可用');
  });

  it('GET /user/settings DB 可用（假 DAO）→ {settings, source:database}', async () => {
    const fakeDao = {
      dbEnabled: () => true,
      getUserSettings: async () => ({ theme: 'dark', saveHistory: true }),
    } as unknown as ConfigDao;
    app = await createAppWithDao(fakeDao);
    const res = await app.inject({ method: 'GET', url: '/user/settings' });
    expect(res.statusCode).toBe(200);
    expect(res.json().source).toBe('database');
    expect(res.json().settings).toEqual({ theme: 'dark', saveHistory: true });
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
    const res = await app.inject({
      method: 'PUT',
      url: '/user/settings',
      payload: { settings: { theme: 'dark', maxNewTokens: 512 } },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('ok');
    expect(res.json().synced_fields.sort()).toEqual(['maxNewTokens', 'theme']);
    expect(saved).toEqual({ theme: 'dark', maxNewTokens: 512 });
  });

  it('PUT /user/settings 写入失败（返回 false）→ {status:skipped}', async () => {
    const fakeDao = {
      dbEnabled: () => true,
      setUserSettings: async () => false,
    } as unknown as ConfigDao;
    app = await createAppWithDao(fakeDao);
    const res = await app.inject({
      method: 'PUT',
      url: '/user/settings',
      payload: { settings: { theme: 'dark' } },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('skipped');
  });

  it('GET /health 探活', async () => {
    app = await createApp();
    const res = await app.inject({ method: 'GET', url: '/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('ok');
    expect(res.json().service).toBe('control-svc');
  });

  it('未知路由 → JSON 404 detail', async () => {
    app = await createApp();
    const res = await app.inject({ method: 'GET', url: '/nope' });
    expect(res.statusCode).toBe(404);
    expect(res.json()).toHaveProperty('detail');
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
  // 与 createApp() 相同的必需步骤，但顺序必须先 init 再 ready：
  // NestFactory.create 内部已 init，而 Test.createTestingModule 的
  // createNestApplication 不会自动 init，必须先 app.init()（注册路由/
  // 中间件），再 fastify.ready()（幂等，构建路由 preParsing 等 hooks
  // 上下文）；反序会触发 "Root plugin has already booted"。
  await testApp.init();
  await testApp.getHttpAdapter().getInstance().ready();
  return testApp;
}
