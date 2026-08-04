/**
 * control-svc db/health 域契约测试（阶段 3.2 db-health 域）
 *
 * 语义对齐 api_server.py:6669-6700：
 *  - GET /db/health → 三态：
 *    - enabled=false（QLH_DB_ENABLED=0）→ not_configured
 *    - enabled=true 但连接失败 → connection_failed（message 含驱动错误）
 *    - 连接成功（SELECT 1）→ {status:'ok', host, port, db}
 *
 * 说明：ok 分支无真实 PostgreSQL，用 stub ConfigDao（ping 恒 ok）验证
 * 响应形状；connection_failed 用真实 ConfigDao + 无效端口（连接超时 3s 内
 * 失败）；driver_missing 分支 TS 侧不存在（pg 为编译期依赖，见控制器注释）。
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { ConfigDao } from '../src/data/config-dao';

describe('control-svc db/health 域（阶段 3.2 db-health）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpBase: string;

  beforeEach(() => {
    tmpBase = fs.mkdtempSync(path.join(os.tmpdir(), 'control-db-'));
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
    fs.rmSync(tmpBase, { recursive: true, force: true });
  });

  async function createTestApp(dao: unknown): Promise<NestFastifyApplication> {
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
    const { RequestIdInterceptor } = require('../src/common/request-id');
    testApp.useGlobalFilters(new JsonDetailFilter());
    testApp.useGlobalInterceptors(new RequestIdInterceptor());
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  it('GET /db/health 未配置（QLH_DB_ENABLED=0）→ not_configured', async () => {
    const dao = new ConfigDao({
      host: 'localhost',
      port: 5432,
      name: 'qlh_edge_inference',
      user: 'postgres',
      password: '',
      enabled: false,
      sslmode: 'prefer',
    });
    app = await createTestApp(dao);
    const res = await app.inject({ method: 'GET', url: '/db/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      status: 'unavailable',
      reason: 'not_configured',
      message: '数据库未配置，正在使用本地文件存储',
      retry_in_seconds: 0,
    });
  });

  it('GET /db/health 已配置但连接失败 → connection_failed（含驱动错误消息）', async () => {
    const dao = new ConfigDao({
      host: '127.0.0.1',
      port: 59999, // 必然不可达端口
      name: 'x',
      user: 'postgres',
      password: '',
      enabled: true,
      sslmode: 'disable',
    });
    app = await createTestApp(dao);
    const res = await app.inject({ method: 'GET', url: '/db/health' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('unavailable');
    expect(body.reason).toBe('connection_failed');
    expect(typeof body.message).toBe('string');
    expect(body.message.length).toBeGreaterThan(0);
    expect(body.retry_in_seconds).toBe(0);
  });

  it('GET /db/health 连接成功 → {status:ok, host, port, db}', async () => {
    const dao = {
      dbEnabled: () => true,
      ping: async () => ({ ok: true }),
      getConnectionInfo: () => ({ host: 'db.example', port: 5432, db: 'qlh_edge_inference' }),
    };
    app = await createTestApp(dao);
    const res = await app.inject({ method: 'GET', url: '/db/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      status: 'ok',
      host: 'db.example',
      port: 5432,
      db: 'qlh_edge_inference',
    });
  });
});
