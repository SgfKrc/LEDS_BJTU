/**
 * M1 storage/health 契约测试（本地/远端双健康，§12.3）。
 * /db/health 三态契约保持不变（阶段 3.2）；本端点提供新结构。
 */
import { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { AppModule } from '../src/app';
import { ConfigDao } from '../src/data/config-dao';
import { SqliteStore } from '../src/data/sqlite-store';
import { OutboxService } from '../src/data/outbox.service';

class FakeConfigDao {
  enabled = false;
  pingOk = false;
  dbEnabled() { return this.enabled; }
  getConnectionInfo() { return { host: '127.0.0.1', port: 5432, db: 'qlh' }; }
  async ping() {
    return this.pingOk ? { ok: true } : { ok: false, error: 'fake down' };
  }
}

describe('storage/health 契约（M1 双健康）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpBase: string;
  let configDao: FakeConfigDao;

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'control-m1-'));
    configDao = new FakeConfigDao();
    process.env.QLH_SQLITE_PATH = join(tmpBase, 'qlh-control.sqlite3');
  });

  afterEach(async () => {
    if (app) {
      // 先关 SQLite（WAL 句柄占用会阻止 Windows 删除临时目录）
      try {
        app.get(SqliteStore).close();
      } catch {
        // 未实例化则跳过
      }
      await app.close();
      app = null;
    }
    delete process.env.QLH_SQLITE_PATH;
    rmSync(tmpBase, { recursive: true, force: true });
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(ConfigDao)
      .useValue(configDao)
      .compile();
    const fastify = new (require('@nestjs/platform-fastify').FastifyAdapter)();
    const testApp = moduleRef.createNestApplication(fastify) as unknown as NestFastifyApplication;
    const { JsonDetailFilter } = require('../src/common/json-detail.filter');
    const { RequestIdInterceptor } = require('../src/common/request-id');
    testApp.useGlobalFilters(new JsonDetailFilter());
    testApp.useGlobalInterceptors(new RequestIdInterceptor());
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  it('GET /storage/health → local_only 结构（远端未配置）', async () => {
    app = await createTestApp();
    configDao.enabled = false;
    const res = await app.inject({ method: 'GET', url: '/storage/health' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.local).toEqual({
      status: 'ok',
      backend: 'sqlite',
      writable: true,
      path: join(tmpBase, 'qlh-control.sqlite3'),
      schema_version: 2,
    });
    expect(body.remote).toEqual({ status: 'not_configured', backend: 'postgresql' });
    expect(body.projection.pending_events).toBe(0);
    expect(body.effective_mode).toBe('local_only');
  });

  it('GET /storage/health → local_primary（远端 ok 且无积压）', async () => {
    app = await createTestApp();
    configDao.enabled = true;
    configDao.pingOk = true;
    const res = await app.inject({ method: 'GET', url: '/storage/health' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.remote.status).toBe('ok');
    expect(body.effective_mode).toBe('local_primary');
  });

  it('GET /storage/health → local_primary_pending（outbox 有积压）', async () => {
    app = await createTestApp();
    configDao.enabled = true;
    configDao.pingOk = true;
    // 直接向同一 SQLite 实例入队一条事件
    const store = app.get(SqliteStore);
    new OutboxService(store).enqueue('model_registry', 'created', {});
    const res = await app.inject({ method: 'GET', url: '/storage/health' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.projection.pending_events).toBe(1);
    expect(body.effective_mode).toBe('local_primary_pending');
  });

  it('GET /storage/health → local_only（远端配置但不可达）', async () => {
    app = await createTestApp();
    configDao.enabled = true;
    configDao.pingOk = false;
    const res = await app.inject({ method: 'GET', url: '/storage/health' });
    const body = res.json();
    expect(body.remote.status).toBe('unavailable');
    expect(body.effective_mode).toBe('local_only');
  });
});
