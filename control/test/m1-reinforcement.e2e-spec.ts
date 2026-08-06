/**
 * M1 复核补强测试：
 *  1. 启动自动迁移接线（旧 model_registry.json 升级后对 GET /models/registry 可见）；
 *  2. backup 数据一致性（WAL 未 checkpoint 的数据进入备份）；
 *  3. projector clientFactory 幂等投影；
 *  4. settings pg 投影抛异常时本地事实源成功；
 *  5. cluster_endpoints 同 cluster_id 去重（UNIQUE 约束）。
 */
import { mkdtempSync, writeFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Test } from '@nestjs/testing';
import { FastifyAdapter, NestFastifyApplication } from '@nestjs/platform-fastify';
import { AppModule } from '../src/app';
import { SqliteStore } from '../src/data/sqlite-store';
import { OutboxService } from '../src/data/outbox.service';
import { PostgresProjector } from '../src/data/postgres-projector';
import { ClusterSettingsRepository } from '../src/data/cluster-settings-repository';
import { ModelRegistryRepository } from '../src/data/model-registry-repository';
import { ClusterEndpointsRepository } from '../src/data/cluster-endpoints-repository';
import { ConfigDao } from '../src/data/config-dao';

function tempDir(): string {
  return mkdtempSync(join(tmpdir(), 'qlh-m1rev-'));
}

describe('M1 启动自动迁移接线', () => {
  let app: NestFastifyApplication | null = null;
  let tmpBase: string;
  let sqliteStore: SqliteStore;

  beforeEach(() => {
    tmpBase = tempDir();
    sqliteStore = new SqliteStore(join(tmpBase, 'control.sqlite3'));
    sqliteStore.open();
    // 旧 JSON 注册表（升级场景：存量数据）
    writeFileSync(join(tmpBase, 'model_registry.json'), JSON.stringify([
      { model_id: 'legacy-model', name: '旧模型', model_path: '/legacy',
        model_type: 'safetensors', quant_types: ['int4'] },
    ]), 'utf-8');
    process.env.QLH_LEGACY_REGISTRY_PATH = join(tmpBase, 'model_registry.json');
    process.env.QLH_CATALOG_SEED_PATH = join(tmpBase, 'no-seed.json');
  });

  afterEach(async () => {
    if (app) { await app.close(); app = null; }
    sqliteStore.close();
    delete process.env.QLH_LEGACY_REGISTRY_PATH;
    delete process.env.QLH_CATALOG_SEED_PATH;
    rmSync(tmpBase, { recursive: true, force: true });
  });

  it('应用启动时自动导入旧注册表，GET /models/registry 可见存量数据', async () => {
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(sqliteStore)
      .overrideProvider(ConfigDao).useValue({
        enabled: false, dbEnabled: () => false,
        getConnectionInfo: () => ({}), ping: async () => ({ ok: true }),
      })
      .compile();
    app = moduleRef.createNestApplication(new FastifyAdapter()) as NestFastifyApplication;
    await app.init(); // 触发 onApplicationBootstrap
    const res = await (app as any).inject({ method: 'GET', url: '/models/registry' });
    expect(res.statusCode).toBe(200);
    const models = res.json().models;
    expect(models.some((m: { model_id: string }) => m.model_id === 'legacy-model')).toBe(true);
    // 启动迁移幂等：再次启动不产生重复
    await app.close();
    const moduleRef2 = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(sqliteStore)
      .overrideProvider(ConfigDao).useValue({
        enabled: false, dbEnabled: () => false,
        getConnectionInfo: () => ({}), ping: async () => ({ ok: true }),
      })
      .compile();
    app = moduleRef2.createNestApplication(new FastifyAdapter()) as NestFastifyApplication;
    await app.init();
    const res2 = await (app as any).inject({ method: 'GET', url: '/models/registry' });
    const count = res2.json().models.filter(
      (m: { model_id: string }) => m.model_id === 'legacy-model',
    ).length;
    expect(count).toBe(1); // 无重复
  });
});

describe('backup WAL 一致性', () => {
  it('写入后未 checkpoint 的数据出现在备份文件中', async () => {
    const dir = tempDir();
    try {
      const store = new SqliteStore(join(dir, 'a.sqlite3'));
      store.open();
      new ClusterSettingsRepository(store).set('max_nodes', '7');
      const backupPath = join(dir, 'b.sqlite3');
      await store.backupTo(backupPath);
      store.close();
      // 用只读方式验证备份内容
      const backup = new (require('node:sqlite').DatabaseSync)(backupPath, { readOnly: true });
      const row = backup.prepare(
        "SELECT value FROM cluster_settings WHERE key='max_nodes'",
      ).get() as { value: string } | undefined;
      backup.close();
      expect(row?.value).toBe('7');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('projector clientFactory 幂等投影', () => {
  it('投影后事件被标记；重复执行不重复投影', async () => {
    const store = new SqliteStore(join(tempDir(), 'p.sqlite3'));
    store.open();
    const outbox = new OutboxService(store);
    const event = outbox.enqueue('model_registry', 'created', { model_id: 'm' });
    expect(outbox.pendingCount()).toBe(1);

    const queries: Array<{ sql: string; params: unknown[] }> = [];
    const fakeClient = {
      async connect() { /* noop */ },
      async query(sql: string, params: unknown[]) {
        queries.push({ sql, params });
        return { rowCount: 1, rows: [] }; // 首次投影成功
      },
      async end() { /* noop */ },
    };
    const config = { dbEnabled: () => true, getConnectionInfo: () => ({ host: 'x', port: 1, db: 'd' }) };
    const projector = new PostgresProjector(config as any, outbox, {
      baseIntervalMs: 1000, maxIntervalMs: 2000,
      clientFactory: () => fakeClient as any,
    });
    const first = await projector.runOnce();
    expect(first.projected).toBe(1);
    expect(outbox.pendingCount()).toBe(0); // 已标记投影
    // 重复执行：无 pending，不产生投影 INSERT（CREATE TABLE IF NOT EXISTS 仍会执行）
    const second = await projector.runOnce();
    expect(second.projected).toBe(0);
    const inserts = queries.filter((q) => q.sql.includes('INSERT INTO outbox_projection'));
    expect(inserts.length).toBe(1);
    store.close();
  });
});

describe('settings pg 投影抛异常', () => {
  it('setUserSettings 抛异常 → 本地事实源成功 ok', async () => {
    const store = new SqliteStore(join(tempDir(), 's.sqlite3'));
    store.open();
    const fakeDao = {
      dbEnabled: () => true,
      setUserSettings: async () => { throw new Error('pg exploded'); },
    } as unknown as ConfigDao;
    const { Test } = require('@nestjs/testing') as typeof import('@nestjs/testing');
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(ConfigDao).useValue(fakeDao)
      .overrideProvider(SqliteStore).useValue(store)
      .compile();
    const app = moduleRef.createNestApplication(new FastifyAdapter());
    await app.init();
    const res = await (app as any).inject({
      method: 'PUT', url: '/user/settings',
      payload: { settings: { theme: 'dark' } },
    });
    expect(res.json().status).toBe('ok');
    const row = store.prepare(
      "SELECT value FROM cluster_settings WHERE key='user_settings'",
    ).get() as { value: string } | undefined;
    expect(JSON.parse(row!.value)).toEqual({ theme: 'dark' });
    await app.close();
    store.close();
  });
});

describe('cluster_endpoints cluster_id 去重', () => {
  it('同 cluster_id 不同 endpoint_id 只保留一条（UNIQUE 约束 + ON CONFLICT）', () => {
    const store = new SqliteStore(join(tempDir(), 'e.sqlite3'));
    store.open();
    const repo = new ClusterEndpointsRepository(store);
    repo.upsert({
      endpoint_id: 'ep_a', cluster_id: 'c1', name: 'n1',
      scheme: 'http', host: '10.0.0.1', port: 8888, status: 'active',
    });
    repo.upsert({
      endpoint_id: 'ep_b', cluster_id: 'c1', name: 'n2',
      scheme: 'http', host: '10.0.0.2', port: 8888, status: 'active',
    });
    expect(repo.list().length).toBe(1);
    expect(repo.list()[0].endpoint_id).toBe('ep_b');
    store.close();
  });
});
