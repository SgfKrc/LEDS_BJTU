/**
 * M4 多集群档案测试：repository CRUD/幂等、verify 探测（mock fetch）、
 * 多集群并存不串用、endpoint 自检。
 */
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Test } from '@nestjs/testing';
import { FastifyAdapter } from '@nestjs/platform-fastify';
import { AppModule } from '../src/app';
import { SqliteStore } from '../src/data/sqlite-store';
import { ClusterProfileRepository } from '../src/data/cluster-profile-repository';
import { ClusterEndpointsRepository } from '../src/data/cluster-endpoints-repository';

function tempStore(): { dir: string; store: SqliteStore } {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-m4-'));
  const store = new SqliteStore(join(dir, 'ctl.sqlite3'));
  store.open();
  return { dir, store };
}

describe('ClusterProfileRepository（M4）', () => {
  it('CRUD 与 cluster_id 幂等（不重复建群）', () => {
    const { dir, store } = tempStore();
    const repo = new ClusterProfileRepository(store);
    const p1 = repo.create({
      cluster_id: 'cluster_a', name: '家里集群', master_endpoint: 'http://100.64.0.1:8000',
    });
    const p2 = repo.create({
      cluster_id: 'cluster_a', name: '家里集群（更新）', master_endpoint: 'http://100.64.0.2:8000',
    });
    // 同 cluster_id：不重复建群（ON CONFLICT 更新）
    expect(repo.list().length).toBe(1);
    expect(repo.get(p1.profile_id)?.master_endpoint).toBe('http://100.64.0.2:8000');
    expect(p2.cluster_id).toBe('cluster_a');

    // 多集群并存
    repo.create({
      cluster_id: 'cluster_b', name: '实验室集群', master_endpoint: 'http://100.64.0.9:8000',
    });
    expect(repo.list().length).toBe(2);
    expect(repo.getByCluster('cluster_a')?.name).toContain('家里');
    expect(repo.getByCluster('cluster_b')?.cluster_id).toBe('cluster_b');

    // markVerified + delete
    const verified = repo.markVerified(p1.profile_id, 'active');
    expect(verified?.status).toBe('active');
    expect(verified?.last_verified_at).toBeTruthy();
    expect(repo.delete(p1.profile_id)).toBe(true);
    expect(repo.delete(p1.profile_id)).toBe(false);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });
});

describe('cluster profiles API（M4）', () => {
  let app: any;
  let tmpBase: string;
  let sqliteStore: SqliteStore;

  beforeEach(async () => {
    tmpBase = mkdtempSync(join(tmpdir(), 'qlh-m4api-'));
    sqliteStore = new SqliteStore(join(tmpBase, 'ctl.sqlite3'));
    sqliteStore.open();
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(sqliteStore)
      .compile();
    app = moduleRef.createNestApplication(new FastifyAdapter());
    await app.init();
  });

  afterEach(async () => {
    await app.close();
    sqliteStore.close();
    rmSync(tmpBase, { recursive: true, force: true });
  });

  it('POST /cluster/profiles/verify 不可达 → unreachable（无副作用）', async () => {
    const res = await app.inject({
      method: 'POST', url: '/cluster/profiles/verify',
      payload: { name: 'x', master_endpoint: 'http://127.0.0.1:1' },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('unreachable');
    expect(res.json().error).toBeTruthy();
    // 无副作用：未创建档案
    const list = await app.inject({ method: 'GET', url: '/cluster/profiles' });
    expect(list.json().profiles).toEqual([]);
  });

  it('POST/GET/DELETE profiles 全流程 + 多集群不串用', async () => {
    const created = await app.inject({
      method: 'POST', url: '/cluster/profiles',
      payload: { cluster_id: 'c1', name: 'A', master_endpoint: 'http://100.64.0.1:8000' },
    });
    expect(created.statusCode).toBe(201);
    const profileId = created.json().profile.profile_id;

    await app.inject({
      method: 'POST', url: '/cluster/profiles',
      payload: { cluster_id: 'c2', name: 'B', master_endpoint: 'http://100.64.0.2:8000' },
    });
    const list = await app.inject({ method: 'GET', url: '/cluster/profiles' });
    expect(list.json().profiles.length).toBe(2);
    // 多集群并存：cluster_id 互不串用
    const ids = list.json().profiles.map((p: { cluster_id: string }) => p.cluster_id);
    expect(ids).toContain('c1');
    expect(ids).toContain('c2');

    const del = await app.inject({ method: 'DELETE', url: `/cluster/profiles/${profileId}` });
    expect(del.json().status).toBe('deleted');
    const after = await app.inject({ method: 'GET', url: '/cluster/profiles' });
    expect(after.json().profiles.length).toBe(1);
  });

  it('POST /cluster/profiles 缺字段 → 422', async () => {
    const res = await app.inject({
      method: 'POST', url: '/cluster/profiles',
      payload: { name: 'x' },
    });
    expect(res.statusCode).toBe(422);
  });

  it('POST /cluster/endpoints/verify 自检本机 health', async () => {
    const res = await app.inject({
      method: 'POST', url: '/cluster/endpoints/verify',
      payload: { scheme: 'http', host: '127.0.0.1', port: 1 },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().reachable).toBe(false); // 端口 1 必不可达
    expect(res.json().detail).toBeTruthy();
  });

  it('first-connect 登记的 endpoint 在 /cluster/endpoints 可见', async () => {
    // 直接经 repository 模拟 first-connect 登记（与 bootstrap 控制器同路径）
    new ClusterEndpointsRepository(sqliteStore).upsert({
      endpoint_id: 'ep_c1', cluster_id: 'c1', name: 'bootstrap-registered',
      scheme: 'http', host: '100.64.0.1', port: 8888, status: 'active',
    });
    const res = await app.inject({ method: 'GET', url: '/cluster/endpoints' });
    expect(res.json().endpoints.some((e: { cluster_id: string }) => e.cluster_id === 'c1')).toBe(true);
  });
});
