/**
 * control-svc bootstrap 域契约测试（阶段 3.2 bootstrap 域）
 *
 * 语义对齐 api_server.py:5365-5468 + bootstrap.py：
 *  - POST /bootstrap/first-connect：
 *    * QLH_BOOTSTRAP_ENABLED=false → 403 'bootstrap disabled'
 *    * 信任校验（QLH_BOOTSTRAP_REQUIRE_TAILSCALE 默认 true）：非信任
 *      来源 → 403 'source network is not trusted'；默认信任集
 *      100.64.0.0/10、127.0.0.0/8、::1/128、fd7a:115c:a1e0::/48，
 *      QLH_TRUSTED_BOOTSTRAP_CIDRS 可覆盖
 *    * node_id 归一化（非法字符 → '_'、64 截断、空/'master' → client_<host>）
 *    * 响应结构 {status:'ok', cluster{cluster_id, master_api_host/port,
 *      master_tcp_host/port, cluster_secret}, node{node_id, role:'client',
 *      node_type, pipeline_worker}, android{presence_interval_seconds:45,
 *      pipeline_worker:false, model_manifest_url}}
 *    * node_type: android/mobile → 'android'，其余 → 'pc'（pipeline_worker）
 *  - GET /bootstrap/info：无条件信任校验；{status:'ok', is_master:true,
 *    node_id, master_api_port, master_tcp_port}
 *  - CIDR 单元：isInCidr IPv4/IPv6 匹配
 */
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { isInCidr, isTrustedBootstrapSource } from '../src/common/bootstrap-trust';

describe('control-svc bootstrap 域（阶段 3.2）', () => {
  let app: NestFastifyApplication | null = null;

  beforeEach(() => {
    delete process.env.QLH_BOOTSTRAP_ENABLED;
    delete process.env.QLH_BOOTSTRAP_REQUIRE_TAILSCALE;
    delete process.env.QLH_TRUSTED_BOOTSTRAP_CIDRS;
    delete process.env.QLH_CLUSTER_ID;
    delete process.env.QLH_CLUSTER_SECRET;
    delete process.env.QLH_NODE_ID;
    delete process.env.QLH_MASTER_API_PORT;
    delete process.env.QLH_MASTER_TCP_PORT;
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const { Test } = require('@nestjs/testing');
    const { AppModule } = require('../src/app');
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();
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

  // ---------- CIDR 单元 ----------

  it('isInCidr：IPv4 前缀匹配', () => {
    expect(isInCidr('127.0.0.1', '127.0.0.0/8')).toBe(true);
    expect(isInCidr('127.255.255.255', '127.0.0.0/8')).toBe(true);
    expect(isInCidr('128.0.0.1', '127.0.0.0/8')).toBe(false);
    expect(isInCidr('100.64.5.9', '100.64.0.0/10')).toBe(true);
    expect(isInCidr('100.127.255.255', '100.64.0.0/10')).toBe(true);
    expect(isInCidr('100.128.0.1', '100.64.0.0/10')).toBe(false);
    expect(isInCidr('192.168.1.5', '192.168.1.0/24')).toBe(true);
    expect(isInCidr('192.168.2.5', '192.168.1.0/24')).toBe(false);
  });

  it('isInCidr：IPv6 与版本不一致', () => {
    expect(isInCidr('::1', '::1/128')).toBe(true);
    expect(isInCidr('::2', '::1/128')).toBe(false);
    expect(isInCidr('::ffff:127.0.0.1', '::ffff:127.0.0.1/128')).toBe(true);
    // 版本不一致不匹配
    expect(isInCidr('127.0.0.1', '::1/128')).toBe(false);
    expect(isInCidr('::1', '127.0.0.0/8')).toBe(false);
  });

  it('isTrustedBootstrapSource：默认信任集 + hostname 不信任', () => {
    expect(isTrustedBootstrapSource('127.0.0.1')).toBe(true);
    expect(isTrustedBootstrapSource('100.64.1.2')).toBe(true);
    expect(isTrustedBootstrapSource('::1')).toBe(true);
    expect(isTrustedBootstrapSource('fd7a:115c:a1e0::1')).toBe(true);
    expect(isTrustedBootstrapSource('::ffff:127.0.0.1')).toBe(true);
    expect(isTrustedBootstrapSource('192.168.1.5')).toBe(false);
    expect(isTrustedBootstrapSource('localhost')).toBe(false); // hostname 非 IP
    expect(isTrustedBootstrapSource('')).toBe(false);
  });

  // ---------- POST /bootstrap/first-connect ----------

  it('first-connect 本机 → 200 + 完整 cluster/node/android 结构', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'node-abc', node_type: 'pc', hostname: 'worker-host' },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('ok');
    expect(body.cluster.cluster_id).toBe('qlh-default');
    expect(body.cluster.master_api_host).toBeTruthy();
    expect(body.cluster.master_api_port).toBe(8000);
    expect(body.cluster.master_tcp_port).toBe(8888);
    expect(body.cluster.cluster_secret).toBe('local-bootstrap-not-configured');
    expect(body.node).toEqual({
      node_id: 'node-abc',
      role: 'client',
      node_type: 'pc',
      pipeline_worker: true,
    });
    expect(body.android.presence_interval_seconds).toBe(45);
    expect(body.android.pipeline_worker).toBe(false);
    expect(body.android.model_manifest_url).toContain('/api/models/downloadable');
  });

  it('first-connect 远程非信任 → 403 source network is not trusted', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      remoteAddress: '192.168.1.50',
      payload: { node_id: 'n1', node_type: 'pc' },
    });
    expect(res.statusCode).toBe(403);
    expect(res.json().detail).toBe('source network is not trusted');
  });

  it('first-connect Tailscale IPv6 → 裸 host + 合法方括号 URL', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      remoteAddress: 'fd7a:115c:a1e0::10',
      headers: { host: '[fd7a:115c:a1e0::1]:8030' },
      payload: { node_id: 'node-v6', node_type: 'android' },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().cluster.master_api_host).toBe('fd7a:115c:a1e0::1');
    expect(res.json().android.model_manifest_url).toBe(
      'http://[fd7a:115c:a1e0::1]:8000/api/models/downloadable',
    );
  });

  it('first-connect QLH_TRUSTED_BOOTSTRAP_CIDRS 覆盖 → 远程放行', async () => {
    process.env.QLH_TRUSTED_BOOTSTRAP_CIDRS = '192.168.1.0/24';
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      remoteAddress: '192.168.1.50',
      payload: { node_id: 'n1', node_type: 'pc' },
    });
    expect(res.statusCode).toBe(200);
    // 默认集被覆盖后 127.0.0.1 反而不可信
    const local = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'n1', node_type: 'pc' },
    });
    expect(local.statusCode).toBe(403);
  });

  it('first-connect 开关关闭 → 403 bootstrap disabled', async () => {
    process.env.QLH_BOOTSTRAP_ENABLED = 'false';
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'n1' },
    });
    expect(res.statusCode).toBe(403);
    expect(res.json().detail).toBe('bootstrap disabled');
  });

  it('first-connect REQUIRE_TAILSCALE=false → 跳过信任校验', async () => {
    process.env.QLH_BOOTSTRAP_REQUIRE_TAILSCALE = 'false';
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      remoteAddress: '192.168.1.50',
      payload: { node_id: 'n1' },
    });
    expect(res.statusCode).toBe(200);
  });

  it('first-connect node_type android → pipeline_worker false + node_id 前缀', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'galaxy-s21', node_type: 'android' },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().node.node_type).toBe('android');
    expect(res.json().node.pipeline_worker).toBe(false);
    // 空 node_id + android → android_<hostname>
    const empty = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_type: 'mobile' },
    });
    expect(empty.json().node.node_type).toBe('android');
    expect(empty.json().node.node_id.startsWith('android_')).toBe(true);
  });

  it('first-connect node_id 归一化：非法字符 + 64 截断', async () => {
    app = await createTestApp();
    const weird = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'a b/c$d', node_type: 'pc' },
    });
    expect(weird.json().node.node_id).toBe('a_b_c_d');
    const long = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'x'.repeat(80), node_type: 'pc' },
    });
    expect(long.json().node.node_id).toHaveLength(64);
  });

  it('first-connect env 覆盖 cluster_id / cluster_secret / 端口', async () => {
    process.env.QLH_CLUSTER_ID = 'my-cluster';
    process.env.QLH_CLUSTER_SECRET = 'super-secret';
    process.env.QLH_MASTER_API_PORT = '8080';
    process.env.QLH_MASTER_TCP_PORT = '7777';
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/bootstrap/first-connect',
      payload: { node_id: 'n1' },
    });
    const body = res.json();
    expect(body.cluster.cluster_id).toBe('my-cluster');
    expect(body.cluster.cluster_secret).toBe('super-secret');
    expect(body.cluster.master_api_port).toBe(8080);
    expect(body.cluster.master_tcp_port).toBe(7777);
  });

  // ---------- GET /bootstrap/info ----------

  it('bootstrap/info 本机 → 200 发现契约', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/bootstrap/info' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      status: 'ok',
      is_master: true,
      node_id: 'master',
      master_api_port: 8000,
      master_tcp_port: 8888,
    });
  });

  it('bootstrap/info 远程 → 403（无条件信任校验）', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'GET',
      url: '/bootstrap/info',
      remoteAddress: '10.0.0.5',
    });
    expect(res.statusCode).toBe(403);
  });

  it('bootstrap/info QLH_NODE_ID 覆盖 node_id', async () => {
    process.env.QLH_NODE_ID = 'my-master';
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/bootstrap/info' });
    expect(res.json().node_id).toBe('my-master');
  });
});
