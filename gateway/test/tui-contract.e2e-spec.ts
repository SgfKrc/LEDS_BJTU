/**
 * TUI 契约测试（阶段 2 网关专项）
 *
 * 用例清单与 docs/TUI适配实施计划.md §4.1 一一对应：
 *   - 用例 1-38：tui_admin.py 全部 HTTP 调用点（§2.2 全表）
 *   - 用例 39-43：5 项细节断言 + 错误契约
 *
 * 打开规则：按任务逐个打开——
 *   T2（已完成）：用例 1、41、43（用例 41 的 logs 段当前由 catch-all 404 JSON 兜底，
 *        T6 打开真端点后补 status=200 断言）
 *   T3（已完成）：用例 4-32、39（scheduler-svc 测试桩 fake-scheduler.ts）
 *   T4（device 代理）：用例 33-35
 *   T5（状态聚合）：用例 2、3、40、42（用例 40 断言跨 /status、/models/current、
 *        /cluster/nodes、/cluster/queue 四个端点，按"不允许部分断言"整体归 T5）
 *   T6（logs 代理）：用例 36-38、41(logs 段补充 status 断言)
 * 打开一个用例即要求其断言全绿，不允许部分断言。
 */
import request from 'supertest';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { startFakeScheduler } from './fake-scheduler';
import type { FakeScheduler } from './fake-scheduler';

describe('TUI 契约（阶段 2 网关）', () => {
  let app: NestFastifyApplication;
  let fake: FakeScheduler;

  beforeAll(async () => {
    fake = await startFakeScheduler();
    process.env.QLH_SCHEDULER_URL = `http://127.0.0.1:${fake.port}`;
    app = await createApp();
    await app.init();
    // fastify 的 preReady（fourOhFour 404 context 的 hooks 初始化）在 ready() 时完成；
    // NestJS app.init() 不等待它，supertest 的 server.listen 可能在 ready 完成前
    // 放行首个请求，导致 context.preParsing 未初始化而崩溃
    // （"Cannot read properties of undefined (reading 'length')"）。
    await (app.getHttpAdapter().getInstance() as any).ready();
  });

  afterAll(async () => {
    delete process.env.QLH_SCHEDULER_URL;
    await app.close();
    await fake.close();
  });

  const server = () => app.getHttpServer();

  // ============================================================
  // T0 骨架 smoke（非契约用例，T1 起常绿）
  // ============================================================
  describe('T0 骨架 smoke', () => {
    it('app 可创建，/api 前缀生效（未挂路由时 404 亦为 JSON）', async () => {
      const res = await request(server()).get('/api/nonexistent');
      expect(res.status).toBe(404);
      expect(res.headers['content-type']).toContain('application/json');
    });
  });

  // ============================================================
  // 用例 1-38：§2.2 端点全表（38 个调用点）
  // ============================================================
  describe('端点全表', () => {
    it('用例 1: GET /api/health 返回 200 且 status=ok', async () => {
      const res = await request(server()).get('/api/health');
      expect(res.status).toBe(200);
      expect(res.body.status).toBe('ok');
    });

    it.skip('用例 2: GET /api/status 聚合字段齐全（T5）', async () => {
      const res = await request(server()).get('/api/status');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('run_mode');
      expect(res.body).toHaveProperty('node_role');
      expect(res.body).toHaveProperty('model_loaded');
    });

    it.skip('用例 3: GET /api/models/current（T5）', async () => {
      const res = await request(server()).get('/api/models/current');
      expect(res.status).toBe(200);
    });

    it('用例 4: GET /api/cluster/my-role（T3）', async () => {
      const res = await request(server()).get('/api/cluster/my-role');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('is_master');
      expect(res.body).toHaveProperty('node_id');
    });

    it('用例 5: GET /api/cluster/nodes 节点统计与列表（T3）', async () => {
      const res = await request(server()).get('/api/cluster/nodes');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('count');
      expect(res.body).toHaveProperty('online_count');
      expect(res.body).toHaveProperty('offline_count');
      expect(Array.isArray(res.body.nodes)).toBe(true);
    });

    it('用例 6: GET /api/cluster/invite（T3）', async () => {
      const res = await request(server()).get('/api/cluster/invite');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('master_host');
      expect(res.body).toHaveProperty('max_nodes');
    });

    it('用例 7: GET /api/cluster/spare-master（T3）', async () => {
      const res = await request(server()).get('/api/cluster/spare-master');
      expect(res.status).toBe(200);
    });

    it('用例 8: GET /api/cluster/master-health（T3，从节点视角）', async () => {
      const res = await request(server()).get('/api/cluster/master-health');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('master_online');
    });

    it('用例 9: GET /api/cluster/discover（T3）', async () => {
      const res = await request(server()).get('/api/cluster/discover');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('found');
    });

    it('用例 10: POST /api/cluster/connect（T3）', async () => {
      const res = await request(server())
        .post('/api/cluster/connect')
        .send({ host: '127.0.0.1', port: 8888 });
      expect(res.status).toBe(200);
    });

    it('用例 11: POST /api/cluster/nodes/register（T3）', async () => {
      const res = await request(server())
        .post('/api/cluster/nodes/register')
        .send({});
      expect(res.status).toBe(200);
    });

    it('用例 12: POST /api/cluster/nodes/:id/deregister（T3，路径参数须 quote 兼容）', async () => {
      const res = await request(server()).post(
        '/api/cluster/nodes/test-node/deregister',
      );
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('用例 13: DELETE /api/cluster/nodes/:id 返回 JSON 非空体', async () => {
      const res = await request(server()).delete(
        '/api/cluster/nodes/test-node',
      );
      // 对齐 api_server.py:5266-5285：节点不存在 → 404 + detail（JSON 非空体，禁止 204）
      expect(res.status).toBe(404);
      expect(res.body).toHaveProperty('detail');
      expect(res.headers['content-type']).toContain('application/json');
      expect(res.text).not.toBe('');
    });

    it('用例 14: POST /api/cluster/transfer-master（T3）', async () => {
      const res = await request(server())
        .post('/api/cluster/transfer-master')
        .send({ target_node_id: 'test-node' });
      expect(res.status).toBe(200);
    });

    it('用例 15: POST /api/cluster/spare-master（T3）', async () => {
      const res = await request(server())
        .post('/api/cluster/spare-master')
        .send({ node_id: 'test-node' });
      expect(res.status).toBe(200);
    });

    it('用例 16: DELETE /api/cluster/spare-master 返回 JSON（T3）', async () => {
      const res = await request(server()).delete('/api/cluster/spare-master');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('用例 17: PUT /api/cluster/config/max-nodes（T3）', async () => {
      const res = await request(server())
        .put('/api/cluster/config/max-nodes')
        .send({ max_nodes: 8 });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('用例 18: GET /api/cluster/transfer-logs（T3）', async () => {
      const res = await request(server()).get('/api/cluster/transfer-logs');
      expect(res.status).toBe(200);
    });

    it('用例 19: POST /api/cluster/email-test（T3）', async () => {
      const res = await request(server()).post('/api/cluster/email-test');
      expect(res.status).toBe(200);
    });

    it('用例 20: POST /api/cluster/reset-identity（T3）', async () => {
      const res = await request(server()).post('/api/cluster/reset-identity');
      expect(res.status).toBe(200);
    });

    it('用例 21: GET /api/cluster/config/distributed-inference（T3）', async () => {
      const res = await request(server()).get(
        '/api/cluster/config/distributed-inference',
      );
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('enabled');
    });

    it('用例 22: PUT /api/cluster/config/distributed-inference（T3）', async () => {
      const res = await request(server())
        .put('/api/cluster/config/distributed-inference')
        .send({ enabled: true });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('用例 23: GET /api/cluster/layers 层分配（T3）', async () => {
      const res = await request(server()).get('/api/cluster/layers');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('assignments');
    });

    it('用例 24: PUT /api/cluster/layers 手工分层覆盖（T3）', async () => {
      const res = await request(server())
        .put('/api/cluster/layers')
        .send({ assignment: [] });
      expect(res.status).toBe(200);
    });

    it('用例 25: DELETE /api/cluster/layers 返回 JSON（T3）', async () => {
      const res = await request(server()).delete('/api/cluster/layers');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('用例 26: GET /api/cluster/config（T3）', async () => {
      const res = await request(server()).get('/api/cluster/config');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('network');
      expect(res.body).toHaveProperty('model');
    });

    it('用例 27: GET /api/cluster/queue MLFQ 队列快照（T3）', async () => {
      const res = await request(server()).get('/api/cluster/queue');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('paused');
      expect(res.body).toHaveProperty('strategy');
      expect(Array.isArray(res.body.q0)).toBe(true);
      expect(Array.isArray(res.body.q1)).toBe(true);
      expect(Array.isArray(res.body.q2)).toBe(true);
    });

    it('用例 28: POST /api/cluster/queue/strategy（T3）', async () => {
      const res = await request(server())
        .post('/api/cluster/queue/strategy')
        .send({ strategy: 'mlfq' });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('strategy');
    });

    it('用例 29: POST /api/cluster/queue/pause（T3，响应不读字段但须 JSON）', async () => {
      const res = await request(server()).post('/api/cluster/queue/pause');
      expect(res.status).toBe(200);
    });

    it('用例 30: POST /api/cluster/queue/resume（T3，响应不读字段但须 JSON）', async () => {
      const res = await request(server()).post('/api/cluster/queue/resume');
      expect(res.status).toBe(200);
    });

    it('用例 31: POST /api/cluster/queue/clear（T3）', async () => {
      const res = await request(server()).post('/api/cluster/queue/clear');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('cleared');
    });

    it('用例 32: DELETE /api/cluster/queue/task/:id 返回 JSON 非空体', async () => {
      const res = await request(server()).delete(
        '/api/cluster/queue/task/nonexistent-task',
      );
      // 对齐 api_server.py:5715-5732：不存在任务返回 200 + {success:false, message}（非 404）
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('success');
      expect(res.body).toHaveProperty('message');
      expect(res.headers['content-type']).toContain('application/json');
      expect(res.text).not.toBe(''); // 禁止 204 空体
    });

    it.skip('用例 33: GET /api/device/profile 画像字段（T4）', async () => {
      const res = await request(server()).get('/api/device/profile');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('hostname');
      expect(res.body).toHaveProperty('tier');
      expect(res.body).toHaveProperty('score_total');
    });

    it.skip('用例 34: POST /api/device/select-gpu（T4）', async () => {
      const res = await request(server())
        .post('/api/device/select-gpu')
        .send({ gpu_index: 0 });
      expect(res.status).toBe(200);
    });

    it.skip('用例 35: POST /api/device/auto-configure（T4）', async () => {
      const res = await request(server()).post('/api/device/auto-configure');
      expect(res.status).toBe(200);
    });

    it.skip('用例 36: GET /api/logs/recent?limit&level（T6，允许无 token）', async () => {
      const res = await request(server()).get(
        '/api/logs/recent?limit=50&level=INFO',
      );
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('logs');
      expect(Array.isArray(res.body.logs)).toBe(true);
    });

    it.skip('用例 37: GET /api/logs 文件列表（T6）', async () => {
      const res = await request(server()).get('/api/logs');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.files)).toBe(true);
    });

    it.skip('用例 38: GET /api/logs/stats（T6）', async () => {
      const res = await request(server()).get('/api/logs/stats');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('log_dir');
    });
  });

  // ============================================================
  // 用例 39-43：5 项细节断言 + 错误契约
  // ============================================================
  describe('5 项细节断言', () => {
    it('用例 39: 四个 DELETE 端点均返回 JSON 非空体（T3）', async () => {
      const paths = [
        '/api/cluster/queue/task/x',
        '/api/cluster/nodes/x',
        '/api/cluster/spare-master',
        '/api/cluster/layers',
      ];
      for (const p of paths) {
        const res = await request(server()).delete(p);
        expect(res.headers['content-type']).toContain('application/json');
        expect(res.text).not.toBe('');
      }
    });

    it.skip('用例 40: 数值字段类型必须为 number（T3/T5）', async () => {
      const status = await request(server()).get('/api/status');
      expect(typeof status.body.gpu?.utilization).toBe('number');
      expect(typeof status.body.device?.score).toBe('number');

      const nodes = await request(server()).get('/api/cluster/nodes');
      const first = nodes.body.nodes?.[0];
      if (first) {
        expect(typeof first.avg_rtt_ms).toBe('number');
      }

      const queue = await request(server()).get('/api/cluster/queue');
      for (const level of ['q0', 'q1', 'q2']) {
        const item = queue.body[level]?.[0];
        if (item) {
          expect(typeof item.wait_seconds).toBe('number');
        }
      }

      const current = await request(server()).get('/api/models/current');
      // 用 != null 同时放行 null 与 undefined（字段缺失时 JSON.stringify 会省略）
      if (current.body.gpu_allocated_gb != null) {
        expect(typeof current.body.gpu_allocated_gb).toBe('number');
      }
    });

    it('用例 41: /api/health JSON ok；无 token 日志请求返回 JSON 而非 302', async () => {
      // logs 段当前由 catch-all 404 JSON 兜底（非 302 + JSON 即过）；
      // T6 打开真端点后补 status=200 断言。
      const health = await request(server()).get('/api/health');
      expect(health.status).toBe(200);
      expect(health.body.status).toBe('ok');

      const logs = await request(server()).get('/api/logs/recent');
      expect(logs.status).not.toBe(302);
      expect(logs.headers['content-type']).toContain('application/json');
    });

    it.skip('用例 42: /api/status 含 gpu/kv_cache/device 嵌套对象（T5）', async () => {
      const res = await request(server()).get('/api/status');
      expect(res.body.gpu).toBeDefined();
      expect(res.body.kv_cache).toBeDefined();
      expect(res.body.device).toBeDefined();
    });

    it('用例 43: 错误响应体必须为 JSON detail', async () => {
      const res = await request(server()).get('/api/nonexistent-route');
      expect(res.status).toBe(404);
      expect(res.headers['content-type']).toContain('application/json');
      expect(res.body).toHaveProperty('detail');
      // 前端 client.js:84 读取 X-Request-ID 响应头（对齐 FastAPI request_id 中间件）
      expect(res.headers['x-request-id']).toBeDefined();
    });
  });
});
