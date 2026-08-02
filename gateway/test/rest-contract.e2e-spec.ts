/**
 * 阶段 2 其余域契约测试（非 TUI 域）
 *
 * 覆盖：chat（含 SSE 流式）、models（load/switch/available）、
 *       experimental/speculative、控制面域（sessions/conversations/settings/
 *       review/workflows/bootstrap/models registry/downloadable/gguf/files/download
 *       → legacy-control 桩）。
 * 与 tui-contract.e2e-spec.ts 共享同一套桩（fake-scheduler / fake-inference /
 * src/legacy_control.py），但独立文件、独立测试实例。
 */
import request from 'supertest';
import { spawn, type ChildProcess } from 'child_process';
import path from 'path';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { startFakeScheduler } from './fake-scheduler';
import { startFakeInference } from './fake-inference';

const REPO_ROOT = path.resolve(__dirname, '..', '..');

async function startLegacyControl(): Promise<{ port: number; close(): Promise<void> }> {
  const port = 20000 + Math.floor(Math.random() * 30000);
  const proc: ChildProcess = spawn(
    'python',
    ['src/legacy_control.py', '--port', String(port)],
    { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  const actualPort = await new Promise<string>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('legacy-control 启动超时（15s）')),
      15000,
    );
    let buf = '';
    proc.stdout?.on('data', (d: Buffer) => {
      buf += d.toString();
      const m = buf.match(/LEGACY_CONTROL_LISTENING:(\d+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
    proc.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`legacy-control 提前退出 code=${code}`));
    });
    proc.stderr?.on('data', (d: Buffer) => {
      buf += d.toString();
    });
  });
  return {
    port: Number(actualPort),
    close: () =>
      new Promise<void>((resolve) => {
        proc.kill();
        proc.on('exit', () => resolve());
      }),
  };
}

describe('阶段 2 其余域契约', () => {
  let app: NestFastifyApplication | undefined;
  let fake: Awaited<ReturnType<typeof startFakeScheduler>> | undefined;
  let fakeInf: Awaited<ReturnType<typeof startFakeInference>> | undefined;
  let legacy: { port: number; close(): Promise<void> } | undefined;

  beforeAll(async () => {
    fake = await startFakeScheduler();
    fakeInf = await startFakeInference();
    legacy = await startLegacyControl();
    process.env.QLH_SCHEDULER_URL = `http://127.0.0.1:${fake.port}`;
    process.env.QLH_INFERENCE_URL = `http://127.0.0.1:${fakeInf.port}`;
    process.env.QLH_LEGACY_CONTROL_URL = `http://127.0.0.1:${legacy.port}`;
    app = await createApp();
    await app.init();
    await (app.getHttpAdapter().getInstance() as any).ready();
  });

  afterAll(async () => {
    delete process.env.QLH_SCHEDULER_URL;
    delete process.env.QLH_INFERENCE_URL;
    delete process.env.QLH_LEGACY_CONTROL_URL;
    await app?.close();
    await fake?.close();
    await fakeInf?.close();
    await legacy?.close();
  });

  const server = () => app!.getHttpServer();

  describe('chat 域', () => {
    it('POST /api/chat 完整对话（→ inference /v1/chat）', async () => {
      const res = await request(server())
        .post('/api/chat')
        .send({ message: '你好', session_id: 's1' });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('reply');
    });

    it('POST /api/chat/stream SSE 逐字节透传（含 data 事件与 [DONE]）', async () => {
      const res = await request(server())
        .post('/api/chat/stream')
        .send({ message: '你好' })
        .timeout({ response: 10000 });
      expect(res.status).toBe(200);
      expect(res.headers['content-type']).toContain('text/event-stream');
      expect(res.text).toContain('data: {"delta":"你"}');
      expect(res.text).toContain('data: [DONE]');
    });

    it('POST /api/chat/clear 清会话历史', async () => {
      const res = await request(server()).post('/api/chat/clear');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('cleared');
    });

    it('POST /api/chat/generations/:id/cancel 取消生成', async () => {
      const res = await request(server()).post(
        '/api/chat/generations/gen-1/cancel',
      );
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });
  });

  describe('models 域', () => {
    it('POST /api/models/load', async () => {
      const res = await request(server())
        .post('/api/models/load')
        .send({ engine: 'torch', quant_type: 'int4' });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('POST /api/models/switch', async () => {
      const res = await request(server())
        .post('/api/models/switch')
        .send({ model_id: 'qwen-1_8b-chat-gguf' });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('GET /api/models/available', async () => {
      const res = await request(server()).get('/api/models/available');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.models)).toBe(true);
    });

    it('GET /api/models 模型配置列表（前端 fetchModels 依赖，含 active_model_id）', async () => {
      const res = await request(server()).get('/api/models');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.models)).toBe(true);
      expect(res.body).toHaveProperty('active_model_id');
    });
  });

  describe('experimental 域', () => {
    it('POST /api/experimental/speculative → inference /v1/speculative/run', async () => {
      const res = await request(server())
        .post('/api/experimental/speculative')
        .send({ prompt: 'test' });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('accepted');
    });
  });

  describe('控制面域（legacy-control 桩）', () => {
    it('GET /api/sessions 空列表', async () => {
      const res = await request(server()).get('/api/sessions');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.sessions)).toBe(true);
    });

    it('POST /api/sessions 创建', async () => {
      const res = await request(server())
        .post('/api/sessions')
        .send({ title: 't' });
      expect(res.status).toBe(200);
    });

    it('GET /api/review/tickets 空列表', async () => {
      const res = await request(server()).get('/api/review/tickets');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.tickets)).toBe(true);
    });

    it('POST /api/review/create', async () => {
      const res = await request(server())
        .post('/api/review/create')
        .send({ target_node_id: 'n1', reason: 'r' });
      expect(res.status).toBe(200);
    });

    it('GET /api/workflows 空列表', async () => {
      const res = await request(server()).get('/api/workflows');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.workflows)).toBe(true);
    });

    it('GET /api/settings', async () => {
      const res = await request(server()).get('/api/settings');
      expect(res.status).toBe(200);
    });

    it('GET /api/bootstrap/info', async () => {
      const res = await request(server()).get('/api/bootstrap/info');
      expect(res.status).toBe(200);
    });

    it('GET /api/presets 预设列表（前端 ChatPanel 依赖）', async () => {
      const res = await request(server()).get('/api/presets');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.presets)).toBe(true);
      expect(res.body.presets[0]).toHaveProperty('label');
      expect(res.body.presets[0]).toHaveProperty('question');
      expect(res.body).toHaveProperty('current_speed_tok_s');
    });

    it('GET /api/user/settings（数据库不可用时空 settings）', async () => {
      const res = await request(server()).get('/api/user/settings');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('settings');
    });

    it('PUT /api/user/settings 保存设置', async () => {
      const res = await request(server())
        .put('/api/user/settings')
        .send({ settings: { theme: 'dark' } });
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('GET /api/db/health 数据库健康', async () => {
      const res = await request(server()).get('/api/db/health');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('status');
    });

    it('GET /api/models/registry 空列表', async () => {
      const res = await request(server()).get('/api/models/registry');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.registry)).toBe(true);
    });

    it('POST /api/models/registry 注册', async () => {
      const res = await request(server())
        .post('/api/models/registry')
        .send({ model_id: 'm1' });
      expect(res.status).toBe(200);
    });

    it('GET /api/models/downloadable 空列表', async () => {
      const res = await request(server()).get('/api/models/downloadable');
      expect(res.status).toBe(200);
    });

    it('GET /api/models/gguf', async () => {
      const res = await request(server()).get('/api/models/gguf');
      expect(res.status).toBe(200);
    });
  });
});
