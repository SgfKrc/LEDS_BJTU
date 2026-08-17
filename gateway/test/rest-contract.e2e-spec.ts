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

    it('POST /api/chat/upload 文本文件解析（txt）', async () => {
      const res = await request(server())
        .post('/api/chat/upload')
        .attach('file', Buffer.from('第一行\n第二行\n第三行'), 'demo.txt');
      expect(res.status).toBe(200);
      expect(res.body.filename).toBe('demo.txt');
      expect(res.body.extension).toBe('.txt');
      expect(res.body.language).toBe('plaintext');
      expect(res.body.content).toBe('第一行\n第二行\n第三行');
      expect(res.body.line_count).toBe(3);
      expect(res.body.total_lines).toBe(3);
      expect(res.body.truncated).toBe(false);
      expect(res.body.size_bytes).toBeGreaterThan(0);
    });

    it('POST /api/chat/upload 语言检测（py → python）', async () => {
      const res = await request(server())
        .post('/api/chat/upload')
        .attach('file', Buffer.from('def main():\n    pass\n'), 'main.py');
      expect(res.status).toBe(200);
      expect(res.body.language).toBe('python');
      expect(res.body.extension).toBe('.py');
    });

    it('POST /api/chat/upload 不支持扩展名 → 400', async () => {
      const res = await request(server())
        .post('/api/chat/upload')
        .attach('file', Buffer.from('x'), 'demo.exe');
      expect(res.status).toBe(400);
      expect(res.body.detail).toContain('不支持的文件类型');
    });

    it('POST /api/chat/upload 超 5000 行截断', async () => {
      const content = Array.from({ length: 5100 }, (_, i) => `line-${i}`).join(
        '\n',
      );
      const res = await request(server())
        .post('/api/chat/upload')
        .attach('file', Buffer.from(content), 'big.log');
      expect(res.status).toBe(200);
      expect(res.body.truncated).toBe(true);
      expect(res.body.line_count).toBe(5000);
      expect(res.body.total_lines).toBe(5100);
      expect(res.body.truncated_lines).toBe(100);
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

    it('GET /api/models/local-assets 透传只读本地资产目录', async () => {
      const res = await request(server()).get('/api/models/local-assets');
      expect(res.status).toBe(200);
      expect(res.body.summary).toMatchObject({ total: 1 });
      expect(res.body.assets[0]).toMatchObject({ model_id: 'qwen3-4b' });
      expect(fakeInf?.requests.some(item => item.path === '/v1/models/local-assets')).toBe(true);
    });

    it('POST /api/models/unload 显式释放本地 LLM', async () => {
      const res = await request(server()).post('/api/models/unload');
      expect(res.status).toBe(200);
      expect(res.body).toMatchObject({ success: true, loaded: false });
      expect(fakeInf?.requests.some(item => item.path === '/v1/models/unload')).toBe(true);
    });
  });

  describe('diffusion 域', () => {
    it('GET capabilities 与 assets 走 inference-svc', async () => {
      const capabilities = await request(server()).get('/api/diffusion/capabilities');
      expect(capabilities.status).toBe(200);
      expect(capabilities.body.loaded).toBe(false);

      const artifacts = await request(server()).get('/api/diffusion/artifacts');
      expect(artifacts.status).toBe(200);
      expect(Array.isArray(artifacts.body.artifacts)).toBe(true);
    });

    it('本机可检查并登记 SD 路径', async () => {
      const inspected = await request(server())
        .post('/api/diffusion/artifacts/inspect')
        .send({ path: 'C:/models/sd15' });
      expect(inspected.status).toBe(200);
      expect(inspected.body.artifact_kind).toBe('sd15_pipeline');

      const registered = await request(server())
        .post('/api/diffusion/artifacts/register')
        .send({ path: 'C:/models/sd15', artifact_id: 'sd-local' });
      expect(registered.status).toBe(200);
      expect(registered.body.artifact_id).toBe('sd-local');
    });

    it('load/generate/job/cancel/unload 保持状态码和 JSON 契约', async () => {
      const loaded = await request(server())
        .post('/api/diffusion/load')
        .send({ artifact_id: 'sd-local', profile: 'balanced' });
      expect(loaded.status).toBe(200);
      expect(loaded.body.loaded).toBe(true);

      const generated = await request(server())
        .post('/api/diffusion/generate')
        .send({ preset_id: 'sd15_original_v1', seed: 9 });
      expect(generated.status).toBe(202);
      expect(generated.body.job_id).toBe('sdjob_test');

      const job = await request(server()).get('/api/diffusion/jobs/sdjob_test');
      expect(job.status).toBe(200);
      expect(job.body.state).toBe('completed');

      const cancelled = await request(server()).post(
        '/api/diffusion/jobs/sdjob_test/cancel',
      );
      expect(cancelled.status).toBe(200);
      expect(cancelled.body.accepted).toBe(true);

      const unloaded = await request(server()).post('/api/diffusion/unload');
      expect(unloaded.status).toBe(200);
      expect(unloaded.body.loaded).toBe(false);
    });

    it('PNG blob 保留二进制内容和 content-type', async () => {
      const blob = await request(server()).get('/api/diffusion/blobs/img_test');
      expect(blob.status).toBe(200);
      expect(blob.headers['content-type']).toContain('image/png');
      expect(Buffer.isBuffer(blob.body)).toBe(true);
      expect(blob.body.toString()).toBe('fake-png');

      const deleted = await request(server()).delete('/api/diffusion/blobs/img_test');
      expect(deleted.status).toBe(200);
      expect(deleted.body.deleted).toBe(true);
    });

    it('multipart 输入 blob、img2img、inpaint 与 instruction 经过网关', async () => {
      const uploaded = await request(server())
        .post('/api/diffusion/blobs')
        .field('purpose', 'input_image')
        .attach('file', Buffer.from('png-body'), {
          filename: 'source.png',
          contentType: 'image/png',
        });
      expect(uploaded.status).toBe(201);
      expect(uploaded.body).toMatchObject({
        blob_id: 'img_input',
        purpose: 'input_image',
      });
      const uploadRequest = fakeInf?.requests.find(
        item => item.method === 'POST' && item.path === '/v1/diffusion/blobs',
      );
      expect(String(uploadRequest?.body)).toContain('name="purpose"');
      expect(String(uploadRequest?.body)).toContain('input_image');

      const edit = await request(server())
        .post('/api/diffusion/edit')
        .send({
          mode: 'img2img',
          source_blob_id: 'img_input',
          prompt: 'sketch',
        });
      expect(edit.status).toBe(202);
      expect(edit.body).toMatchObject({
        job_id: 'sdedit_test',
        kind: 'edit',
      });

      const inpaint = await request(server())
        .post('/api/diffusion/edit')
        .send({
          mode: 'inpaint',
          source_blob_id: 'img_input',
          mask_blob_id: 'img_mask',
          edit_adapter_id: 'sd15_inpaint_v1',
          prompt: 'repair the selected area',
        });
      expect(inpaint.status).toBe(202);
      expect(inpaint.body).toMatchObject({ kind: 'edit' });

      const instruction = await request(server())
        .post('/api/diffusion/edit')
        .send({
          mode: 'instruction',
          source_blob_id: 'img_input',
          instruction: 'turn the car red',
          edit_adapter_id: 'sd15_instruct_pix2pix_v1',
          image_guidance_scale: 1.0,
        });
      expect(instruction.status).toBe(202);
      expect(instruction.body).toMatchObject({ kind: 'edit' });
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
