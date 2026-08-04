/**
 * 控制面域渐进切换测试（阶段 3.2 → 3.4 过渡）
 *
 * 设置 QLH_CONTROL_URL 指向真实 control-svc 进程后：
 *  - 已迁移域（sessions / conversations / user/settings / logs / workflows /
 *    bootstrap / models registry / gguf / download / cluster/review）→
 *    control-svc 真实响应
 *  - 未迁移域（presets / db/health / models downloadable / files）→
 *    legacy-control 桩（并行共存基线）
 * 不设置 QLH_CONTROL_URL 的行为由 rest-contract/tui-contract 既有测试锁定。
 *
 * control-svc 以随机端口 + 临时数据目录启动（QLH_*_FILE/DIR 全部隔离，
 * 不触碰仓库运行时数据）。
 */
import request from 'supertest';
import { spawn, type ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
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

async function startControlSvc(dataDir: string): Promise<{ port: number; close(): Promise<void> }> {
  const port = 30000 + Math.floor(Math.random() * 20000);
  const proc: ChildProcess = spawn('node', ['control/dist/main.js'], {
    cwd: REPO_ROOT,
    env: {
      ...process.env,
      QLH_CONTROL_PORT: String(port),
      QLH_DB_ENABLED: '0', // ConfigDao 默认启用 DB；测试隔离强制降级路径
      QLH_CHAT_HISTORY_DIR: path.join(dataDir, 'chat_history'),
      QLH_LOG_DIR: path.join(dataDir, 'logs'),
      QLH_REVIEW_STORE: path.join(dataDir, 'review_tickets.json'),
      QLH_MODEL_REGISTRY_FILE: path.join(dataDir, 'model_registry.json'),
      QLH_WORKFLOW_JOURNAL_FILE: path.join(dataDir, 'workflow_journal.json'),
      QLH_MODELS_DIR: path.join(dataDir, 'models'),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const actualPort = await new Promise<string>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('control-svc 启动超时（20s）')),
      20000,
    );
    let buf = '';
    proc.stdout?.on('data', (d: Buffer) => {
      buf += d.toString();
      const m = buf.match(/CONTROL_SVC_LISTENING:(\d+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
    proc.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`control-svc 提前退出 code=${code}`));
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

describe('控制面域渐进切换（QLH_CONTROL_URL → control-svc）', () => {
  let app: NestFastifyApplication | undefined;
  let fake: Awaited<ReturnType<typeof startFakeScheduler>> | undefined;
  let fakeInf: Awaited<ReturnType<typeof startFakeInference>> | undefined;
  let legacy: { port: number; close(): Promise<void> } | undefined;
  let control: { port: number; close(): Promise<void> } | undefined;
  let dataDir: string;

  beforeAll(async () => {
    dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-svc-switch-'));
    fs.mkdirSync(path.join(dataDir, 'models'), { recursive: true });
    fake = await startFakeScheduler();
    fakeInf = await startFakeInference();
    legacy = await startLegacyControl();
    control = await startControlSvc(dataDir);
    process.env.QLH_SCHEDULER_URL = `http://127.0.0.1:${fake.port}`;
    process.env.QLH_INFERENCE_URL = `http://127.0.0.1:${fakeInf.port}`;
    process.env.QLH_LEGACY_CONTROL_URL = `http://127.0.0.1:${legacy.port}`;
    process.env.QLH_CONTROL_URL = `http://127.0.0.1:${control.port}`;
    app = await createApp();
    await app.init();
    await (app.getHttpAdapter().getInstance() as any).ready();
  });

  afterAll(async () => {
    delete process.env.QLH_SCHEDULER_URL;
    delete process.env.QLH_INFERENCE_URL;
    delete process.env.QLH_LEGACY_CONTROL_URL;
    delete process.env.QLH_CONTROL_URL;
    await app?.close();
    await fake?.close();
    await fakeInf?.close();
    await legacy?.close();
    await control?.close();
    fs.rmSync(dataDir, { recursive: true, force: true });
  });

  const server = () => app!.getHttpServer();

  describe('已迁移域 → control-svc 真实响应', () => {
    it('GET /api/user/settings → control-svc（DB 禁用 → source:none）', async () => {
      const res = await request(server()).get('/api/user/settings');
      expect(res.status).toBe(200);
      expect(res.body).toEqual({ settings: {}, source: 'none' });
    });

    it('GET /api/sessions → control-svc 空列表结构', async () => {
      const res = await request(server()).get('/api/sessions');
      expect(res.status).toBe(200);
      expect(res.body).toEqual({
        sessions: [],
        active_session_id: null,
        total: 0,
      });
    });

    it('GET /api/logs/recent → control-svc（buffer 结构）', async () => {
      const res = await request(server()).get('/api/logs/recent');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.logs)).toBe(true);
      expect(res.body).toHaveProperty('buffer_capacity');
      expect(res.body).toHaveProperty('total_seen');
    });

    it('GET /api/workflows → control-svc（enabled/templates）', async () => {
      const res = await request(server()).get('/api/workflows');
      expect(res.status).toBe(200);
      expect(res.body.enabled).toBe(true);
      expect(res.body.templates).toEqual(['dual_candidate']);
    });

    it('GET /api/bootstrap/info → control-svc（发现契约）', async () => {
      const res = await request(server()).get('/api/bootstrap/info');
      expect(res.status).toBe(200);
      expect(res.body.status).toBe('ok');
      expect(res.body.is_master).toBe(true);
    });

    it('GET /api/cluster/review/tickets → control-svc（空列表）', async () => {
      const res = await request(server()).get('/api/cluster/review/tickets');
      expect(res.status).toBe(200);
      expect(res.body).toEqual({ tickets: [], count: 0 });
    });

    it('GET /api/models/registry → control-svc（空注册表）', async () => {
      const res = await request(server()).get('/api/models/registry');
      expect(res.status).toBe(200);
      expect(res.body).toEqual({ models: [] });
    });

    it('POST /api/sessions 创建 → control-svc 真实 uuid 会话', async () => {
      const res = await request(server())
        .post('/api/sessions')
        .send({ title: '切换测试' });
      expect(res.status).toBe(200);
      expect(res.body.title).toBe('切换测试');
      expect(res.body.id).toMatch(/^[0-9a-f-]{36}$/);
      // 数据落 control-svc 的隔离目录
      expect(
        fs.existsSync(
          path.join(dataDir, 'chat_history', '_sessions.json'),
        ),
      ).toBe(true);
    });
  });

  describe('未迁移域 → legacy-control 桩（基线不变）', () => {
    it('GET /api/presets → legacy 桩', async () => {
      const res = await request(server()).get('/api/presets');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('presets');
    });

    it('GET /api/models/downloadable → legacy 桩', async () => {
      const res = await request(server()).get('/api/models/downloadable');
      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('downloadable');
    });

    it('GET /api/db/health → legacy 桩', async () => {
      const res = await request(server()).get('/api/db/health');
      expect(res.status).toBe(200);
    });
  });
});
