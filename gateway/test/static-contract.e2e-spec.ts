/**
 * 阶段 2.3 静态前端托管契约测试
 *
 * 对齐 api_server.py:7569-7571 的 FastAPI StaticFiles(html=True) 语义：
 *   - GET /            → index.html（text/html）
 *   - GET /assets/*    → 真实静态资源（按文件返回）
 *   - GET 未命中文件   → SPA 回退 index.html（前端路由）
 *   - GET /api/*       → 不落入静态托管（API 404 保持 JSON detail）
 *   - 非 GET           → 不静态托管（404 JSON）
 *
 * 无需后端桩：仅验证网关自身的静态托管与路由优先级（/api/health 内嵌）。
 * 前置条件：frontend/dist 已构建（npm run build），否则测试 skip。
 */
import request from 'supertest';
import * as fs from 'fs';
import * as path from 'path';
import { createApp, resolveFrontendDist } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';

const REPO_ROOT = path.resolve(__dirname, '..', '..');

describe('阶段 2.3 静态前端托管', () => {
  let app: NestFastifyApplication | undefined;
  let dist: string;

  beforeAll(async () => {
    dist = resolveFrontendDist();
    if (!fs.existsSync(dist)) {
      // eslint-disable-next-line no-console
      console.warn(`frontend/dist 不存在（${dist}），静态托管用例 skip——先执行 npm run build`);
      return;
    }
    process.env.QLH_FRONTEND_DIST = dist;
    app = await createApp();
    await app.init();
    await (app.getHttpAdapter().getInstance() as any).ready();
  });

  afterAll(async () => {
    delete process.env.QLH_FRONTEND_DIST;
    await app?.close();
  });

  const server = () => app!.getHttpServer();
  const hasDist = () => fs.existsSync(dist);

  it('GET / 返回 index.html（text/html，含 #root）', async () => {
    if (!hasDist()) return;
    const res = await request(server()).get('/');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toContain('text/html');
    expect(res.text).toContain('root');
  });

  it('GET /assets/* 按真实文件返回（JS 资源，content-type 正确）', async () => {
    if (!hasDist()) return;
    const assets = fs.readdirSync(path.join(dist, 'assets'));
    const js = assets.find((f) => f.endsWith('.js'));
    if (!js) return; // 无 JS 资源时跳过（构建异常情况）
    const res = await request(server()).get(`/assets/${js}`);
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toContain('javascript');
    expect(res.text.length).toBeGreaterThan(1000);
  });

  it('GET 未命中文件 → SPA 回退 index.html（前端路由）', async () => {
    if (!hasDist()) return;
    const res = await request(server()).get('/some/frontend/route');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toContain('text/html');
    expect(res.text).toContain('root');
  });

  it('GET /api/health 仍由 API 处理（不被静态托管吞掉）', async () => {
    if (!hasDist()) return;
    const res = await request(server()).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: 'ok' });
  });

  it('GET /api/未匹配路由 → JSON 404 detail（API 语义不被 SPA 回退破坏）', async () => {
    if (!hasDist()) return;
    const res = await request(server()).get('/api/nonexistent-route');
    expect(res.status).toBe(404);
    expect(res.body).toHaveProperty('detail');
  });

  it('路径穿越（/../）→ 404', async () => {
    if (!hasDist()) return;
    const res = await request(server()).get('/..%2f..%2fpackage.json');
    expect(res.status).toBe(404);
  });

  it('POST / → 不静态托管（404 JSON）', async () => {
    if (!hasDist()) return;
    const res = await request(server()).post('/');
    expect(res.status).toBe(404);
    expect(res.body).toHaveProperty('detail');
  });
});
