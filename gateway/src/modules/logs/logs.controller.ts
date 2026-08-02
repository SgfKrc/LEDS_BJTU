/**
 * logs 域控制器（TUI 契约用例 36-38、41 logs 段）
 *
 * 透传代理：对外 /api/logs/* → legacy-control /logs/*（去 /api 前缀）。
 * 关键语义（docs/TUI适配实施计划.md §3.1 细节②⑤）：
 *   - 透传 X-QLH-Log-Token（可选：无 token 也允许请求，对齐 tui_admin.py:228-229）
 *   - 响应必须 JSON，禁止 302/204 空体
 */
import { All, Controller, NotFoundException, Req } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { LegacyControlClient } from '../../clients/legacy.client';

const LOGS_PREFIX = '/api/logs';

@Controller()
export class LogsController {
  constructor(private readonly legacy: LegacyControlClient) {}

  // /api/logs 与 /api/logs/* 都命中（TUI 调用 /api/logs 本身，tui_admin.py:1198）；
  // 同一方法叠加多个 @All 在 fastify adapter 下会被覆盖，故拆两个方法共享 proxyInner
  @All('logs')
  async proxyRoot(@Req() req: FastifyRequest): Promise<unknown> {
    return this.proxyInner(req);
  }

  @All('logs/*')
  async proxySub(@Req() req: FastifyRequest): Promise<unknown> {
    return this.proxyInner(req);
  }

  private async proxyInner(req: FastifyRequest): Promise<unknown> {
    const full = req.url; // 含 query，如 /api/logs 或 /api/logs/recent?limit=50&level=INFO
    if (full !== LOGS_PREFIX && !full.startsWith(LOGS_PREFIX + '/')) {
      throw new NotFoundException(`Route ${req.method}:${full} not found`);
    }
    // 内部端点路径 = 对外路径去掉 /api 前缀；/api/logs 本身 → /logs
    const subPath = full === LOGS_PREFIX ? '/logs' : full.slice('/api'.length);
    const rawToken = req.headers['x-qlh-log-token'];
    const token = Array.isArray(rawToken) ? rawToken[0] : rawToken;
    const extraHeaders: Record<string, string> = {};
    if (token) {
      extraHeaders['x-qlh-log-token'] = token;
    }
    const body = (req as { body?: unknown }).body;
    return this.legacy.request(req.method, subPath, body, extraHeaders);
  }
}
