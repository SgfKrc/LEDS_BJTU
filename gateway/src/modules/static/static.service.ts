/**
 * 静态前端托管服务（阶段 2.3）
 *
 * 对齐 api_server.py:7561-7571 的 FastAPI StaticFiles 语义：
 *   - GET / 返回 frontend/dist/index.html
 *   - GET /assets/* 等静态资源按文件返回
 *   - 未命中文件的 GET 回退 index.html（SPA 路由，等价 StaticFiles(html=True)）
 *
 * 调用方：CatchAllController（@All('*') 兜底路由）——GET 先尝试静态托管，
 * 其余方法或 /api/* 保持 JSON 404 detail 语义。
 */
import { Injectable } from '@nestjs/common';
import type { FastifyReply, FastifyRequest } from 'fastify';
import * as fs from 'fs';
import * as path from 'path';
import * as process from 'process';
import { normalizeRequestId } from '../../common/request-id';

export function resolveFrontendDist(): string {
  const fromEnv = process.env.QLH_FRONTEND_DIST;
  if (fromEnv && fs.existsSync(fromEnv)) {
    return path.resolve(fromEnv);
  }
  // 默认：gateway/dist/modules/static 上溯 4 级 = 项目根/frontend/dist
  const fallback = path.resolve(__dirname, '..', '..', '..', '..', 'frontend', 'dist');
  return fallback;
}

@Injectable()
export class StaticService {
  private readonly root: string;

  constructor() {
    this.root = resolveFrontendDist();
  }

  /** 尝试以静态文件响应 GET；返回 true 表示已响应。 */
  async tryServe(req: FastifyRequest, reply: FastifyReply): Promise<boolean> {
    const urlPath = (req.url.split('?')[0] || '/').replace(/\/+$/, '') || '/';
    // /api/* 不做静态托管（保持 API JSON 404 语义）
    if (urlPath === '/api' || urlPath.startsWith('/api/')) {
      return false;
    }
    if (!fs.existsSync(this.root)) {
      return false;
    }
    const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '');
    // 路径穿越防御：解码后按段检查 ..（覆盖 /../、/%2e%2e/、..%2f 等变体）
    let decodedRel: string;
    try {
      decodedRel = decodeURIComponent(rel);
    } catch {
      this.sendNotFound(req, reply);
      return true;
    }
    if (decodedRel.split('/').includes('..')) {
      this.sendNotFound(req, reply);
      return true;
    }
    const resolved = path.resolve(this.root, decodedRel);
    const rootBase = path.resolve(this.root) + path.sep;
    if (resolved !== path.join(this.root, 'index.html') && !resolved.startsWith(rootBase)) {
      // 兜底：仍不在 root 内（防御双保险）
      this.sendNotFound(req, reply);
      return true;
    }
    let file = resolved;
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      file = path.join(this.root, 'index.html'); // SPA 回退（html=True 语义）
    }
    await reply.sendFile(path.relative(this.root, file), this.root);
    return true;
  }

  /** 以 JSON detail 404 响应（setNotFoundHandler / 路径穿越共用）。 */
  sendNotFound(req: FastifyRequest, reply: FastifyReply): void {
    const raw = req.headers['x-request-id'];
    const requestId = normalizeRequestId(Array.isArray(raw) ? raw[0] : raw);
    reply.header('X-Request-ID', requestId);
    reply.code(404).send({
      detail: `Route ${req.method}:${req.url} not found`,
      request_id: requestId,
    });
  }
}
