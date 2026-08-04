/**
 * 日志接口访问控制 — 对齐 api_server.py _require_log_api_access (6906-6928)
 *
 * L0 安全边界：日志可能包含隐私与调试细节，默认只允许本机访问。
 * 远程管理员访问需要显式配置 QLH_LOG_ADMIN_TOKEN，并在请求头传入
 * X-QLH-Log-Token。client-error 端点豁免（前端上报，无鉴权）。
 */
import {
  CanActivate,
  ExecutionContext,
  HttpException,
  Injectable,
} from '@nestjs/common';
import type { FastifyRequest } from 'fastify';

const LOCAL_LOG_CLIENTS = new Set([
  '127.0.0.1',
  '::1',
  '::ffff:127.0.0.1',
  'localhost',
  'testclient',
]);

@Injectable()
export class LogAccessGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    const req = context.switchToHttp().getRequest<FastifyRequest>();
    const host = (req.ip || 'unknown').toLowerCase();
    if (LOCAL_LOG_CLIENTS.has(host)) return true;

    const adminToken = (process.env.QLH_LOG_ADMIN_TOKEN || '').trim();
    const requestToken = String(req.headers['x-qlh-log-token'] ?? '').trim();
    if (adminToken && requestToken && requestToken === adminToken) {
      return true;
    }
    throw new HttpException('日志接口仅允许本机访问；远程访问需管理员授权', 403);
  }
}
