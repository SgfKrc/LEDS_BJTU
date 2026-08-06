/**
 * 全局异常过滤器 — 统一输出 JSON `detail` 结构
 *
 * 对齐 src/api_server.py:351-366（http_exception_with_request_id）：
 *   - HTTPException：status 透传，detail 取 message
 *   - 其他异常：500 + "服务器内部错误，请查看后端日志"
 *   - 响应体 {detail, request_id}，响应头 X-Request-ID
 * 这是 TUI 契约用例 43 与前端错误解析（client.js:84-91 读 detail）的硬依赖。
 */
import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  Logger,
} from '@nestjs/common';
import type { FastifyReply } from 'fastify';
import { normalizeRequestId, RequestWithRequestId } from './request-id';

@Catch()
export class JsonDetailFilter implements ExceptionFilter {
  private readonly logger = new Logger('Exception');

  catch(exception: unknown, host: ArgumentsHost): void {
    const ctx = host.switchToHttp();
    const res = ctx.getResponse<FastifyReply>();
    const req = ctx.getRequest<RequestWithRequestId>();
    // 拦截器未执行时（fastify 层提前拒绝：malformed JSON body、body 过大等），
    // req.requestId 为 undefined —— 对齐 Python 中间件（call_next 外层生成 uuid），
    // 此处 fallback 生成真实 request_id，而非回写 '-'
    // （api_server.py:319-341 在 call_next 外层 try/finally 中回写 X-Request-ID）。
    const requestId = req.requestId || normalizeRequestId(undefined);

    let status = 500;
    let detail: unknown = '服务器内部错误，请查看后端日志';
    if (exception instanceof HttpException) {
      status = exception.getStatus();
      const body = exception.getResponse();
      if (typeof body === 'string') {
        detail = body;
      } else if (body && typeof body === 'object') {
        const msg = (body as { message?: unknown }).message;
        detail = msg ?? body;
      }
    }

    if (status >= 500) {
      this.logger.error(
        `event=http_exception request_id=${requestId} method=${req.method}` +
          ` path=${req.url} status=${status}`,
        exception instanceof Error ? exception.stack : undefined,
      );
    }

    res
      .status(status)
      .header('X-Request-ID', requestId)
      .send({ detail, request_id: requestId });
  }
}
