/**
 * 请求 ID 基础设施
 *
 * 对齐 src/api_server.py:310-316（_normalize_request_id）与 :319-348
 * （request_id_logging_middleware）的语义：
 *   - X-Request-ID 头读取，非法字符剔除，空值回退 uuid4，截断 64 字符
 *   - 响应头回写 X-Request-ID
 *   - event=http_request 结构化日志（含耗时）
 */
import {
  CallHandler,
  ExecutionContext,
  HttpException,
  Injectable,
  Logger,
  NestInterceptor,
} from '@nestjs/common';
import { randomUUID } from 'crypto';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { Observable, catchError, tap } from 'rxjs';

export interface RequestWithRequestId extends FastifyRequest {
  requestId?: string;
}

export function normalizeRequestId(value: string | undefined): string {
  if (!value) {
    return randomUUID();
  }
  // 对齐 Python：re.sub(r"[^A-Za-z0-9_.:-]", "", value.strip())
  const cleaned = value.trim().replace(/[^A-Za-z0-9_.:-]/g, '');
  if (!cleaned) {
    return randomUUID();
  }
  return cleaned.slice(0, 64);
}

@Injectable()
export class RequestIdInterceptor implements NestInterceptor {
  private readonly logger = new Logger('HTTP');

  intercept(context: ExecutionContext, next: CallHandler): Observable<unknown> {
    const ctx = context.switchToHttp();
    const req = ctx.getRequest<RequestWithRequestId>();
    const res = ctx.getResponse<FastifyReply>();
    const raw = req.headers['x-request-id'];
    const requestId = normalizeRequestId(
      Array.isArray(raw) ? raw[0] : raw,
    );
    req.requestId = requestId;
    res.header('X-Request-ID', requestId);

    const start = Date.now();
    return next.handle().pipe(
      tap(() => this.log(req, res.statusCode, requestId, start)),
      catchError((err: unknown) => {
        this.log(req, this.statusOf(err), requestId, start, err);
        throw err;
      }),
    );
  }

  private statusOf(err: unknown): number {
    if (err instanceof HttpException) {
      return err.getStatus();
    }
    return 500;
  }

  private log(
    req: RequestWithRequestId,
    status: number,
    requestId: string,
    start: number,
    err?: unknown,
  ): void {
    const durationMs = Date.now() - start;
    const line =
      `event=http_request request_id=${requestId} method=${req.method}` +
      ` path=${req.url} status=${status} duration_ms=${durationMs}`;
    if (err && status >= 500) {
      this.logger.error(
        line,
        err instanceof Error ? err.stack : undefined,
      );
    } else {
      this.logger.log(line);
    }
  }
}
