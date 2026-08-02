/**
 * 内部服务 HTTP 转发客户端（SchedulerClient / InferenceClient 的公共基类）
 *
 * 语义对齐 tui_admin.py ApiClient 的容错行为：
 *   - 2xx 空体 → {}；非 JSON → {detail: text}
 *   - 非 2xx → HttpException(status, detail)（由全局 JsonDetailFilter 输出）
 *   - 上游不可达 → 502
 */
import { HttpException } from '@nestjs/common';

export abstract class ForwardClient {
  protected constructor(
    protected readonly baseUrl: string,
    protected readonly timeoutMs = 15000,
  ) {}

  async request(method: string, path: string, body?: unknown): Promise<unknown> {
    const headers: Record<string, string> = { accept: 'application/json' };
    let payload: string | undefined;
    if (body !== undefined) {
      payload = JSON.stringify(body);
      headers['content-type'] = 'application/json';
    }

    let res: Response;
    try {
      res = await fetch(this.baseUrl + path, {
        method,
        headers,
        body: payload,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err) {
      throw new HttpException(
        `上游服务不可达（${this.baseUrl}）: ${
          err instanceof Error ? err.message : String(err)
        }`,
        502,
      );
    }

    const text = await res.text();
    let data: unknown = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { detail: text };
      }
    }
    if (!res.ok) {
      const detail =
        (data as { detail?: unknown } | null)?.detail ??
        `upstream ${res.status}`;
      throw new HttpException(detail, res.status);
    }
    return data;
  }
}
