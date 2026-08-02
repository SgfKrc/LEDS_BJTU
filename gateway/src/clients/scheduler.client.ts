/**
 * scheduler-svc 客户端（HTTP 透传代理）
 *
 * 设计（docs/微服务架构改造计划.md §4.2 对外适配原则）：
 *   - 网关对 /api/cluster/* 做 1:1 透传：内部端点路径 = 对外路径去掉 /api 前缀
 *     （如 GET /api/cluster/nodes → GET /cluster/nodes）
 *   - scheduler-svc 按 docs/TUI适配实施计划.md §2.2 字段契约提供响应
 *   - 数值字段必须保持 number|null（契约用例 40），禁止字符串化
 *
 * 配置：QLH_SCHEDULER_URL（默认 http://127.0.0.1:8020）
 */
import { HttpException } from '@nestjs/common';

/** SchedulerApiTypes：对外响应契约的关键形状（TUI §2.2 字段），供开发期类型参考 */
export interface SchedulerApiTypes {
  // 数值字段（number | null 是硬契约，见用例 40）：
  //   nodes[].avg_rtt_ms、queue.q*/[].wait_seconds、status.device.score、
  //   status.gpu.utilization、models.current.gpu_allocated_gb
  node: {
    node_id: string;
    role: string;
    node_type: string;
    state: string;
    address: string;
    network_type: string;
    avg_rtt_ms: number | null;
    task_count: number;
    error_count: number;
    last_heartbeat: number | null;
  };
  queueItem: {
    task_id: string;
    original_level: number;
    wait_seconds: number;
    max_new_tokens: number;
    is_aged: boolean;
    session_id: string | null;
  };
}

export class SchedulerClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(baseUrl?: string, timeoutMs = 15000) {
    this.baseUrl =
      baseUrl || process.env.QLH_SCHEDULER_URL || 'http://127.0.0.1:8020';
    this.timeoutMs = timeoutMs;
  }

  /**
   * 透传请求到 scheduler-svc。
   * 非 2xx：抛 HttpException(status, detail)，由全局 JsonDetailFilter
   * 输出 {"detail", "request_id"}（对齐 FastAPI HTTPException 语义）。
   * 2xx 空体：返回 {}（对齐 tui_admin.py:254-255 的空响应容错）。
   */
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
        `scheduler-svc 不可达（${this.baseUrl}）: ${
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
        `scheduler upstream ${res.status}`;
      throw new HttpException(detail, res.status);
    }
    return data;
  }
}
