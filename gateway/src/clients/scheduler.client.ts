/**
 * scheduler-svc 客户端
 *
 * 设计（docs/微服务架构改造计划.md §4.2 对外适配原则 + T3 落地决策）：
 *   - 网关对 /api/cluster/* 做 1:1 透传：内部端点路径 = 对外路径去掉 /api 前缀
 *   - scheduler-svc 按 docs/TUI适配实施计划.md §2.2 字段契约提供响应
 *   - 数值字段必须保持 number|null（契约用例 40），禁止字符串化
 *
 * 配置：QLH_SCHEDULER_URL（默认 http://127.0.0.1:8020）
 */
import { ForwardClient } from './forward-client';

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

export class SchedulerClient extends ForwardClient {
  constructor(baseUrl?: string, timeoutMs?: number) {
    super(
      baseUrl || process.env.QLH_SCHEDULER_URL || 'http://127.0.0.1:8020',
      timeoutMs,
    );
  }
}
