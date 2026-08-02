/**
 * legacy-control 客户端（阶段 2 控制面遗留进程，见 docs/TUI适配实施计划.md §3.3）
 *
 * 配置：QLH_LEGACY_CONTROL_URL（默认 http://127.0.0.1:8040）。
 * 内部端点：/logs/*（对外 /api/logs/* 去掉 /api 前缀后透传）。
 */
import { ForwardClient } from './forward-client';

export class LegacyControlClient extends ForwardClient {
  constructor(baseUrl?: string, timeoutMs?: number) {
    super(
      baseUrl ||
        process.env.QLH_LEGACY_CONTROL_URL ||
        'http://127.0.0.1:8040',
      timeoutMs,
    );
  }
}
