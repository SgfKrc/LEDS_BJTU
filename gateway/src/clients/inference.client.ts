/**
 * inference-svc 客户端
 *
 * 配置：QLH_INFERENCE_URL（默认 http://127.0.0.1:8010）。
 * 内部契约见 docs/微服务架构改造计划.md §4.1（/v1/status、/v1/models/current 等）。
 */
import { ForwardClient } from './forward-client';

export class InferenceClient extends ForwardClient {
  constructor(baseUrl?: string, timeoutMs?: number) {
    super(
      baseUrl || process.env.QLH_INFERENCE_URL || 'http://127.0.0.1:8010',
      timeoutMs,
    );
  }
}
