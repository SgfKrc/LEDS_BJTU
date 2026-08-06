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

  /**
   * SSE 流式 chat 原始转发（/api/chat/stream → /v1/chat/stream）。
   * 返回未消费的 fetch Response，由控制器管道到客户端响应；支持中断。
   */
  async chatStreamRaw(
    body: unknown,
    signal: AbortSignal,
  ): Promise<Response> {
    return fetch(this.baseUrl + '/v1/chat/stream', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        accept: 'text/event-stream',
      },
      body: JSON.stringify(body ?? {}),
      signal,
    });
  }

  /** Binary-preserving fetch used by diffusion image blobs. */
  async diffusionBlobRaw(blobId: string): Promise<Response> {
    return fetch(
      this.baseUrl + `/v1/diffusion/blobs/${encodeURIComponent(blobId)}`,
      {
        method: 'GET',
        headers: { accept: 'image/png' },
        signal: AbortSignal.timeout(this.timeoutMs),
      },
    );
  }
}
