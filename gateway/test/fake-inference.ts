/**
 * inference-svc 测试桩（T5 契约测试用）
 *
 * 内部契约对齐 docs/微服务架构改造计划.md §4.1：
 *   GET /v1/status → 模型/GPU/KV 缓存状态（数值字段 number）
 *   GET /v1/models/current → 对外形状对齐 api_server.py:2159-2178
 */
import http from 'http';
import type { AddressInfo } from 'net';

export interface FakeInference {
  port: number;
  requests: Array<{ method: string; path: string; body?: unknown }>;
  close(): Promise<void>;
}

const routes: Array<{ method: string; match: RegExp; data: unknown }> = [
  {
    method: 'GET',
    match: /^\/v1\/status$/,
    data: {
      model_loaded: true,
      current_quant: 'int4',
      model_name: 'Qwen/Qwen-1.8B-Chat',
      active_model_id: 'qwen-1_8b-chat',
      engine: 'torch',
      gpu: {
        name: 'NVIDIA GeForce RTX 3060',
        total_mb: 12288,
        allocated_mb: 1792.0,
        reserved_mb: 2048.0,
        utilization: 35.0,
      },
      kv_cache: {
        total_tokens: 512,
        max_tokens: 65536,
        allocated_pages: 4,
        free_pages: 124,
        max_pages: 128,
        page_size: 512,
        utilization: 0.0313,
        estimated_memory_mb: 64.0,
        rounds: 3,
        total_time_s: 12.5,
      },
    },
  },
  {
    method: 'GET',
    match: /^\/v1\/models\/current$/,
    data: {
      loaded: true,
      model_id: 'qwen-1_8b-chat',
      quant_type: 'int4',
      model_name: 'Qwen/Qwen-1.8B-Chat',
      model_path: 'models/qwen-1_8b-chat',
      engine: 'torch',
      total_params: 1842333696,
      device: 'cuda',
      gpu_allocated_gb: 1.75,
      gpu_reserved_gb: 2.0,
    },
  },
];

export async function startFakeInference(): Promise<FakeInference> {
  const requests: FakeInference['requests'] = [];
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const raw = Buffer.concat(chunks).toString('utf-8');
    let body: unknown;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      body = raw;
    }
    const path = (req.url || '/').split('?')[0];
    const method = req.method || 'GET';
    requests.push({ method, path, body });

    const route = routes.find((r) => r.method === method && r.match.test(path));
    const status = route ? 200 : 404;
    const data = route
      ? route.data
      : { detail: `fake inference: Route ${method}:${path} not found` };
    res.writeHead(status, { 'content-type': 'application/json' });
    res.end(JSON.stringify(data));
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  return {
    port,
    requests,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}
