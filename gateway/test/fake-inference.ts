/**
 * inference-svc 测试桩（T5/T6 + 阶段 2 其余域用）
 *
 * 内部契约对齐 docs/微服务架构改造计划.md §4.1：
 *   GET  /v1/status、/v1/models/current、/v1/models/available
 *   POST /v1/chat、/v1/chat/stream(SSE)、/v1/chat/cancel、/v1/chat/clear、
 *        /v1/models/load、/v1/models/switch、/v1/speculative/run
 */
import http from 'http';
import type { AddressInfo } from 'net';

export interface FakeInference {
  port: number;
  requests: Array<{ method: string; path: string; body?: unknown }>;
  close(): Promise<void>;
}

const SSE_BODY = [
  'data: {"delta":"你"}',
  '',
  'data: {"delta":"好"}',
  '',
  'data: [DONE]',
  '',
].join('\n');

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
  {
    method: 'GET',
    match: /^\/v1\/models\/local-assets$/,
    data: {
      assets: [{ model_id: 'qwen3-4b', available_formats: ['safetensors', 'gguf'] }],
      summary: { total: 1, total_bytes: 1 },
    },
  },
  {
    method: 'GET',
    match: /^\/v1\/models\/available$/,
    data: {
      models: [
        { model_id: 'qwen-1_8b-chat', name: 'Qwen 1.8B Chat', engine: 'torch' },
        { model_id: 'qwen-1_8b-chat-gguf', name: 'Qwen 1.8B Chat GGUF', engine: 'llama_cpp' },
      ],
    },
  },
  {
    method: 'GET',
    match: /^\/v1\/models$/,
    data: {
      models: [
        { model_id: 'qwen-1_8b-chat', name: 'Qwen 1.8B Chat', engine: 'torch' },
        { model_id: 'qwen-1_8b-chat-gguf', name: 'Qwen 1.8B Chat GGUF', engine: 'llama_cpp' },
      ],
      active_model_id: 'qwen-1_8b-chat',
    },
  },
  {
    method: 'POST',
    match: /^\/v1\/chat$/,
    data: { reply: '桩回复：你好，我是 QLH。', session_id: 'stub-session' },
  },
  {
    method: 'POST',
    match: /^\/v1\/chat\/clear$/,
    data: { cleared: true },
  },
  {
    method: 'POST',
    match: /^\/v1\/chat\/cancel$/,
    data: { status: 'cancelled', generation_id: 'stub-gen' },
  },
  {
    method: 'POST',
    match: /^\/v1\/models\/load$/,
    data: { status: 'loaded', model_id: 'qwen-1_8b-chat', engine: 'torch' },
  },
  {
    method: 'POST',
    match: /^\/v1\/models\/unload$/,
    data: { success: true, loaded: false, unloaded: true, message: '模型已卸载' },
  },
  {
    method: 'POST',
    match: /^\/v1\/models\/switch$/,
    data: { status: 'switched', model_id: 'qwen-1_8b-chat-gguf', engine: 'llama_cpp' },
  },
  {
    method: 'POST',
    match: /^\/v1\/speculative\/run$/,
    data: { accepted: 0, drafted: 0, verified: 1, note: 'stub' },
  },
  {
    method: 'GET',
    match: /^\/v1\/diffusion\/capabilities$/,
    data: { state: 'unloaded', loaded: false, presets: [{ preset_id: 'sd15_original_v1' }] },
  },
  {
    method: 'POST',
    match: /^\/v1\/diffusion\/artifacts\/inspect$/,
    data: { path: 'C:/models/sd15', artifact_kind: 'sd15_pipeline', loadable: true },
  },
  {
    method: 'POST',
    match: /^\/v1\/diffusion\/artifacts\/register$/,
    data: { artifact_id: 'sd-local', name: 'SD local', artifact: { artifact_kind: 'sd15_pipeline' } },
  },
  {
    method: 'GET',
    match: /^\/v1\/diffusion\/artifacts$/,
    data: { artifacts: [{ artifact_id: 'sd-local', artifact: { artifact_kind: 'sd15_pipeline' } }] },
  },
  {
    method: 'POST',
    match: /^\/v1\/diffusion\/load$/,
    data: { state: 'loaded', loaded: true },
  },
  {
    method: 'POST',
    match: /^\/v1\/diffusion\/unload$/,
    data: { state: 'unloaded', loaded: false },
  },
  {
    method: 'POST',
    match: /^\/v1\/diffusion\/generate$/,
    data: { job_id: 'sdjob_test', state: 'queued' },
  },
  {
    method: 'GET',
    match: /^\/v1\/diffusion\/jobs\/sdjob_test$/,
    data: { job_id: 'sdjob_test', state: 'completed', blob: { blob_id: 'img_test' } },
  },
  {
    method: 'POST',
    match: /^\/v1\/diffusion\/jobs\/sdjob_test\/cancel$/,
    data: { accepted: true, job: { job_id: 'sdjob_test', state: 'running' } },
  },
  {
    method: 'DELETE',
    match: /^\/v1\/diffusion\/blobs\/img_test$/,
    data: { deleted: true, blob_id: 'img_test' },
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

    // SSE 流式端点：text/event-stream 逐行透传
    if (method === 'POST' && path === '/v1/chat/stream') {
      res.writeHead(200, {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      res.end(SSE_BODY);
      return;
    }

    if (method === 'GET' && path === '/v1/diffusion/blobs/img_test') {
      res.writeHead(200, {
        'content-type': 'image/png',
        'cache-control': 'private, no-store',
        etag: '"fake-image"',
      });
      res.end(Buffer.from('fake-png'));
      return;
    }

    if (method === 'POST' && path === '/v1/diffusion/blobs') {
      res.writeHead(201, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        blob_id: 'img_input',
        purpose: 'input_image',
        content_type: 'image/png',
        size_bytes: 8,
        width: 8,
        height: 8,
      }));
      return;
    }

    if (method === 'POST' && path === '/v1/diffusion/edit') {
      const mode = body && typeof body === 'object'
        ? (body as { mode?: unknown }).mode
        : undefined;
      if (mode === 'img2img' || mode === 'inpaint' || mode === 'instruction') {
        res.writeHead(202, { 'content-type': 'application/json' });
        res.end(JSON.stringify({
          job_id: 'sdedit_test',
          state: 'queued',
          kind: 'edit',
        }));
        return;
      }
      res.writeHead(501, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        detail: {
          code: 'DIFFUSION_UNSUPPORTED',
          message: 'edit executor is not installed',
        },
      }));
      return;
    }

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
