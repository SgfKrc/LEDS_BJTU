/**
 * scheduler-svc 测试桩（T3 契约测试用）
 *
 * 响应字段对齐 docs/TUI适配实施计划.md §2.2 全表；
 * DELETE 语义对齐 src/api_server.py 真实行为：
 *   - queue/task/{id} 不存在 → 200 + {success:false, message}（:5715-5732）
 *   - nodes/{id} 不存在 → 404 + detail（:5266-5285）
 * 数值字段一律 number（契约用例 40 的 fake 侧保证）。
 */
import http from 'http';
import type { AddressInfo } from 'net';

export interface FakeScheduler {
  port: number;
  requests: Array<{ method: string; path: string; body?: unknown }>;
  close(): Promise<void>;
}

type Handler = (body: unknown, path: string) => { status: number; data: unknown };

function json(status: number, data: unknown): { status: number; data: unknown } {
  return { status, data };
}

const routes: Array<{ method: string; match: RegExp; handler: Handler }> = [
  { method: 'GET', match: /^\/cluster\/my-role$/, handler: () => json(200, { is_master: true, is_provisional: false, runtime_node_role: 'master', node_role: 'master', node_id: 'test-master' }) },
  { method: 'GET', match: /^\/cluster\/nodes$/, handler: () => json(200, {
    count: 2, online_count: 1, offline_count: 1,
    nodes: [
      { node_id: 'test-master', role: 'master', node_type: 'pc', state: 'online', address: '100.64.0.1', network_type: 'tailscale', avg_rtt_ms: 12.5, task_count: 1, error_count: 0, last_heartbeat: 1754100000 },
      { node_id: 'test-client', role: 'client', node_type: 'pc', state: 'offline', address: '100.64.0.2', network_type: 'tailscale', avg_rtt_ms: null, task_count: 0, error_count: 2, last_heartbeat: 1754000000 },
    ],
  }) },
  { method: 'GET', match: /^\/cluster\/invite$/, handler: () => json(200, { master_host: '100.64.0.1', master_port: 8888, node_count: 2, max_nodes: 8, identity_verified: true }) },
  { method: 'GET', match: /^\/cluster\/spare-master$/, handler: () => json(200, { spare_master: { node_id: 'test-client', hostname: 'spare-pc', is_online: true } }) },
  { method: 'GET', match: /^\/cluster\/master-health$/, handler: () => json(200, { master_online: true, master_host: '100.64.0.1', master_port: 8888, last_seen_seconds_ago: 0, stale: false }) },
  { method: 'GET', match: /^\/cluster\/discover$/, handler: () => json(200, { found: true, master_host: '100.64.0.1', master_port: 8888, source: 'fake', stale: false }) },
  { method: 'POST', match: /^\/cluster\/connect$/, handler: () => json(200, { status: 'connected', message: '已连接主节点' }) },
  { method: 'POST', match: /^\/cluster\/nodes\/register$/, handler: () => json(200, { status: 'registered', reason: '', message: '注册成功' }) },
  { method: 'POST', match: /^\/cluster\/nodes\/[^/]+\/deregister$/, handler: () => json(200, { status: 'deregistered' }) },
  { method: 'DELETE', match: /^\/cluster\/nodes\/[^/]+$/, handler: () => json(404, { detail: '节点不存在' }) },
  { method: 'POST', match: /^\/cluster\/transfer-master$/, handler: () => json(200, { status: 'transferred', message: '主节点身份已转让' }) },
  { method: 'POST', match: /^\/cluster\/spare-master$/, handler: () => json(200, { status: 'set', message: '已设置备用主节点' }) },
  { method: 'DELETE', match: /^\/cluster\/spare-master$/, handler: () => json(200, { status: 'cleared', message: '已清除备用主节点' }) },
  { method: 'PUT', match: /^\/cluster\/config\/max-nodes$/, handler: () => json(200, { status: 'updated' }) },
  { method: 'GET', match: /^\/cluster\/transfer-logs$/, handler: () => json(200, { count: 1, logs: [{ direction: 'master->client', from_role: 'master', to_role: 'client', related_node: 'test-client', timestamp: 1754100000, outcome: 'ok' }] }) },
  { method: 'POST', match: /^\/cluster\/email-test$/, handler: () => json(200, { message: '测试邮件已发送', status: 'sent' }) },
  { method: 'POST', match: /^\/cluster\/reset-identity$/, handler: () => json(200, { status: 'reset' }) },
  { method: 'GET', match: /^\/cluster\/config\/distributed-inference$/, handler: () => json(200, { enabled: true, default: false }) },
  { method: 'PUT', match: /^\/cluster\/config\/distributed-inference$/, handler: () => json(200, { status: 'updated' }) },
  { method: 'GET', match: /^\/cluster\/layers$/, handler: () => json(200, { total: 24, strategy: 'graph', computed_at: 1754100000, assignments: [{ node_id: 'test-master', role: 'master', start_layer: 0, end_layer: 11, has_embedding: true, has_lm_head: false, score: 0.9 }] }) },
  { method: 'PUT', match: /^\/cluster\/layers$/, handler: () => json(200, { status: 'applied', message: '已应用分层' }) },
  { method: 'DELETE', match: /^\/cluster\/layers$/, handler: () => json(200, { status: 'cleared' }) },
  { method: 'GET', match: /^\/cluster\/config$/, handler: () => json(200, { network: { server_ip: '0.0.0.0', server_port: 8888, heartbeat_interval_s: 5 }, model: { quant_type: 'int4', page_size: 512, max_page_num: 100, max_seq_len: 8192 } }) },
  { method: 'GET', match: /^\/cluster\/queue$/, handler: () => json(200, {
    paused: false, strategy: 'mlfq', current_task: null, queue_size: 1, max_size: 100,
    q0_depth: 1, q1_depth: 0, q2_depth: 0, completed_count: 5,
    aging_params: { q0_max_tokens: 256, q1_max_tokens: 512, q1_to_q0_s: 30, q2_to_q1_s: 60 },
    preempt_stats: { count: 1, total_overhead_ms: 120 },
    q0: [{ task_id: 't1', original_level: 0, wait_seconds: 3.2, max_new_tokens: 256, is_aged: false, session_id: 's1' }],
    q1: [], q2: [],
  }) },
  { method: 'POST', match: /^\/cluster\/queue\/strategy$/, handler: () => json(200, { strategy: 'mlfq' }) },
  { method: 'POST', match: /^\/cluster\/queue\/pause$/, handler: () => json(200, { paused: true }) },
  { method: 'POST', match: /^\/cluster\/queue\/resume$/, handler: () => json(200, { paused: false }) },
  { method: 'POST', match: /^\/cluster\/queue\/clear$/, handler: () => json(200, { success: true, cleared: 3 }) },
  // 对齐 api_server.py:5715-5732：不存在任务返回 200 + success:false（非 404）
  { method: 'DELETE', match: /^\/cluster\/queue\/task\/[^/]+$/, handler: () => json(200, { success: false, task_id: 'nonexistent-task', message: '任务不存在或已经完成，无法取消' }) },
  // ---- device 域（TUI §2.2 #33-35；画像采集留 Python，字段对齐旧网关） ----
  { method: 'GET', match: /^\/device\/profile$/, handler: () => json(200, {
    os: { system: 'Windows', release: '10.0.22631' },
    hostname: 'test-pc',
    cpu: { model: 'Intel(R) Core(TM) i5-12400F', brand: 'Intel', physical_cores: 6, logical_cores: 12 },
    ram: { total_gb: 16.0, available_gb: 8.5 },
    memory: { total_gb: 16.0, available_gb: 8.5 },
    disk: { free_gb: 100.0, total_gb: 512.0 },
    gpus: [{ name: 'NVIDIA GeForce RTX 3060', gpu_type: 'nvidia', cuda_available: true, vram_total_gb: 12.0 }],
    selected_gpu_index: 0,
    tier_label: 'laptop',
    tier: 2,
    score_total: 85.5,
    recommendations: ['推荐使用 INT4 量化档位'],
    warnings: [],
  }) },
  { method: 'POST', match: /^\/device\/select-gpu$/, handler: () => json(200, { selected_gpu: { name: 'NVIDIA GeForce RTX 3060' }, selected_gpu_index: 0, warning: '' }) },
  { method: 'POST', match: /^\/device\/auto-configure$/, handler: () => json(200, { applied_config: { description: 'laptop 档配置已应用' }, tier: 2, score: 85.5 }) },
];

export async function startFakeScheduler(): Promise<FakeScheduler> {
  const requests: FakeScheduler['requests'] = [];
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
    const { status, data } = route
      ? route.handler(body, path)
      : { status: 404, data: { detail: `fake scheduler: Route ${method}:${path} not found` } };
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
