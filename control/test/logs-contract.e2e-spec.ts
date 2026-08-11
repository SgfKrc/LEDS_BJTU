/**
 * control-svc logs 域契约测试（阶段 3.2 日志域）
 *
 * 语义对齐 api_server.py:6883-7585：
 *  - GET /logs → {files:[{name,size,modified}]} mtime 降序，仅 .log
 *  - GET /logs/recent → 内存缓冲，limit clamp 1-1000，level/name/node_id/
 *    request_id 过滤（level 按 levelno >=，未知 level 精确匹配）
 *  - GET /logs/stats → {log_dir, files_count, files_total_bytes,
 *    buffer_*, levels, loggers, nodes, node_id, device_ip}
 *  - GET /logs/download?name= → 附件下载 text/plain；400 非法名 / 404 不存在
 *  - DELETE /logs → 清空全部
 *  - GET /logs/export → ZIP（Content-Disposition qlh-logs-*.zip）；无文件 404
 *  - POST /logs/client-error → 无鉴权，{status:'ok',logged:true}
 *  - GET /logs/node/{id}/recent → 本机/master 直读；远程降级 remote-unavailable
 *  - GET /logs/nodes-summary → {local, workers:[], total_workers:0}
 *  - GET|DELETE /logs/{filename} → 末 1MB 内容 / 删除；400 非法名 / 404 不存在
 *
 * 鉴权矩阵（LogAccessGuard 对齐 _require_log_api_access）：
 *  本机 IP 放行；远程需 QLH_LOG_ADMIN_TOKEN == X-QLH-Log-Token，否则 403。
 *  client-error 豁免。
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import JSZip from 'jszip';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { LogBuffer } from '../src/data/log-buffer';
import { LogFileStore } from '../src/data/log-file-store';

describe('control-svc logs 域（阶段 3.2）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpLogDir: string;
  let buffer: LogBuffer;
  const savedToken = process.env.QLH_LOG_ADMIN_TOKEN;

  beforeEach(() => {
    tmpLogDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-logs-'));
    buffer = new LogBuffer();
    // 隔离环境变量：避免开发机/CI 的 QLH_NODE_ID/QLH_DEVICE_IP/QLH_LOG_DIR
    // 影响 node_id/device_ip/log_dir 断言
    delete process.env.QLH_NODE_ID;
    delete process.env.QLH_DEVICE_IP;
    delete process.env.QLH_LOG_DIR;
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
    fs.rmSync(tmpLogDir, { recursive: true, force: true });
    if (savedToken === undefined) delete process.env.QLH_LOG_ADMIN_TOKEN;
    else process.env.QLH_LOG_ADMIN_TOKEN = savedToken;
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const { Test } = require('@nestjs/testing');
    const { AppModule } = require('../src/app');
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(LogFileStore)
      .useValue(new LogFileStore(tmpLogDir))
      .overrideProvider(LogBuffer)
      .useValue(buffer)
      .compile();
    const fastifyAdapter = new (require('@nestjs/platform-fastify').FastifyAdapter)();
    const testApp = moduleRef.createNestApplication(fastifyAdapter);
    const { JsonDetailFilter } = require('../src/common/json-detail.filter');
    const { RequestIdInterceptor } = require('../src/common/request-id');
    testApp.useGlobalFilters(new JsonDetailFilter());
    testApp.useGlobalInterceptors(new RequestIdInterceptor()); // 对齐 createApp()：client-error 依赖 req.requestId
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  function writeLogFile(name: string, content: string): void {
    fs.writeFileSync(path.join(tmpLogDir, name), content, 'utf-8');
  }

  function seedBuffer(): void {
    buffer.append({ level: 'INFO', levelno: 20, name: 'uvicorn.access', message: 'GET / 200', node_id: 'master' });
    buffer.append({ level: 'WARNING', levelno: 30, name: 'scheduler', message: '节点离线重试', node_id: 'master' });
    buffer.append({ level: 'ERROR', levelno: 40, name: 'api_server', message: '推理失败', request_id: 'req-1', node_id: 'master' });
  }

  // ---------- 鉴权矩阵 ----------

  it('本机访问放行（默认 127.0.0.1）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/logs' });
    expect(res.statusCode).toBe(200);
  });

  it('远程无 token → 403 detail', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'GET',
      url: '/logs',
      remoteAddress: '192.168.1.50',
    });
    expect(res.statusCode).toBe(403);
    expect(res.json().detail).toBe('日志接口仅允许本机访问；远程访问需管理员授权');
  });

  it('远程错误 token → 403', async () => {
    process.env.QLH_LOG_ADMIN_TOKEN = 'secret-token';
    app = await createTestApp();
    const res = await app.inject({
      method: 'GET',
      url: '/logs',
      remoteAddress: '192.168.1.50',
      headers: { 'x-qlh-log-token': 'wrong' },
    });
    expect(res.statusCode).toBe(403);
  });

  it('远程正确 token → 放行', async () => {
    process.env.QLH_LOG_ADMIN_TOKEN = 'secret-token';
    app = await createTestApp();
    const res = await app.inject({
      method: 'GET',
      url: '/logs',
      remoteAddress: '192.168.1.50',
      headers: { 'x-qlh-log-token': 'secret-token' },
    });
    expect(res.statusCode).toBe(200);
  });

  it('未配置 token 时远程一律 403（即使带 token 头）', async () => {
    delete process.env.QLH_LOG_ADMIN_TOKEN;
    app = await createTestApp();
    const res = await app.inject({
      method: 'GET',
      url: '/logs',
      remoteAddress: '192.168.1.50',
      headers: { 'x-qlh-log-token': 'anything' },
    });
    expect(res.statusCode).toBe(403);
  });

  it('本机 IP + 错误 token → 本机优先放行（对齐 Python 先查本机）', async () => {
    process.env.QLH_LOG_ADMIN_TOKEN = 'secret-token';
    app = await createTestApp();
    const res = await app.inject({
      method: 'GET',
      url: '/logs',
      headers: { 'x-qlh-log-token': 'wrong' },
    });
    expect(res.statusCode).toBe(200);
  });

  it('client-error 无鉴权：远程无 token → 200', async () => {
    delete process.env.QLH_LOG_ADMIN_TOKEN;
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/logs/client-error',
      remoteAddress: '192.168.1.50',
      payload: { message: '前端炸了' },
    });
    expect(res.statusCode).toBe(200);
  });

  it('QLH_NODE_ID 环境变量覆盖 node_id（stats + nodes-summary）', async () => {
    process.env.QLH_NODE_ID = 'my-node';
    app = await createTestApp();
    const stats = await app.inject({ method: 'GET', url: '/logs/stats' });
    expect(stats.json().node_id).toBe('my-node');
    const summary = await app.inject({ method: 'GET', url: '/logs/nodes-summary' });
    expect(summary.json().local.node_id).toBe('my-node');
    const local = await app.inject({ method: 'GET', url: '/logs/node/my-node/recent' });
    expect(local.json().source).toBe('local');
  });

  // ---------- GET /logs ----------

  it('GET /logs → 仅 .log 文件，按 mtime 降序', async () => {
    app = await createTestApp();
    writeLogFile('b.log', 'b');
    writeLogFile('a.log', 'a');
    writeLogFile('notes.txt', 'not-a-log');
    writeLogFile('c.log.1', 'rotated'); // RotatingFileHandler 备份
    // Windows 文件系统 mtime 精度低，显式设置 mtime 保证排序确定
    const base = Date.now() - 10000;
    const files = ['a.log', 'b.log', 'c.log.1', 'notes.txt'];
    files.forEach((f, i) => {
      const p = path.join(tmpLogDir, f);
      const t = new Date(base + i * 1000);
      fs.utimesSync(p, t, t);
    });
    const res = await app.inject({ method: 'GET', url: '/logs' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    const names = body.files.map((f: { name: string }) => f.name);
    expect(names).not.toContain('notes.txt');
    expect(names).toContain('c.log.1');
    expect(names).toHaveLength(3);
    // mtime 降序：最新修改的在前（c.log.1 的 mtime 最新）
    expect(names[0]).toBe('c.log.1');
    const first = body.files[0];
    expect(typeof first.size).toBe('number');
    expect(typeof first.modified).toBe('string');
  });

  // ---------- GET /logs/recent ----------

  it('GET /logs/recent → 缓冲快照 + 过滤 + 截断标记', async () => {
    app = await createTestApp();
    seedBuffer();
    const res = await app.inject({ method: 'GET', url: '/logs/recent?limit=2' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.count).toBe(2);
    expect(body.matched).toBe(3);
    expect(body.limit).toBe(2);
    expect(body.buffer_size).toBe(3);
    expect(body.buffer_capacity).toBe(5000);
    expect(body.total_seen).toBe(3);
    expect(body.truncated).toBe(true);
    expect(body.logs[1].message).toBe('推理失败');
    expect(body.filters).toEqual({
      level: null,
      name: null,
      node_id: null,
      request_id: null,
    });
  });

  it('GET /logs/recent level 过滤（按 levelno >=）', async () => {
    app = await createTestApp();
    seedBuffer();
    const res = await app.inject({ method: 'GET', url: '/logs/recent?level=WARNING' });
    const body = res.json();
    expect(body.matched).toBe(2);
    expect(body.logs.map((l: { level: string }) => l.level)).toEqual(['WARNING', 'ERROR']);
  });

  it('GET /logs/recent 未知 level 名 → 精确匹配', async () => {
    app = await createTestApp();
    seedBuffer();
    const res = await app.inject({ method: 'GET', url: '/logs/recent?level=NOTALEVEL' });
    expect(res.json().matched).toBe(0);
  });

  it('GET /logs/recent name 子串 / request_id 精确过滤', async () => {
    app = await createTestApp();
    seedBuffer();
    const byName = await app.inject({ method: 'GET', url: '/logs/recent?name=sched' });
    expect(byName.json().matched).toBe(1);
    const byRid = await app.inject({ method: 'GET', url: '/logs/recent?request_id=req-1' });
    expect(byRid.json().matched).toBe(1);
    expect(byRid.json().logs[0].message).toBe('推理失败');
  });

  it('GET /logs/recent node_id 精确过滤', async () => {
    app = await createTestApp();
    seedBuffer();
    buffer.append({ level: 'INFO', levelno: 20, name: 'worker', message: 'worker 日志', node_id: 'worker-1' });
    const res = await app.inject({ method: 'GET', url: '/logs/recent?node_id=worker-1' });
    expect(res.statusCode).toBe(200);
    expect(res.json().matched).toBe(1);
    expect(res.json().logs[0].message).toBe('worker 日志');
  });

  it('GET /logs/recent limit clamp 1-1000，非法值回退默认 200', async () => {
    app = await createTestApp();
    seedBuffer();
    const big = await app.inject({ method: 'GET', url: '/logs/recent?limit=5000' });
    expect(big.json().limit).toBe(1000);
    const bad = await app.inject({ method: 'GET', url: '/logs/recent?limit=abc' });
    expect(bad.json().limit).toBe(200);
  });

  // ---------- GET /logs/stats ----------

  it('GET /logs/stats → 文件 + 缓冲统计', async () => {
    app = await createTestApp();
    writeLogFile('a.log', 'hello');
    seedBuffer();
    const res = await app.inject({ method: 'GET', url: '/logs/stats' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.log_dir).toBe(tmpLogDir);
    expect(body.files_count).toBe(1);
    expect(body.files_total_bytes).toBe(5);
    expect(body.buffer_size).toBe(3);
    expect(body.buffer_capacity).toBe(5000);
    expect(body.levels).toEqual({ INFO: 1, WARNING: 1, ERROR: 1 });
    expect(body.loggers).toHaveProperty('api_server');
    expect(body.nodes).toEqual({ master: 3 });
    expect(typeof body.node_id).toBe('string');
  });

  // ---------- GET /logs/download ----------

  it('GET /logs/download → 附件流 + 内容', async () => {
    app = await createTestApp();
    writeLogFile('qlh-test.log', 'line1\nline2\n');
    const res = await app.inject({ method: 'GET', url: '/logs/download?name=qlh-test.log' });
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-disposition']).toContain('attachment; filename="qlh-test.log"');
    expect(res.headers['content-type']).toContain('text/plain');
    expect(res.body).toBe('line1\nline2\n');
  });

  it('GET /logs/download 非法名 → 400；不存在 → 404', async () => {
    app = await createTestApp();
    const bad = await app.inject({
      method: 'GET',
      url: '/logs/download?name=..%2Fsecret.txt',
    });
    expect(bad.statusCode).toBe(400);
    expect(bad.json().detail).toBe('无效的日志文件名');
    const missing = await app.inject({ method: 'GET', url: '/logs/download?name=nope.log' });
    expect(missing.statusCode).toBe(404);
    expect(missing.json().detail).toBe('文件不存在');
  });

  // ---------- GET /logs/{filename} ----------

  it('GET /logs/{filename} → 内容；反斜杠/穿越名 400', async () => {
    app = await createTestApp();
    writeLogFile('qlh-test.log', '你好\nworld\n');
    const res = await app.inject({ method: 'GET', url: '/logs/qlh-test.log' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ name: 'qlh-test.log', content: '你好\nworld\n', truncated: false });
    const bad = await app.inject({ method: 'GET', url: '/logs/..%2F..%2Fetc%2Fpasswd' });
    expect(bad.statusCode).toBe(400);
    const bs = await app.inject({ method: 'GET', url: '/logs/a%5Cb.log' });
    expect(bs.statusCode).toBe(400);
    const missing = await app.inject({ method: 'GET', url: '/logs/nope.log' });
    expect(missing.statusCode).toBe(404);
  });

  it('GET /logs/{filename} 大文件 → 跳过不完整首行（对齐 Python f.readline()）', async () => {
    app = await createTestApp();
    const big = 'A'.repeat(1024 * 1024) + '\n' + 'B'.repeat(100); // 总长 > 1MB
    writeLogFile('big.log', big);
    const res = await app.inject({ method: 'GET', url: '/logs/big.log' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.truncated).toBe(true);
    expect(body.content).toBe('B'.repeat(100)); // 半行 'A' 被跳过
  });

  it('LogBuffer 容量满时丢弃最旧条目', () => {
    const b = new LogBuffer(3);
    b.append({ level: 'INFO', levelno: 20, message: 'a' });
    b.append({ level: 'INFO', levelno: 20, message: 'b' });
    b.append({ level: 'INFO', levelno: 20, message: 'c' });
    b.append({ level: 'INFO', levelno: 20, message: 'd' });
    expect(b.size()).toBe(3);
    expect(b.snapshot().entries[0].message).toBe('b');
    expect(b.stats().buffer_total_seen).toBe(4);
    expect(b.stats().buffer_dropped_estimate).toBe(1);
  });

  it('GET /logs/{filename} 大文件 → 末 1MB + truncated 标记', async () => {
    app = await createTestApp();
    const big = 'x'.repeat(1024 * 1024 + 1000); // 1MB + 1000
    writeLogFile('big.log', big);
    const res = await app.inject({ method: 'GET', url: '/logs/big.log' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.truncated).toBe(true);
    expect(body.content.length).toBeLessThanOrEqual(1024 * 1024);
    expect(body.content.endsWith('x'.repeat(1000))).toBe(true);
  });

  // ---------- DELETE /logs/{filename} / DELETE /logs ----------

  it('DELETE /logs/{filename} → 删除；不存在 404', async () => {
    app = await createTestApp();
    writeLogFile('del.log', 'x');
    const res = await app.inject({ method: 'DELETE', url: '/logs/del.log' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'ok', deleted: 'del.log', failed: [] });
    expect(fs.existsSync(path.join(tmpLogDir, 'del.log'))).toBe(false);
    const again = await app.inject({ method: 'DELETE', url: '/logs/del.log' });
    expect(again.statusCode).toBe(404);
  });

  it('DELETE /logs → 清空全部 .log（保留非日志文件）', async () => {
    app = await createTestApp();
    writeLogFile('a.log', 'a');
    writeLogFile('b.log', 'b');
    writeLogFile('keep.txt', 'keep');
    const res = await app.inject({ method: 'DELETE', url: '/logs' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('ok');
    expect(body.deleted_count).toBe(2);
    expect(fs.existsSync(path.join(tmpLogDir, 'keep.txt'))).toBe(true);
  });

  // ---------- GET /logs/export ----------

  it('GET /logs/export → ZIP 含全部 .log + 附件头', async () => {
    app = await createTestApp();
    writeLogFile('a.log', 'content-a');
    writeLogFile('b.log', 'content-b');
    const res = await app.inject({ method: 'GET', url: '/logs/export' });
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toContain('application/zip');
    expect(res.headers['content-disposition']).toMatch(/attachment; filename="qlh-logs-.+-\d{8}-\d{6}\.zip"/);
    const zip = await JSZip.loadAsync(res.rawPayload as Buffer);
    expect(Object.keys(zip.files).sort()).toEqual(['a.log', 'b.log']);
    expect(await zip.file('a.log')!.async('string')).toBe('content-a');
  });

  it('GET /logs/export 无 .log 文件 → 空 ZIP 200（对齐 Python）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/logs/export' });
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toContain('application/zip');
    const zip = await JSZip.loadAsync(res.rawPayload as Buffer);
    expect(Object.keys(zip.files).filter((f) => !f.endsWith('/'))).toEqual([]);
  });

  // ---------- POST /logs/client-error ----------

  it('POST /logs/client-error → ok + 写入缓冲（字段截断）', async () => {
    app = await createTestApp();
    const before = buffer.size();
    const res = await app.inject({
      method: 'POST',
      url: '/logs/client-error',
      payload: {
        message: '前端炸了',
        source: 'window.onerror',
        stack: 'at x (y:1:1)',
        url: 'http://localhost/',
        extra: { session_id: 's1' },
      },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'ok', logged: true });
    expect(buffer.size()).toBe(before + 1);
  });

  it('POST /logs/client-error 超长字段 → 截断 + [truncated] 后缀', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/logs/client-error',
      payload: { message: 'e'.repeat(600) },
    });
    expect(res.statusCode).toBe(200);
    const { entries } = buffer.snapshot();
    const entry = entries[entries.length - 1];
    expect(entry.message).toContain('message=');
    // message 字段截 500 字符后仍有 request_id= 后缀，故用 toContain
    expect(entry.message).toContain('...[truncated]');
    // message= 字段截 500 字符：前缀 + 500 + 后缀，总长有界
    expect(entry.message.length).toBeLessThan(600 + '...[truncated]'.length + 300);
  });

  it('POST /logs/client-error 空 body 也接受（全部字段有默认值）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/logs/client-error' });
    expect(res.statusCode).toBe(200);
    expect(res.json().logged).toBe(true);
  });

  it('POST /logs/client-error 写入 request_id（可被 recent?request_id 关联）', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/logs/client-error',
      payload: { message: '带请求id的错误' },
      headers: { 'x-request-id': 'client-err-1' },
    });
    expect(res.statusCode).toBe(200);
    const recent = await app.inject({ method: 'GET', url: '/logs/recent?request_id=client-err-1' });
    expect(recent.json().matched).toBe(1);
    expect(recent.json().logs[0].message).toContain('request_id=client-err-1');
  });

  // ---------- 多节点（降级） ----------

  it('GET /logs/node/{id}/recent 本机（master）→ source:local', async () => {
    app = await createTestApp();
    seedBuffer();
    const res = await app.inject({ method: 'GET', url: '/logs/node/master/recent' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.source).toBe('local');
    expect(body.node_id).toBe('master');
    expect(body.count).toBe(3);
  });

  it('GET /logs/node/{id}/recent 远程节点 → 降级 remote-unavailable', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/logs/node/worker-1/recent' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      node_id: 'worker-1',
      source: 'remote-unavailable',
      logs: [],
      count: 0,
      matched: 0,
      buffer_size: 0,
    });
  });

  it('GET /logs/nodes-summary → local + 空 workers（聚合未迁移）', async () => {
    app = await createTestApp();
    seedBuffer();
    const res = await app.inject({ method: 'GET', url: '/logs/nodes-summary' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.local).toEqual({ node_id: 'master', buffer_size: 3, buffer_capacity: 5000 });
    expect(body.workers).toEqual([]);
    expect(body.total_workers).toBe(0);
  });
});
