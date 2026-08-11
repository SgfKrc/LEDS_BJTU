/**
 * control-svc sessions/conversations 域契约测试（阶段 3.2 首迁域）
 *
 * 语义对齐 api_server.py:6245-6662（local_store 降级路径）：
 *   - GET  /conversations/sync-status → {save_history, db_connected,
 *     local_save_enabled, local_store_enabled, cloud_sync_enabled}
 *   - GET  /conversations → {messages:[{role,content}], count, source:'local_store'}
 *   - DELETE /conversations → {status:'cleared', session_id, deleted_count}
 *     （session_id=default 时解析为活跃会话，对齐 api_server.py:6331）
 *   - POST /sessions → {id, title, message_count:0, active:true}（自动激活）
 *   - GET  /sessions → {sessions, active_session_id, total}（updated_at DESC）
 *   - GET  /sessions/:id → 单会话；未知会话返回空壳（无 404，对齐 :6505）
 *   - PUT  /sessions/:id → 重命名；不存在 404；title 1-256 校验 400
 *   - DELETE /sessions/:id → {status:'deleted', session_id}
 *   - POST /sessions/:id/activate → {session_id, messages, count}
 *   - DELETE /sessions/:id/turns/:turnIndex → 正常 2 条；越界 400；无消息 404
 *
 * 硬验收：预置 Python 旧格式 JSON（_sessions.json + {id}.json）必须可读。
 * 存储目录注入临时目录，避免触碰仓库运行时 chat_history/。
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { SessionStore } from '../src/data/session-store';

describe('control-svc sessions/conversations 域（阶段 3.2 首迁）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpDir: string;
  let store: SessionStore;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-sessions-'));
    store = new SessionStore(tmpDir);
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const { Test } = require('@nestjs/testing');
    const { AppModule } = require('../src/app');
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(SessionStore)
      .useValue(store)
      .compile();
    const fastifyAdapter = new (require('@nestjs/platform-fastify').FastifyAdapter)();
    const testApp = moduleRef.createNestApplication(fastifyAdapter);
    const { JsonDetailFilter } = require('../src/common/json-detail.filter');
    testApp.useGlobalFilters(new JsonDetailFilter());
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  /** 预置 Python 旧格式数据（与 local_store.py 写出的布局一致） */
  function presetLegacyData(sessions: unknown[], messagesBySession: Record<string, unknown[]>): void {
    fs.writeFileSync(
      path.join(tmpDir, '_sessions.json'),
      JSON.stringify(sessions, null, 2),
      'utf-8',
    );
    for (const [sid, msgs] of Object.entries(messagesBySession)) {
      fs.writeFileSync(path.join(tmpDir, `${sid}.json`), JSON.stringify(msgs, null, 2), 'utf-8');
    }
  }

  // ---------- POST /sessions ----------

  it('POST /sessions → 创建并自动激活（默认标题）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/sessions', payload: {} });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.title).toBe('新对话');
    expect(body.message_count).toBe(0);
    expect(body.active).toBe(true);
    expect(typeof body.id).toBe('string');
    expect(body.id).toMatch(/^[0-9a-f-]{36}$/);
    // 自动激活
    expect(store.activeSessionId).toBe(body.id);
    // 持久化到 _sessions.json
    const persisted = JSON.parse(
      fs.readFileSync(path.join(tmpDir, '_sessions.json'), 'utf-8'),
    );
    expect(persisted).toHaveLength(1);
    expect(persisted[0].id).toBe(body.id);
  });

  it('POST /sessions 指定 title', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/sessions',
      payload: { title: '我的会话' },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().title).toBe('我的会话');
  });

  it('POST /sessions first_message 自动生成标题（>30 字截断 + ...）', async () => {
    app = await createTestApp();
    const long = '第'.repeat(40);
    const res = await app.inject({
      method: 'POST',
      url: '/sessions',
      payload: { first_message: long },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().title).toBe(`${'第'.repeat(30)}...`);
  });

  it('POST /sessions 无 body 也能创建（可选 body）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/sessions' });
    expect(res.statusCode).toBe(200);
    expect(res.json().title).toBe('新对话');
  });

  // ---------- GET /sessions ----------

  it('GET /sessions → 列表按 updated_at DESC + active_session_id + total', async () => {
    app = await createTestApp();
    const a = await app.inject({ method: 'POST', url: '/sessions', payload: { title: 'A' } });
    const b = await app.inject({ method: 'POST', url: '/sessions', payload: { title: 'B' } });
    expect(a.statusCode).toBe(200);
    expect(b.statusCode).toBe(200);
    const res = await app.inject({ method: 'GET', url: '/sessions' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.sessions).toHaveLength(2);
    expect(body.sessions[0].title).toBe('B'); // 新会话在头部（unshift）
    expect(body.sessions[1].title).toBe('A');
    expect(body.total).toBe(2);
    expect(body.active_session_id).toBe(b.json().id);
  });

  it('GET /sessions?limit=1&offset=1 分页', async () => {
    app = await createTestApp();
    await app.inject({ method: 'POST', url: '/sessions', payload: { title: 'A' } });
    await app.inject({ method: 'POST', url: '/sessions', payload: { title: 'B' } });
    const res = await app.inject({ method: 'GET', url: '/sessions?limit=1&offset=1' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.sessions).toHaveLength(1);
    expect(body.sessions[0].title).toBe('A');
    expect(body.total).toBe(2);
  });

  // ---------- GET /sessions/:id ----------

  it('GET /sessions/:id → 单会话元数据 + active 字段', async () => {
    app = await createTestApp();
    const created = await app.inject({
      method: 'POST',
      url: '/sessions',
      payload: { title: '单会话' },
    });
    const id = created.json().id;
    const res = await app.inject({ method: 'GET', url: `/sessions/${id}` });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.id).toBe(id);
    expect(body.title).toBe('单会话');
    expect(body.message_count).toBe(0);
    expect(body.active).toBe(true);
    expect(body.created_at).toBeTruthy();
  });

  it('GET /sessions/:id 未知会话 → 空壳（200，无 404）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/sessions/nope' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ id: 'nope', title: '新对话', message_count: 0, active: false });
  });

  // ---------- PUT /sessions/:id ----------

  it('PUT /sessions/:id 重命名', async () => {
    app = await createTestApp();
    const created = await app.inject({
      method: 'POST',
      url: '/sessions',
      payload: { title: '旧名' },
    });
    const id = created.json().id;
    const res = await app.inject({
      method: 'PUT',
      url: `/sessions/${id}`,
      payload: { title: '新名' },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().title).toBe('新名');
    expect(res.json().id).toBe(id);
    // 持久化
    const persisted = JSON.parse(
      fs.readFileSync(path.join(tmpDir, '_sessions.json'), 'utf-8'),
    );
    expect(persisted[0].title).toBe('新名');
  });

  it('PUT /sessions/:id 不存在 → 404 detail', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'PUT',
      url: '/sessions/nope',
      payload: { title: '新名' },
    });
    expect(res.statusCode).toBe(404);
    expect(res.json().detail).toBe('会话不存在: nope');
  });

  it('PUT /sessions/:id 空 title → 422（pydantic 1-256 校验对齐）', async () => {
    app = await createTestApp();
    const created = await app.inject({ method: 'POST', url: '/sessions' });
    const id = created.json().id;
    const res = await app.inject({
      method: 'PUT',
      url: `/sessions/${id}`,
      payload: { title: '' },
    });
    expect(res.statusCode).toBe(422);
    expect(typeof res.json().detail).toBe('string');
  });

  it('PUT /sessions/:id title 超 256 字符 → 422', async () => {
    app = await createTestApp();
    const created = await app.inject({ method: 'POST', url: '/sessions' });
    const id = created.json().id;
    const res = await app.inject({
      method: 'PUT',
      url: `/sessions/${id}`,
      payload: { title: '长'.repeat(257) },
    });
    expect(res.statusCode).toBe(422);
  });

  // ---------- DELETE /sessions/:id ----------

  it('DELETE /sessions/:id → 删除会话与消息文件', async () => {
    app = await createTestApp();
    const created = await app.inject({
      method: 'POST',
      url: '/sessions',
      payload: { title: '待删' },
    });
    const id = created.json().id;
    store.saveMessage(id, 'user', 'hi');
    expect(fs.existsSync(path.join(tmpDir, `${id}.json`))).toBe(true);
    const res = await app.inject({ method: 'DELETE', url: `/sessions/${id}` });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'deleted', session_id: id });
    expect(fs.existsSync(path.join(tmpDir, `${id}.json`))).toBe(false);
    expect(store.getSession(id)).toBeNull();
  });

  it('DELETE /sessions/:id 活跃会话 → active 清空', async () => {
    app = await createTestApp();
    const created = await app.inject({ method: 'POST', url: '/sessions' });
    const id = created.json().id;
    expect(store.activeSessionId).toBe(id);
    const res = await app.inject({ method: 'DELETE', url: `/sessions/${id}` });
    expect(res.statusCode).toBe(200);
    expect(store.activeSessionId).toBeNull();
  });

  // ---------- POST /sessions/:id/activate ----------

  it('POST /sessions/:id/activate → 切换活跃 + 返回消息历史', async () => {
    app = await createTestApp();
    const a = await app.inject({ method: 'POST', url: '/sessions', payload: { title: 'A' } });
    const b = await app.inject({ method: 'POST', url: '/sessions', payload: { title: 'B' } });
    store.saveMessage(a.json().id, 'user', '你好');
    store.saveMessage(a.json().id, 'assistant', '你好！');
    const res = await app.inject({
      method: 'POST',
      url: `/sessions/${a.json().id}/activate`,
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.session_id).toBe(a.json().id);
    expect(body.count).toBe(2);
    expect(body.messages).toEqual([
      { role: 'user', content: '你好' },
      { role: 'assistant', content: '你好！' },
    ]);
    expect(store.activeSessionId).toBe(a.json().id);
    expect(b.statusCode).toBe(200);
  });

  // ---------- GET /conversations ----------

  it('GET /conversations 空会话 → {messages:[], count:0, source:local_store}', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/conversations?session_id=empty1' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ messages: [], count: 0, source: 'local_store' });
  });

  it('GET /conversations 有消息 → messages/count（旧数据格式可读）', async () => {
    app = await createTestApp();
    presetLegacyData([], {
      old1: [
        { role: 'user', content: 'hi', created_at: '2026-08-03T22:42:28' },
        { role: 'assistant', content: 'hello', created_at: '2026-08-03T22:42:30' },
      ],
    });
    const res = await app.inject({ method: 'GET', url: '/conversations?session_id=old1' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.count).toBe(2);
    expect(body.source).toBe('local_store');
    expect(body.messages).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ]);
  });

  it('GET /conversations?limit=1 截断（保留末尾）', async () => {
    app = await createTestApp();
    const sid = 'lim1';
    store.saveMessage(sid, 'user', 'm1');
    store.saveMessage(sid, 'user', 'm2');
    const res = await app.inject({ method: 'GET', url: `/conversations?session_id=${sid}&limit=1` });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.count).toBe(1);
    expect(body.messages[0].content).toBe('m2');
  });

  // ---------- DELETE /conversations ----------

  it('DELETE /conversations?session_id=default → 解析为活跃会话', async () => {
    app = await createTestApp();
    const created = await app.inject({ method: 'POST', url: '/sessions' });
    const id = created.json().id;
    store.saveMessage(id, 'user', '要清空的消息');
    const res = await app.inject({ method: 'DELETE', url: '/conversations' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('cleared');
    expect(body.session_id).toBe(id);
    expect(body.deleted_count).toBe(1);
    expect(store.loadMessages(id, 0)).toHaveLength(0);
  });

  it('DELETE /conversations?session_id=显式指定', async () => {
    app = await createTestApp();
    store.saveMessage('explicit1', 'user', 'x');
    const res = await app.inject({
      method: 'DELETE',
      url: '/conversations?session_id=explicit1',
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().session_id).toBe('explicit1');
    expect(res.json().deleted_count).toBe(1);
  });

  // ---------- GET /conversations/sync-status ----------

  it('GET /conversations/sync-status → 字段齐全（DB 禁用时）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/conversations/sync-status' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      save_history: true,
      db_connected: false,
      local_save_enabled: true,
      local_store_enabled: true,
      cloud_sync_enabled: false,
    });
  });

  // ---------- DELETE /sessions/:id/turns/:turnIndex ----------

  it('DELETE turns → 删除 user+assistant 一轮，返回 remaining_turns', async () => {
    app = await createTestApp();
    const created = await app.inject({
      method: 'POST',
      url: '/sessions',
      payload: { title: '轮次' },
    });
    const id = created.json().id;
    store.saveMessage(id, 'user', 'q1');
    store.saveMessage(id, 'assistant', 'a1');
    store.saveMessage(id, 'user', 'q2');
    store.saveMessage(id, 'assistant', 'a2');
    const res = await app.inject({
      method: 'DELETE',
      url: `/sessions/${id}/turns/0`,
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('deleted');
    expect(body.session_id).toBe(id);
    expect(body.turn_index).toBe(0);
    expect(body.deleted_count).toBe(2);
    expect(body.remaining_turns).toBe(1);
    // 剩余消息 = q2/a2
    const msgs = store.loadMessages(id, 0);
    expect(msgs.map((m) => m.content)).toEqual(['q2', 'a2']);
  });

  it('DELETE turns 越界 → 400 带有效范围', async () => {
    app = await createTestApp();
    const created = await app.inject({ method: 'POST', url: '/sessions' });
    const id = created.json().id;
    store.saveMessage(id, 'user', 'q1');
    store.saveMessage(id, 'assistant', 'a1');
    const res = await app.inject({
      method: 'DELETE',
      url: `/sessions/${id}/turns/5`,
    });
    expect(res.statusCode).toBe(400);
    expect(res.json().detail).toBe('无效的轮次索引: 5（有效范围: 0-0）');
  });

  it('DELETE turns 无消息 → 404', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'DELETE',
      url: '/sessions/nomsg/turns/0',
    });
    expect(res.statusCode).toBe(404);
    expect(res.json().detail).toBe('会话不存在或无消息: nomsg');
  });

  it('DELETE turns 非整数索引 → 422（FastAPI 路径参数 int 校验对齐）', async () => {
    app = await createTestApp();
    const created = await app.inject({ method: 'POST', url: '/sessions' });
    const id = created.json().id;
    store.saveMessage(id, 'user', 'q1');
    store.saveMessage(id, 'assistant', 'a1');
    const res = await app.inject({
      method: 'DELETE',
      url: `/sessions/${id}/turns/abc`,
    });
    expect(res.statusCode).toBe(422);
  });

  // ---------- 路径穿越防护 ----------

  it('session_id 穿越 → 400，目录外 .json 不受影响', async () => {
    const outside = path.join(os.tmpdir(), `probe-${Date.now()}.json`);
    fs.writeFileSync(outside, JSON.stringify([{ role: 'user', content: 'topsecret' }]), 'utf-8');
    try {
      app = await createTestApp();
      const probe = `../${path.basename(outside, '.json')}`;
      const encoded = encodeURIComponent(probe);
      // 读：拒绝
      const read = await app.inject({ method: 'GET', url: `/conversations?session_id=${encoded}` });
      expect(read.statusCode).toBe(400);
      expect(read.json().detail).toBe(`无效的会话 id: ${probe}`);
      // 清空：拒绝，文件未被覆盖
      const clear = await app.inject({ method: 'DELETE', url: `/conversations?session_id=${encoded}` });
      expect(clear.statusCode).toBe(400);
      expect(JSON.parse(fs.readFileSync(outside, 'utf-8'))[0].content).toBe('topsecret');
      // 路径参数：拒绝，文件未被删除
      const del = await app.inject({ method: 'DELETE', url: `/sessions/${encoded}` });
      expect(del.statusCode).toBe(400);
      expect(fs.existsSync(outside)).toBe(true);
    } finally {
      fs.rmSync(outside, { force: true });
    }
  });

  it('合法会话 id 不受影响（default/短 id/uuid 均通过）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/conversations?session_id=default' });
    expect(res.statusCode).toBe(200);
    const res2 = await app.inject({ method: 'GET', url: '/conversations?session_id=a1' });
    expect(res2.statusCode).toBe(200);
  });

  // ---------- 旧数据兼容（硬验收） ----------

  it('旧 Python _sessions.json 预置数据可读、排序正确', async () => {
    app = await createTestApp();
    presetLegacyData(
      [
        {
          id: 'old-a',
          title: '旧会话A',
          created_at: '2026-08-01T10:00:00',
          updated_at: '2026-08-01T10:00:00',
          message_count: 2,
        },
        {
          id: 'old-b',
          title: '旧会话B',
          created_at: '2026-08-02T10:00:00',
          updated_at: '2026-08-02T10:00:00',
          message_count: 4,
        },
      ],
      {
        'old-a': [
          { role: 'user', content: 'hi', created_at: '2026-08-01T10:00:00' },
          { role: 'assistant', content: 'hello', created_at: '2026-08-01T10:00:01' },
        ],
      },
    );
    const res = await app.inject({ method: 'GET', url: '/sessions' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.total).toBe(2);
    expect(body.sessions[0].id).toBe('old-b'); // updated_at DESC
    expect(body.sessions[1].id).toBe('old-a');
    expect(body.sessions[1].message_count).toBe(2);
    // 旧消息文件可读
    const conv = await app.inject({ method: 'GET', url: '/conversations?session_id=old-a' });
    expect(conv.statusCode).toBe(200);
    expect(conv.json().count).toBe(2);
    expect(conv.json().source).toBe('local_store');
    // 旧会话可重命名、可删除（写路径不破坏旧数据）
    const renamed = await app.inject({
      method: 'PUT',
      url: '/sessions/old-a',
      payload: { title: '旧会话A改名' },
    });
    expect(renamed.statusCode).toBe(200);
    expect(renamed.json().title).toBe('旧会话A改名');
    const del = await app.inject({ method: 'DELETE', url: '/sessions/old-b' });
    expect(del.statusCode).toBe(200);
    expect(fs.existsSync(path.join(tmpDir, '_sessions.json'))).toBe(true);
  });

  it('损坏的 _sessions.json → 重建为空（对齐 local_store _read_json 重建语义）', async () => {
    app = await createTestApp();
    fs.writeFileSync(path.join(tmpDir, '_sessions.json'), '{ 坏 JSON', 'utf-8');
    const res = await app.inject({ method: 'GET', url: '/sessions' });
    expect(res.statusCode).toBe(200);
    expect(res.json().sessions).toEqual([]);
    expect(res.json().total).toBe(0);
  });
});
