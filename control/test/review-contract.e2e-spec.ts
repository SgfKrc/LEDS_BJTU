/**
 * control-svc review 域契约测试（阶段 3.2 review 域）
 *
 * 语义对齐 api_server.py:5990-6180 + review.py ReviewManager：
 *  - create → ticket.to_dict() 11 字段；target_node_id 必填 400
 *  - vote → 值 ∉{-1,0,1} 400；不存在/已关闭/自投 404；阈值 ±2 裁决
 *    （score>=+2 approved / <=-2 rejected，置 resolved_at）
 *  - 同节点重复投票覆盖旧票；创建者/目标节点不可投票
 *  - tickets 列表 created_at DESC + status 过滤；get 404
 *  - expire-check：pending 超时 → expired
 *  - delete 单个 404/成功；deleteResolved 计数
 *  - can-vote 降级放行；mail-poll 降级 skipped；email-test 依赖 mailer
 *  - notification_sent：mailer 成功才置位；JSON 持久化可重读
 *
 * 注意：controller 的投票者身份取 QLH_NODE_ID（默认 master），与 created_by
 * 相同即自投 404——投票用例需切换 QLH_NODE_ID 模拟从节点。
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { ConfigDao } from '../src/data/config-dao';
import { ReviewStore } from '../src/data/review-store';
import { ReviewMailer, ReviewService } from '../src/modules/review/review.service';

class FakeMailer implements ReviewMailer {
  calls: string[] = [];
  ok = false; // 默认对齐 NoopReviewMailer（降级）：发送失败不置 notification_sent

  async sendReviewCreated(): Promise<boolean> {
    this.calls.push('created');
    return this.ok;
  }

  async sendReviewResolved(): Promise<boolean> {
    this.calls.push('resolved');
    return this.ok;
  }

  async sendTestEmail(): Promise<boolean> {
    return this.ok;
  }
}

describe('control-svc review 域（阶段 3.2）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpFile: string;
  let store: ReviewStore;
  let mailer: FakeMailer;

  const dbDisabledDao = new ConfigDao({
    host: 'localhost',
    port: 5432,
    name: 'x',
    user: 'postgres',
    password: '',
    enabled: false,
    sslmode: 'prefer',
  });

  beforeEach(() => {
    tmpFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'control-review-')), 'review_tickets.json');
    store = new ReviewStore(tmpFile);
    mailer = new FakeMailer();
    delete process.env.QLH_NODE_ID; // 默认 master
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
    fs.rmSync(path.dirname(tmpFile), { recursive: true, force: true });
    delete process.env.QLH_NODE_ID;
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const { Test } = require('@nestjs/testing');
    const { AppModule } = require('../src/app');
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(ReviewStore)
      .useValue(store)
      .overrideProvider(ReviewService)
      .useValue(new ReviewService(store, mailer))
      .overrideProvider(ConfigDao)
      .useValue(dbDisabledDao)
      .compile();
    const fastifyAdapter = new (require('@nestjs/platform-fastify').FastifyAdapter)();
    const testApp = moduleRef.createNestApplication(fastifyAdapter);
    const { JsonDetailFilter } = require('../src/common/json-detail.filter');
    const { RequestIdInterceptor } = require('../src/common/request-id');
    testApp.useGlobalFilters(new JsonDetailFilter());
    testApp.useGlobalInterceptors(new RequestIdInterceptor());
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  /** 创建工单（master 发起） */
  async function createTicket(target = 'worker-1', reason = '测试转让'): Promise<string> {
    const res = await app!.inject({
      method: 'POST',
      url: '/cluster/review/create',
      payload: { target_node_id: target, reason },
    });
    expect(res.statusCode).toBe(200);
    return res.json().ticket_id;
  }

  /** 以指定节点投票 */
  async function vote(node: string, ticketId: string, value: number, comment = ''): Promise<{ status: number; body: any }> {
    process.env.QLH_NODE_ID = node;
    const res = await app!.inject({
      method: 'POST',
      url: '/cluster/review/vote',
      payload: { ticket_id: ticketId, vote: value, comment },
    });
    return { status: res.statusCode, body: res.json() };
  }

  // ---------- create ----------

  it('POST create → 11 字段工单（pending/score 0/expires 约 48h）', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/cluster/review/create',
      payload: { target_node_id: 'worker-1', reason: '升级维护' },
    });
    expect(res.statusCode).toBe(200);
    const t = res.json();
    expect(t.ticket_id).toMatch(/^review_[0-9a-f]{12}$/);
    expect(t.status).toBe('pending');
    expect(t.created_by).toBe('master');
    expect(t.target_node_id).toBe('worker-1');
    expect(t.transfer_reason).toBe('升级维护');
    expect(t.score).toBe(0);
    expect(t.votes).toEqual([]);
    expect(t.notification_sent).toBe(false);
    expect(t.resolved_at).toBeNull();
    // expires_at ≈ created_at + 48h
    expect(t.expires_at - t.created_at).toBeCloseTo(48 * 3600, 0);
    // 持久化到文件
    expect(fs.existsSync(tmpFile)).toBe(true);
    expect(store.loadAll()).toHaveLength(1);
  });

  it('POST create 缺 target_node_id → 400', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/cluster/review/create', payload: {} });
    expect(res.statusCode).toBe(400);
    expect(res.json().detail).toBe('target_node_id 必填');
  });

  it('create timeout_hours 自定义（10h）生效', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/cluster/review/create',
      payload: { target_node_id: 'worker-1', timeout_hours: 10 },
    });
    const t = res.json();
    expect(t.expires_at - t.created_at).toBeCloseTo(10 * 3600, 0);
  });

  it('create 时 mailer 成功 → notification_sent=true', async () => {
    mailer.ok = true;
    app = await createTestApp();
    const id = await createTicket();
    const t = store.get(id)!;
    expect(t.notification_sent).toBe(true);
    expect(mailer.calls).toContain('created');
  });

  // ---------- vote 状态机 ----------

  it('两票 +1 → approved（阈值 +2）', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    const v1 = await vote('worker-2', id, 1);
    expect(v1.status).toBe(200);
    expect(v1.body.status).toBe('pending');
    expect(v1.body.score).toBe(1);
    const v2 = await vote('worker-3', id, 1);
    expect(v2.status).toBe(200);
    expect(v2.body.status).toBe('approved');
    expect(v2.body.score).toBe(2);
    expect(v2.body.resolved_at).not.toBeNull();
    expect(v2.body.votes).toHaveLength(2);
    // 结果邮件已发送
    expect(mailer.calls).toContain('resolved');
  });

  it('两票 -1 → rejected（阈值 -2）', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    await vote('worker-2', id, -1);
    const v2 = await vote('worker-3', id, -1);
    expect(v2.body.status).toBe('rejected');
    expect(v2.body.score).toBe(-2);
  });

  it('弃权 0 不改变 score', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    const v = await vote('worker-2', id, 0);
    expect(v.body.score).toBe(0);
    expect(v.body.status).toBe('pending');
    expect(v.body.votes[0].value).toBe(0);
  });

  it('同节点重复投票覆盖旧票（票数不变，score 更新）', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    await vote('worker-2', id, 1);
    const v2 = await vote('worker-2', id, -1);
    expect(v2.body.votes).toHaveLength(1);
    expect(v2.body.score).toBe(-1);
  });

  it('vote 非法值 → 400', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    process.env.QLH_NODE_ID = 'worker-2';
    const res = await app.inject({
      method: 'POST',
      url: '/cluster/review/vote',
      payload: { ticket_id: id, vote: 5 },
    });
    expect(res.statusCode).toBe(400);
    expect(res.json().detail).toBe('投票值必须为 -1、0 或 +1');
  });

  it('vote 不存在工单 → 404', async () => {
    app = await createTestApp();
    const res = await vote('worker-2', 'review_doesnotexist', 1);
    expect(res.status).toBe(404);
    expect(res.body.detail).toBe("工单 'review_doesnotexist' 不存在或已关闭");
  });

  it('vote 已关闭工单（approved 后）→ 404', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    await vote('worker-2', id, 1);
    await vote('worker-3', id, 1); // approved
    const late = await vote('worker-4', id, 1);
    expect(late.status).toBe(404);
  });

  it('创建者自投 → 404；目标节点自投 → 404', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    const creator = await vote('master', id, 1); // created_by == master
    expect(creator.status).toBe(404);
    const target = await vote('worker-1', id, 1); // target_node_id == worker-1
    expect(target.status).toBe(404);
  });

  // ---------- list / get ----------

  it('GET tickets → created_at DESC + status 过滤', async () => {
    app = await createTestApp();
    const id1 = await createTicket('worker-1');
    const id2 = await createTicket('worker-2');
    const all = await app.inject({ method: 'GET', url: '/cluster/review/tickets' });
    expect(all.statusCode).toBe(200);
    expect(all.json().count).toBe(2);
    expect(all.json().tickets[0].ticket_id).toBe(id2); // 新在前
    expect(all.json().tickets[1].ticket_id).toBe(id1);
    const pending = await app.inject({
      method: 'GET',
      url: '/cluster/review/tickets?status=pending',
    });
    expect(pending.json().count).toBe(2);
    const approved = await app.inject({
      method: 'GET',
      url: '/cluster/review/tickets?status=approved',
    });
    expect(approved.json().count).toBe(0);
  });

  it('GET tickets/:id → 详情；不存在 404', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    const res = await app.inject({ method: 'GET', url: `/cluster/review/tickets/${id}` });
    expect(res.statusCode).toBe(200);
    expect(res.json().ticket_id).toBe(id);
    const missing = await app.inject({ method: 'GET', url: '/cluster/review/tickets/review_nope' });
    expect(missing.statusCode).toBe(404);
    expect(missing.json().detail).toBe("工单 'review_nope' 不存在");
  });

  // ---------- can-vote / expire-check ----------

  it('GET can-vote → 降级放行（node_id + can_vote:true）', async () => {
    app = await createTestApp();
    process.env.QLH_NODE_ID = 'master';
    const res = await app.inject({ method: 'GET', url: '/cluster/review/can-vote' });
    expect(res.statusCode).toBe(200);
    expect(res.json().node_id).toBe('master');
    expect(res.json().can_vote).toBe(true);
    expect(typeof res.json().reason).toBe('string');
  });

  it('POST expire-check → 过期工单转 expired', async () => {
    app = await createTestApp();
    // 预置一张已过期工单（直接改存储的 expires_at）
    const id = await createTicket('worker-1');
    const t = store.get(id)!;
    t.expires_at = Date.now() / 1000 - 10;
    store.upsert(t);
    const res = await app.inject({ method: 'POST', url: '/cluster/review/expire-check' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ expired: [id], count: 1 });
    expect(store.get(id)!.status).toBe('expired');
    expect(store.get(id)!.resolved_at).not.toBeNull();
  });

  it('POST expire-check 无过期 → {expired:[], count:0}', async () => {
    app = await createTestApp();
    await createTicket('worker-1');
    const res = await app.inject({ method: 'POST', url: '/cluster/review/expire-check' });
    expect(res.json()).toEqual({ expired: [], count: 0 });
  });

  // ---------- delete ----------

  it('DELETE tickets/:id → 删除；不存在 404', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    const res = await app.inject({ method: 'DELETE', url: `/cluster/review/tickets/${id}` });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'deleted', ticket_id: id });
    const missing = await app.inject({ method: 'DELETE', url: `/cluster/review/tickets/${id}` });
    expect(missing.statusCode).toBe(404);
  });

  it('DELETE tickets → 批量删除已解决（pending 保留）', async () => {
    app = await createTestApp();
    const pendingId = await createTicket('worker-1');
    const resolvedId = await createTicket('worker-2');
    const t = store.get(resolvedId)!;
    t.status = 'approved';
    store.upsert(t);
    const res = await app.inject({ method: 'DELETE', url: '/cluster/review/tickets' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'deleted', count: 1 });
    expect(store.get(pendingId)).not.toBeNull();
    expect(store.get(resolvedId)).toBeNull();
  });

  // ---------- mail-poll / email-test ----------

  it('POST mail-poll → 降级 skipped（IMAP 未迁移）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/cluster/review/mail-poll' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'ok', polled: 0, skipped: 'imap-not-migrated' });
  });

  it('POST email-test：mailer 成功 → 200；失败（Noop）→ 500', async () => {
    mailer.ok = true;
    app = await createTestApp();
    const ok = await app.inject({ method: 'POST', url: '/cluster/email-test' });
    expect(ok.statusCode).toBe(200);
    expect(ok.json().status).toBe('ok');
    await app.close();
    mailer.ok = false;
    app = await createTestApp();
    const fail = await app.inject({ method: 'POST', url: '/cluster/email-test' });
    expect(fail.statusCode).toBe(500);
    expect(fail.json().detail).toBe('邮件发送失败，请检查后端日志了解详情');
  });

  // ---------- 持久化（重启模拟） ----------

  it('JSON 持久化：新 store 实例读同一文件数据完整', async () => {
    app = await createTestApp();
    const id = await createTicket('worker-1');
    await vote('worker-2', id, 1);
    // 模拟重启：同一文件新建 store + service
    const store2 = new ReviewStore(tmpFile);
    const t2 = store2.get(id)!;
    expect(t2.status).toBe('pending');
    expect(t2.votes).toHaveLength(1);
    expect(t2.votes[0].voter_node_id).toBe('worker-2');
    expect(t2.score).toBe(1);
  });
});
