/**
 * control-svc workflows 域契约测试（阶段 3.2 任务图控制面）
 *
 * 语义对齐 api_server.py:4571-4700 + task_graph.py list/get/request_cancel/
 * cleanup_journal：
 *  - GET /workflows → {enabled, available, role, templates, providers,
 *    provider_status, provider_error, worker_protocol, journal, workflows[]}
 *    workflows 按 created_at DESC + session_id 过滤 + limit
 *  - GET /workflows/:id → snapshot + observability（完整推导）+ journal；404
 *  - POST /workflows/:id/cancel → 非法 ID 400（wf_ 前缀 + 8-96 safe 字符）；
 *    未注册 → cancel_pending/pending_registration；非终态 → cancel_requested
 *    并本地标记；终态 → 返回 state
 *  - POST /workflows/journal/cleanup → clamp(30/1000, 上限 3650/100000)、
 *    仅删终态、{deleted_workflows, deleted_by_age, deleted_by_limit,
 *    remaining_terminal}
 *
 * observability 断言覆盖：state/terminal/result_ready/retry_count/
 * same_provider_retry_count/reassignment_count/retrying/rejection 字段/
 * winner_count/actual_providers/actual_nodes/partial_result。
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { ConfigDao } from '../src/data/config-dao';
import {
  WorkflowJournalStore,
  WorkflowSnapshot,
} from '../src/data/workflow-journal-store';

/** 构造一张最小快照（字段对齐 WorkflowRecord.snapshot） */
function makeSnapshot(partial: Partial<WorkflowSnapshot> & { workflow_id: string }): WorkflowSnapshot {
  return {
    request_id: 'req-1',
    session_id: 's1',
    model_identity: null,
    template: 'dual_candidate',
    state: 'running',
    last_sequence: 1,
    final_stage_id: 'aggregate',
    created_at: Date.now() / 1000,
    started_at: Date.now() / 1000,
    result_ready_at: null,
    finished_at: null,
    duration_seconds: 0,
    error: '',
    stage_count: 0,
    completed_stage_count: 0,
    failed_stage_count: 0,
    skipped_stage_count: 0,
    partial_result: false,
    cancelled_stage_count: 0,
    attempt_count: 0,
    retry_count: 0,
    same_provider_retry_count: 0,
    result_rejection_count: 0,
    cancel_requested: false,
    stages: [],
    ...partial,
  };
}

/** 双候选已完成的快照（含 attempts，测 observability） */
function completedSnapshot(id: string): WorkflowSnapshot {
  return makeSnapshot({
    workflow_id: id,
    state: 'completed',
    started_at: 100.0,
    result_ready_at: 110.0,
    finished_at: 115.0,
    duration_seconds: 15,
    stage_count: 3,
    completed_stage_count: 3,
    retry_count: 2,
    same_provider_retry_count: 1,
    result_rejection_count: 1,
    stages: [
      {
        stage_id: 'candidate_a',
        stage_type: 'candidate',
        depends_on: [],
        provider: 'local_full_model',
        requested_provider: 'local_full_model',
        fallback_providers: [],
        pure: false,
        accept_timeout_seconds: 30,
        lease_timeout_seconds: 60,
        minimum_successful_dependencies: 0,
        max_same_provider_retries: 1,
        retry_safe: false,
        lease_epoch: 1,
        winner_attempt_id: 'att-a1',
        retry_count: 1,
        same_provider_retry_count: 1,
        last_retry_error_code: '',
        result_rejection_count: 1,
        last_result_rejection_reason: 'quality too low',
        last_result_rejected_at: 105.0,
        state: 'completed',
        started_at: 100.0,
        finished_at: 108.0,
        duration_seconds: 8,
        error: '',
        attempts: [
          {
            attempt_id: 'att-a1',
            provider: 'local_full_model',
            provider_kind: 'local',
            provider_node_id: 'master',
            reservation_id: '',
            lease_id: '',
            lease_epoch: 0,
            lease_expires_at: 0,
            state: 'completed',
            started_at: 100.0,
            finished_at: 108.0,
            error: '',
            result_metadata: {},
            result_sha256: 'abc',
          },
        ],
        output_available: true,
        output_sha256: 'abc',
        output_size_bytes: 12,
      },
      {
        stage_id: 'candidate_b',
        stage_type: 'candidate',
        depends_on: [],
        provider: 'task_worker:node-2',
        requested_provider: 'task_worker:node-2',
        fallback_providers: [],
        pure: false,
        accept_timeout_seconds: 30,
        lease_timeout_seconds: 60,
        minimum_successful_dependencies: 0,
        max_same_provider_retries: 0,
        retry_safe: false,
        lease_epoch: 1,
        winner_attempt_id: 'att-b1',
        retry_count: 0,
        same_provider_retry_count: 0,
        last_retry_error_code: '',
        result_rejection_count: 0,
        last_result_rejection_reason: '',
        last_result_rejected_at: null,
        state: 'completed',
        started_at: 100.0,
        finished_at: 109.0,
        duration_seconds: 9,
        error: '',
        attempts: [
          {
            attempt_id: 'att-b1',
            provider: 'task_worker:node-2',
            provider_kind: 'remote',
            provider_node_id: 'node-2',
            reservation_id: '',
            lease_id: '',
            lease_epoch: 0,
            lease_expires_at: 0,
            state: 'completed',
            started_at: 100.0,
            finished_at: 109.0,
            error: '',
            result_metadata: {},
            result_sha256: 'def',
          },
        ],
        output_available: true,
        output_sha256: 'def',
        output_size_bytes: 34,
      },
      {
        stage_id: 'aggregate',
        stage_type: 'aggregate',
        depends_on: ['candidate_a', 'candidate_b'],
        provider: 'local_full_model',
        requested_provider: 'local_full_model',
        fallback_providers: [],
        pure: false,
        accept_timeout_seconds: 30,
        lease_timeout_seconds: 60,
        minimum_successful_dependencies: 2,
        max_same_provider_retries: 0,
        retry_safe: false,
        lease_epoch: 2,
        winner_attempt_id: 'att-c1',
        retry_count: 1,
        same_provider_retry_count: 0,
        last_retry_error_code: 'timeout',
        result_rejection_count: 0,
        last_result_rejection_reason: '',
        last_result_rejected_at: null,
        state: 'completed',
        started_at: 110.0,
        finished_at: 115.0,
        duration_seconds: 5,
        error: '',
        attempts: [
          {
            attempt_id: 'att-c1',
            provider: 'local_full_model',
            provider_kind: 'local',
            provider_node_id: 'master',
            reservation_id: '',
            lease_id: '',
            lease_epoch: 0,
            lease_expires_at: 0,
            state: 'completed',
            started_at: 110.0,
            finished_at: 115.0,
            error: '',
            result_metadata: {},
            result_sha256: 'caf',
          },
        ],
        output_available: true,
        output_sha256: 'caf',
        output_size_bytes: 56,
      },
    ],
  });
}

describe('control-svc workflows 域（阶段 3.2 任务图控制面）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpFile: string;
  let store: WorkflowJournalStore;

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
    tmpFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'control-wf-')), 'journal.json');
    store = new WorkflowJournalStore(tmpFile);
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
    fs.rmSync(path.dirname(tmpFile), { recursive: true, force: true });
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const { Test } = require('@nestjs/testing');
    const { AppModule } = require('../src/app');
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(WorkflowJournalStore)
      .useValue(store)
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

  // ---------- GET /workflows ----------

  it('GET /workflows 空 journal → 完整字段骨架', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/workflows' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.enabled).toBe(true);
    expect(body.available).toBe(true);
    expect(body.role).toBe('master');
    expect(body.templates).toEqual(['dual_candidate']);
    expect(body.workflows).toEqual([]);
    expect(body.journal.available).toBe(true);
    expect(body.journal.backend).toBe('json-file');
    expect(body.journal.record_count).toBe(0);
    // journal 不含 path（对齐 _public_task_journal）
    expect(body.journal.path).toBeUndefined();
  });

  it('GET /workflows 列表按 created_at DESC + session 过滤 + limit', async () => {
    app = await createTestApp();
    store.upsertSnapshot(makeSnapshot({ workflow_id: 'wf_old00001', session_id: 's1', created_at: 100 }));
    store.upsertSnapshot(makeSnapshot({ workflow_id: 'wf_new00001', session_id: 's2', created_at: 200 }));
    store.upsertSnapshot(makeSnapshot({ workflow_id: 'wf_mid00001', session_id: 's1', created_at: 150 }));
    const all = await app.inject({ method: 'GET', url: '/workflows' });
    expect(all.json().workflows.map((w: { workflow_id: string }) => w.workflow_id)).toEqual([
      'wf_new00001',
      'wf_mid00001',
      'wf_old00001',
    ]);
    const s1 = await app.inject({ method: 'GET', url: '/workflows?session_id=s1' });
    expect(s1.json().workflows.map((w: { workflow_id: string }) => w.workflow_id)).toEqual([
      'wf_mid00001',
      'wf_old00001',
    ]);
    const limited = await app.inject({ method: 'GET', url: '/workflows?limit=2' });
    expect(limited.json().workflows).toHaveLength(2);
  });

  // ---------- GET /workflows/:id + observability ----------

  it('GET /workflows/:id → 快照 + observability 完整推导；404', async () => {
    app = await createTestApp();
    store.upsertSnapshot(completedSnapshot('wf_complete01'));
    const res = await app.inject({ method: 'GET', url: '/workflows/wf_complete01' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.workflow_id).toBe('wf_complete01');
    expect(body.template).toBe('dual_candidate');
    const o = body.observability;
    expect(o.state).toBe('completed');
    expect(o.terminal).toBe(true);
    expect(o.result_ready).toBe(false);
    // retry_count 用快照顶层（2），非 stages 求和（1+0+1=2 相同）
    expect(o.retry_count).toBe(2);
    expect(o.same_provider_retry_count).toBe(1);
    expect(o.reassignment_count).toBe(1);
    expect(o.retrying).toBe(false);
    expect(o.result_rejection_count).toBe(1);
    expect(o.last_result_rejection_reason).toBe('quality too low');
    expect(o.last_result_rejected_at).toBe(105.0);
    expect(o.winner_count).toBe(3);
    expect(o.actual_providers).toEqual(['local_full_model', 'task_worker:node-2']);
    expect(o.actual_nodes).toEqual(['master', 'node-2']);
    expect(o.partial_result).toBe(false);
    const missing = await app.inject({ method: 'GET', url: '/workflows/wf_missing000' });
    expect(missing.statusCode).toBe(404);
    expect(missing.json().detail).toBe('工作流不存在: wf_missing000');
  });

  it('observability 顶层计数缺失时从 stages 求和（retry_count=0 时）', async () => {
    app = await createTestApp();
    const snap = completedSnapshot('wf_sum0000001');
    snap.retry_count = 0;
    snap.same_provider_retry_count = 0;
    snap.result_rejection_count = 0;
    store.upsertSnapshot(snap);
    const res = await app.inject({ method: 'GET', url: '/workflows/wf_sum0000001' });
    const o = res.json().observability;
    expect(o.retry_count).toBe(2); // stages 求和 1+0+1
    expect(o.same_provider_retry_count).toBe(1);
    expect(o.result_rejection_count).toBe(1);
  });

  // ---------- POST /workflows/:id/cancel ----------

  it('POST cancel 非法 ID → 400', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/workflows/not-a-wf-id/cancel' });
    expect(res.statusCode).toBe(400);
    expect(res.json().detail).toBe(
      'workflow_id must start with wf_ and contain 8-96 safe characters',
    );
  });

  it('POST cancel 未注册 ID → cancel_pending/pending_registration', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'POST', url: '/workflows/wf_unreg000001/cancel' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      status: 'cancel_pending',
      workflow: {
        workflow_id: 'wf_unreg000001',
        state: 'pending_registration',
        cancel_requested: true,
      },
    });
  });

  it('POST cancel 运行中 → cancel_requested + 本地标记持久化', async () => {
    app = await createTestApp();
    store.upsertSnapshot(makeSnapshot({ workflow_id: 'wf_running001', state: 'running' }));
    const res = await app.inject({ method: 'POST', url: '/workflows/wf_running001/cancel' });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('cancel_requested');
    expect(res.json().workflow.cancel_requested).toBe(true);
    expect(store.getSnapshot('wf_running001')!.cancel_requested).toBe(true);
  });

  it('POST cancel 终态 → 返回 state（completed）', async () => {
    app = await createTestApp();
    store.upsertSnapshot(completedSnapshot('wf_done000001'));
    const res = await app.inject({ method: 'POST', url: '/workflows/wf_done000001/cancel' });
    expect(res.statusCode).toBe(200);
    expect(res.json().status).toBe('completed');
    expect(res.json().workflow.cancel_requested).toBe(false);
  });

  // ---------- POST /workflows/journal/cleanup ----------

  it('cleanup 只删终态（age + limit 双策略）', async () => {
    app = await createTestApp();
    // 直接写存储文件控制 _updated_at（对齐 SQLite updated_at 列语义）
    const oldTerminal = completedSnapshot('wf_oldterm001');
    const newTerminal = completedSnapshot('wf_newterm001');
    const running = makeSnapshot({ workflow_id: 'wf_running002', state: 'running' });
    const now = Date.now() / 1000;
    fs.writeFileSync(
      tmpFile,
      JSON.stringify(
        [
          { _updated_at: now - 40 * 86400, snapshot: oldTerminal },
          { _updated_at: now - 1, snapshot: newTerminal },
          { _updated_at: now, snapshot: running },
        ],
        null,
        2,
      ),
      'utf-8',
    );
    const res = await app.inject({
      method: 'POST',
      url: '/workflows/journal/cleanup?max_age_days=30',
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('completed');
    expect(body.policy).toEqual({ max_age_days: 30, max_records: 1000 });
    expect(body.result.deleted_workflows).toBe(1);
    expect(body.result.deleted_by_age).toBe(1);
    expect(body.result.deleted_by_limit).toBe(0);
    expect(body.result.remaining_terminal).toBe(1); // wf_newterm001
    expect(store.getSnapshot('wf_oldterm001')).toBeNull();
    expect(store.getSnapshot('wf_newterm001')).not.toBeNull();
    expect(store.getSnapshot('wf_running002')).not.toBeNull(); // 非终态不删
  });

  it('cleanup 无参数 → 默认 policy + 无删除（返回 remaining_terminal）', async () => {
    app = await createTestApp();
    store.upsertSnapshot(completedSnapshot('wf_term000001'));
    const res = await app.inject({ method: 'POST', url: '/workflows/journal/cleanup' });
    const body = res.json();
    expect(body.policy).toEqual({ max_age_days: 30, max_records: 1000 });
    expect(body.result.deleted_workflows).toBe(0);
    expect(body.result.remaining_terminal).toBe(1);
  });

  it('cleanup 参数 clamp（>3650 → 3650，>100000 → 100000）', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/workflows/journal/cleanup?max_age_days=9999&max_records=999999',
    });
    expect(res.json().policy).toEqual({ max_age_days: 3650, max_records: 100000 });
  });

  // ---------- 持久化 ----------

  it('JSON 持久化：新 store 实例读同一文件（含 cancel 标记）', async () => {
    app = await createTestApp();
    store.upsertSnapshot(completedSnapshot('wf_persist001'));
    store.upsertSnapshot(makeSnapshot({ workflow_id: 'wf_running003', state: 'running' }));
    await app.inject({ method: 'POST', url: '/workflows/wf_running003/cancel' });
    const store2 = new WorkflowJournalStore(tmpFile);
    expect(store2.getSnapshot('wf_persist001')!.state).toBe('completed');
    expect(store2.getSnapshot('wf_running003')!.cancel_requested).toBe(true);
  });
});
