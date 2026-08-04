/**
 * 任务图控制面控制器 — 阶段 3.2 任务图域（语义对齐 api_server.py:4571-4700）
 *
 * 端点（4 个，全部纯控制面；执行段在 inference-svc engine_host.py）：
 *   GET  /workflows                    → {enabled, available, role, templates,
 *                                       providers, provider_status,
 *                                       provider_error, worker_protocol,
 *                                       journal, workflows[]}
 *   POST /workflows/journal/cleanup    → {status:'completed', policy, result}
 *   GET  /workflows/:workflowId        → snapshot + observability + journal；404
 *   POST /workflows/:workflowId/cancel → 状态机（pending 登记 / 已注册非终态 /
 *                                       终态）；非法 ID 400
 *
 * observability 完整复刻 _workflow_observability（api_server.py:4475-4561，
 * 从快照推导，不依赖执行段运行时）。
 *
 * 降级说明（已记录计划文档）：role/provider 状态依赖 scheduler 与执行段——
 * role 恒 'master'、providers 空、worker_protocol 空对象；cancel 的真实跨
 * 进程取消需经 control-svc → inference-svc 消息中转（执行段不在本进程），
 * 此处仅更新本地 journal 的 cancel_requested 标记；journal 为 JSON 降级
 * （SQLite 真实数据读取待清理阶段桥接）。
 */
import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpException,
  Param,
  Post,
  Query,
} from '@nestjs/common';
import {
  TERMINAL_WORKFLOW_STATES,
  WorkflowJournalStore,
  WorkflowSnapshot,
  WORKFLOW_ID_PATTERN,
} from '../../data/workflow-journal-store';

@Controller()
export class WorkflowsController {
  constructor(private readonly journal: WorkflowJournalStore) {}

  // ---------- 列表（对齐 api_server.py:4571-4619） ----------

  @Get('workflows')
  listWorkflows(
    @Query('limit') limitRaw?: string,
    @Query('session_id') sessionId = '',
  ): Record<string, unknown> {
    const limit = this.parseInt(limitRaw, 20);
    const journal = this.journal.status();
    const workflows = this.journal
      .listSnapshots(limit, sessionId)
      .map((w) => this.publicWorkflow(w, journal));
    return {
      enabled: true,
      available: true, // 降级：执行段 provider 状态未迁移，恒可用
      role: 'master', // 降级：control-svc 假定部署于主节点
      templates: ['dual_candidate'],
      providers: [], // 降级：provider 注册表在执行段
      provider_status: [],
      provider_error: '',
      worker_protocol: {}, // 降级：scheduler 侧状态未迁移
      journal: journal,
      workflows,
    };
  }

  // ---------- journal 清理（对齐 api_server.py:4622-4653） ----------

  @Post('workflows/journal/cleanup')
  @HttpCode(200)
  cleanupJournal(
    @Query('max_age_days') ageRaw?: string,
    @Query('max_records') recordsRaw?: string,
  ): Record<string, unknown> {
    // clamp 对齐 :4631-4638（默认 30 天 / 1000 条，上限 3650 / 100000）
    const ageDays = this.clampInt(ageRaw, 30, 0, 3650);
    const maxRecords = this.clampInt(recordsRaw, 1000, 0, 100000);
    const result = this.journal.cleanupTerminal(ageDays, maxRecords);
    return {
      status: 'completed',
      policy: { max_age_days: ageDays, max_records: maxRecords },
      result,
    };
  }

  // ---------- 详情（对齐 api_server.py:4664-4673） ----------

  @Get('workflows/:workflowId')
  getWorkflow(@Param('workflowId') workflowId: string): Record<string, unknown> {
    const workflow = this.journal.getSnapshot(workflowId);
    if (!workflow) {
      throw new HttpException(`工作流不存在: ${workflowId}`, 404);
    }
    return this.publicWorkflow(workflow, this.journal.status());
  }

  // ---------- 取消（对齐 api_server.py:4676-4700 + request_cancel） ----------

  @Post('workflows/:workflowId/cancel')
  @HttpCode(200)
  cancelWorkflow(@Param('workflowId') workflowId: string): Record<string, unknown> {
    if (!WORKFLOW_ID_PATTERN.test(workflowId)) {
      // 对齐 task_graph.py:2852 的 TaskGraphError → 400
      throw new HttpException(
        'workflow_id must start with wf_ and contain 8-96 safe characters',
        400,
      );
    }
    const workflow = this.journal.getSnapshot(workflowId);
    if (!workflow) {
      // 未注册 ID：fencing（对齐 request_cancel 返回 None 的分支）
      return {
        status: 'cancel_pending',
        workflow: {
          workflow_id: workflowId,
          state: 'pending_registration',
          cancel_requested: true,
        },
      };
    }
    let updated: WorkflowSnapshot;
    if (!TERMINAL_WORKFLOW_STATES.includes(workflow.state)) {
      // 本地标记（真实跨进程取消需经 inference-svc 消息中转，遗留记录）
      updated = { ...workflow, cancel_requested: true };
      this.journal.upsertSnapshot(updated);
    } else {
      updated = workflow;
    }
    return {
      status: TERMINAL_WORKFLOW_STATES.includes(updated.state)
        ? updated.state
        : 'cancel_requested',
      workflow: updated,
    };
  }

  // ---------- 工具 ----------

  /** 对齐 _public_workflow：snapshot + observability + journal */
  private publicWorkflow(
    snapshot: WorkflowSnapshot,
    journal: Record<string, unknown>,
  ): Record<string, unknown> {
    return {
      ...snapshot,
      observability: this.observability(snapshot),
      journal: this.publicJournal(journal),
    };
  }

  /** 对齐 _public_task_journal：剔除 path（JSON 降级无此字段，原样返回） */
  private publicJournal(status: Record<string, unknown>): Record<string, unknown> {
    const { path: _path, ...rest } = status;
    return rest;
  }

  /** 完整复刻 _workflow_observability（api_server.py:4475-4561） */
  private observability(snapshot: WorkflowSnapshot): Record<string, unknown> {
    const stages = Array.isArray(snapshot.stages) ? snapshot.stages : [];
    const attempts = stages.flatMap((s) => s.attempts ?? []);
    let retryCount = this.safeCount(snapshot.retry_count);
    if (!retryCount) {
      retryCount = stages.reduce((sum, s) => sum + this.safeCount(s.retry_count), 0);
    }
    let sameProviderRetry = this.safeCount(snapshot.same_provider_retry_count);
    if (!sameProviderRetry) {
      sameProviderRetry = stages.reduce(
        (sum, s) => sum + this.safeCount(s.same_provider_retry_count),
        0,
      );
    }
    let rejectionCount = this.safeCount(snapshot.result_rejection_count);
    if (!rejectionCount) {
      rejectionCount = stages.reduce(
        (sum, s) => sum + this.safeCount(s.result_rejection_count),
        0,
      );
    }
    const rejectedStages = stages
      .filter((s) => s.last_result_rejection_reason)
      .sort(
        (a, b) =>
          this.safeTimestamp(a.last_result_rejected_at) -
          this.safeTimestamp(b.last_result_rejected_at),
      );
    const lastRejection = rejectedStages[rejectedStages.length - 1] ?? {};
    const actualProviders = [
      ...new Set(attempts.map((a) => String(a.provider ?? '')).filter(Boolean)),
    ].sort();
    const actualNodes = [
      ...new Set(attempts.map((a) => String(a.provider_node_id ?? '')).filter(Boolean)),
    ].sort();
    const state = String(snapshot.state ?? 'unknown');
    const recoveredAfterRestart = Boolean(snapshot.recovered_after_restart ?? false);
    return {
      state,
      result_ready: state === 'result_ready',
      terminal: TERMINAL_WORKFLOW_STATES.includes(state as WorkflowSnapshot['state']),
      partial_result: Boolean(snapshot.partial_result),
      recovered_after_restart: recoveredAfterRestart,
      recovery_reason: recoveredAfterRestart
        ? String(snapshot.recovery_reason ?? snapshot.error_code ?? '')
        : '',
      retry_count: retryCount,
      same_provider_retry_count: sameProviderRetry,
      reassignment_count: Math.max(0, retryCount - sameProviderRetry),
      retrying: retryCount > 0 && (state === 'running' || state === 'created'),
      result_rejection_count: rejectionCount,
      last_result_rejection_reason: String(
        lastRejection.last_result_rejection_reason ?? '',
      ),
      last_result_rejected_at: lastRejection.last_result_rejected_at ?? null,
      winner_count: stages.filter((s) => s.winner_attempt_id).length,
      actual_providers: actualProviders,
      actual_nodes: actualNodes,
    };
  }

  private safeCount(value: unknown): number {
    const n = Number(value);
    return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
  }

  private safeTimestamp(value: unknown): number {
    const t = Number(value);
    return Number.isFinite(t) && t >= 0 && t <= 1e15 ? t : 0;
  }

  private parseInt(raw: string | undefined, def: number): number {
    if (raw === undefined || raw === '') return def;
    const n = Number(raw);
    return Number.isFinite(n) ? Math.max(1, Math.floor(n)) : def;
  }

  private clampInt(raw: string | undefined, def: number, min: number, max: number): number {
    if (raw === undefined || raw === '') return def;
    const n = Number(raw);
    if (!Number.isFinite(n)) return def;
    return Math.max(min, Math.min(Math.floor(n), max));
  }
}
