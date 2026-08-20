/**
 * 任务 — 推理队列、工作流列表与执行提供者。
 *
 * 覆盖加载 / 空 / 错误 / 无权限 / 成功五种状态；写操作有明确反馈（§5.3、Phase 3）。
 */

import { useCallback, useMemo, useState } from 'react';
import { Ban, Pause, Play, SlidersHorizontal } from 'lucide-react';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { TaskTable, type Column } from '../components/TaskTable';
import { StatusBadge } from '../components/StatusBadge';
import { CommandButton } from '../components/CommandButton';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { Drawer } from '../components/Drawer';
import { pushToast } from '../components/Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import { useReveal } from '../motion/useReveal';
import { formatDuration, formatRelative } from '../motion/countUp';
import { labelForState, toneForState } from '../components/statusTone';
import { useMyRole, useQueue, useWorkflows } from '../data/hooks';
import { fixturesEnabled } from '../data/fixtures';
import * as api from '../data/api';
import type { QueueTask, WorkflowRecord } from '../data/types';

/** 队列任务 + 所在层级，便于在一张表里展示 Q0/Q1/Q2。 */
interface LeveledTask extends QueueTask {
  level: 'Q0' | 'Q1' | 'Q2';
}

const WORKFLOW_FILTERS = [
  { id: 'all', label: '全部' },
  { id: 'running', label: '运行中' },
  { id: 'succeeded', label: '已完成' },
  { id: 'failed', label: '失败' },
] as const;

type WorkflowFilter = (typeof WORKFLOW_FILTERS)[number]['id'];

export function TasksPage() {
  const role = useMyRole();
  const queueEnabled = role.state === 'ready' && role.data?.is_master === true;
  const queue = useQueue(5_000, queueEnabled);
  const workflows = useWorkflows(20, 8_000);

  const [filter, setFilter] = useState<WorkflowFilter>('all');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowRecord | null>(null);
  const [busyAction, setBusyAction] = useState('');

  const refresh = useCallback(() => {
    role.refresh();
    queue.refresh();
    workflows.refresh();
  }, [role.refresh, queue.refresh, workflows.refresh]);
  useRegisterRefresh(refresh);

  useReveal([queue.data, workflows.data]);

  const usingFixtures = fixturesEnabled();

  /** fixture 模式下拦截写操作，避免误以为真的改了集群状态。 */
  const guardWrite = useCallback((): boolean => {
    if (usingFixtures) {
      pushToast('演示数据模式下不执行真实写操作。', 'warn');
      return false;
    }
    return true;
  }, [usingFixtures]);

  const runAction = useCallback(
    async (key: string, label: string, fn: () => Promise<unknown>) => {
      if (!guardWrite()) return;
      setBusyAction(key);
      try {
        await fn();
        pushToast(`${label}已生效。`, 'ok');
        queue.refresh();
      } catch (err) {
        pushToast(`${label}失败：${api.describeError(err)}`, 'danger');
      } finally {
        setBusyAction('');
      }
    },
    [guardWrite, queue.refresh],
  );

  const queueTasks: LeveledTask[] = useMemo(() => {
    const q = queue.data;
    if (!q) return [];
    return [
      ...q.q0.map((t) => ({ ...t, level: 'Q0' as const })),
      ...q.q1.map((t) => ({ ...t, level: 'Q1' as const })),
      ...q.q2.map((t) => ({ ...t, level: 'Q2' as const })),
    ];
  }, [queue.data]);

  const filteredWorkflows = useMemo(() => {
    const list = workflows.data?.workflows ?? [];
    if (filter === 'all') return list;
    return list.filter((w) => String(w.state || '').toLowerCase() === filter);
  }, [workflows.data, filter]);

  const queueColumns: Column<LeveledTask>[] = useMemo(
    () => [
      {
        key: 'task',
        header: '任务',
        render: (row) => <span className="cell-mono">{row.task_id || '—'}</span>,
      },
      {
        key: 'level',
        header: '层级',
        render: (row) => <StatusBadge label={row.level} tone={row.aged ? 'warn' : 'info'} size="sm" />,
      },
      {
        key: 'session',
        header: '会话',
        secondary: true,
        render: (row) => <span className="cell-mono">{row.session_id || '—'}</span>,
      },
      {
        key: 'wait',
        header: '等待',
        numeric: true,
        render: (row) => (
          <span className="num-display">{formatDuration(row.wait_s ?? 0)}</span>
        ),
      },
      {
        key: 'estimate',
        header: '预估',
        numeric: true,
        secondary: true,
        render: (row) => (
          <span className="num-display">{formatDuration(row.estimated_s ?? 0)}</span>
        ),
      },
      {
        key: 'tokens',
        header: '提示 tokens',
        numeric: true,
        secondary: true,
        render: (row) => <span className="num-display">{row.prompt_tokens ?? 0}</span>,
      },
    ],
    [],
  );

  const workflowColumns: Column<WorkflowRecord>[] = useMemo(
    () => [
      {
        key: 'id',
        header: '工作流',
        render: (row) => <span className="cell-mono">{row.workflow_id}</span>,
      },
      {
        key: 'state',
        header: '状态',
        render: (row) => (
          <StatusBadge state={row.state} pulse={String(row.state).toLowerCase() === 'running'} size="sm" />
        ),
      },
      {
        key: 'template',
        header: '模板',
        secondary: true,
        render: (row) => row.template || '—',
      },
      {
        key: 'stages',
        header: '阶段',
        numeric: true,
        render: (row) => {
          const stages = row.stages ?? [];
          const done = stages.filter((s) =>
            ['succeeded', 'success', 'completed'].includes(String(s.state).toLowerCase()),
          ).length;
          return (
            <span className="num-display">
              {done} / {stages.length}
            </span>
          );
        },
      },
      {
        key: 'updated',
        header: '更新',
        secondary: true,
        render: (row) => formatRelative(row.updated_at ?? 0),
      },
    ],
    [],
  );

  const toggleSelect = useCallback((key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelected((prev) => {
      const keys = queueTasks.map((t) => t.task_id || '').filter(Boolean);
      if (keys.length > 0 && keys.every((k) => prev.has(k))) return new Set();
      return new Set(keys);
    });
  }, [queueTasks]);

  const cancelSelected = useCallback(async () => {
    if (!guardWrite()) return;
    const ids = Array.from(selected);
    setBusyAction('bulk-cancel');
    let ok = 0;
    let failed = 0;
    for (const id of ids) {
      try {
        await api.cancelQueueTask(id);
        ok += 1;
      } catch {
        failed += 1;
      }
    }
    setBusyAction('');
    setSelected(new Set());
    queue.refresh();
    if (failed === 0) pushToast(`已取消 ${ok} 个任务。`, 'ok');
    else pushToast(`已取消 ${ok} 个，${failed} 个失败。`, failed === ids.length ? 'danger' : 'warn');
  }, [guardWrite, selected, queue.refresh]);

  const q = queue.data;
  // 403 表示当前节点不是主节点，属于「无权限」而不是错误。
  const queueDenied = role.state === 'ready' && role.data?.is_master !== true;
  const roleUnavailable = role.state === 'error';

  return (
    <>
      <PageHeader
        tag="TASKS"
        title="任务"
        description="推理请求的排队与执行情况。队列控制仅在主节点可用。"
        actions={
          <>
            <CommandButton
              variant="ghost"
              size="sm"
              icon={SlidersHorizontal}
              busy={busyAction === 'strategy'}
              onClick={() =>
                void runAction('strategy', '切换调度策略', () =>
                  api.setQueueStrategy(q?.strategy === 'mlfq' ? 'fifo' : 'mlfq'),
                )
              }
            >
              {q?.strategy === 'mlfq' ? '切换为 FIFO' : '切换为 MLFQ'}
            </CommandButton>
            {q?.paused ? (
              <CommandButton
                icon={Play}
                size="sm"
                busy={busyAction === 'resume'}
                onClick={() => void runAction('resume', '恢复队列', api.resumeQueue)}
              >
                恢复队列
              </CommandButton>
            ) : (
              <CommandButton
                icon={Pause}
                size="sm"
                variant="ghost"
                busy={busyAction === 'pause'}
                onClick={() => void runAction('pause', '暂停队列', api.pauseQueue)}
              >
                暂停队列
              </CommandButton>
            )}
          </>
        }
      />

      {/* ---- 队列概况 ---- */}
      <section className="band" data-reveal>
        <SectionHead
          title="推理队列"
          hint="三级反馈队列的实时深度与当前执行任务。"
          actions={
            q ? (
              <div className="chiprow">
                <StatusBadge
                  label={labelForState(q.strategy)}
                  tone="info"
                  size="sm"
                />
                <StatusBadge
                  label={q.paused ? '已暂停' : '运行中'}
                  tone={q.paused ? 'warn' : 'ok'}
                  pulse={!q.paused}
                  size="sm"
                />
              </div>
            ) : null
          }
        />

        {roleUnavailable ? (
          <EmptyState
            kind="error"
            title="节点角色不可用"
            description="无法判断队列权限，暂不请求主节点队列。"
            detail={role.error}
            action={
              <CommandButton variant="ghost" size="sm" onClick={role.refresh}>
                重试角色探针
              </CommandButton>
            }
          />
        ) : queueDenied ? (
          <EmptyState
            kind="denied"
            title="当前节点不使用主节点队列"
            description="请求队列只在主节点开放。工作流和本机对话仍可继续使用。"
            detail={role.data?.node_role ? `当前角色：${role.data.node_role}` : undefined}
          />
        ) : queue.state === 'loading' && !q ? (
          <SkeletonRows rows={3} columns={4} />
        ) : (
          <>
            {q ? (
              <div className="queuebar">
                <div className="queuebar__levels">
                  {(['q0', 'q1', 'q2'] as const).map((lv) => {
                    const depth = q[`${lv}_depth`];
                    const pct = q.max_size > 0 ? Math.min(100, (depth / q.max_size) * 100) : 0;
                    return (
                      <div className="qlevel" key={lv}>
                        <div className="qlevel__head">
                          <span className="mono-label">{lv.toUpperCase()}</span>
                          <span className="num-display">{depth}</span>
                        </div>
                        <div className="qlevel__track">
                          <span className="qlevel__fill" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <dl className="queuebar__facts">
                  <div>
                    <dt>当前任务</dt>
                    <dd className="cell-mono">{q.current_task?.task_id || '空闲'}</dd>
                  </div>
                  <div>
                    <dt>已完成</dt>
                    <dd className="num-display">{q.completed_count}</dd>
                  </div>
                  <div>
                    <dt>抢占次数</dt>
                    <dd className="num-display">{q.preempt_stats?.count ?? 0}</dd>
                  </div>
                  <div>
                    <dt>容量</dt>
                    <dd className="num-display">
                      {q.queue_size} / {q.max_size}
                    </dd>
                  </div>
                </dl>
              </div>
            ) : null}

            <TaskTable<LeveledTask>
              caption="队列中的推理任务"
              columns={queueColumns}
              rows={queueTasks}
              rowKey={(row) => row.task_id || `${row.level}-${row.session_id}`}
              state={queue.state}
              error={queue.error}
              emptyTitle="队列为空"
              emptyDescription="当前没有排队的推理请求。"
              onRetry={queue.refresh}
              selected={selected}
              onToggleSelect={toggleSelect}
              onToggleSelectAll={toggleSelectAll}
              isSelectable={(row) => Boolean(row.task_id)}
              bulkActions={
                <CommandButton
                  variant="danger"
                  size="sm"
                  icon={Ban}
                  busy={busyAction === 'bulk-cancel'}
                  onClick={() => void cancelSelected()}
                >
                  取消所选
                </CommandButton>
              }
            />
          </>
        )}
      </section>

      {/* ---- 工作流 ---- */}
      <section className="band band--alt" data-reveal>
        <SectionHead
          title="工作流"
          hint="任务图执行记录；点击详情查看每个阶段的状态。"
          actions={
            <div className="chiprow" role="group" aria-label="按状态筛选工作流">
              {WORKFLOW_FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  className="chip"
                  data-active={filter === f.id ? 'true' : undefined}
                  aria-pressed={filter === f.id}
                  onClick={() => setFilter(f.id)}
                >
                  {f.label}
                </button>
              ))}
            </div>
          }
        />

        {workflows.data && !workflows.data.enabled ? (
          <EmptyState
            kind="empty"
            title="任务图未启用"
            description="后端 TASK_GRAPH_ENABLED 为关闭状态，工作流不会产生记录。"
          />
        ) : (
          <TaskTable<WorkflowRecord>
            caption="任务图工作流记录"
            columns={workflowColumns}
            rows={filteredWorkflows}
            rowKey={(row) => row.workflow_id}
            state={workflows.state}
            error={workflows.error}
            emptyTitle={filter === 'all' ? '暂无工作流记录' : '该状态下没有工作流'}
            emptyDescription={
              filter === 'all'
                ? '发起带任务图的对话后，执行记录会出现在这里。'
                : '试试切换到「全部」。'
            }
            onRetry={workflows.refresh}
            onOpenRow={setActiveWorkflow}
          />
        )}
      </section>

      {/* ---- 执行提供者 ---- */}
      <section className="band" data-reveal>
        <SectionHead title="执行提供者" hint="可承接任务阶段的本地与远端提供者。" />
        {workflows.state === 'error' ? (
          <EmptyState kind="error" title="提供者状态不可用" detail={workflows.error} compact />
        ) : workflows.data?.provider_status?.length ? (
          <ul className="providerlist">
            {workflows.data.provider_status.map((p) => (
              <li className="provider" key={p.provider_id}>
                <div className="provider__head">
                  <span className="provider__id cell-mono">{p.provider_id}</span>
                  <StatusBadge
                    tone={p.healthy && p.available ? 'ok' : p.healthy ? 'warn' : 'danger'}
                    label={p.healthy && p.available ? '可用' : p.healthy ? '占用中' : '不健康'}
                    size="sm"
                  />
                </div>
                <p className="provider__meta">
                  节点 {p.node_id || '—'} · 并发 {p.active_reservations ?? 0}/{p.max_concurrency ?? 1}
                </p>
                <p className="provider__stages">
                  支持阶段：{(p.supported_stage_types ?? []).join('、') || '—'}
                </p>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState kind="empty" title="没有已注册的提供者" compact />
        )}

        {workflows.data?.provider_error ? (
          <p className="inline-error" role="alert">
            提供者错误：{workflows.data.provider_error}
          </p>
        ) : null}
      </section>

      {/* ---- 工作流详情抽屉 ---- */}
      <Drawer
        open={activeWorkflow !== null}
        tag="WORKFLOW"
        title={activeWorkflow?.workflow_id ?? ''}
        onClose={() => setActiveWorkflow(null)}
        footer={
          activeWorkflow && ['running', 'pending'].includes(String(activeWorkflow.state).toLowerCase()) ? (
            <CommandButton
              variant="danger"
              size="sm"
              icon={Ban}
              busy={busyAction === 'cancel-wf'}
              onClick={() => {
                const id = activeWorkflow.workflow_id;
                void runAction('cancel-wf', '取消工作流', async () => {
                  await api.cancelWorkflow(id);
                  workflows.refresh();
                  setActiveWorkflow(null);
                });
              }}
            >
              取消工作流
            </CommandButton>
          ) : null
        }
      >
        {activeWorkflow ? (
          <>
            <dl className="kvlist">
              <div>
                <dt>状态</dt>
                <dd>
                  <StatusBadge state={activeWorkflow.state} size="sm" />
                </dd>
              </div>
              <div>
                <dt>会话</dt>
                <dd className="cell-mono">{activeWorkflow.session_id || '—'}</dd>
              </div>
              <div>
                <dt>模板</dt>
                <dd>{activeWorkflow.template || '—'}</dd>
              </div>
              <div>
                <dt>创建</dt>
                <dd>{formatRelative(activeWorkflow.created_at ?? 0)}</dd>
              </div>
              <div>
                <dt>更新</dt>
                <dd>{formatRelative(activeWorkflow.updated_at ?? 0)}</dd>
              </div>
            </dl>

            {activeWorkflow.error ? (
              <p className="inline-error" role="alert">
                {activeWorkflow.error}
              </p>
            ) : null}

            <h3 className="drawer__subtitle">执行阶段</h3>
            {(activeWorkflow.stages ?? []).length === 0 ? (
              <EmptyState kind="empty" title="该工作流没有阶段记录" compact />
            ) : (
              <ol className="stagelist">
                {(activeWorkflow.stages ?? []).map((stage, i) => {
                  const duration =
                    stage.started_at && stage.finished_at
                      ? formatDuration(stage.finished_at - stage.started_at)
                      : stage.started_at
                        ? '进行中'
                        : '—';
                  return (
                    <li className={`stage stage--${toneForState(stage.state)}`} key={stage.stage_id || i}>
                      <div className="stage__head">
                        <span className="stage__id cell-mono">{stage.stage_id || `阶段 ${i + 1}`}</span>
                        <StatusBadge state={stage.state} size="sm" />
                      </div>
                      <p className="stage__meta">
                        {stage.stage_type || '—'} · {stage.provider_id || '未分配'} ·{' '}
                        {stage.node_id || '—'} · {duration}
                      </p>
                      {stage.error ? (
                        <p className="stage__error">{stage.error}</p>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
            )}
          </>
        ) : null}
      </Drawer>
    </>
  );
}
