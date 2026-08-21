/** Task operations workspace: queue controls, workflow index, and a persistent execution detail. */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Ban, Pause, Play, RefreshCw, SlidersHorizontal } from 'lucide-react';
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
import type { QueueResponse, QueueTask, WorkflowRecord } from '../data/types';
import { PageBackdrop } from '../visual/PageBackdrop';

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
type WorkspaceId = 'queue' | 'workflows';

function QueueDetail({ task }: { task: LeveledTask }) {
  return (
    <dl className="kvlist">
      <div><dt>任务 ID</dt><dd className="cell-mono">{task.task_id || '—'}</dd></div>
      <div><dt>优先级</dt><dd><StatusBadge label={task.level} tone={task.aged ? 'warn' : 'info'} size="sm" /></dd></div>
      <div><dt>会话</dt><dd className="cell-mono">{task.session_id || '—'}</dd></div>
      <div><dt>等待</dt><dd className="num-display">{formatDuration(task.wait_s ?? 0)}</dd></div>
      <div><dt>预计</dt><dd className="num-display">{formatDuration(task.estimated_s ?? 0)}</dd></div>
      <div><dt>提示 tokens</dt><dd className="num-display">{task.prompt_tokens ?? 0}</dd></div>
    </dl>
  );
}

function WorkflowDetail({ workflow }: { workflow: WorkflowRecord }) {
  return (
    <>
      <dl className="kvlist">
        <div><dt>状态</dt><dd><StatusBadge state={workflow.state} size="sm" /></dd></div>
        <div><dt>会话</dt><dd className="cell-mono">{workflow.session_id || '—'}</dd></div>
        <div><dt>模板</dt><dd>{workflow.template || '—'}</dd></div>
        <div><dt>创建</dt><dd>{formatRelative(workflow.created_at ?? 0)}</dd></div>
        <div><dt>更新</dt><dd>{formatRelative(workflow.updated_at ?? 0)}</dd></div>
      </dl>
      {workflow.error ? <p className="inline-error" role="alert">{workflow.error}</p> : null}
      <h3 className="drawer__subtitle">执行阶段</h3>
      {(workflow.stages ?? []).length === 0 ? <EmptyState kind="empty" title="没有阶段记录" compact /> : <ol className="stagelist">{(workflow.stages ?? []).map((stage, index) => {
        const duration = stage.started_at && stage.finished_at ? formatDuration(stage.finished_at - stage.started_at) : stage.started_at ? '进行中' : '—';
        return <li className={`stage stage--${toneForState(stage.state)}`} key={stage.stage_id || index}><div className="stage__head"><span className="stage__id cell-mono">{stage.stage_id || `阶段 ${index + 1}`}</span><StatusBadge state={stage.state} size="sm" /></div><p className="stage__meta">{stage.stage_type || '—'} · {stage.provider_id || '未分配'} · {stage.node_id || '—'} · {duration}</p>{stage.error ? <p className="stage__error">{stage.error}</p> : null}</li>;
      })}</ol>}
    </>
  );
}

function QueueOverview({ queue }: { queue: QueueResponse }) {
  return (
    <div className="queuebar tasks-queuebar">
      <div className="queuebar__levels">
        {(['q0', 'q1', 'q2'] as const).map((level) => {
          const depth = queue[`${level}_depth`];
          const pct = queue.max_size > 0 ? Math.min(100, (depth / queue.max_size) * 100) : 0;
          return <div className="qlevel" key={level}><div className="qlevel__head"><span className="mono-label">{level.toUpperCase()}</span><span className="num-display">{depth}</span></div><div className="qlevel__track"><span className="qlevel__fill" style={{ width: `${pct}%` }} /></div></div>;
        })}
      </div>
      <dl className="queuebar__facts">
        <div><dt>当前任务</dt><dd className="cell-mono">{queue.current_task?.task_id || '空闲'}</dd></div>
        <div><dt>已完成</dt><dd className="num-display">{queue.completed_count}</dd></div>
        <div><dt>抢占次数</dt><dd className="num-display">{queue.preempt_stats?.count ?? 0}</dd></div>
        <div><dt>容量</dt><dd className="num-display">{queue.queue_size} / {queue.max_size}</dd></div>
      </dl>
    </div>
  );
}

export function TasksPage() {
  const role = useMyRole();
  const canManageQueue = role.state === 'ready' && role.data?.is_master === true;
  const queue = useQueue(5_000, canManageQueue);
  const workflows = useWorkflows(20, 8_000);
  const usingFixtures = fixturesEnabled();

  const [workspace, setWorkspace] = useState<WorkspaceId>('queue');
  const [filter, setFilter] = useState<WorkflowFilter>('all');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowRecord | null>(null);
  const [activeTask, setActiveTask] = useState<LeveledTask | null>(null);
  const [busyAction, setBusyAction] = useState('');
  const [compactDetails, setCompactDetails] = useState(false);

  useEffect(() => {
    const media = window.matchMedia('(max-width: 1199px)');
    const update = () => setCompactDetails(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);

  const refresh = useCallback(() => {
    role.refresh();
    queue.refresh();
    workflows.refresh();
  }, [queue, role, workflows]);
  useRegisterRefresh(refresh);
  useReveal([queue.data, workflows.data]);

  const guardWrite = useCallback((): boolean => {
    if (usingFixtures) {
      pushToast('演示数据模式下不执行真实写操作。', 'warn');
      return false;
    }
    return true;
  }, [usingFixtures]);

  const runAction = useCallback(async (key: string, label: string, fn: () => Promise<unknown>) => {
    if (!guardWrite()) return;
    setBusyAction(key);
    try {
      await fn();
      pushToast(`${label}已生效。`, 'ok');
      queue.refresh();
      workflows.refresh();
    } catch (error) {
      pushToast(`${label}失败：${api.describeError(error)}`, 'danger');
    } finally {
      setBusyAction('');
    }
  }, [guardWrite, queue, workflows]);

  const q = queue.data;
  const queueTasks: LeveledTask[] = useMemo(() => !q ? [] : [
    ...q.q0.map((task) => ({ ...task, level: 'Q0' as const })),
    ...q.q1.map((task) => ({ ...task, level: 'Q1' as const })),
    ...q.q2.map((task) => ({ ...task, level: 'Q2' as const })),
  ], [q]);
  const filteredWorkflows = useMemo(() => {
    const list = workflows.data?.workflows ?? [];
    return filter === 'all' ? list : list.filter((workflow) => String(workflow.state || '').toLowerCase() === filter);
  }, [filter, workflows.data]);

  const queueColumns: Column<LeveledTask>[] = useMemo(() => [
    { key: 'task', header: '任务', render: (row) => <span className="cell-mono">{row.task_id || '—'}</span> },
    { key: 'level', header: '层级', render: (row) => <StatusBadge label={row.level} tone={row.aged ? 'warn' : 'info'} size="sm" /> },
    { key: 'session', header: '会话', secondary: true, render: (row) => <span className="cell-mono">{row.session_id || '—'}</span> },
    { key: 'wait', header: '等待', numeric: true, render: (row) => <span className="num-display">{formatDuration(row.wait_s ?? 0)}</span> },
    { key: 'estimate', header: '预计', numeric: true, secondary: true, render: (row) => <span className="num-display">{formatDuration(row.estimated_s ?? 0)}</span> },
  ], []);
  const workflowColumns: Column<WorkflowRecord>[] = useMemo(() => [
    { key: 'id', header: '工作流', render: (row) => <span className="cell-mono">{row.workflow_id}</span> },
    { key: 'state', header: '状态', render: (row) => <StatusBadge state={row.state} pulse={String(row.state).toLowerCase() === 'running'} size="sm" /> },
    { key: 'template', header: '模板', secondary: true, render: (row) => row.template || '—' },
    { key: 'stages', header: '阶段', numeric: true, render: (row) => <span className="num-display">{(row.stages ?? []).filter((stage) => ['succeeded', 'success', 'completed'].includes(String(stage.state).toLowerCase())).length} / {(row.stages ?? []).length}</span> },
    { key: 'updated', header: '更新', secondary: true, render: (row) => formatRelative(row.updated_at ?? 0) },
  ], []);

  const toggleSelect = useCallback((key: string) => setSelected((current) => {
    const next = new Set(current);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  }), []);
  const toggleSelectAll = useCallback(() => setSelected((current) => {
    const ids = queueTasks.map((task) => task.task_id || '').filter(Boolean);
    return ids.length && ids.every((id) => current.has(id)) ? new Set() : new Set(ids);
  }), [queueTasks]);
  const cancelSelected = useCallback(async () => {
    if (!guardWrite()) return;
    const ids = Array.from(selected);
    setBusyAction('bulk-cancel');
    const outcomes = await Promise.allSettled(ids.map((id) => api.cancelQueueTask(id)));
    const success = outcomes.filter((result) => result.status === 'fulfilled').length;
    setSelected(new Set());
    setBusyAction('');
    queue.refresh();
    pushToast(success === ids.length ? `已取消 ${success} 个任务。` : `已取消 ${success} 个，${ids.length - success} 个失败。`, success === ids.length ? 'ok' : 'warn');
  }, [guardWrite, queue, selected]);

  const cleanupJournal = useCallback(async () => {
    if (!guardWrite() || !window.confirm('Clean task journal records older than the retention window?')) return;
    setBusyAction('journal-cleanup');
    try {
      await api.cleanupWorkflowJournal({ max_age_days: 30, max_records: 5000 });
      pushToast('Task journal cleanup completed', 'ok');
      workflows.refresh();
    } catch (error) {
      pushToast(`Journal cleanup failed: ${api.describeError(error)}`, 'danger');
    } finally { setBusyAction(''); }
  }, [guardWrite, workflows]);

  const selectTask = useCallback((task: LeveledTask) => {
    setActiveTask(task);
    setActiveWorkflow(null);
  }, []);
  const selectWorkflow = useCallback((workflow: WorkflowRecord) => {
    setActiveWorkflow(workflow);
    setActiveTask(null);
  }, []);
  const detailTitle = activeWorkflow?.workflow_id || activeTask?.task_id || '';
  const detailContent = activeWorkflow ? <WorkflowDetail workflow={activeWorkflow} /> : activeTask ? <QueueDetail task={activeTask} /> : <EmptyState kind="empty" title="未选择执行项" description="从队列或工作流表中选择一行以固定详情。" compact />;
  const queueDenied = role.state === 'ready' && !canManageQueue;

  return (
    <div className="tasks-page" data-testid="tasks-page">
      <PageBackdrop scene="tasks" className="tasks-page__bg" />
      <PageHeader tag="TASKS" title="任务" description="队列控制、工作流执行与阶段详情分区呈现，避免上下文被长列表冲散。" actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={queue.refreshing || workflows.refreshing} onClick={refresh}>刷新</CommandButton>} />

      <div className="tasks-layout">
        <aside className="tasks-rail" aria-label="Queue controls and task navigation">
          <section className="tasks-panel tasks-identity">
            <span className="mono-label">SCHEDULER LINK</span>
            <strong>{role.data?.node_id || 'detecting node'}</strong>
            <StatusBadge label={canManageQueue ? 'MASTER / QUEUE CONTROL' : 'WORKFLOW READ ONLY'} tone={canManageQueue ? 'ok' : 'info'} size="sm" />
            <p>{canManageQueue ? 'Queue controls are available on this primary node.' : 'Workflow history remains readable without the primary queue.'}</p>
          </section>

          <nav className="tasks-nav" aria-label="Task workspaces">
            <button type="button" data-active={workspace === 'queue' ? 'true' : undefined} onClick={() => setWorkspace('queue')}><span>Inference queue</span><em>{q?.queue_size ?? '—'}</em></button>
            <button type="button" data-active={workspace === 'workflows' ? 'true' : undefined} onClick={() => setWorkspace('workflows')}><span>Workflows</span><em>{(workflows.data?.workflows.length ?? 0).toString().padStart(2, '0')}</em></button>
          </nav>

          {canManageQueue ? <section className="tasks-panel tasks-controls"><SectionHead title="Queue control" hint={q ? `${labelForState(q.strategy)} · ${q.paused ? 'paused' : 'running'}` : 'Waiting for queue state'} />
            <div className="tasks-control-buttons"><CommandButton variant="ghost" size="sm" icon={SlidersHorizontal} busy={busyAction === 'strategy'} onClick={() => void runAction('strategy', '切换调度策略', () => api.setQueueStrategy(q?.strategy === 'mlfq' ? 'fifo' : 'mlfq'))}>{q?.strategy === 'mlfq' ? '切换为 FIFO' : '切换为 MLFQ'}</CommandButton>{q?.paused ? <CommandButton size="sm" icon={Play} busy={busyAction === 'resume'} onClick={() => void runAction('resume', '恢复队列', api.resumeQueue)}>恢复队列</CommandButton> : <CommandButton variant="ghost" size="sm" icon={Pause} busy={busyAction === 'pause'} onClick={() => void runAction('pause', '暂停队列', api.pauseQueue)}>暂停队列</CommandButton>}</div>
          </section> : <section className="tasks-panel tasks-controls"><SectionHead title="Queue authority" hint="Primary-only mutation domain" /><p className="tasks-readonly">This node cannot alter the primary queue.</p></section>}

          <section className="tasks-panel tasks-provider-list"><SectionHead title="Providers" hint="Stage execution availability" />
            {workflows.state === 'error' ? <EmptyState kind="error" title="提供者不可用" detail={workflows.error} compact /> : workflows.data?.provider_status?.length ? <ul className="tasks-providers">{workflows.data.provider_status.map((provider) => <li key={provider.provider_id}><div><strong className="cell-mono">{provider.provider_id}</strong><StatusBadge tone={provider.healthy && provider.available ? 'ok' : provider.healthy ? 'warn' : 'danger'} label={provider.healthy && provider.available ? 'READY' : provider.healthy ? 'BUSY' : 'DOWN'} size="sm" /></div><span>{provider.node_id || '—'} · {provider.active_reservations ?? 0}/{provider.max_concurrency ?? 1}</span></li>)}</ul> : <EmptyState kind="empty" title="没有提供者" compact />}
            {workflows.data?.provider_error ? <p className="inline-error" role="alert">{workflows.data.provider_error}</p> : null}
          </section>
        </aside>

        <main className="tasks-main">
          <section className="tasks-panel tasks-workspace" data-workspace={workspace}>
            {workspace === 'queue' ? <><SectionHead title="推理队列" hint="三级反馈队列的深度、排队任务与批量取消。" actions={q ? <div className="chiprow"><StatusBadge label={labelForState(q.strategy)} tone="info" size="sm" /><StatusBadge label={q.paused ? '已暂停' : '运行中'} tone={q.paused ? 'warn' : 'ok'} pulse={!q.paused} size="sm" /></div> : null} />
              <div className="tasks-workspace__scroll">
                {role.state === 'error' ? <EmptyState kind="error" title="节点角色不可用" description="暂不请求主节点队列。" detail={role.error} errorKind={role.errorKind} errorStatus={role.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={role.refresh}>重试角色探针</CommandButton>} /> : queueDenied ? <EmptyState kind="denied" title="当前节点不使用主节点队列" description="请求队列只在主节点开放；工作流历史仍可继续使用。" detail={role.data?.node_role ? `当前角色：${role.data.node_role}` : undefined} /> : queue.state === 'loading' && !q ? <SkeletonRows rows={4} columns={3} /> : q ? <><QueueOverview queue={q} /><div className="tasks-table-scroll"><TaskTable<LeveledTask> caption="队列中的推理任务" columns={queueColumns} rows={queueTasks} rowKey={(row) => row.task_id || `${row.level}-${row.session_id}`} state={queue.state} error={queue.error} errorKind={queue.errorKind} errorStatus={queue.errorStatus} emptyTitle="队列为空" emptyDescription="当前没有排队的推理请求。" onRetry={queue.refresh} selected={selected} onToggleSelect={toggleSelect} onToggleSelectAll={toggleSelectAll} isSelectable={(row) => Boolean(row.task_id)} bulkActions={<CommandButton variant="danger" size="sm" icon={Ban} busy={busyAction === 'bulk-cancel'} onClick={() => void cancelSelected()}>取消所选</CommandButton>} onOpenRow={selectTask} /></div></> : <EmptyState kind="empty" title="队列状态未返回" description="刷新后重试。" />}
              </div>
            </> : <><SectionHead title="工作流" hint="任务图执行记录；选择一项可查看阶段、提供者和失败信息。" actions={<div className="chiprow" role="group" aria-label="按状态筛选工作流">{canManageQueue ? <CommandButton variant="ghost" size="sm" busy={busyAction === 'journal-cleanup'} onClick={() => void cleanupJournal()}>Clean journal</CommandButton> : null}{WORKFLOW_FILTERS.map((entry) => <button key={entry.id} type="button" className="chip" data-active={filter === entry.id ? 'true' : undefined} aria-pressed={filter === entry.id} onClick={() => setFilter(entry.id)}>{entry.label}</button>)}</div>} />
              <div className="tasks-workspace__scroll tasks-workspace__scroll--table">{workflows.data && !workflows.data.enabled ? <EmptyState kind="empty" title="任务图未启用" description="后端 TASK_GRAPH_ENABLED 处于关闭状态。" /> : <TaskTable<WorkflowRecord> caption="任务图工作流记录" columns={workflowColumns} rows={filteredWorkflows} rowKey={(row) => row.workflow_id} state={workflows.state} error={workflows.error} errorKind={workflows.errorKind} errorStatus={workflows.errorStatus} emptyTitle={filter === 'all' ? '暂无工作流记录' : '该状态下没有工作流'} emptyDescription={filter === 'all' ? '发起带任务图的对话后，执行记录会出现在这里。' : '尝试切换到“全部”。'} onRetry={workflows.refresh} onOpenRow={selectWorkflow} />}</div>
            </>}
          </section>
        </main>

        <aside className="tasks-details" aria-label="Selected execution detail">
          <section className="tasks-panel tasks-detail-panel"><SectionHead title="执行详情" hint={activeWorkflow ? 'WORKFLOW' : activeTask ? `QUEUE ${activeTask.level}` : '选择一项'} />{detailContent}</section>
        </aside>
      </div>

      <Drawer open={compactDetails && (activeWorkflow !== null || activeTask !== null)} tag={activeWorkflow ? 'WORKFLOW' : 'QUEUE TASK'} title={detailTitle} onClose={() => { setActiveWorkflow(null); setActiveTask(null); }} footer={activeWorkflow && ['running', 'pending'].includes(String(activeWorkflow.state).toLowerCase()) ? <CommandButton variant="danger" size="sm" icon={Ban} busy={busyAction === 'cancel-wf'} onClick={() => { const id = activeWorkflow.workflow_id; void runAction('cancel-wf', '取消工作流', async () => { await api.cancelWorkflow(id); setActiveWorkflow(null); }); }}>取消工作流</CommandButton> : activeTask?.task_id ? <CommandButton variant="danger" size="sm" icon={Ban} busy={busyAction === 'cancel-task'} onClick={() => { const id = activeTask.task_id || ''; if (!id) return; void runAction('cancel-task', '取消任务', async () => { await api.cancelQueueTask(id); setActiveTask(null); }); }}>取消任务</CommandButton> : undefined}>{activeWorkflow ? <WorkflowDetail workflow={activeWorkflow} /> : activeTask ? <QueueDetail task={activeTask} /> : null}</Drawer>
    </div>
  );
}
