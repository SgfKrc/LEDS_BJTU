/**
 * 工作台 — 左控制台 / 右对话的分屏首页。
 *
 * 分工：左侧回答「集群现在什么状态」，右侧回答「让它做点什么」。
 * 两侧比例由用户拖动决定（见 SplitPane），比例记在 localStorage。
 *
 * 左栏是概览的浓缩版：只保留发请求前后真正需要盯的四件事
 * —— 关键数字、流水线准入、队列、最近事件。要看全貌去概览/任务/活动页。
 */

import { useCallback, useMemo, useState } from 'react';
import { ArrowUpRight, Gauge, ListTree, Radio } from 'lucide-react';
import { SplitPane } from '../components/SplitPane';
import { ChatPane } from '../components/ChatPane';
import { MetricStrip, type Metric } from '../components/MetricStrip';
import { StatusBadge } from '../components/StatusBadge';
import { EmptyState } from '../components/EmptyState';
import { CommandButton } from '../components/CommandButton';
import { useRegisterRefresh } from '../app/refreshBus';
import { routeHref } from '../app/routes';
import { labelForState, toneForLogLevel } from '../components/statusTone';
import { formatRelative } from '../motion/countUp';
import { useReveal } from '../motion/useReveal';
import {
  useClusterNodes,
  usePipelineCapacity,
  useQueue,
  useRecentLogs,
  useSystemStatus,
} from '../data/hooks';

/** 左栏的紧凑分区标题；比 SectionHead 更矮，适合窄栏。 */
function PanelHead({
  title,
  tag,
  hint,
  action,
}: {
  title: string;
  tag: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="panel__head">
      <div className="panel__headtext">
        <p className="panel__tag mono-label">{tag}</p>
        <h2 className="panel__title">{title}</h2>
        {hint ? <p className="panel__hint">{hint}</p> : null}
      </div>
      {action ? <div className="panel__headaction">{action}</div> : null}
    </div>
  );
}

export function WorkbenchPage() {
  const status = useSystemStatus();
  const nodes = useClusterNodes();
  const queue = useQueue(8_000);
  const capacity = usePipelineCapacity();
  const logs = useRecentLogs({ limit: 8 }, 15_000);
  const [lastTurnAt, setLastTurnAt] = useState(0);

  const refresh = useCallback(() => {
    status.refresh();
    nodes.refresh();
    queue.refresh();
    capacity.refresh();
    logs.refresh();
  }, [status.refresh, nodes.refresh, queue.refresh, capacity.refresh, logs.refresh]);
  useRegisterRefresh(refresh);

  // 对话产生新一轮后立刻刷新左栏：这是分屏的核心价值，
  // 用户能直接看到自己那句话对队列/日志的影响，而不用等轮询。
  const onTurnComplete = useCallback(() => {
    setLastTurnAt(Date.now());
    queue.refresh();
    logs.refresh();
    status.refresh();
  }, [queue.refresh, logs.refresh, status.refresh]);

  // MetricStrip 带 [data-reveal]，没有这行它会一直停在隐藏态。
  useReveal([status.data, queue.data]);

  const s = status.data;
  const onlineCount = nodes.data?.online_count ?? 0;
  const totalNodes = nodes.data?.count ?? 0;
  const cap = capacity.data;

  const metrics: Metric[] = useMemo(() => {
    const gpuUtil = s?.gpu?.utilization ?? 0;
    const kvUtil = (s?.kv_cache?.utilization ?? 0) * 100;
    const queueDepth = queue.data?.queue_size ?? 0;
    return [
      {
        label: '在线节点',
        value: onlineCount,
        suffix: totalNodes ? ` / ${totalNodes}` : '',
        hint: onlineCount === totalNodes ? '全部在线' : '存在离线节点',
        tone: totalNodes > 0 && onlineCount < totalNodes ? 'warn' : 'ok',
      },
      {
        label: '显存',
        value: gpuUtil,
        suffix: '%',
        decimals: 1,
        hint: s?.gpu?.name ? s.gpu.name.replace('NVIDIA GeForce ', '') : '无 GPU',
        tone: gpuUtil > 85 ? 'danger' : gpuUtil > 60 ? 'warn' : 'ok',
      },
      {
        label: 'KV 缓存',
        value: kvUtil,
        suffix: '%',
        decimals: 1,
        hint: `${s?.kv_cache?.allocated_pages ?? 0}/${s?.kv_cache?.max_pages ?? 0} 页`,
        tone: kvUtil > 85 ? 'danger' : kvUtil > 60 ? 'warn' : 'ok',
      },
      {
        label: '队列深度',
        value: queueDepth,
        hint:
          queue.state === 'error'
            ? '队列接口不可用'
            : `已完成 ${queue.data?.completed_count ?? 0}`,
        tone: queueDepth > 0 ? 'info' : 'idle',
      },
    ];
  }, [s, onlineCount, totalNodes, queue.data, queue.state]);

  const statusSentence = (() => {
    if (status.state === 'loading') return '正在读取集群状态…';
    if (status.state === 'error') return '无法读取集群状态。';
    if (!s) return '暂无状态数据。';
    const modelPart = s.model_loaded
      ? `已加载 ${s.model_name}`
      : s.pipeline_prepared
        ? '流水线已就绪，尚未加载权重'
        : '尚未加载模型';
    return `${labelForState(s.node_role)} · ${labelForState(s.run_mode)}模式 · ${modelPart}。`;
  })();

  const queueDenied = queue.state === 'error' && queue.error.startsWith('无权限');
  const recentLogs = logs.data?.logs ?? [];

  const left = (
    <div className="console" data-enter>
      <header className="console__head">
        <div className="console__headmain">
          <p className="console__tag mono-label">NODE / {s?.node_id ?? '—'}</p>
          <h1 className="console__title">控制台</h1>
          <p className="console__status" aria-live="polite">
            {statusSentence}
          </p>
        </div>
        <CommandButton
          href={routeHref('overview')}
          variant="ghost"
          size="sm"
          icon={ArrowUpRight}
        >
          完整概览
        </CommandButton>
      </header>

      <MetricStrip metrics={metrics} caption="集群关键指标" />

      {/* ---- 流水线准入 ---- */}
      <section className="panel" aria-labelledby="wb-capacity">
        <PanelHead
          tag="PIPELINE"
          title="流水线准入"
          hint="决定这次请求走分布式还是回落单机。"
        />
        <span id="wb-capacity" className="sr-only">
          流水线准入
        </span>
        {capacity.state === 'error' ? (
          <EmptyState kind="error" title="准入信息不可用" detail={capacity.error} compact />
        ) : cap ? (
          <div className="capsule" data-tone={cap.status === 'ready' ? 'ok' : 'warn'}>
            <div className="capsule__top">
              <StatusBadge
                label={cap.status === 'ready' ? '已准入' : '未准入'}
                tone={cap.status === 'ready' ? 'ok' : 'warn'}
                size="sm"
              />
              <p className="capsule__reason">{cap.reason || '—'}</p>
            </div>
            <dl className="capsule__specs">
              <div>
                <dt>模型</dt>
                <dd className="mono-label">{cap.model_id || '—'}</dd>
              </div>
              <div>
                <dt>参与节点</dt>
                <dd className="num-display">
                  {cap.participating_node_count ?? 0} / {cap.candidate_node_count ?? 0}
                </dd>
              </div>
              <div>
                <dt>权重体积</dt>
                <dd className="num-display">
                  {cap.raw_model_bytes
                    ? `${(cap.raw_model_bytes / 1024 ** 3).toFixed(1)} GB`
                    : '—'}
                </dd>
              </div>
            </dl>
          </div>
        ) : (
          <EmptyState kind="loading" title="读取准入状态" compact />
        )}
      </section>

      {/* ---- 队列 ---- */}
      <section className="panel" aria-labelledby="wb-queue">
        <PanelHead
          tag="QUEUE"
          title="推理队列"
          hint={
            queue.data?.paused
              ? '队列已暂停，新请求不会被调度。'
              : '三级队列的当前深度。'
          }
          action={
            <CommandButton href={routeHref('tasks')} variant="ghost" size="sm" icon={ListTree}>
              任务页
            </CommandButton>
          }
        />
        <span id="wb-queue" className="sr-only">
          推理队列
        </span>
        {queueDenied ? (
          <EmptyState
            kind="denied"
            title="当前节点无权查看队列"
            description="队列接口仅主节点开放。"
            compact
          />
        ) : queue.state === 'error' ? (
          <EmptyState kind="error" title="队列信息不可用" detail={queue.error} compact />
        ) : queue.data ? (
          <ul className="qbars">
            {(['q0', 'q1', 'q2'] as const).map((lv) => {
              const depth = queue.data?.[`${lv}_depth`] ?? 0;
              const max = queue.data?.max_size || 1;
              const pct = Math.min(100, (depth / max) * 100);
              return (
                <li
                  className="qbars__item"
                  key={lv}
                  data-tone={pct > 85 ? 'danger' : pct > 60 ? 'warn' : undefined}
                >
                  <span className="qbars__label mono-label">{lv.toUpperCase()}</span>
                  <span className="qbars__track" aria-hidden="true">
                    <span className="qbars__fill" style={{ width: `${pct}%` }} />
                  </span>
                  <span className="qbars__value num-display">{depth}</span>
                </li>
              );
            })}
          </ul>
        ) : (
          <EmptyState kind="loading" title="读取队列" compact />
        )}
      </section>

      {/* ---- 最近事件 ---- */}
      <section className="panel" aria-labelledby="wb-events">
        <PanelHead
          tag="EVENTS"
          title="最近事件"
          hint={lastTurnAt ? '已随最新一轮对话刷新。' : '每 15 秒刷新。'}
          action={
            <CommandButton href={routeHref('activity')} variant="ghost" size="sm" icon={Radio}>
              活动页
            </CommandButton>
          }
        />
        <span id="wb-events" className="sr-only">
          最近事件
        </span>
        {logs.state === 'error' ? (
          <EmptyState
            kind={logs.error.startsWith('无权限') ? 'denied' : 'error'}
            title={logs.error.startsWith('无权限') ? '日志需要管理令牌' : '日志不可用'}
            detail={logs.error}
            compact
          />
        ) : recentLogs.length === 0 ? (
          <EmptyState kind={logs.state === 'loading' ? 'loading' : 'empty'} title="暂无事件" compact />
        ) : (
          <ol className="evlist">
            {recentLogs.map((log) => (
              <li className={`evlist__item evlist__item--${toneForLogLevel(log.level)}`} key={log.seq}>
                <span className="evlist__time mono-label">{log.timestamp.slice(11)}</span>
                <span className="evlist__level mono-label">{log.level}</span>
                <span className="evlist__msg">{log.message}</span>
              </li>
            ))}
          </ol>
        )}
      </section>

      <footer className="console__foot">
        <span className="mono-label">
          <Gauge size={12} strokeWidth={2} aria-hidden="true" />{' '}
          最近更新 {status.updatedAt ? formatRelative(status.updatedAt / 1000) : '—'}
        </span>
      </footer>
    </div>
  );

  return (
    <SplitPane
      left={left}
      right={<ChatPane onTurnComplete={onTurnComplete} />}
      leftLabel="集群控制台"
      rightLabel="对话"
    />
  );
}
