/**
 * 概览 — 首屏「当前状态 + 一项主行动 + 关键数字」三段式（§4.4）。
 *
 * 不使用全屏营销口号；首屏底部露出下一段标题，提示可继续滚动。
 */

import { useCallback, useMemo } from 'react';
import { ArrowRight, Cpu, HardDrive, Layers, Server } from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { MetricStrip, type Metric } from '../components/MetricStrip';
import { StatusBadge } from '../components/StatusBadge';
import { SectionHead } from '../components/PageHeader';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { ActivityTimeline, type TimelineItem } from '../components/ActivityTimeline';
import { AccentCanvas } from '../visual/AccentCanvas';
import { useRegisterRefresh } from '../app/refreshBus';
import { routeHref } from '../app/routes';
import { useReveal } from '../motion/useReveal';
import { formatBytes, formatRelative } from '../motion/countUp';
import { labelForState, toneForLogLevel } from '../components/statusTone';
import {
  useClusterNodes,
  usePipelineCapacity,
  useQueue,
  useRecentLogs,
  useMyRole,
  useSystemStatus,
} from '../data/hooks';
import type { ClusterNode } from '../data/types';

function NodeCard({ node }: { node: ClusterNode }) {
  const dev = node.device_info || {};
  const gpu = dev.gpu;
  const ram = dev.ram;

  return (
    <li className="nodecard" data-reveal data-online={node.is_available ? 'true' : 'false'}>
      <div className="nodecard__head">
        <span className="nodecard__role mono-label">{labelForState(node.role)}</span>
        <StatusBadge state={node.state} size="sm" pulse={node.is_available} />
      </div>
      <p className="nodecard__name">{node.hostname || node.node_id}</p>
      <p className="nodecard__id">{node.node_id}</p>
      <dl className="nodecard__specs">
        <div>
          <dt>设备档位</dt>
          <dd>{dev.tier_label || dev.tier || '未知'}</dd>
        </div>
        <div>
          <dt>处理器</dt>
          <dd>{dev.cpu?.model_name || '未知'}</dd>
        </div>
        <div>
          <dt>显卡</dt>
          <dd>
            {gpu?.name || '无独立显卡'}
            {gpu?.vram_total_gb ? ` · ${gpu.vram_total_gb}GB` : ''}
          </dd>
        </div>
        <div>
          <dt>内存占用</dt>
          <dd>
            {ram?.percent_used != null
              ? `${ram.percent_used}%（${ram.used_gb ?? '?'}/${ram.total_gb ?? '?'}GB）`
              : '未知'}
          </dd>
        </div>
        <div>
          <dt>往返延迟</dt>
          <dd className="num-display">
            {node.is_available ? `${node.avg_rtt_ms.toFixed(1)} ms` : '—'}
          </dd>
        </div>
        <div>
          <dt>最近心跳</dt>
          <dd>{formatRelative(node.last_heartbeat)}</dd>
        </div>
      </dl>
      {node.error_count > 0 ? (
        <p className="nodecard__warn">累计错误 {node.error_count} 次</p>
      ) : null}
    </li>
  );
}

export function OverviewPage() {
  const status = useSystemStatus();
  const nodes = useClusterNodes();
  const role = useMyRole();
  const queueEnabled = role.state === 'ready' && role.data?.is_master === true;
  const queue = useQueue(8_000, queueEnabled);
  const capacity = usePipelineCapacity();
  const logs = useRecentLogs({ limit: 6 }, 15_000);
  const queueNotApplicable = role.state === 'ready' && role.data?.is_master !== true;
  const roleUnavailable = role.state === 'error';

  const refresh = useCallback(() => {
    status.refresh();
    nodes.refresh();
    role.refresh();
    queue.refresh();
    capacity.refresh();
    logs.refresh();
  }, [status.refresh, nodes.refresh, role.refresh, queue.refresh, capacity.refresh, logs.refresh]);
  useRegisterRefresh(refresh);

  useReveal([nodes.data, logs.data]);

  const s = status.data;
  const nodeList = nodes.data?.nodes ?? [];
  const onlineCount = nodes.data?.online_count ?? 0;
  const totalNodes = nodes.data?.count ?? 0;

  const metrics: Metric[] = useMemo(() => {
    const gpuUtil = s?.gpu?.utilization ?? 0;
    const kvUtil = (s?.kv_cache?.utilization ?? 0) * 100;
    const queueDepth = queueEnabled ? queue.data?.queue_size ?? 0 : 0;

    return [
      {
        label: '在线节点',
        value: onlineCount,
        suffix: totalNodes ? ` / ${totalNodes}` : '',
        hint: onlineCount === totalNodes ? '全部节点在线' : '存在离线节点',
        tone: totalNodes > 0 && onlineCount < totalNodes ? 'warn' : 'ok',
      },
      {
        label: '显存占用',
        value: gpuUtil,
        suffix: '%',
        decimals: 1,
        hint: s?.gpu?.name ? s.gpu.name.replace('NVIDIA GeForce ', '') : '无 GPU 信息',
        tone: gpuUtil > 85 ? 'danger' : gpuUtil > 60 ? 'warn' : 'ok',
      },
      {
        label: 'KV 缓存',
        value: kvUtil,
        suffix: '%',
        decimals: 1,
        hint: `${s?.kv_cache?.allocated_pages ?? 0} / ${s?.kv_cache?.max_pages ?? 0} 页`,
        tone: kvUtil > 85 ? 'danger' : kvUtil > 60 ? 'warn' : 'ok',
      },
      {
        label: '队列深度',
        value: queueDepth,
        hint: queueNotApplicable
          ? '单机模式不适用'
          : roleUnavailable
            ? '节点角色不可用'
          : role.state !== 'ready'
            ? '正在确认节点角色'
            : queue.state === 'error'
              ? '队列接口不可用'
              : `已完成 ${queue.data?.completed_count ?? 0}`,
        tone: queueDepth > 0 ? 'info' : 'idle',
      },
    ];
  }, [s, onlineCount, totalNodes, queue.data, queue.state, queueEnabled, queueNotApplicable, role.state, roleUnavailable]);

  const timelineItems: TimelineItem[] = useMemo(
    () =>
      (logs.data?.logs ?? []).map((log) => ({
        id: String(log.seq),
        time: log.timestamp.slice(11),
        title: log.name,
        detail: log.message,
        tone: toneForLogLevel(log.level),
        badge: log.level,
        ...(log.node_id ? { source: log.node_id } : {}),
      })),
    [logs.data],
  );

  // 首屏状态句：用一句中文说明当前集群在做什么。
  const statusSentence = (() => {
    if (status.state === 'loading') return '正在读取集群状态…';
    if (status.state === 'error') return '无法读取集群状态。';
    if (!s) return '暂无状态数据。';
    const mode = labelForState(s.run_mode);
    const role = labelForState(s.node_role);
    const modelPart = s.model_loaded
      ? `已加载 ${s.model_name}`
      : s.pipeline_prepared
        ? `流水线已就绪（${s.pipeline_descriptor?.model_id ?? s.model_name}），尚未加载权重`
        : '尚未加载模型';
    return `本机以${role}身份运行在${mode}模式，${modelPart}。`;
  })();

  const hasActiveWork = queueEnabled && ((queue.data?.queue_size ?? 0) > 0 || Boolean(queue.data?.current_task));

  return (
    <>
      {/* ---- Hero：状态 + 主行动 + 关键数字 ---- */}
      <section className="hero" data-enter>
        <div className="hero__grid">
          <div className="hero__main">
            <span className="hero__tag mono-label">NODE / {s?.node_id ?? '—'}</span>
            <h1 className="hero__title">
              集群概览
              <span className="hero__rule" aria-hidden="true" />
            </h1>
            <p className="hero__status lede" aria-live="polite">
              {statusSentence}
            </p>

            <div className="hero__actions">
              <CommandButton href={routeHref('tasks')} icon={ArrowRight}>
                查看任务队列
              </CommandButton>
              <CommandButton href={routeHref('activity')} variant="ghost" icon={Layers}>
                运行活动
              </CommandButton>
            </div>

            {s?.device?.warnings?.length ? (
              <ul className="hero__warnings">
                {s.device.warnings.slice(0, 2).map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            ) : null}
          </div>

          <div className="hero__visual">
            <AccentCanvas nodeCount={Math.max(2, totalNodes)} active={hasActiveWork} />
            <div className="hero__visual-labels">
              <span className="floatlabel floatlabel--a">
                <span className="floatlabel__dot" aria-hidden="true" />
                {labelForState(s?.run_mode)}
              </span>
              <span className="floatlabel floatlabel--b">
                {onlineCount} 节点在线
              </span>
            </div>
          </div>
        </div>

        <MetricStrip metrics={metrics} caption="集群关键指标" />

        {/* 首屏底部露出下一段标题（§4.4） */}
        <p className="hero__next" aria-hidden="true">
          向下查看节点与最近活动
        </p>
      </section>

      {/* ---- 流水线准入 ---- */}
      <section className="band" data-reveal>
        <SectionHead
          title="流水线准入"
          hint="分布式推理的层分配结果；未准入时会回落到单机全模型。"
        />
        {roleUnavailable ? (
          <EmptyState
            kind="error"
            title="节点角色不可用"
            description="无法判断本机是否为主节点，队列数据暂不请求。"
            detail={role.error}
            action={
              <CommandButton variant="ghost" size="sm" onClick={role.refresh}>
                重试角色探针
              </CommandButton>
            }
          />
        ) : queueNotApplicable ? (
          <EmptyState
            kind="denied"
            title="单机模式不使用主节点队列"
            description="本机状态和对话仍可用；请求队列只在主节点提供。"
            detail={role.data?.node_role ? `当前角色：${role.data.node_role}` : undefined}
            action={
              <CommandButton variant="ghost" size="sm" href={routeHref('settings')}>
                查看设置
              </CommandButton>
            }
          />
        ) : capacity.state === 'loading' && !capacity.data ? (
          <SkeletonRows rows={2} columns={3} />
        ) : capacity.state === 'error' ? (
          <EmptyState
            kind="error"
            title="准入信息不可用"
            description="该接口仅主节点可用。"
            detail={capacity.error}
            compact
            action={
              <CommandButton variant="ghost" size="sm" onClick={capacity.refresh}>
                重试
              </CommandButton>
            }
          />
        ) : capacity.data ? (
          <div className="capacity">
            <div className="capacity__verdict">
              <StatusBadge
                state={capacity.data.status}
                tone={capacity.data.admitted ? 'ok' : 'warn'}
                label={capacity.data.admitted ? '已准入' : '未准入'}
              />
              <p className="capacity__reason">
                {capacity.data.reason ||
                  capacity.data.reason_code ||
                  (capacity.data.admitted
                    ? '层分配已生效。'
                    : '可参与节点不足，当前使用单机全模型执行。')}
              </p>
            </div>
            <dl className="capacity__facts">
              <div>
                <dt>模型</dt>
                <dd>{capacity.data.model_id}</dd>
              </div>
              <div>
                <dt>总层数</dt>
                <dd className="num-display">{capacity.data.total_layers}</dd>
              </div>
              <div>
                <dt>权重体积</dt>
                <dd className="num-display">{formatBytes(capacity.data.raw_model_bytes)}</dd>
              </div>
              <div>
                <dt>参与节点</dt>
                <dd className="num-display">
                  {capacity.data.participating_node_count} / {capacity.data.candidate_node_count}
                </dd>
              </div>
              <div>
                <dt>仅控制面</dt>
                <dd>
                  {(capacity.data.control_only_nodes ?? []).length
                    ? (capacity.data.control_only_nodes ?? []).join('、')
                    : '无'}
                </dd>
              </div>
              <div>
                <dt>计算时间</dt>
                <dd>{formatRelative(capacity.data.computed_at)}</dd>
              </div>
            </dl>
          </div>
        ) : null}
      </section>

      {/* ---- 节点 ---- */}
      <section className="band band--alt" data-reveal>
        <SectionHead
          title="节点"
          hint={
            nodes.data
              ? `${nodes.data.count} 个已注册节点，${nodes.data.online_count} 个在线。`
              : '集群成员与设备能力。'
          }
          actions={
            <span className="mono-label">
              {nodes.updatedAt ? `更新于 ${new Date(nodes.updatedAt).toLocaleTimeString()}` : ''}
            </span>
          }
        />
        {nodes.state === 'error' ? (
          <EmptyState
            kind="error"
            title="节点列表加载失败"
            detail={nodes.error}
            action={
              <CommandButton variant="ghost" size="sm" onClick={nodes.refresh}>
                重试
              </CommandButton>
            }
          />
        ) : nodes.state === 'loading' && nodeList.length === 0 ? (
          <SkeletonRows rows={2} columns={3} />
        ) : nodeList.length === 0 ? (
          <EmptyState
            kind="empty"
            title="尚无注册节点"
            description="其他设备加入集群后会显示在这里。"
          />
        ) : (
          <ul className="nodegrid">
            {nodeList.map((node) => (
              <NodeCard key={node.node_id} node={node} />
            ))}
          </ul>
        )}
      </section>

      {/* ---- 最近活动 ---- */}
      <section className="band" data-reveal>
        <SectionHead
          title="最近活动"
          hint="来自运行日志的最新事件。"
          actions={
            <CommandButton href={routeHref('activity')} variant="ghost" size="sm" icon={ArrowRight}>
              全部活动
            </CommandButton>
          }
        />
        <ActivityTimeline
          items={timelineItems}
          state={logs.state}
          error={logs.error}
          onRetry={logs.refresh}
          live
        />
      </section>

      {/* ---- 本机资源 ---- */}
      <section className="band band--alt" data-reveal>
        <SectionHead title="本机资源" hint="主节点的设备画像与缓存使用情况。" />
        {status.state === 'error' ? (
          <EmptyState kind="error" title="状态接口不可用" detail={status.error} compact />
        ) : (
          <div className="reslist">
            <div className="rescard">
              <span className="rescard__icon" aria-hidden="true">
                <Cpu size={18} strokeWidth={2.25} />
              </span>
              <div>
                <p className="rescard__label mono-label">设备档位</p>
                <p className="rescard__value">
                  {s?.device?.tier_label || s?.device?.tier || '未知'}
                </p>
                <p className="rescard__hint">评分 {s?.device?.score ?? '—'}</p>
              </div>
            </div>
            <div className="rescard">
              <span className="rescard__icon" aria-hidden="true">
                <HardDrive size={18} strokeWidth={2.25} />
              </span>
              <div>
                <p className="rescard__label mono-label">显存</p>
                <p className="rescard__value num-display">
                  {s?.gpu?.allocated_mb ?? 0} / {s?.gpu?.total_mb ?? 0} MB
                </p>
                <p className="rescard__hint">{s?.gpu?.name || '无 GPU'}</p>
              </div>
            </div>
            <div className="rescard">
              <span className="rescard__icon" aria-hidden="true">
                <Server size={18} strokeWidth={2.25} />
              </span>
              <div>
                <p className="rescard__label mono-label">对话轮次</p>
                <p className="rescard__value num-display">{s?.conversation_turns ?? 0}</p>
                <p className="rescard__hint">
                  累计 {s?.kv_cache?.total_tokens ?? 0} tokens
                </p>
              </div>
            </div>
          </div>
        )}
      </section>
    </>
  );
}
