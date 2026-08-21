/** Cluster overview workspace: runtime context, bounded data panes, and persistent inspection detail. */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowRight, Cpu, HardDrive, Layers, RefreshCw, Server } from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { Drawer } from '../components/Drawer';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { MetricStrip, type Metric } from '../components/MetricStrip';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { ActivityTimeline, type TimelineItem } from '../components/ActivityTimeline';
import { StatusBadge } from '../components/StatusBadge';
import { useRegisterRefresh } from '../app/refreshBus';
import { routeHref } from '../app/routes';
import { useReveal } from '../motion/useReveal';
import { formatBytes, formatRelative } from '../motion/countUp';
import { labelForState, toneForLogLevel } from '../components/statusTone';
import {
  useClusterNodes,
  useMyRole,
  usePipelineCapacity,
  useQueue,
  useRecentLogs,
  useSystemStatus,
} from '../data/hooks';
import type { ClusterNode, LogEntry } from '../data/types';
import { PageBackdrop } from '../visual/PageBackdrop';

type OverviewWorkspace = 'summary' | 'pipeline' | 'nodes' | 'activity';

const WORKSPACES: Array<{ id: OverviewWorkspace; label: string; code: string }> = [
  { id: 'summary', label: '运行总览', code: 'STATE' },
  { id: 'pipeline', label: '流水线准入', code: 'PLAN' },
  { id: 'nodes', label: '节点', code: 'NODES' },
  { id: 'activity', label: '最近活动', code: 'LOGS' },
];

function NodeCard({ node, active, onSelect }: { node: ClusterNode; active: boolean; onSelect: (node: ClusterNode) => void }) {
  const device = node.device_info ?? {};
  const gpu = device.gpu;

  return (
    <li className="overview-nodecard" data-online={node.is_available ? 'true' : 'false'}>
      <button type="button" data-active={active ? 'true' : undefined} onClick={() => onSelect(node)}>
        <div className="overview-nodecard__head">
          <span className="mono-label">{labelForState(node.role)}</span>
          <StatusBadge state={node.state} size="sm" pulse={node.is_available} />
        </div>
        <strong>{node.hostname || node.node_id}</strong>
        <span className="cell-mono">{node.node_id}</span>
        <span className="overview-nodecard__meta">{device.tier_label || device.tier || '未知设备'} · {gpu?.name || '无独立显卡'}</span>
        <span className="overview-nodecard__meta">{node.is_available ? `${node.avg_rtt_ms.toFixed(1)} ms` : '离线'} · {node.error_count ? `${node.error_count} errors` : 'no errors'}</span>
      </button>
    </li>
  );
}

function NodeDetail({ node }: { node: ClusterNode }) {
  const device = node.device_info ?? {};
  const ram = device.ram;
  const gpu = device.gpu;
  return (
    <dl className="kvlist">
      <div><dt>节点</dt><dd>{node.hostname || node.node_id}</dd></div>
      <div><dt>状态</dt><dd><StatusBadge state={node.state} size="sm" /></dd></div>
      <div><dt>角色</dt><dd>{labelForState(node.role)}</dd></div>
      <div><dt>地址</dt><dd className="cell-mono">{node.address || '—'}</dd></div>
      <div><dt>设备</dt><dd>{device.tier_label || device.tier || '未知'}</dd></div>
      <div><dt>CPU</dt><dd>{device.cpu?.model_name || '未知'}</dd></div>
      <div><dt>GPU</dt><dd>{gpu?.name || '无独立显卡'}</dd></div>
      <div><dt>内存</dt><dd>{ram?.percent_used != null ? `${ram.percent_used}% (${ram.used_gb ?? '?'}/${ram.total_gb ?? '?'} GB)` : '未知'}</dd></div>
      <div><dt>往返延迟</dt><dd className="num-display">{node.is_available ? `${node.avg_rtt_ms.toFixed(1)} ms` : '—'}</dd></div>
      <div><dt>最近心跳</dt><dd>{formatRelative(node.last_heartbeat)}</dd></div>
    </dl>
  );
}

function LogDetail({ log }: { log: LogEntry }) {
  return (
    <>
      <dl className="kvlist">
        <div><dt>时间</dt><dd className="cell-mono">{log.timestamp}</dd></div>
        <div><dt>级别</dt><dd><StatusBadge label={log.level} tone={toneForLogLevel(log.level)} size="sm" /></dd></div>
        <div><dt>来源</dt><dd className="cell-mono">{log.name}</dd></div>
        <div><dt>节点</dt><dd className="cell-mono">{log.node_id || '—'}</dd></div>
        <div><dt>请求 ID</dt><dd className="cell-mono">{log.request_id || '—'}</dd></div>
      </dl>
      <h3 className="overview-detail__subtitle">消息</h3>
      <pre className="codeblock">{log.message}</pre>
    </>
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

  const [workspace, setWorkspace] = useState<OverviewWorkspace>('summary');
  const [activeNode, setActiveNode] = useState<ClusterNode | null>(null);
  const [activeLog, setActiveLog] = useState<LogEntry | null>(null);
  const [compactDetails, setCompactDetails] = useState(false);

  useEffect(() => {
    const media = window.matchMedia('(max-width: 1199px)');
    const update = () => setCompactDetails(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);

  const refresh = useCallback(() => {
    status.refresh();
    nodes.refresh();
    role.refresh();
    queue.refresh();
    capacity.refresh();
    logs.refresh();
  }, [capacity, logs, nodes, queue, role, status]);
  useRegisterRefresh(refresh);
  useReveal([workspace, nodes.data, logs.data]);

  const system = status.data;
  const nodeList = nodes.data?.nodes ?? [];
  const onlineCount = nodes.data?.online_count ?? 0;
  const totalNodes = nodes.data?.count ?? 0;
  const activeWorkspace = WORKSPACES.find((item) => item.id === workspace) ?? WORKSPACES[0];

  const metrics: Metric[] = useMemo(() => {
    const gpuUtil = system?.gpu?.utilization ?? 0;
    const kvUtil = (system?.kv_cache?.utilization ?? 0) * 100;
    const queueDepth = queueEnabled ? queue.data?.queue_size ?? 0 : 0;
    return [
      { label: '在线节点', value: onlineCount, suffix: totalNodes ? ` / ${totalNodes}` : '', hint: onlineCount === totalNodes ? '全部节点在线' : '存在离线节点', tone: totalNodes > 0 && onlineCount < totalNodes ? 'warn' : 'ok' },
      { label: '显存占用', value: gpuUtil, suffix: '%', decimals: 1, hint: system?.gpu?.name ? system.gpu.name.replace('NVIDIA GeForce ', '') : '无 GPU 信息', tone: gpuUtil > 85 ? 'danger' : gpuUtil > 60 ? 'warn' : 'ok' },
      { label: 'KV 缓存', value: kvUtil, suffix: '%', decimals: 1, hint: `${system?.kv_cache?.allocated_pages ?? 0} / ${system?.kv_cache?.max_pages ?? 0} 页`, tone: kvUtil > 85 ? 'danger' : kvUtil > 60 ? 'warn' : 'ok' },
      { label: '队列深度', value: queueDepth, hint: queueNotApplicable ? '单机模式不适用' : roleUnavailable ? '节点角色不可用' : role.state !== 'ready' ? '正在确认节点角色' : queue.state === 'error' ? '队列接口不可用' : `已完成 ${queue.data?.completed_count ?? 0}`, tone: queueDepth > 0 ? 'info' : 'idle' },
    ];
  }, [onlineCount, queue.data, queue.state, queueEnabled, queueNotApplicable, role.state, roleUnavailable, system, totalNodes]);

  const timelineItems: TimelineItem[] = useMemo(() => (logs.data?.logs ?? []).map((log) => ({
    id: String(log.seq),
    time: log.timestamp.slice(11),
    title: log.name,
    detail: log.message,
    tone: toneForLogLevel(log.level),
    badge: log.level,
    ...(log.node_id ? { source: log.node_id } : {}),
  })), [logs.data]);

  const statusSentence = (() => {
    if (status.state === 'loading') return '正在读取集群状态。';
    if (status.state === 'error') return '无法读取集群状态。';
    if (!system) return '暂无状态数据。';
    const modelPart = system.model_loaded ? `已加载 ${system.model_name}` : system.pipeline_prepared ? `流水线已就绪（${system.pipeline_descriptor?.model_id ?? system.model_name}）` : '尚未加载模型';
    return `本机以 ${labelForState(system.node_role)} 身份运行在 ${labelForState(system.run_mode)} 模式，${modelPart}。`;
  })();

  const selectWorkspace = useCallback((next: OverviewWorkspace) => {
    setWorkspace(next);
    setActiveNode(null);
    setActiveLog(null);
  }, []);
  const selectNode = useCallback((node: ClusterNode) => {
    setActiveNode(node);
    setActiveLog(null);
  }, []);
  const selectLog = useCallback((id: string) => {
    setActiveLog((logs.data?.logs ?? []).find((log) => String(log.seq) === id) ?? null);
    setActiveNode(null);
  }, [logs.data]);

  const detailTitle = activeNode ? (activeNode.hostname || activeNode.node_id) : activeLog ? `${activeLog.level} · ${activeLog.name}` : '运行上下文';
  const drawerOpen = compactDetails && (activeNode !== null || activeLog !== null);

  return (
    <div className="overview-page" data-testid="overview-page">
      <PageBackdrop scene="overview" className="overview-page__bg" />
      <PageHeader
        tag="OVERVIEW"
        title="集群概览"
        description={statusSentence}
        actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={status.refreshing || nodes.refreshing} onClick={refresh}>刷新</CommandButton>}
      />

      <div className="overview-layout">
        <aside className="overview-rail" aria-label="概览导航">
          <section className="overview-panel overview-identity">
            <span className="mono-label">LOCAL OBSERVATORY</span>
            <strong>{system?.node_id || role.data?.node_id || 'checking runtime'}</strong>
            <span>{labelForState(system?.run_mode)} · {labelForState(system?.node_role)}</span>
          </section>

          <nav className="overview-nav" aria-label="概览领域">
            {WORKSPACES.map((item) => <button key={item.id} type="button" data-active={workspace === item.id ? 'true' : undefined} aria-pressed={workspace === item.id} onClick={() => selectWorkspace(item.id)}><span>{item.label}</span><em>{item.id === 'nodes' ? `${onlineCount}/${totalNodes}` : item.id === 'activity' ? `${timelineItems.length}` : item.code}</em></button>)}
          </nav>

          <section className="overview-panel overview-rail-health">
            <SectionHead title="本机状态" hint={system?.model_loaded ? 'MODEL LOADED' : system?.pipeline_prepared ? 'PIPELINE READY' : 'RUNTIME IDLE'} />
            <dl className="overview-rail-facts">
              <div><dt>模型</dt><dd>{system?.model_loaded ? '已加载' : '未加载'}</dd></div>
              <div><dt>活跃任务</dt><dd>{queueEnabled && queue.data?.current_task ? '执行中' : '空闲'}</dd></div>
              <div><dt>告警</dt><dd>{system?.device?.warnings?.length ?? 0}</dd></div>
            </dl>
          </section>

          <div className="overview-rail-actions">
            <CommandButton href={routeHref('tasks')} variant="ghost" size="sm" icon={ArrowRight}>任务队列</CommandButton>
            <CommandButton href={routeHref('activity')} variant="ghost" size="sm" icon={Layers}>全部活动</CommandButton>
          </div>
        </aside>

        <main className="overview-main">
          <section className="overview-panel overview-workspace" data-workspace={workspace} aria-labelledby="overview-workspace-title">
            <div className="overview-workspace__head">
              <SectionHead id="overview-workspace-title" title={activeWorkspace.label} hint={workspace === 'summary' ? '本机运行状态与关键资源指标。' : workspace === 'pipeline' ? '主节点计算的分布式推理准入计划。' : workspace === 'nodes' ? '选择节点以固定查看设备与连通性详情。' : '选择日志条目以固定查看其完整上下文。'} />
            </div>
            <div className="overview-workspace__scroll">
              {workspace === 'summary' ? <>
                <MetricStrip metrics={metrics} caption="集群关键指标" />
                <section className="overview-subsection">
                  <SectionHead title="本地资源" hint="主节点的设备画像与缓存使用情况。" />
                  {status.state === 'error' ? <EmptyState kind="error" title="状态接口不可用" detail={status.error} errorKind={status.errorKind} errorStatus={status.errorStatus} compact action={<CommandButton variant="ghost" size="sm" onClick={status.refresh}>重试</CommandButton>} /> : <div className="overview-resource-list">
                    <div><Cpu size={17} aria-hidden="true" /><span><small>设备档位</small><strong>{system?.device?.tier_label || system?.device?.tier || '未知'}</strong><em>评分 {system?.device?.score ?? '—'}</em></span></div>
                    <div><HardDrive size={17} aria-hidden="true" /><span><small>显存</small><strong>{system?.gpu?.allocated_mb ?? 0} / {system?.gpu?.total_mb ?? 0} MB</strong><em>{system?.gpu?.name || '无 GPU'}</em></span></div>
                    <div><Server size={17} aria-hidden="true" /><span><small>对话轮次</small><strong>{system?.conversation_turns ?? 0}</strong><em>累计 {system?.kv_cache?.total_tokens ?? 0} tokens</em></span></div>
                  </div>}
                </section>
                {system?.device?.warnings?.length ? <section className="overview-subsection"><SectionHead title="运行提醒" hint="来自设备检测的本机约束。" /><ul className="overview-warnings">{system.device.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></section> : null}
              </> : null}

              {workspace === 'pipeline' ? <>
                {roleUnavailable ? <EmptyState kind="error" title="节点角色不可用" description="无法判断本机是否为主节点，队列数据暂不请求。" detail={role.error} errorKind={role.errorKind} errorStatus={role.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={role.refresh}>重试角色探针</CommandButton>} /> : queueNotApplicable ? <EmptyState kind="denied" title="单机模式不使用主节点队列" description="本机状态和对话仍可用；请求队列只在主节点提供。" detail={role.data?.node_role ? `当前角色：${role.data.node_role}` : undefined} action={<CommandButton variant="ghost" size="sm" href={routeHref('settings')}>查看设置</CommandButton>} /> : capacity.state === 'loading' && !capacity.data ? <SkeletonRows rows={2} columns={3} /> : capacity.state === 'error' ? <EmptyState kind="error" title="准入信息不可用" description="该接口仅主节点可用。" detail={capacity.error} errorKind={capacity.errorKind} errorStatus={capacity.errorStatus} compact action={<CommandButton variant="ghost" size="sm" onClick={capacity.refresh}>重试</CommandButton>} /> : capacity.data ? <div className="capacity overview-capacity">
                  <div className="capacity__verdict"><StatusBadge state={capacity.data.status} tone={capacity.data.admitted ? 'ok' : 'warn'} label={capacity.data.admitted ? '已准入' : '未准入'} /><p className="capacity__reason">{capacity.data.reason || capacity.data.reason_code || (capacity.data.admitted ? '层分配已生效。' : '可参与节点不足，当前使用单机全模型执行。')}</p></div>
                  <dl className="capacity__facts"><div><dt>模型</dt><dd>{capacity.data.model_id || '—'}</dd></div><div><dt>总层数</dt><dd className="num-display">{capacity.data.total_layers ?? 0}</dd></div><div><dt>权重体积</dt><dd className="num-display">{formatBytes(capacity.data.raw_model_bytes ?? 0)}</dd></div><div><dt>参与节点</dt><dd className="num-display">{capacity.data.participating_node_count ?? 0} / {capacity.data.candidate_node_count ?? 0}</dd></div><div><dt>仅控制面</dt><dd>{(capacity.data.control_only_nodes ?? []).length ? capacity.data.control_only_nodes.join('、') : '无'}</dd></div><div><dt>计算时间</dt><dd>{formatRelative(capacity.data.computed_at)}</dd></div></dl>
                </div> : null}
              </> : null}

              {workspace === 'nodes' ? <>
                {nodes.state === 'error' ? <EmptyState kind="error" title="节点列表加载失败" detail={nodes.error} errorKind={nodes.errorKind} errorStatus={nodes.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={nodes.refresh}>重试</CommandButton>} /> : nodes.state === 'loading' && nodeList.length === 0 ? <SkeletonRows rows={3} columns={2} /> : nodeList.length === 0 ? <EmptyState kind="empty" title="尚无注册节点" description="其他设备加入集群后会显示在这里。" /> : <ul className="overview-nodegrid">{nodeList.map((node) => <NodeCard key={node.node_id} node={node} active={activeNode?.node_id === node.node_id} onSelect={selectNode} />)}</ul>}
              </> : null}

              {workspace === 'activity' ? <div className="overview-timeline"><ActivityTimeline items={timelineItems} state={logs.state} error={logs.error} errorKind={logs.errorKind} errorStatus={logs.errorStatus} onRetry={logs.refresh} onSelect={selectLog} live /></div> : null}
            </div>
          </section>
        </main>

        <aside className="overview-details" aria-label="当前概览详情">
          <section className="overview-panel overview-detail-panel">
            <SectionHead title="当前详情" hint={activeNode ? 'NODE' : activeLog ? 'LOG' : activeWorkspace.code} />
            {activeNode ? <NodeDetail node={activeNode} /> : activeLog ? <LogDetail log={activeLog} /> : <>
              <p className="overview-detail__title">{detailTitle}</p>
              <dl className="kvlist">
                <div><dt>运行模式</dt><dd><StatusBadge state={system?.run_mode} size="sm" /></dd></div>
                <div><dt>节点角色</dt><dd><StatusBadge state={system?.node_role} size="sm" /></dd></div>
                <div><dt>模型状态</dt><dd>{system?.model_loaded ? '已加载' : system?.pipeline_prepared ? '流水线已就绪' : '待加载'}</dd></div>
                <div><dt>节点在线</dt><dd className="num-display">{onlineCount} / {totalNodes}</dd></div>
                <div><dt>日志缓冲</dt><dd className="num-display">{logs.data?.count ?? 0} 条</dd></div>
              </dl>
            </>}
          </section>
        </aside>
      </div>

      <Drawer open={drawerOpen} tag={activeNode ? 'NODE' : 'LOG'} title={detailTitle} onClose={() => { setActiveNode(null); setActiveLog(null); }}>
        {activeNode ? <NodeDetail node={activeNode} /> : activeLog ? <LogDetail log={activeLog} /> : null}
      </Drawer>
    </div>
  );
}
