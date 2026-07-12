import { useState, useEffect, useCallback } from 'react';
import {
  fetchClusterStatus, fetchClusterNodes,
  fetchClusterConfig, deregisterNode, deleteClusterNode, updateMaxNodes,
  fetchInviteInfo, connectToMaster,
  discoverMaster, resetMasterIdentity,
  manualRegisterNode, fetchMasterHealth,
  testEmailNotification,
  fetchDistributedInferenceConfig, updateDistributedInferenceConfig,
  fetchLayerAssignment, updateLayerAssignment, resetLayerAssignments,
  fetchConversationSyncStatus,
  transferMasterRole, fetchTransferLogs,
  fetchSpareMaster, designateSpareMaster,
  removeSpareMaster, fetchSpareMasterLogs,
  fetchQueue, setQueueStrategy, pauseQueue,
  resumeQueue, clearQueue, cancelQueueTask,
} from '../api/client';

const ROLE_LABELS = { master: '主节点', client: '从节点' };
const ROLE_ICONS = { master: '🖥️', client: '💻' };

// 动态加载的模块引用（避免 window 全局污染）
let _deleteReviewTicketFn = null;
let _deleteResolvedReviewTicketsFn = null;
const TYPE_ICONS = { pc: '💻', android: '📱' };
const TYPE_LABELS = { pc: 'PC', android: 'Android' };

const STATE_LABELS = { online: '在线', busy: '忙碌', offline: '离线', error: '异常' };
const STATE_COLORS = {
  online: 'var(--success)', busy: 'var(--warning)',
  offline: 'var(--text-muted)', error: 'var(--danger)',
};

const NETWORK_LABELS = {
  wifi: '📶 WiFi', ethernet: '🔌 以太网', mobile: '📱 移动网络',
  vpn: '🔐 VPN', other: '🌐 其他', localhost: '🏠 本地', unknown: '❓ 未知',
};
const NETWORK_CLASSES = {
  wifi: 'net-wifi', ethernet: 'net-eth', mobile: 'net-mobile',
  vpn: 'net-vpn', other: 'net-other', localhost: 'net-local', unknown: 'net-unknown',
};

function formatTime(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleTimeString();
}
function formatUptime(ts) {
  if (!ts) return '—';
  const sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 60) return `${Math.floor(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
function formatRTT(rttMs) {
  if (rttMs == null || rttMs === 0) return '—';
  if (rttMs < 1) return '<1ms';
  if (rttMs < 10) return `${rttMs.toFixed(1)}ms`;
  return `${Math.round(rttMs)}ms`;
}
function isTcpActive(clientInfo) {
  // TCP 判定为活跃：最近一次心跳在 10 秒内
  if (!clientInfo || !clientInfo.last_heartbeat) return false;
  return (Date.now() / 1000 - clientInfo.last_heartbeat) < 10;
}

// ================================================================
// 流水线拓扑子组件
// ================================================================

function PipelineTopology({ assignments, nodes, isDistributed, runMode }) {
  // 按 start_layer 排序，确保流水线顺序
  const sorted = [...assignments].sort((a, b) => (a.start_layer ?? 0) - (b.start_layer ?? 0));

  // 分离主节点和从节点
  const pipelineNodes = sorted.filter(a => a.node_id !== 'master');
  const masterNode = sorted.find(a => a.node_id === 'master');

  // 检查流水线就绪状态
  const nodeMap = {};
  (nodes || []).forEach(n => { nodeMap[n.node_id] = n; });

  const allWorkersOnline = pipelineNodes.every(a => {
    const n = nodeMap[a.node_id];
    return n && n.state === 'online';
  });
  const anyWorkerOffline = pipelineNodes.some(a => {
    const n = nodeMap[a.node_id];
    return n && n.state === 'offline';
  });

  const isPipelineActive = isDistributed && runMode === 'distributed' && pipelineNodes.length > 0;
  const isPipelineHealthy = isPipelineActive && allWorkersOnline;
  const isPipelineDegraded = isPipelineActive && anyWorkerOffline && !allWorkersOnline;

  // 单节点模式（无从节点）
  if (pipelineNodes.length === 0) {
    return (
      <div className="pipeline-topology">
        <div className="pipeline-status-banner" style={{ background: 'var(--warning-bg)', border: '1px solid var(--warning)' }}>
          <span>⚠️ 未检测到从节点，流水线模式不可用。</span>
          <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
            需要至少 1 个从节点参与分布式推理才能启用流水线。
          </span>
        </div>
        {masterNode && (
          <div className="pipeline-single-node">
            <PipelineNodeCard
              node={masterNode}
              nodeInfo={nodeMap[masterNode.node_id]}
              isFirst={true}
              isLast={true}
              isMaster={true}
            />
          </div>
        )}
      </div>
    );
  }

  // 构建完整流水线：主节点（若有分配） + 从节点（按层序）
  const fullPipeline = [];
  if (masterNode && masterNode.layers_count > 0) {
    fullPipeline.push({ ...masterNode, _isMaster: true });
  }
  pipelineNodes.forEach(n => fullPipeline.push({ ...n, _isMaster: false }));

  return (
    <div className="pipeline-topology">
      {/* 流水线状态横幅 */}
      <div className={`pipeline-status-banner ${
        !isPipelineActive ? 'pipeline-inactive' :
        isPipelineHealthy ? 'pipeline-healthy' :
        isPipelineDegraded ? 'pipeline-degraded' : 'pipeline-error'
      }`}>
        <span className="pipeline-status-icon">
          {!isPipelineActive ? '⏸️' :
           isPipelineHealthy ? '✅' :
           isPipelineDegraded ? '⚠️' : '❌'}
        </span>
        <span className="pipeline-status-text">
          {!isPipelineActive ? '流水线未激活 — 请启用分布式推理并确保有从节点在线' :
           isPipelineHealthy ? '流水线就绪 — 所有节点在线，分布式层推理正常工作' :
           isPipelineDegraded ? '流水线降级 — 部分从节点离线，将自动回退到主节点全模型推理' :
           '流水线不可用 — 所有从节点离线'}
        </span>
      </div>

      {/* 拓扑图 */}
      <div className="pipeline-flow">
        {fullPipeline.map((node, idx) => (
          <div key={node.node_id} className="pipeline-flow-item">
            <PipelineNodeCard
              node={node}
              nodeInfo={nodeMap[node.node_id]}
              isFirst={idx === 0}
              isLast={idx === fullPipeline.length - 1}
              isMaster={node._isMaster}
            />
            {idx < fullPipeline.length - 1 && (
              <div className="pipeline-arrow">
                <div className="pipeline-arrow-line" />
                <div className="pipeline-arrow-head">▶</div>
                <div className="pipeline-arrow-label">hidden_states</div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 图例 */}
      <div className="pipeline-legend">
        <span className="pipeline-legend-item">
          <span className="pipeline-legend-swatch master" /> 主节点
        </span>
        <span className="pipeline-legend-item">
          <span className="pipeline-legend-swatch client-online" /> 从节点 (在线)
        </span>
        <span className="pipeline-legend-item">
          <span className="pipeline-legend-swatch client-offline" /> 从节点 (离线)
        </span>
        <span className="pipeline-legend-item">
          <span className="pipeline-legend-swatch arrow" /> 隐藏状态传输
        </span>
      </div>
    </div>
  );
}

function PipelineNodeCard({ node, nodeInfo, isFirst, isLast, isMaster }) {
  const isOnline = nodeInfo?.state === 'online';
  const isOffline = nodeInfo?.state === 'offline';
  const rtt = nodeInfo?.avg_rtt_ms ?? nodeInfo?.last_rtt_ms ?? 0;
  const roleIcon = isMaster ? '🖥️' : '💻';
  const roleLabel = isMaster ? '主节点' : '从节点';

  return (
    <div className={`pipeline-node-card ${
      isOnline ? 'node-online' :
      isOffline ? 'node-offline' : 'node-unknown'
    } ${isMaster ? 'node-master' : 'node-worker'}`}>
      {/* 节点头部 */}
      <div className="pipeline-node-header">
        <span className="pipeline-node-role">
          {roleIcon} {roleLabel}
        </span>
        <span className={`pipeline-node-status ${
          isOnline ? 'status-online' :
          isOffline ? 'status-offline' : 'status-unknown'
        }`}>
          {isOnline ? '🟢 在线' :
           isOffline ? '🔴 离线' : '⚪ 未知'}
        </span>
      </div>

      {/* 节点标识 */}
      <div className="pipeline-node-id mono">{node.node_id}</div>

      {/* 层范围 */}
      <div className="pipeline-node-layers">
        <span className="pipeline-layer-label">层范围</span>
        <span className="pipeline-layer-range mono">
          Layer {node.start_layer} – {node.end_layer}
          <span className="pipeline-layer-count">
            ({node.layers_count ?? (node.end_layer - node.start_layer)} 层)
          </span>
        </span>
      </div>

      {/* 特殊层标记 */}
      <div className="pipeline-node-flags">
        {node.has_embedding && <span className="pipeline-flag flag-embed">📥 Embedding</span>}
        {node.has_lm_head && <span className="pipeline-flag flag-lmhead">📤 LM Head</span>}
        {!node.has_embedding && !node.has_lm_head && (
          <span className="pipeline-flag flag-transformer">🔄 Transformer</span>
        )}
        {isFirst && <span className="pipeline-flag flag-entry">🚪 入口</span>}
        {isLast && <span className="pipeline-flag flag-exit">🚪 出口</span>}
      </div>

      {/* RTT 延迟 */}
      {isOnline && rtt > 0 && (
        <div className="pipeline-node-rtt">
          <span className="pipeline-rtt-label">网络延迟</span>
          <span className={`pipeline-rtt-value mono ${
            rtt < 10 ? 'rtt-good' : rtt < 50 ? 'rtt-ok' : 'rtt-slow'
          }`}>
            {rtt.toFixed(1)} ms
          </span>
        </div>
      )}

      {/* 设备信息 */}
      {nodeInfo?.device_info?.gpu?.name && (
        <div className="pipeline-node-gpu" title={nodeInfo.device_info.gpu.name}>
          🎮 {nodeInfo.device_info.gpu.name}
        </div>
      )}
    </div>
  );
}


export default function AdminPanel({ onToast, myRole, hasDedicatedGpu }) {
  const [status, setStatus] = useState(null);
  const [nodes, setNodes] = useState(null);
  const [config, setConfig] = useState(null);
  const [invite, setInvite] = useState(null);
  const [loading, setLoading] = useState(true);
  const [deregistering, setDeregistering] = useState(null);
  const [maxNodesInput, setMaxNodesInput] = useState('');
  const [updatingMaxNodes, setUpdatingMaxNodes] = useState(false);

  // 从节点连接表单
  const [masterHost, setMasterHost] = useState('');
  const [masterPort, setMasterPort] = useState('8888');
  const [connecting, setConnecting] = useState(false);

  // 主节点自动发现
  const [discovery, setDiscovery] = useState(null);  // { found, master_host, master_port, stale, source }
  const [discovering, setDiscovering] = useState(false);

  // 手动注册从节点表单（主节点）
  const [manualNodeId, setManualNodeId] = useState('');
  const [manualHostname, setManualHostname] = useState('');
  const [manualAddress, setManualAddress] = useState('');
  const [manualNetworkType, setManualNetworkType] = useState('ethernet');
  const [manualNodeType, setManualNodeType] = useState('pc');
  const [registering, setRegistering] = useState(false);
  const [deletingNode, setDeletingNode] = useState(null);
  const [showOnlineOnly, setShowOnlineOnly] = useState(false);

  // 主节点健康状态（从节点监控）
  const [masterHealth, setMasterHealth] = useState(null);  // { master_online, stale, last_seen_seconds_ago }

  // 分布式推理开关
  const [distributedEnabled, setDistributedEnabled] = useState(true);
  const [togglingDistributed, setTogglingDistributed] = useState(false);

  // 动态分层
  const [layerAssignment, setLayerAssignment] = useState(null);
  const [layerOverrides, setLayerOverrides] = useState({});  // { node_id: { start, end } }

  // 云同步状态
  const [syncStatus, setSyncStatus] = useState(null);

  // 角色转让
  const [transferTarget, setTransferTarget] = useState('');
  const [transferring, setTransferring] = useState(false);
  const [transferLogs, setTransferLogs] = useState([]);

  // 备用主节点
  const [spareMaster, setSpareMaster] = useState(null);  // { node_id, hostname, is_online, state, ... }
  const [spareTarget, setSpareTarget] = useState('');
  const [designatingSpare, setDesignatingSpare] = useState(false);
  const [spareMasterLogs, setSpareMasterLogs] = useState([]);

  // P3: 审查投票
  const [reviewTickets, setReviewTickets] = useState([]);
  const [canVote, setCanVote] = useState(false);
  const [reviewTarget, setReviewTarget] = useState('');
  const [reviewReason, setReviewReason] = useState('');
  const [creatingReview, setCreatingReview] = useState(false);
  const [votingTicket, setVotingTicket] = useState(null);

  // P3.5: 推理调度队列
  const [queueDetail, setQueueDetail] = useState(null);
  const [queueLoading, setQueueLoading] = useState(false);

  const isMaster = myRole?.is_master ?? false;

  // 拉取全部数据
  const refresh = useCallback(() => {
    setLoading(true);
    Promise.all([
      fetchClusterStatus().catch(() => null),
      fetchClusterNodes().catch(() => null),
      fetchClusterConfig().catch(() => null),
      isMaster ? fetchInviteInfo().catch(() => null) : Promise.resolve(null),
      isMaster ? fetchLayerAssignment().catch(() => null) : Promise.resolve(null),
      !isMaster ? fetchMasterHealth().catch(() => null) : Promise.resolve(null),
      isMaster ? fetchQueue().catch(() => null) : Promise.resolve(null),
    ]).then(([s, n, c, inv, layerData, mh, qd]) => {
      setStatus(s || {});
      setNodes(n || { nodes: [] });
      setConfig(c || {});
      setInvite(inv);
      if (layerData) {
        setLayerAssignment(layerData);
        const ov = {};
        (layerData.assignments || []).forEach(a => {
          ov[a.node_id] = { start: a.start_layer, end: a.end_layer };
        });
        setLayerOverrides(ov);
      }
      if (mh) {
        setMasterHealth(mh);
      }
      if (c?.max_nodes && !maxNodesInput) {
        setMaxNodesInput(String(c.max_nodes));
      }
      if (qd) {
        setQueueDetail(qd);
      }
    }).finally(() => setLoading(false));
  }, [isMaster]);

  // 初始加载 + 5 秒自动刷新
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleDeregister = async (nodeId) => {
    setDeregistering(nodeId);
    try {
      await deregisterNode(nodeId);
      onToast?.({ type: 'success', msg: `已注销节点: ${nodeId}` });
      refresh();
    } catch (err) {
      onToast?.({ type: 'error', msg: `注销失败: ${err.message}` });
    } finally {
      setDeregistering(null);
    }
  };

  const handleDeleteNode = async (nodeId) => {
    if (!window.confirm(`确认删除离线节点 ${nodeId}？此操作会移除节点记录。`)) return;
    setDeletingNode(nodeId);
    try {
      await deleteClusterNode(nodeId);
      onToast?.({ type: 'success', msg: `已删除节点: ${nodeId}` });
      refresh();
    } catch (err) {
      onToast?.({ type: 'error', msg: `删除失败: ${err.message}` });
    } finally {
      setDeletingNode(null);
    }
  };

  const handleDeleteAllOffline = async () => {
    const offlineNodes = (nodes?.nodes || []).filter(
      n => n.role !== 'master' && !n.is_available
    );
    if (offlineNodes.length === 0) {
      onToast?.({ type: 'info', msg: '没有可删除的离线节点' });
      return;
    }
    if (!window.confirm(
      `确认删除全部 ${offlineNodes.length} 个离线节点？\n\n此操作不可撤销。`
    )) return;
    let ok = 0, fail = 0;
    for (const n of offlineNodes) {
      try {
        await deleteClusterNode(n.node_id);
        ok++;
      } catch (_) {
        fail++;
      }
    }
    onToast?.({ type: 'success', msg: `已删除 ${ok} 个，失败 ${fail} 个` });
    refresh();
  };

  const handleUpdateMaxNodes = async () => {
    const v = parseInt(maxNodesInput, 10);
    if (!v || v < 1 || v > 64) {
      onToast?.({ type: 'error', msg: '请输入 1-64 之间的整数' });
      return;
    }
    setUpdatingMaxNodes(true);
    try {
      const result = await updateMaxNodes(v);
      if (result.status === 'ok') {
        onToast?.({ type: 'success', msg: `最大节点数已更新: ${result.max_nodes}` });
        setConfig(prev => prev ? { ...prev, max_nodes: result.max_nodes } : prev);
        setMaxNodesInput(String(result.max_nodes));
        refresh();
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `更新失败: ${err.message}` });
    } finally {
      setUpdatingMaxNodes(false);
    }
  };

  const handleResetIdentity = async () => {
    if (!window.confirm('⚠️ 确定要重置主节点身份标识吗？\n\n这将清除数据库中记录的 MAC 地址。\n下次启动时将重新记录本机 MAC 作为新的身份标识。\n\n此操作通常在更换主节点机器或网卡后使用。')) {
      return;
    }
    try {
      const result = await resetMasterIdentity('reset');
      if (result.status === 'ok') {
        onToast?.({ type: 'success', msg: result.message || '身份已重置' });
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `重置失败: ${err.message}` });
    }
  };

  // ---- 发送测试邮件 ----
  const handleEmailTest = async () => {
    setRegistering(true);  // 复用 registering 状态显示 loading
    try {
      const result = await testEmailNotification();
      if (result.status === 'ok') {
        onToast?.({ type: 'success', msg: result.message || '测试邮件已发送' });
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `邮件发送失败: ${err.message}` });
    } finally {
      setRegistering(false);
    }
  };

  // ---- 角色转让 ----
  const handleTransferMaster = async () => {
    if (!transferTarget) {
      onToast?.({ type: 'error', msg: '请选择目标从节点' });
      return;
    }
    const targetNode = (nodes?.nodes || []).find(n => n.node_id === transferTarget);
    const targetName = targetNode ? `${targetNode.node_id} (${targetNode.hostname || '未知主机'})` : transferTarget;
    if (!window.confirm(
      `⚠️ 确定要将主节点身份转让给从节点「${targetName}」吗？\n\n` +
      `此操作将：\n` +
      `  1. 通知目标从节点准备升级为主节点\n` +
      `  2. 激活备用主节点「${spareMaster?.node_id || '未知'}」暂代监政\n` +
      `  3. 保存降级日志和备用激活日志到数据库\n` +
      `  4. 更新数据库中的主节点信息\n\n` +
      `转让后建议三方重启服务：\n` +
      `  • 目标节点以主节点模式运行\n` +
      `  • 本节点以从节点模式运行\n` +
      `  • 新主节点上线后将自动通知备用主节点退出暂代`
    )) return;

    setTransferring(true);
    try {
      const result = await transferMasterRole(transferTarget);
      if (result.status === 'ok') {
        onToast?.({ type: 'success', msg: result.message || '角色转让已发起' });
        setTransferTarget('');
        refresh();
        // 刷新转让日志
        fetchTransferLogs()
          .then(data => setTransferLogs(data.logs || []))
          .catch(() => {});
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `角色转让失败: ${err.message}` });
    } finally {
      setTransferring(false);
    }
  };

  const loadTransferLogs = useCallback(() => {
    if (isMaster) {
      fetchTransferLogs()
        .then(data => setTransferLogs(data.logs || []))
        .catch(() => onToast?.({ type: 'warning', msg: '无法加载转让日志' }));
    }
    // P3: 加载审查数据（主节点始终加载，投票资格由服务端检查）
    if (isMaster) {
      import('../api/client').then(({ fetchReviewTickets, checkCanVote, deleteReviewTicket, deleteResolvedReviewTickets }) => {
        fetchReviewTickets('pending')
          .then(data => setReviewTickets(data.tickets || []))
          .catch(() => onToast?.({ type: 'warning', msg: '无法加载审查工单' }));
        checkCanVote()
          .then(data => setCanVote(data.can_vote || false))
          .catch(() => {});  // canVote 默认 false，静默降级即可
        // 缓存动态导入的函数引用
        _deleteReviewTicketFn = deleteReviewTicket;
        _deleteResolvedReviewTicketsFn = deleteResolvedReviewTickets;
      }).catch(() => onToast?.({ type: 'warning', msg: '审查模块加载失败' }));
    }
  }, [isMaster, onToast]);

  // ---- 备用主节点 ----
  const loadSpareMasterData = useCallback(() => {
    if (isMaster) {
      fetchSpareMaster()
        .then(data => {
          setSpareMaster(data.spare_master || null);
        })
        .catch(() => {});
      fetchSpareMasterLogs()
        .then(data => setSpareMasterLogs(data.logs || []))
        .catch(() => {});
    }
  }, [isMaster]);

  const handleDesignateSpareMaster = async () => {
    if (!spareTarget.trim()) {
      onToast?.({ type: 'error', msg: '请选择备用主节点' });
      return;
    }
    if (!window.confirm(
      `确定要将 '${spareTarget}' 指定为备用主节点吗？\n\n` +
      '备用主节点是主节点身份转让的指定接班人。'
    )) return;
    setDesignatingSpare(true);
    try {
      const result = await designateSpareMaster(spareTarget.trim());
      if (result.status === 'ok' || result.status === 'duplicate') {
        onToast?.({ type: 'success', msg: result.message || `已指定备用主节点: ${spareTarget}` });
        setSpareTarget('');
        loadSpareMasterData();
        refresh();
      } else {
        onToast?.({ type: 'error', msg: result.reason || '指定失败' });
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `指定失败: ${err.message}` });
    } finally {
      setDesignatingSpare(false);
    }
  };

  const handleRemoveSpareMaster = async () => {
    if (!window.confirm('确定要取消备用主节点指定吗？取消后将无法转让主节点身份。')) return;
    try {
      const result = await removeSpareMaster();
      if (result.status === 'ok') {
        onToast?.({ type: 'success', msg: result.message || '已取消备用主节点' });
        setTransferTarget('');
        loadSpareMasterData();
        refresh();
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `取消失败: ${err.message}` });
    }
  };

  // ---- P3: 审查工单操作 ----

  const handleCreateReview = async () => {
    if (!reviewTarget.trim()) {
      onToast?.({ type: 'error', msg: '请选择转让目标节点' });
      return;
    }
    setCreatingReview(true);
    try {
      const { createReviewTicket } = await import('../api/client');
      const result = await createReviewTicket(reviewTarget, reviewReason, 48);
      onToast?.({ type: 'success', msg: `审查工单 ${result.ticket_id} 已创建` });
      setReviewTarget('');
      setReviewReason('');
      // 刷新工单列表
      const { fetchReviewTickets } = await import('../api/client');
      const data = await fetchReviewTickets('pending');
      setReviewTickets(data.tickets || []);
    } catch (err) {
      onToast?.({ type: 'error', msg: `创建工单失败: ${err.message}` });
    } finally {
      setCreatingReview(false);
    }
  };

  const handleVote = async (ticketId, voteValue) => {
    setVotingTicket(ticketId);
    try {
      const { castVote } = await import('../api/client');
      await castVote(ticketId, voteValue);
      onToast?.({ type: 'success', msg: `投票成功: ${voteValue > 0 ? '+' : ''}${voteValue}` });
      // 刷新
      const { fetchReviewTickets } = await import('../api/client');
      const data = await fetchReviewTickets('pending');
      setReviewTickets(data.tickets || []);
    } catch (err) {
      onToast?.({ type: 'error', msg: `投票失败: ${err.message}` });
    } finally {
      setVotingTicket(null);
    }
  };

  const handleDeleteTicket = async (ticketId) => {
    if (!window.confirm(`确定删除工单 ${ticketId}？`)) return;
    try {
      const fn = _deleteReviewTicketFn || (await import('../api/client')).deleteReviewTicket;
      await fn(ticketId);
      onToast?.({ type: 'success', msg: `工单 ${ticketId} 已删除` });
      const { fetchReviewTickets } = await import('../api/client');
      const data = await fetchReviewTickets('pending');
      setReviewTickets(data.tickets || []);
    } catch (err) {
      onToast?.({ type: 'error', msg: `删除失败: ${err.message}` });
    }
  };

  const handleDeleteResolvedTickets = async () => {
    if (!window.confirm('确定删除所有已解决/已过期/已拒绝的审查工单？')) return;
    try {
      const fn = _deleteResolvedReviewTicketsFn || (await import('../api/client')).deleteResolvedReviewTickets;
      const result = await fn();
      onToast?.({ type: 'success', msg: `已清理 ${result.count} 个工单` });
      const { fetchReviewTickets } = await import('../api/client');
      const data = await fetchReviewTickets('pending');
      setReviewTickets(data.tickets || []);
    } catch (err) {
      onToast?.({ type: 'error', msg: `清理失败: ${err.message}` });
    }
  };

  const formatRemaining = (expiresAt) => {
    if (!expiresAt) return '—';
    const remaining = expiresAt * 1000 - Date.now();
    if (remaining <= 0) return '已过期';
    const hours = Math.floor(remaining / 3600000);
    const mins = Math.floor((remaining % 3600000) / 60000);
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  // ---- 以太网自动填充 ----
  useEffect(() => {
    if (manualNetworkType === 'ethernet' && invite?.master_host) {
      // 以太网选项下自动填充主节点地址和端口
      if (!manualAddress.trim()) {
        setManualAddress(`${invite.master_host}:${invite.master_port}`);
      }
      // 自动生成节点 ID 建议
      if (!manualNodeId.trim()) {
        const existingCount = (nodes?.nodes || []).filter(n => n.network_type === 'ethernet').length;
        setManualNodeId(`eth-node-${existingCount + 1}`);
      }
    }
  }, [manualNetworkType]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- 分布式推理开关 ----
  const fetchDistributedConfig = useCallback(() => {
    fetchDistributedInferenceConfig()
      .then(config => setDistributedEnabled(config.enabled))
      .catch(() => {});
  }, []);

  const handleToggleDistributed = async () => {
    setTogglingDistributed(true);
    try {
      const result = await updateDistributedInferenceConfig(!distributedEnabled);
      if (result.status === 'ok') {
        setDistributedEnabled(result.enabled);
        onToast?.({ type: 'success', msg: `分布式推理已${result.enabled ? '启用' : '禁用'}` });
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `切换失败: ${err.message}` });
    } finally {
      setTogglingDistributed(false);
    }
  };

  // ---- 动态分层 ----
  const fetchLayerConfig = useCallback(() => {
    fetchLayerAssignment()
      .then(data => {
        setLayerAssignment(data);
        // 初始化覆盖值为当前值
        const ov = {};
        (data.assignments || []).forEach(a => {
          ov[a.node_id] = { start: a.start_layer, end: a.end_layer };
        });
        setLayerOverrides(ov);
      })
      .catch(() => {});
  }, []);

  const handleApplyLayerOverride = async () => {
    const assignments = Object.entries(layerOverrides).map(([node_id, val]) => ({
      node_id,
      start_layer: val.start,
      end_layer: val.end,
    }));
    try {
      const result = await updateLayerAssignment(assignments);
      if (result.status === 'ok') {
        onToast?.({ type: 'success', msg: '分层配置已更新（手动模式）' });
        fetchLayerConfig();
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `分层更新失败: ${err.message}` });
    }
  };

  const handleResetLayerStrategy = async () => {
    if (!window.confirm('确定要恢复自动分层吗？这将清除手动覆盖并重新根据硬件配置动态计算。')) return;
    try {
      await resetLayerAssignments();
      await fetchLayerConfig();
      onToast?.({ type: 'success', msg: '已恢复自动分层策略' });
    } catch (err) {
      onToast?.({ type: 'error', msg: `操作失败: ${err.message}` });
    }
  };

  // ---- 云同步状态 ----
  const fetchSyncStatus = useCallback(() => {
    fetchConversationSyncStatus()
      .then(setSyncStatus)
      .catch(() => {});
  }, []);

  // ---- 主节点：手动注册从节点 ----
  const handleManualRegister = async () => {
    if (!manualNodeId.trim()) {
      onToast?.({ type: 'error', msg: '请输入节点 ID' });
      return;
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(manualNodeId.trim())) {
      onToast?.({ type: 'error', msg: '节点 ID 只能包含字母、数字、下划线和连字符' });
      return;
    }
    setRegistering(true);
    try {
      const result = await manualRegisterNode(
        manualNodeId.trim(),
        manualHostname.trim() || manualNodeId.trim(),
        manualAddress.trim(),
        manualNetworkType,
        manualNodeType,
      );
      if (result.status === 'registered' || result.status === 'updated') {
        onToast?.({ type: 'success', msg: result.message || `节点 '${result.node_id}' 已保存` });
        setManualNodeId('');
        setManualHostname('');
        setManualAddress('');
        refresh();
      } else if (result.status === 'exists') {
        onToast?.({ type: 'warning', msg: result.message || `节点 '${result.node_id}' 已存在` });
      } else if (result.status === 'conflict') {
        onToast?.({ type: 'warning', msg: result.reason || `节点 '${result.node_id}' 已有真实连接记录，请先注销或删除后再重建` });
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `注册失败: ${err.message}` });
    } finally {
      setRegistering(false);
    }
  };

  // ---- 从节点：自动发现主节点（数据库查询） ----
  const handleDiscover = async (showToast = true) => {
    setDiscovering(true);
    try {
      const result = await discoverMaster();
      setDiscovery(result);
      if (result.found && !result.stale) {
        setMasterHost(result.master_host);
        setMasterPort(String(result.master_port || 8888));
        if (showToast) {
          onToast?.({ type: 'success', msg: `🔍 在数据库中发现了主节点: ${result.master_host}:${result.master_port}` });
        }
      } else if (result.found && result.stale) {
        // 找到了但心跳过期
        setMasterHost(result.master_host);
        setMasterPort(String(result.master_port || 8888));
        if (showToast) {
          onToast?.({ type: 'warning', msg: `⚠️ 主节点记录已过期 (${result.master_host}:${result.master_port})，请确认主节点在线` });
        }
      } else if (showToast) {
        onToast?.({ type: 'error', msg: '未在数据库中发现主节点，请手动输入地址' });
      }
    } catch (err) {
      setDiscovery({ found: false });
      if (showToast) {
        onToast?.({ type: 'error', msg: `自动发现失败: ${err.message}` });
      }
    } finally {
      setDiscovering(false);
    }
  };

  // 从节点挂载时自动尝试发现主节点（静默）
  useEffect(() => {
    if (!isMaster && myRole?.is_client) {
      // 如果 master_discovery 已在 myRole 中返回（来自 get_my_role），直接使用
      if (myRole.master_discovery?.found) {
        const md = myRole.master_discovery;
        setDiscovery(md);
        if (!md.stale) {
          setMasterHost(md.master_host);
          setMasterPort(String(md.master_port || 8888));
        }
      } else {
        // 否则主动查询
        handleDiscover(false);
      }
    }
  }, [isMaster, myRole?.is_client]); // eslint-disable-line react-hooks/exhaustive-deps

  // 从节点：周期性检查主节点健康状态（配合 5s 刷新）
  useEffect(() => {
    if (!isMaster && myRole?.is_client) {
      fetchMasterHealth()
        .then(setMasterHealth)
        .catch(() => setMasterHealth(null));
    }
  }, [isMaster, myRole?.is_client, nodes]); // nodes 变化时重新检查

  // 主节点：获取分布式推理开关、分层配置、转让日志和备用主节点
  useEffect(() => {
    if (isMaster) {
      fetchDistributedConfig();
      fetchLayerConfig();
      fetchSyncStatus();
      loadTransferLogs();
      loadSpareMasterData();
    }
  }, [isMaster, fetchDistributedConfig, fetchLayerConfig, fetchSyncStatus, loadTransferLogs, loadSpareMasterData]);

  // ---- 从节点：连接主节点 ----
  const handleConnect = async () => {
    if (!masterHost.trim()) {
      onToast?.({ type: 'error', msg: '请输入主节点 IP 地址' });
      return;
    }
    const port = parseInt(masterPort, 10) || 8888;
    setConnecting(true);
    try {
      const result = await connectToMaster(masterHost.trim(), port);
      if (result.status === 'connected') {
        onToast?.({ type: 'success', msg: result.message || '连接成功！' });
        refresh();
        // 更新 myRole 的 node_id
        if (myRole && result.node_id) {
          myRole.node_id = result.node_id;
        }
      }
    } catch (err) {
      onToast?.({ type: 'error', msg: `连接失败: ${err.message}` });
    } finally {
      setConnecting(false);
    }
  };

  if (loading && !status) {
    return (
      <div className="admin-panel">
        <div className="admin-loading">
          <div className="admin-spinner" />
          <p>加载集群状态...</p>
        </div>
      </div>
    );
  }

  const nodeList = nodes?.nodes || [];
  const filteredNodeList = (isMaster
    ? (showOnlineOnly ? nodeList.filter(n => n.is_available) : nodeList)
    : nodeList.filter(n => n.node_id === myRole?.node_id));
  const onlineCount = nodes?.online_count || 0;
  const offlineCount = nodes?.offline_count || 0;
  const nodesReady = status?.nodes_ready || false;
  const runMode = status?.run_mode || 'single';
  const currentTask = status?.current_task;
  const tcpServer = status?.tcp_server;
  const hasCapacity = invite?.has_capacity ?? true;
  const macAddresses = invite?.mac_addresses || myRole?.mac_addresses || [];
  const identityVerified = invite?.identity_verified ?? myRole?.identity_verified ?? null;
  const identityReason = invite?.identity_reason ?? myRole?.identity_reason ?? '';

  return (
    <div className="admin-panel">
      {/* Header */}
      <div className="chat-header">
        <h2>⚙️ 后台管理 · 节点管理</h2>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className={`cluster-mode-badge ${runMode}`}>
            {runMode === 'distributed' ? '🌐 分布式' : '🏠 单机'}
          </span>
          {isMaster ? (
            <span className="role-badge" data-role="master">🖥️ 主节点</span>
          ) : (
            <span className="role-badge" data-role="client">💻 {myRole?.node_id || '从节点'}</span>
          )}
          <button className="btn-ghost" onClick={refresh} title="刷新">🔄</button>
        </div>
      </div>

      <div className="admin-content">
        {/* ---- 从节点：主节点宕机告警 ---- */}
        {!isMaster && masterHealth && !masterHealth.master_online && (
          <div className="master-down-banner">
            <span className="master-down-icon">⚠️</span>
            <div className="master-down-text">
              <strong>主节点已离线</strong>
              {masterHealth.last_seen_seconds_ago != null && (
                <span> · 上次心跳 {masterHealth.last_seen_seconds_ago} 秒前</span>
              )}
            </div>
            <span className="master-down-hint">
              主节点恢复后将自动重连。您也可以手动检查主节点状态。
            </span>
          </div>
        )}

        {/* ---- 从节点：连接主节点面板 ---- */}
        {!isMaster && (
          <section className="admin-section">
            <h3>🔗 连接主节点</h3>
            <div className="connect-panel">
              <p className="connect-desc">
                输入主节点的 IP 地址和端口以注册到分布式推理集群。
                主节点的连接信息可在主节点后台管理的「注册新节点」区域找到。
              </p>

              {/* 自动发现提示 */}
              {discovery?.found && (
                <div className={`discovery-banner ${discovery.stale ? 'stale' : 'fresh'}`}>
                  <span className="discovery-icon">{discovery.stale ? '⚠️' : '✅'}</span>
                  <span className="discovery-text">
                    {discovery.stale
                      ? `数据库中发现主节点记录 (${discovery.master_host}:${discovery.master_port})，但心跳已过期，请确认主节点在线`
                      : `已在数据库中自动发现主节点: ${discovery.master_host}:${discovery.master_port}`}
                  </span>
                  <span className="discovery-source">
                    {discovery.source === 'database' ? '📡 数据库' : '⚙️ 配置文件'}
                  </span>
                </div>
              )}

              <div className="connect-form">
                <div className="connect-field">
                  <label>主节点 IP</label>
                  <input
                    type="text"
                    className="connect-input"
                    placeholder="例如 192.168.1.100"
                    value={masterHost}
                    onChange={(e) => setMasterHost(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleConnect(); }}
                  />
                </div>
                <div className="connect-field connect-field-port">
                  <label>端口</label>
                  <input
                    type="number"
                    className="connect-input"
                    placeholder="8888"
                    value={masterPort}
                    min={1}
                    max={65535}
                    onChange={(e) => setMasterPort(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleConnect(); }}
                  />
                </div>
                <button
                  className="btn-primary connect-btn"
                  onClick={handleConnect}
                  disabled={connecting || !masterHost.trim()}
                >
                  {connecting ? '⏳ 连接中...' : '🔗 连接主节点'}
                </button>
                <button
                  className="btn-ghost discover-btn"
                  onClick={() => handleDiscover(true)}
                  disabled={discovering}
                  title="从数据库查询主节点地址"
                >
                  {discovering ? '⏳ 查询中...' : '🔍 自动发现'}
                </button>
              </div>
            </div>
          </section>
        )}

        {/* ---- 主节点：手动注册从节点 ---- */}
        {isMaster && (
          <section className="admin-section">
            <h3>📝 手动注册从节点</h3>
            <div className="connect-panel">
              <p className="connect-desc">
                手动录入从节点信息以预留槽位。注册后节点显示为离线状态，
                待从节点通过 TCP 连接后自动激活。从节点也可通过「连接主节点」自行注册。
              </p>
              <div className="manual-register-form">
                <div className="connect-field">
                  <label>节点 ID *</label>
                  <input
                    type="text"
                    className="connect-input"
                    placeholder="例如 jetson-nano-01"
                    value={manualNodeId}
                    onChange={(e) => setManualNodeId(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleManualRegister(); }}
                  />
                </div>
                <div className="connect-field">
                  <label>主机名</label>
                  <input
                    type="text"
                    className="connect-input"
                    placeholder="默认同节点 ID"
                    value={manualHostname}
                    onChange={(e) => setManualHostname(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleManualRegister(); }}
                  />
                </div>
                <div className="connect-field connect-field-port">
                  <label>预留地址</label>
                  <input
                    type="text"
                    className="connect-input"
                    placeholder="IP:Port（可选）"
                    value={manualAddress}
                    onChange={(e) => setManualAddress(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleManualRegister(); }}
                  />
                </div>
                <div className="connect-field">
                  <label>网络类型</label>
                  <select
                    className="connect-input"
                    value={manualNetworkType}
                    onChange={(e) => setManualNetworkType(e.target.value)}
                  >
                    <option value="ethernet">🔌 以太网</option>
                    <option value="wifi">📶 WiFi</option>
                    <option value="localhost">🏠 本地</option>
                    <option value="unknown">❓ 未知</option>
                  </select>
                </div>
                <div className="connect-field">
                  <label>节点类型</label>
                  <select
                    className="connect-input"
                    value={manualNodeType}
                    onChange={(e) => setManualNodeType(e.target.value)}
                  >
                    <option value="pc">💻 PC</option>
                    <option value="android">📱 Android</option>
                  </select>
                </div>
                <button
                  className="btn-primary connect-btn"
                  onClick={handleManualRegister}
                  disabled={registering || !manualNodeId.trim()}
                >
                  {registering ? '⏳ 注册中...' : '📝 注册节点'}
                </button>
              </div>
            </div>
          </section>
        )}

        {/* ---- 集群概览卡片 ---- */}
        <section className="admin-section">
          <h3>📊 {isMaster ? '集群概览' : '本节点状态'}</h3>
          <div className="admin-stats-grid">
            <div className={`admin-stat-card ${nodesReady ? 'ready' : 'not-ready'}`}>
              <div className="stat-icon">{nodesReady ? '✅' : '⚠️'}</div>
              <div className="stat-info">
                <div className="stat-value">{nodesReady ? '就绪' : '未就绪'}</div>
                <div className="stat-label">集群状态</div>
              </div>
            </div>
            <div className="admin-stat-card online">
              <div className="stat-icon">🟢</div>
              <div className="stat-info">
                <div className="stat-value">
                  {onlineCount}
                  {isMaster && <span style={{ fontSize: 12, opacity: 0.7 }}> / {nodeList.length}</span>}
                </div>
                <div className="stat-label">在线节点</div>
              </div>
            </div>
            <div className="admin-stat-card offline">
              <div className="stat-icon">🔴</div>
              <div className="stat-info">
                <div className="stat-value">{offlineCount}</div>
                <div className="stat-label">离线节点</div>
              </div>
            </div>
            <div className={`admin-stat-card ${currentTask ? 'busy' : 'idle'}`}>
              <div className="stat-icon">{currentTask ? '🔄' : '💤'}</div>
              <div className="stat-info">
                <div className="stat-value">{currentTask ? '运行中' : '空闲'}</div>
                <div className="stat-label">
                  {currentTask
                    ? `任务 ${currentTask.task_id?.slice(-8)} · ${currentTask.elapsed?.toFixed(1)}s`
                    : '当前无任务'}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ---- 主节点：注册新节点邀请面板 ---- */}
        {isMaster && invite && (
          <section className="admin-section">
            <h3>📨 注册新节点</h3>
            <div className="invite-panel">
              <p className="connect-desc">
                将以下连接信息提供给新节点。新节点在「设置 → 分布式推理优化」中开启后，
                在后台管理的「连接主节点」中点击「🔍 自动发现」即可自动填充地址并注册。
                {invite.db_registered && (
                  <span style={{ color: 'var(--success)', display: 'block', marginTop: 4 }}>
                    ✅ 主节点信息已自动同步到数据库，从节点可通过「自动发现」找到本节点。
                  </span>
                )}
                {!invite.db_registered && (
                  <span style={{ color: 'var(--warning)', display: 'block', marginTop: 4 }}>
                    ⚠️ 数据库未连接，从节点需手动输入本节点地址。
                  </span>
                )}
                {!hasCapacity && (
                  <span style={{ color: 'var(--danger)', display: 'block', marginTop: 4 }}>
                    ⚠️ 已达到最大节点数上限 ({invite.max_nodes})，请先扩容或注销离线节点。
                  </span>
                )}
              </p>
              <div className="invite-info-grid">
                <div className="invite-info-item">
                  <span className="invite-label">主节点地址</span>
                  <span className="invite-value mono">{invite.master_host}:{invite.master_port}</span>
                </div>
                <div className="invite-info-item">
                  <span className="invite-label">已注册节点</span>
                  <span className="invite-value">{invite.node_count} / {invite.max_nodes}</span>
                </div>
                <div className="invite-info-item">
                  <span className="invite-label">在线节点</span>
                  <span className="invite-value">{invite.online_count}</span>
                </div>
                <div className="invite-info-item">
                  <span className="invite-label">容量状态</span>
                  <span className={`invite-value ${hasCapacity ? 'capacity-ok' : 'capacity-full'}`}>
                    {hasCapacity ? '✅ 可注册' : '❌ 已满'}
                  </span>
                </div>
              </div>

              {/* MAC 地址身份标识 */}
              {macAddresses.length > 0 && (
                <div className="mac-identity-section">
                  <div className="mac-identity-header">
                    <span className="mac-identity-icon">🔒</span>
                    <span className="mac-identity-title">硬件身份标识（MAC 地址）</span>
                    {identityVerified !== null && (
                      <span className={`identity-badge ${identityVerified ? 'verified' : 'unverified'}`}>
                        {identityVerified ? '✅ 已验证' : '⚠️ 未验证'}
                      </span>
                    )}
                  </div>
                  <div className="mac-list">
                    {macAddresses.map((mac) => (
                      <span key={mac} className="mac-chip mono">{mac}</span>
                    ))}
                  </div>
                  <p className="mac-identity-desc">
                    主节点通过 MAC 地址确认身份，IP 地址变化不影响身份识别。
                    {identityReason === 'first_run' && ' 本次为首次启动，MAC 已自动记录。'}
                    {identityReason === 'match' && ' MAC 与数据库记录一致。'}
                    {identityReason === 'mac_mismatch' && (
                      <span style={{ color: 'var(--danger)' }}>
                        ⚠️ MAC 与数据库记录不匹配！可能是另一台机器冒充主节点，或本机更换了网卡。
                        如需更换主节点机器，请使用下方「重置主节点身份」功能。
                      </span>
                    )}
                    {identityReason === 'db_unavailable' && ' 数据库不可用，跳过身份验证。'}
                  </p>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ---- 节点列表 ---- */}
        <section className="admin-section">
          <h3>🖥️ {isMaster ? '已注册节点' : '本节点详情'}</h3>
          {isMaster && (
            <div className="queue-control-row" style={{ marginBottom: 8 }}>
              <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
                <input type="checkbox" checked={showOnlineOnly}
                  onChange={(e) => setShowOnlineOnly(e.target.checked)} />
                仅显示在线
              </label>
              <button className="btn btn-sm btn-danger-outline"
                onClick={handleDeleteAllOffline}
                title="删除所有离线节点记录"
              >🗑 清理离线节点</button>
            </div>
          )}
          <div className="admin-table-wrap">
            <table className="admin-node-table">
              <thead>
                <tr>
                  <th>节点</th>
                  <th>角色</th>
                  <th>类型</th>
                  <th>状态</th>
                  <th>🔗 连接</th>
                  <th>⏱ 延迟</th>
                  <th>🌐 网络</th>
                  <th>地址</th>
                  <th>主机名</th>
                  <th>在线时长</th>
                  <th>任务</th>
                  <th>错误</th>
                  {isMaster && <th>操作</th>}
                </tr>
              </thead>
              <tbody>
                {filteredNodeList.length === 0 ? (
                  <tr>
                    <td colSpan={isMaster ? 13 : 12} style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
                      {isMaster ? '暂无已注册的从节点，使用上方「注册新节点」添加' : '尚未连接到主节点，请在上方输入主节点地址'}
                    </td>
                  </tr>
                ) : (
                  filteredNodeList.map((node) => {
                    const isOffline = node.state === 'offline';
                    const isMasterNode = node.role === 'master';
                    const isAndroidThin = node.node_type === 'android' && node.device_info?.connection_type === 'http_thin';
                    const netLabel = NETWORK_LABELS[node.network_type] || NETWORK_LABELS.unknown;
                    const netClass = NETWORK_CLASSES[node.network_type] || NETWORK_CLASSES.unknown;

                    // TCP 连接状态
                    const tcpDetail = status?.tcp_server?.client_details?.[node.node_id];
                    const tcpConnected = isAndroidThin ? node.state === 'online' : isTcpActive(tcpDetail);
                    const tcpMissed = tcpDetail?.heartbeat_missed || 0;
                    const isSelfMaster = isMasterNode && isMaster;

                    // 延迟 / RTT
                    let rttDisplay = '—';
                    let rttMs = null;
                    if (isMaster && !isSelfMaster) {
                      // 主节点视角：显示从节点心跳新鲜度
                      if (tcpDetail?.last_heartbeat) {
                        const age = Date.now() / 1000 - tcpDetail.last_heartbeat;
                        rttMs = age * 1000;
                        rttDisplay = formatRTT(rttMs);
                      }
                    } else if (!isMaster && isMasterNode) {
                      // 从节点视角：显示到主节点的 RTT
                      rttMs = status?.tcp_client?.avg_rtt_ms;
                      rttDisplay = formatRTT(rttMs);
                    }

                    const typeIcon = TYPE_ICONS[node.node_type] || TYPE_ICONS.pc;
                    const typeLabel = TYPE_LABELS[node.node_type] || TYPE_LABELS.pc;

                    return (
                      <tr key={node.node_id} className={isOffline ? 'row-offline' : ''}>
                        <td>
                          <span className="node-id-cell">
                            {typeIcon} {node.node_id}
                          </span>
                        </td>
                        <td>
                          <span className="role-badge" data-role={node.role === 'master' ? 'master' : 'client'}>
                            {ROLE_LABELS[node.role] || node.role}
                          </span>
                        </td>
                        <td>
                          <span className="type-badge" data-type={node.node_type || 'pc'}>
                            {typeIcon} {typeLabel}
                          </span>
                        </td>
                        <td>
                          <span className="status-dot" style={{ '--dot-color': STATE_COLORS[node.state] || 'var(--text-muted)' }}>
                            {STATE_LABELS[node.state] || node.state}
                          </span>
                        </td>
                        <td>
                          {isSelfMaster ? (
                            <span className="conn-indicator conn-self" title="本机主节点">🖥️ 本地</span>
                          ) : isAndroidThin ? (
                            <span
                              className={`conn-indicator ${tcpConnected ? 'conn-ok' : 'conn-bad'}`}
                              title={`Android HTTP 薄客户端，last seen ${formatTime(node.last_heartbeat)}`}
                            >
                              <span className="conn-dot" />
                              {tcpConnected ? 'HTTP 在线' : 'HTTP 离线'}
                            </span>
                          ) : (
                            <span
                              className={`conn-indicator ${tcpConnected ? 'conn-ok' : 'conn-bad'}`}
                              title={tcpConnected
                                ? `TCP 连接正常，上次心跳 ${formatTime(tcpDetail?.last_heartbeat)}`
                                : tcpDetail
                                  ? `TCP 未连接，心跳丢失 ${tcpMissed} 次`
                                  : '无 TCP 连接信息'}
                            >
                              <span className="conn-dot" />
                              {tcpConnected ? 'TCP 已连' : 'TCP 未连'}
                            </span>
                          )}
                        </td>
                        <td>
                          <span className="rtt-cell" title={rttMs ? `RTT: ${rttMs.toFixed(1)}ms` : undefined}>
                            {rttDisplay}
                          </span>
                        </td>
                        <td><span className={`network-badge ${netClass}`}>{netLabel}</span></td>
                        <td className="mono-cell">
                          {isAndroidThin ? 'HTTP thin client' : (node.address || '—')}
                        </td>
                        <td>{node.hostname || '—'}</td>
                        <td>{isOffline ? '—' : (isAndroidThin ? `seen ${formatTime(node.last_heartbeat)}` : formatUptime(node.connected_at))}</td>
                        <td title={isAndroidThin ? 'Android HTTP 薄客户端请求来源，不参与 pipeline worker 计算' : '实际完成/参与的推理任务数'}>{node.task_count}</td>
                        <td>
                          <span style={{ color: node.error_count > 0 ? 'var(--danger)' : undefined }}>
                            {node.error_count}
                          </span>
                        </td>
                        {isMaster && (
                          <td>
                            {!isMasterNode && !isOffline && (
                              <button
                                className="btn-ghost btn-danger-ghost"
                                onClick={() => handleDeregister(node.node_id)}
                                disabled={deregistering === node.node_id}
                                title="强制注销"
                              >
                                {deregistering === node.node_id ? '⏳' : '✕'}
                              </button>
                            )}
                            {!isMasterNode && isOffline && (
                              <button
                                className="btn-ghost btn-danger-ghost"
                                onClick={() => handleDeleteNode(node.node_id)}
                                disabled={deletingNode === node.node_id}
                                title="删除节点记录"
                              >
                                {deletingNode === node.node_id ? '⏳' : '🗑'}
                              </button>
                            )}
                          </td>
                        )}
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* TCP 服务端信息 */}
          {tcpServer && isMaster && (
            <div className="tcp-server-info">
              <span>🔌 TCP 监听: {tcpServer.host}:{tcpServer.port}</span>
              <span>已连接: {tcpServer.connected_clients?.length || 0} 个客户端</span>
            </div>
          )}
        </section>

        {/* ---- 节点容量配置（仅主节点） ---- */}
        {isMaster && config && (
          <section className="admin-section">
            <h3>🔧 节点容量配置</h3>
            <div className="max-nodes-config">
              <div className="max-nodes-info">
                <span className="config-key">最大节点数</span>
                <span className="config-desc">
                  当前集群最大容纳 <strong>{config.max_nodes || 3}</strong> 个节点。
                  从节点通过 TCP 连接注册后自动加入列表；减少上限时仅移除离线空位。
                </span>
              </div>
              <div className="max-nodes-input-row">
                <input
                  type="number" className="setting-number-input"
                  value={maxNodesInput} min={1} max={64} step={1}
                  onChange={(e) => setMaxNodesInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleUpdateMaxNodes(); }}
                />
                <button className="btn-primary" onClick={handleUpdateMaxNodes} disabled={updatingMaxNodes}>
                  {updatingMaxNodes ? '⏳ 更新中...' : '应用'}
                </button>
              </div>
            </div>
          </section>
        )}

        {/* ---- 主节点身份管理（仅主节点） ---- */}
        {isMaster && (
          <section className="admin-section">
            <h3>🔐 主节点身份管理</h3>
            <div className="identity-management">
              <p className="connect-desc">
                主节点使用物理网卡 MAC 地址作为不可变身份标识。
                IP 地址可能随网络环境变化，但 MAC 地址保持不变。
                如果 MAC 验证失败，说明当前机器与数据库中记录的主节点不是同一台。
              </p>
              <div className="identity-actions">
                <div className="identity-current">
                  <span className="identity-label">当前身份状态：</span>
                  {identityVerified === true && (
                    <span className="identity-badge verified">✅ 身份已验证</span>
                  )}
                  {identityVerified === false && identityReason === 'first_run' && (
                    <span className="identity-badge unverified">🆕 首次启动（已自动记录）</span>
                  )}
                  {identityVerified === false && identityReason === 'mac_mismatch' && (
                    <span className="identity-badge danger">⛔ MAC 不匹配</span>
                  )}
                  {identityVerified === false && identityReason === 'db_unavailable' && (
                    <span className="identity-badge unverified">⚠️ 数据库不可用</span>
                  )}
                  {(identityVerified === null || identityVerified === undefined) && (
                    <span className="identity-badge unverified">⏳ 未检测</span>
                  )}
                </div>
                <button
                  className="btn-primary btn-danger-ghost"
                  onClick={handleResetIdentity}
                  title="仅在更换主节点机器或网卡后使用"
                  style={{ fontSize: 13 }}
                >
                  🔄 重置主节点身份
                </button>
              </div>
            </div>
          </section>
        )}

        {/* ---- 备用主节点（仅主节点） ---- */}
        {isMaster && (
          <section className="admin-section">
            <h3>🛡️ 备用主节点</h3>
            <div className="identity-management">
              <p className="connect-desc">
                指定一个从节点作为备用主节点。转让期间，备用主节点<strong>暂代主节点职责</strong>填补空窗期。
                <br />• 转让前，主节点向备用主节点发送激活通知（TCP），备用暂代监政
                <br />• 新主节点上线后，自动向备用主节点发送接管通知，备用退出暂代
                <br />• 转让目标<strong>不能</strong>是备用主节点本身（请选择其他在线从节点）
              </p>
              {/* 当前备用主节点状态 */}
              {spareMaster ? (
                <div className="spare-master-status" style={{
                  background: spareMaster.is_online ? 'var(--success-bg)' : 'var(--warning-bg)',
                  border: `1px solid ${spareMaster.is_online ? 'var(--success)' : 'var(--warning)'}`,
                  borderRadius: 8, padding: '10px 14px', marginBottom: 12,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
                    <div>
                      <span style={{ fontWeight: 600 }}>当前备用主节点：</span>
                      <span className="mono" style={{ marginLeft: 4 }}>{spareMaster.node_id}</span>
                      {spareMaster.hostname && <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>({spareMaster.hostname})</span>}
                      <span className={`role-badge ${spareMaster.is_online ? 'promotion' : 'demotion'}`}
                            style={{ marginLeft: 8, background: spareMaster.is_online ? 'var(--success)' : 'var(--warning)' }}>
                        {spareMaster.is_online ? '✅ 在线' : '⚠️ 离线'}
                      </span>
                    </div>
                    <button
                      className="btn-primary btn-danger-ghost"
                      onClick={handleRemoveSpareMaster}
                      style={{ fontSize: 12 }}
                    >
                      ✕ 取消指定
                    </button>
                  </div>
                </div>
              ) : (
                <div className="spare-master-status" style={{
                  background: 'var(--warning-bg)', border: '1px solid var(--warning)',
                  borderRadius: 8, padding: '10px 14px', marginBottom: 12,
                }}>
                  <span style={{ color: 'var(--warning)' }}>⚠️ 未指定备用主节点 — 转让功能将被禁用</span>
                </div>
              )}
              {/* 指定备用主节点表单 */}
              <div className="transfer-form">
                <div className="connect-field" style={{ flex: 1 }}>
                  <label>选择从节点作为备用</label>
                  <select
                    className="connect-input"
                    value={spareTarget}
                    onChange={(e) => setSpareTarget(e.target.value)}
                    style={{ width: '100%' }}
                  >
                    <option value="">— 选择在线从节点 —</option>
                    {(nodes?.nodes || [])
                      .filter(n => n.role === 'client' && n.state === 'online')
                      .map(n => (
                        <option key={n.node_id} value={n.node_id}>
                          💻 {n.node_id} {n.hostname ? `(${n.hostname})` : ''} {n.address ? `@ ${n.address}` : ''}
                        </option>
                      ))}
                  </select>
                </div>
                <button
                  className="btn-primary"
                  onClick={handleDesignateSpareMaster}
                  disabled={designatingSpare || !spareTarget.trim()}
                  style={{ fontSize: 13, alignSelf: 'flex-end' }}
                >
                  {designatingSpare ? '⏳ 指定中...' : '🛡️ 指定为备用'}
                </button>
              </div>
              {/* 备用主节点操作日志 */}
              {spareMasterLogs.length > 0 && (
                <div className="transfer-logs-section" style={{ marginTop: 16 }}>
                  <div className="mac-identity-header">
                    <span className="mac-identity-icon">📋</span>
                    <span className="mac-identity-title">备用主节点操作日志</span>
                  </div>
                  <div className="admin-table-wrap" style={{ marginTop: 8 }}>
                    <table className="admin-node-table">
                      <thead>
                        <tr>
                          <th>时间</th>
                          <th>操作</th>
                          <th>详情</th>
                        </tr>
                      </thead>
                      <tbody>
                        {spareMasterLogs.slice(0, 10).map((log, i) => (
                          <tr key={i}>
                            <td className="mono-cell">
                              {log.timestamp ? new Date(log.timestamp * 1000).toLocaleString() : '—'}
                            </td>
                            <td>
                              <span className={`role-badge ${log.direction === 'designated' ? 'promotion' : 'demotion'}`}
                                    style={{ background: log.direction === 'designated' ? 'var(--success)' : 'var(--text-muted)' }}>
                                {log.direction === 'designated' ? '🛡️ 指定' : '✕ 取消'}
                              </span>
                            </td>
                            <td className="mono-cell" style={{ fontSize: 12 }}>
                              {log.details?.target_node_id || log.details?.previous_spare || '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ---- 角色转让（仅主节点） ---- */}
        {isMaster && (
          <section className="admin-section">
            <h3>🔄 转让主节点身份</h3>
            <div className="identity-management">
              <p className="connect-desc">
                将主节点身份转让给任意在线从节点。<strong>备用主节点</strong>在转让空窗期暂代主节点职责。
                <br />• <strong>转让目标</strong>（新主节点）收到 TCP 通知并保存升级日志
                <br />• <strong>备用主节点</strong>收到激活通知，暂代监政直到新主节点上线接管
                <br />• 本节点保存降级日志并更新数据库
                <br />• <strong>建议三方重启服务</strong>：目标节点以主节点模式运行，本节点以从节点模式运行，备用节点退出暂代
              </p>
              {/* 转让前置条件检查 */}
              {!spareMaster ? (
                <div style={{
                  background: 'var(--warning-bg)', border: '1px solid var(--warning)',
                  borderRadius: 8, padding: '10px 14px', marginBottom: 12,
                }}>
                  <span style={{ color: 'var(--warning)' }}>
                    ⚠️ 未指定备用主节点，无法转让。请先在上方「🛡️ 备用主节点」中指定一个从节点作为空窗期监政。
                  </span>
                </div>
              ) : !spareMaster?.is_online ? (
                <div style={{
                  background: 'var(--warning-bg)', border: '1px solid var(--warning)',
                  borderRadius: 8, padding: '10px 14px', marginBottom: 12,
                }}>
                  <span style={{ color: 'var(--warning)' }}>
                    ⚠️ 备用主节点 '{spareMaster.node_id}' 当前离线，无法激活暂代。请等待其上线上再转让。
                  </span>
                </div>
              ) : (
                <>
                  {spareMaster.is_active && (
                    <div style={{
                      background: 'var(--info-bg)', border: '1px solid var(--info)',
                      borderRadius: 8, padding: '10px 14px', marginBottom: 12,
                    }}>
                      <span>🛡️ 备用主节点 <strong>{spareMaster.node_id}</strong> 当前处于<strong>暂代激活</strong>状态，等待新主节点上线接管。</span>
                    </div>
                  )}
                  <div className="transfer-form">
                    <div className="connect-field" style={{ flex: 1 }}>
                      <label>目标从节点（新主节点）</label>
                      <select
                        className="connect-input"
                        value={transferTarget}
                        onChange={(e) => setTransferTarget(e.target.value)}
                        style={{ width: '100%' }}
                      >
                        <option value="">— 选择在线从节点 —</option>
                        {(nodes?.nodes || [])
                          .filter(n => n.role === 'client' && n.state === 'online' && n.node_id !== spareMaster?.node_id)
                          .map(n => (
                            <option key={n.node_id} value={n.node_id}>
                              💻 {n.node_id} {n.hostname ? `(${n.hostname})` : ''} {n.address ? `@ ${n.address}` : ''}
                            </option>
                          ))}
                      </select>
                    </div>
                    <button
                      className="btn-primary"
                      onClick={handleTransferMaster}
                      disabled={transferring || !transferTarget}
                      style={{ fontSize: 13, alignSelf: 'flex-end' }}
                      title="将主节点身份转让给选中的从节点，备用主节点将在空窗期暂代"
                    >
                      {transferring ? '⏳ 转让中...' : '🔄 转让身份'}
                    </button>
                  </div>
                </>
              )}
              {/* 转让日志 */}
              {transferLogs.length > 0 && (
                <div className="transfer-logs-section" style={{ marginTop: 16 }}>
                  <div className="mac-identity-header">
                    <span className="mac-identity-icon">📋</span>
                    <span className="mac-identity-title">转让日志</span>
                  </div>
                  <div className="admin-table-wrap" style={{ marginTop: 8 }}>
                    <table className="admin-node-table">
                      <thead>
                        <tr>
                          <th>时间</th>
                          <th>方向</th>
                          <th>原角色</th>
                          <th>新角色</th>
                          <th>关联节点</th>
                        </tr>
                      </thead>
                      <tbody>
                        {transferLogs.slice(0, 10).map((log, i) => (
                          <tr key={i}>
                            <td className="mono-cell">{log.timestamp_iso || '—'}</td>
                            <td>
                              <span className={`role-badge ${log.direction === 'promotion' ? 'promotion' : 'demotion'}`}
                                    style={{ background: log.direction === 'promotion' ? 'var(--success)' : 'var(--warning)' }}>
                                {log.direction === 'promotion' ? '⬆️ 升级' : '⬇️ 降级'}
                              </span>
                            </td>
                            <td>{log.from_role}</td>
                            <td>{log.to_role}</td>
                            <td className="mono-cell">{log.related_node}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ---- 主节点转让审查（仅主节点可见） ---- */}
        {isMaster && (
          <section className="admin-section">
            <h3>🗳️ 主节点转让审查</h3>
            <p className="connect-desc">
              主节点转让需要经过集群中 PC 独显节点的投票审查。
              投票通过（≥ +2）后，转让操作才会生效。仅 NVIDIA CUDA 独显节点可投票。
            </p>

            {/* 创建工单 */}
            <div className="review-create-bar">
              <select
                value={reviewTarget}
                onChange={e => setReviewTarget(e.target.value)}
                style={{ flex: 1 }}
              >
                <option value="">选择转让目标节点</option>
                {(nodes?.nodes || [])
                  .filter(n => n.role === 'client' && n.state === 'online')
                  .map(n => (
                    <option key={n.node_id} value={n.node_id}>
                      {n.node_id} ({n.hostname || '—'})
                    </option>
                  ))}
              </select>
              <input
                placeholder="转让原因（可选）"
                value={reviewReason}
                onChange={e => setReviewReason(e.target.value)}
                style={{ flex: 1, marginLeft: 8 }}
              />
              <button
                className="btn-primary"
                onClick={handleCreateReview}
                disabled={creatingReview || !reviewTarget.trim()}
                style={{ marginLeft: 8 }}
              >
                {creatingReview ? '⏳ 创建中...' : '📝 创建审查工单'}
              </button>
            </div>

            {/* 待处理工单 */}
            {reviewTickets.length > 0 && (
              <div className="review-ticket-list" style={{ marginTop: 12 }}>
                {reviewTickets.map(ticket => (
                  <div key={ticket.ticket_id} className="review-ticket-card">
                    <div className="ticket-header">
                      <span className="ticket-id">{ticket.ticket_id}</span>
                      <span className="ticket-target">目标: {ticket.target_node_id}</span>
                      <span className={`ticket-score ${
                        ticket.score >= 2 ? 'score-approved' :
                        ticket.score <= -2 ? 'score-rejected' : ''
                      }`}>
                        {ticket.score > 0 ? '+' : ''}{ticket.score} 分
                      </span>
                      <span className="ticket-expiry">
                        {formatRemaining(ticket.expires_at)}
                      </span>
                      <button
                        className="btn-ghost btn-danger-ghost"
                        onClick={() => handleDeleteTicket(ticket.ticket_id)}
                        title="删除此工单"
                        style={{ marginLeft: 'auto', fontSize: 12, padding: '2px 6px' }}
                      >🗑 删除</button>
                    </div>
                    {ticket.transfer_reason && (
                      <div className="ticket-reason">{ticket.transfer_reason}</div>
                    )}
                    {(ticket.votes || []).length > 0 && (
                      <div className="ticket-votes">
                        {(ticket.votes || []).map((v, i) => (
                          <span key={i} className={`vote-badge vote-${v.value >= 0 ? 'pos' : 'neg'}`}>
                            {v.voter_node_id}: {v.value > 0 ? '+' : ''}{v.value}
                          </span>
                        ))}
                      </div>
                    )}
                    {/* 投票按钮 */}
                    {canVote && (
                      <div className="ticket-vote-actions">
                        <button
                          className="setting-btn danger-ghost"
                          onClick={() => handleVote(ticket.ticket_id, -1)}
                          disabled={votingTicket === ticket.ticket_id}
                        >👎 -1</button>
                        <button
                          className="setting-btn secondary"
                          onClick={() => handleVote(ticket.ticket_id, 0)}
                          disabled={votingTicket === ticket.ticket_id}
                        >⏸️ 0</button>
                        <button
                          className="setting-btn primary"
                          onClick={() => handleVote(ticket.ticket_id, 1)}
                          disabled={votingTicket === ticket.ticket_id}
                        >👍 +1</button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
              <button
                className="btn btn-sm btn-danger-outline"
                onClick={handleDeleteResolvedTickets}
                title="删除所有已批准/已拒绝/已过期的工单"
              >
                🗑 清理已解决工单
              </button>
            </div>
            {reviewTickets.length === 0 && (
              <p className="setting-desc" style={{ marginTop: 8 }}>
                暂无待处理的审查工单。
              </p>
            )}

            {/* 非CUDA独显提示 */}
            {!canVote && reviewTickets.length > 0 && (
              <p className="setting-desc" style={{ marginTop: 8, color: 'var(--warning)' }}>
                ⚠️ 当前节点不支持投票。仅 NVIDIA CUDA 独显节点可参与审查投票。
              </p>
            )}
          </section>
        )}

        {/* ---- 邮件告警测试（所有节点） ---- */}
        <section className="admin-section">
          <h3>📧 邮件告警测试</h3>
          <div className="identity-management">
            <p className="connect-desc">
              当从节点检测到主节点宕机超过 {180}s 时，将自动向管理员发送告警邮件。
              恢复后也会发送恢复通知。点击下方按钮验证 SMTP 邮件配置是否正确。
            </p>
            <div className="identity-actions">
              <button
                className="btn-primary"
                onClick={handleEmailTest}
                disabled={registering}
                style={{ fontSize: 13 }}
              >
                {registering ? '⏳ 发送中...' : '📧 发送测试邮件'}
              </button>
            </div>
          </div>
        </section>

        {/* ---- 分布式推理开关（所有节点可见） ---- */}
        <section className="admin-section">
          <h3>🌐 分布式推理</h3>
          <div className="distributed-toggle-panel">
            <p className="connect-desc">
              {isMaster
                ? '启用后，主节点将协调所有注册的从节点进行分布式推理。从节点的对话请求通过 TCP 转发至主节点调度执行。关闭后本节点独立运行。'
                : '启用后，本节点的对话请求将通过 TCP 转发至主节点进行分布式推理。关闭后仅使用本地模型推理。'}
            </p>
            <div className="distributed-toggle-row">
              <span className="distributed-toggle-label">
                当前状态：<strong>{distributedEnabled ? '✅ 已启用' : '❌ 已禁用'}</strong>
              </span>
              <button
                className={`btn-primary${distributedEnabled ? ' btn-danger-ghost' : ''}`}
                onClick={handleToggleDistributed}
                disabled={togglingDistributed}
                style={{ fontSize: 13 }}
              >
                {togglingDistributed ? '⏳ 切换中...' : distributedEnabled ? '🔴 禁用分布式推理' : '🟢 启用分布式推理'}
              </button>
            </div>
          </div>
        </section>

        {/* ---- 动态分层配置（仅主节点） ---- */}
        {config && isMaster && (
          <section className="admin-section">
            <h3>📋 分布式配置</h3>
            <div className="admin-config-grid">
              <div className="config-card">
                <div className="config-card-header">🌐 网络配置</div>
                <div className="config-rows">
                  <div className="config-row">
                    <span className="config-key">服务端地址</span>
                    <span className="config-value mono">{config.network?.server_ip}:{config.network?.server_port}</span>
                  </div>
                  <div className="config-row">
                    <span className="config-key">心跳间隔</span>
                    <span className="config-value">{config.network?.heartbeat_interval_s}s</span>
                  </div>
                  <div className="config-row">
                    <span className="config-key">运行模式</span>
                    <span className="config-value">{config.run_mode}</span>
                  </div>
                </div>
              </div>
              <div className="config-card">
                <div className="config-card-header">🧩 模型分层配置</div>
                <div className="config-rows">
                  <div className="config-row">
                    <span className="config-key">策略</span>
                    <span className="config-value">
                      {config.layers?.strategy === 'graph_orchestrator' ? '🧠 智能编排' :
                       config.layers?.strategy === 'dynamic' ? '🔄 自动分配' : '✏️ 手动覆盖'}
                      <span className="chip-badge" style={{ marginLeft: 8 }}>
                        {config.layers?.total ?? 24} 层
                      </span>
                    </span>
                  </div>
                  {(config.layers?.assignments || layerAssignment?.assignments || []).map((a) => (
                    <div className="config-row" key={a.node_id}>
                      <span className="config-key">
                        {a.role === 'master' ? '🖥️' : '💻'} {a.node_id}
                        {a.score != null && <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 4 }}>({a.score}分)</span>}
                      </span>
                      <span className="config-value mono">
                        Layer {a.start_layer} – {a.end_layer}
                        {a.has_embedding && ' ⬅️Embedding'}
                        {a.has_lm_head && ' ➡️LM Head'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              {/* ---- 流水线请求队列状态 ---- */}
              {status?.pipeline_queue && (
                <div className="config-card">
                  <div className="config-card-header">📋 请求队列</div>
                  <div className="config-rows">
                    <div className="config-row">
                      <span className="config-key">状态</span>
                      <span className="config-value">
                        <span className={`status-dot ${status.pipeline_queue.running ? 'online' : 'offline'}`} />
                        {status.pipeline_queue.running ? '运行中' : '已停止'}
                      </span>
                    </div>
                    <div className="config-row">
                      <span className="config-key">当前任务</span>
                      <span className="config-value mono">
                        {status.pipeline_queue.current_task || '—'}
                      </span>
                    </div>
                    <div className="config-row">
                      <span className="config-key">排队深度</span>
                      <span className="config-value">
                        <span className="chip-badge" style={{
                          background: (status.pipeline_queue.queue_size ?? 0) > 3
                            ? 'var(--warning-bg)' : 'var(--bg-card)'
                        }}>
                          {status.pipeline_queue.queue_size ?? 0}
                        </span>
                      </span>
                    </div>
                    <div className="config-row">
                      <span className="config-key">已完成</span>
                      <span className="config-value">{status.pipeline_queue.completed_count ?? 0}</span>
                    </div>
                  </div>
                </div>
              )}
              <div className="config-card">
                <div className="config-card-header">📈 任务统计</div>
                <div className="config-rows">
                  {config.task_stats && Object.entries(config.task_stats).map(([nid, stats]) => (
                    <div className="config-row" key={nid}>
                      <span className="config-key">{nid}</span>
                      <span className="config-value">✅ {stats.task_count} &nbsp; ❌ {stats.error_count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ---- 推理调度队列 (Phase 3 — 仅主节点) ---- */}
        {isMaster && (
          <section className="admin-section">
            <h3>📋 推理调度</h3>
            <div className="queue-panel">
              {/* Header: strategy + status + controls */}
              <div className="queue-header-row">
                <div className="queue-header-left">
                  <span className="queue-strategy-badge">
                    {(queueDetail?.strategy || 'mlfq').toUpperCase()}
                  </span>
                  <span className={`status-dot ${queueDetail?.running ? 'online' : 'offline'}`} />
                  <span className="queue-status-text">
                    {queueDetail?.paused ? '⏸️ 已暂停' : queueDetail?.running ? '▶️ 运行中' : '⏹️ 已停止'}
                  </span>
                </div>
                <div className="queue-control-row">
                  {queueDetail?.paused ? (
                    <button className="btn btn-sm btn-success" onClick={async () => { await resumeQueue(); refresh(); }}>▶ 恢复</button>
                  ) : (
                    <button className="btn btn-sm btn-warning" onClick={async () => { await pauseQueue(); refresh(); }}>⏸ 暂停</button>
                  )}
                  <button className="btn btn-sm btn-danger-outline" onClick={async () => {
                    if (window.confirm('确认清空所有排队任务？执行中任务不受影响。')) {
                      await clearQueue(); refresh();
                    }
                  }}>🧹 清空</button>
                  <select className="setting-select" style={{width:90,fontSize:12}}
                    value={queueDetail?.strategy || 'mlfq'}
                    onChange={async (e) => {
                    try { await setQueueStrategy(e.target.value); refresh(); }
                    catch (err) { onToast?.({ type: 'error', msg: `队列策略切换失败: ${err.message}` }); }
                  }}>
                    <option value="mlfq">MLFQ</option>
                    <option value="fifo">FIFO</option>
                  </select>
                </div>
              </div>

              {/* Currently running task */}
              {queueDetail?.current_task && (
                <div className="queue-running-card">
                  <span className="queue-running-label">⚡ 执行中</span>
                  <span className="queue-running-id mono">{queueDetail.current_task}</span>
                </div>
              )}

              {/* Q0/Q1/Q2 task lists */}
              {[
                { key: 'q0', label: 'Q0 交互级', cls: 'queue-level-q0', desc: `≤${queueDetail?.aging_params?.q0_max_tokens || 128}tk` },
                { key: 'q1', label: 'Q1 普通级', cls: 'queue-level-q1', desc: `≤${queueDetail?.aging_params?.q1_max_tokens || 512}tk` },
                { key: 'q2', label: 'Q2 批量级', cls: 'queue-level-q2', desc: `>${queueDetail?.aging_params?.q1_max_tokens || 512}tk` },
              ].map(({ key, label, cls, desc }) => {
                const tasks = queueDetail?.[key] || [];
                return (
                  <div className={`queue-level-section ${cls}`} key={key}>
                    <div className="queue-level-header">
                      <span className="queue-level-label">{label}</span>
                      <span className="queue-level-desc">{desc}</span>
                      <span className="queue-level-count">{tasks.length}</span>
                    </div>
                    <div className="queue-task-list">
                      {tasks.length === 0 ? (
                        <div className="queue-empty-state">（空）</div>
                      ) : (
                        tasks.map((t) => (
                          <div className={`queue-task-row ${t.is_aged ? 'aged' : ''}`} key={t.task_id}>
                            <span className="queue-priority-badge">Q{t.priority_level}</span>
                            <span className="queue-task-id mono">{t.task_id.slice(0, 16)}…</span>
                            <span className="queue-task-tokens">{t.max_new_tokens}tk</span>
                            <span className="queue-task-wait">⏳ {t.wait_seconds?.toFixed(0)}s</span>
                            <span className="queue-task-est">~{t.estimated_duration_s?.toFixed(0)}s</span>
                            {t.is_aged && <span className="queue-aging-badge" title="已老化提升">↑</span>}
                            <button className="btn btn-sm btn-danger-outline"
                              style={{padding:'1px 6px',fontSize:11}}
                              onClick={async () => {
                                if (window.confirm(`确认取消任务 ${t.task_id.slice(0, 16)}…？`)) {
                                  await cancelQueueTask(t.task_id); refresh();
                                }
                              }}
                              title="取消任务">×</button>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Footer: stats */}
              <div className="queue-stats-grid">
                <div className="queue-stat-item">
                  <span className="queue-stat-value">{queueDetail?.queue_size ?? 0}</span>
                  <span className="queue-stat-label">排队</span>
                </div>
                <div className="queue-stat-item">
                  <span className="queue-stat-value">{queueDetail?.completed_count ?? 0}</span>
                  <span className="queue-stat-label">完成</span>
                </div>
                <div className="queue-stat-item">
                  <span className="queue-stat-label">老化</span>
                  <span className="queue-stat-value" style={{fontSize:11}}>
                    Q1→Q0 {queueDetail?.aging_params?.q1_to_q0_s ?? 60}s &nbsp;
                    Q2→Q1 {queueDetail?.aging_params?.q2_to_q1_s ?? 120}s
                  </span>
                </div>
                {queueDetail?.preempt_stats && queueDetail.preempt_stats.count > 0 && (
                  <div className="queue-stat-item">
                    <span className="queue-stat-value">{queueDetail.preempt_stats.count}</span>
                    <span className="queue-stat-label">抢占次数</span>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* ---- 手动覆盖分层（仅主节点） ---- */}
        {isMaster && layerAssignment?.assignments && layerAssignment.assignments.length > 0 && (
          <section className="admin-section">
            <h3>✏️ 手动覆盖分层</h3>
            <div className="layer-override-panel">
              <p className="connect-desc">
                修改各节点的层区间将切换为手动覆盖模式。修改后自动推送到所有已连接的从节点。
                区间必须连续且完整覆盖 0-24 层。
              </p>
              {layerAssignment.assignments.map((a) => (
                <div className="layer-override-row" key={a.node_id}>
                  <span className="layer-override-node">{a.role === 'master' ? '🖥️' : '💻'} {a.node_id}</span>
                  <input
                    type="number"
                    className="setting-number-input"
                    value={layerOverrides[a.node_id]?.start ?? a.start_layer}
                    min={0} max={23}
                    onChange={(e) => setLayerOverrides(prev => ({
                      ...prev,
                      [a.node_id]: { ...prev[a.node_id], start: parseInt(e.target.value) || 0 }
                    }))}
                    style={{ width: 60 }}
                  />
                  <span>–</span>
                  <input
                    type="number"
                    className="setting-number-input"
                    value={layerOverrides[a.node_id]?.end ?? a.end_layer}
                    min={1} max={24}
                    onChange={(e) => setLayerOverrides(prev => ({
                      ...prev,
                      [a.node_id]: { ...prev[a.node_id], end: parseInt(e.target.value) || 1 }
                    }))}
                    style={{ width: 60 }}
                  />
                </div>
              ))}
              <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                <button className="btn-primary" onClick={handleApplyLayerOverride} style={{ fontSize: 13 }}>
                  ✅ 应用分层配置
                </button>
                <button className="btn-ghost" onClick={handleResetLayerStrategy} style={{ fontSize: 13 }}>
                  🔄 恢复自动分配
                </button>
              </div>
            </div>
          </section>
        )}

        {/* ---- 流水线拓扑（仅主节点 + 分布式模式） ---- */}
        {isMaster && distributedEnabled && layerAssignment?.assignments && layerAssignment.assignments.length > 0 && (
          <section className="admin-section">
            <h3>🔗 流水线拓扑</h3>
            <PipelineTopology
              assignments={layerAssignment.assignments}
              nodes={nodes?.nodes || []}
              isDistributed={distributedEnabled}
              runMode={status?.run_mode || 'single'}
            />
          </section>
        )}

        {/* ---- 云同步状态 ---- */}
        {distributedEnabled && syncStatus && (
          <section className="admin-section">
            <h3>☁️ 云同步状态</h3>
            <div className="sync-status-panel">
              <div className="sync-status-row">
                <span>本地存储</span>
                <span className="sync-status-ok">✅ 始终可用</span>
              </div>
              <div className="sync-status-row">
                <span>云端同步</span>
                <span className={syncStatus.cloud_sync_enabled ? 'sync-status-ok' : 'sync-status-off'}>
                  {syncStatus.cloud_sync_enabled ? '✅ 已启用' : '⏸️ 已禁用'}
                </span>
              </div>
              <div className="sync-status-row">
                <span>数据库连接</span>
                <span className={syncStatus.db_connected ? 'sync-status-ok' : 'sync-status-error'}>
                  {syncStatus.db_connected ? '✅ 已连接' : '❌ 未连接'}
                </span>
              </div>
            </div>
          </section>
        )}

      </div>
    </div>
  );
}
