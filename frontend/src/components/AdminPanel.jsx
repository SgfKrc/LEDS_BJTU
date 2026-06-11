import { useState, useEffect, useCallback } from 'react';
import {
  fetchClusterStatus, fetchClusterNodes,
  fetchClusterConfig, deregisterNode, updateMaxNodes,
  fetchInviteInfo, connectToMaster,
  discoverMaster, resetMasterIdentity,
  manualRegisterNode, fetchMasterHealth,
  testEmailNotification,
  fetchDistributedInferenceConfig, updateDistributedInferenceConfig,
  fetchLayerAssignment, updateLayerAssignment,
  fetchConversationSyncStatus,
  transferMasterRole, fetchTransferLogs,
  fetchSpareMaster, designateSpareMaster,
  removeSpareMaster, fetchSpareMasterLogs,
} from '../api/client';

const ROLE_LABELS = { master: '主节点', client: '从节点' };
const ROLE_ICONS = { master: '🖥️', client: '💻' };

const STATE_LABELS = { online: '在线', busy: '忙碌', offline: '离线', error: '异常' };
const STATE_COLORS = {
  online: 'var(--success)', busy: 'var(--warning)',
  offline: 'var(--text-muted)', error: 'var(--danger)',
};

const NETWORK_LABELS = {
  wifi: '📶 WiFi', ethernet: '🔌 以太网',
  localhost: '🏠 本地', unknown: '❓ 未知',
};
const NETWORK_CLASSES = {
  wifi: 'net-wifi', ethernet: 'net-eth',
  localhost: 'net-local', unknown: 'net-unknown',
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

export default function AdminPanel({ onToast, myRole }) {
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
  const [registering, setRegistering] = useState(false);

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

  const isMaster = myRole?.is_master ?? true;

  // 拉取全部数据
  const refresh = useCallback(() => {
    Promise.all([
      fetchClusterStatus().catch(() => null),
      fetchClusterNodes().catch(() => null),
      fetchClusterConfig().catch(() => null),
      isMaster ? fetchInviteInfo().catch(() => null) : Promise.resolve(null),
    ]).then(([s, n, c, inv]) => {
      setStatus(s);
      setNodes(n);
      setConfig(c);
      setInvite(inv);
      if (c?.max_nodes && !maxNodesInput) {
        setMaxNodesInput(String(c.max_nodes));
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
      `  2. 保存降级日志到数据库\n` +
      `  3. 更新数据库中的主节点信息\n\n` +
      `转让后建议双方重启服务以应用新角色。\n` +
      `本节点重启后将以从节点模式运行。`
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
        .catch(() => {});
    }
  }, [isMaster]);

  // ---- 备用主节点 ----
  const loadSpareMasterData = useCallback(() => {
    if (isMaster) {
      fetchSpareMaster()
        .then(data => {
          setSpareMaster(data.spare_master || null);
          // 自动同步转让目标为备用主节点
          if (data.spare_master?.node_id) {
            setTransferTarget(data.spare_master.node_id);
          }
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
    // 删除手动覆盖 → 策略回到 dynamic
    try {
      const result = await updateDistributedInferenceConfig(distributedEnabled);  // 触发 sync
      // 删除 layer_override + 恢复 dynamic 策略需要后端接口
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
      );
      if (result.status === 'registered') {
        onToast?.({ type: 'success', msg: result.message || `节点 '${result.node_id}' 已注册` });
        setManualNodeId('');
        setManualHostname('');
        setManualAddress('');
        refresh();
      } else if (result.status === 'exists') {
        onToast?.({ type: 'warning', msg: result.message || `节点 '${result.node_id}' 已存在` });
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
  const filteredNodeList = isMaster
    ? nodeList
    : nodeList.filter(n => n.node_id === myRole?.node_id);
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
          <div className="admin-table-wrap">
            <table className="admin-node-table">
              <thead>
                <tr>
                  <th>节点</th>
                  <th>角色</th>
                  <th>状态</th>
                  <th>网络</th>
                  <th>地址</th>
                  <th>主机名</th>
                  <th>心跳</th>
                  <th>在线时长</th>
                  <th>任务</th>
                  <th>错误</th>
                  {isMaster && <th>操作</th>}
                </tr>
              </thead>
              <tbody>
                {filteredNodeList.length === 0 ? (
                  <tr>
                    <td colSpan={isMaster ? 11 : 10} style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
                      {isMaster ? '暂无已注册的从节点，使用上方「注册新节点」添加' : '尚未连接到主节点，请在上方输入主节点地址'}
                    </td>
                  </tr>
                ) : (
                  filteredNodeList.map((node) => {
                    const isOffline = node.state === 'offline';
                    const isMasterNode = node.role === 'master';
                    const netLabel = NETWORK_LABELS[node.network_type] || NETWORK_LABELS.unknown;
                    const netClass = NETWORK_CLASSES[node.network_type] || NETWORK_CLASSES.unknown;
                    return (
                      <tr key={node.node_id} className={isOffline ? 'row-offline' : ''}>
                        <td>
                          <span className="node-id-cell">
                            {ROLE_ICONS[node.role] || '❓'} {node.node_id}
                          </span>
                        </td>
                        <td>
                          <span className="role-badge" data-role={node.role === 'master' ? 'master' : 'client'}>
                            {ROLE_LABELS[node.role] || node.role}
                          </span>
                        </td>
                        <td>
                          <span className="status-dot" style={{ '--dot-color': STATE_COLORS[node.state] || 'var(--text-muted)' }}>
                            {STATE_LABELS[node.state] || node.state}
                          </span>
                        </td>
                        <td><span className={`network-badge ${netClass}`}>{netLabel}</span></td>
                        <td className="mono-cell">{node.address || '—'}</td>
                        <td>{node.hostname || '—'}</td>
                        <td>{formatTime(node.last_heartbeat)}</td>
                        <td>{isOffline ? '—' : formatUptime(node.connected_at)}</td>
                        <td>{node.task_count}</td>
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
                指定一个从节点作为备用主节点。备用主节点是身份转让的<strong>指定接班人</strong>。
                <br />• 只有指定了备用主节点后，才能转让主节点身份
                <br />• 必须有至少<strong>3 个在线节点</strong>（主 + 备用 + 至少 1 个其他从节点）才能转让
                <br />• 指定操作通过 TCP 通知从节点，并记录操作日志
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
                将主节点身份转让给<strong>备用主节点</strong>。转让前须先指定备用主节点。
                <br />• 目标从节点将收到 TCP 通知并保存升级日志
                <br />• 本节点保存降级日志并更新数据库中的主节点信息
                <br />• <strong>建议双方重启服务</strong>以应用新角色（新主节点重启后以主节点模式运行，原主节点以从节点模式运行）
              </p>
              {/* 转让前置条件检查 */}
              {!spareMaster ? (
                <div style={{
                  background: 'var(--warning-bg)', border: '1px solid var(--warning)',
                  borderRadius: 8, padding: '10px 14px', marginBottom: 12,
                }}>
                  <span style={{ color: 'var(--warning)' }}>
                    ⚠️ 未指定备用主节点，无法转让。请先在上方「🛡️ 备用主节点」中指定。
                  </span>
                </div>
              ) : (
                <div className="transfer-form">
                  <div className="connect-field" style={{ flex: 1 }}>
                    <label>目标从节点（固定为备用主节点）</label>
                    <select
                      className="connect-input"
                      value={transferTarget}
                      onChange={(e) => setTransferTarget(e.target.value)}
                      style={{ width: '100%' }}
                      disabled
                    >
                      <option value={spareMaster.node_id}>
                        🛡️ {spareMaster.node_id} {spareMaster.hostname ? `(${spareMaster.hostname})` : ''} — 备用主节点
                      </option>
                    </select>
                  </div>
                  <button
                    className="btn-primary"
                    onClick={handleTransferMaster}
                    disabled={transferring || !transferTarget || !spareMaster?.is_online}
                    style={{ fontSize: 13, alignSelf: 'flex-end' }}
                    title={
                      !spareMaster?.is_online ? '备用主节点离线，无法转让' :
                      '将主节点身份转让给备用主节点'
                    }
                  >
                    {transferring ? '⏳ 转让中...' : '🔄 转让身份'}
                  </button>
                </div>
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
                      {config.layers?.strategy === 'dynamic' ? '🔄 自动分配' : '✏️ 手动覆盖'}
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
