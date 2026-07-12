import { useState, useEffect, useRef, useCallback } from 'react';
import DevicePanel from './DevicePanel';
import { TIER_PRESETS, TIER_LABELS } from '../App';
import { updateDistributedInferenceConfig } from '../api/client';

// Token 限制档位选项
const TOKEN_OPTIONS = [
  { value: 128,  label: '128',  tiers: [] },
  { value: 256,  label: '256',  tiers: ['mobile'] },
  { value: 512,  label: '512',  tiers: ['mobile', 'edge'] },
  { value: 1024, label: '1024', tiers: ['mobile', 'edge', 'ultrabook'] },
  { value: 2048, label: '2048', tiers: ['edge', 'ultrabook', 'laptop'] },
  { value: 4096, label: '4096', tiers: ['ultrabook', 'laptop', 'workstation'] },
  { value: 8192, label: '8192', tiers: ['laptop', 'workstation'] },
];

// Temperature 选项
const TEMP_OPTIONS = [
  { value: 0.1, label: '0.1 — 极精确' },
  { value: 0.3, label: '0.3 — 精确' },
  { value: 0.5, label: '0.5 — 平衡' },
  { value: 0.7, label: '0.7 — 推荐' },
  { value: 0.9, label: '0.9 — 创意' },
  { value: 1.2, label: '1.2 — 随机' },
];

// Top-P 选项
const TOPP_OPTIONS = [
  { value: 0.5,  label: '0.5' },
  { value: 0.7,  label: '0.7' },
  { value: 0.8,  label: '0.8' },
  { value: 0.9,  label: '0.9 — 推荐' },
  { value: 0.95, label: '0.95' },
  { value: 1.0,  label: '1.0' },
];

export default function SettingsModal({
  open, onClose, deviceRefreshKey, onToast, theme, onToggleTheme,
  themeMode = 'system', onThemeModeChange,
  settings, onSettingsChange, deviceTier, hasDedicatedGpu,
  onDeviceProfileLoaded, onApplyTierPreset,
  myRole,
  // P3: 多模型实验支持
  activeModelId, availableModels, switchingModel,
  onLoadModels, onSwitchModel, onRegisterModel, onUnregisterModel,
}) {
  const overlayRef = useRef(null);

  // Close on Escape
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (open) {
      document.addEventListener('keydown', handler);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handler);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  // Click outside to close
  const handleOverlayClick = (e) => {
    if (e.target === overlayRef.current) onClose();
  };

  // 设备档位检测回调：仅记录设备信息（用于显示推荐徽章），
  // 不自动应用档位预设（避免每次打开设置都覆盖用户手动调整的参数）
  const handleDeviceProfile = useCallback((profile) => {
    if (onDeviceProfileLoaded) {
      onDeviceProfileLoaded(profile);
    }
  }, [onDeviceProfileLoaded]);

  // ================================================================
  // 日志管理状态
  // ================================================================
  const [logFiles, setLogFiles] = useState(null);
  const [viewingLog, setViewingLog] = useState(null);
  // L3: 增强日志查看 — 搜索 / level 过滤 / 最近日志 / 统计
  const [logSearch, setLogSearch] = useState('');
  const [logLevelFilter, setLogLevelFilter] = useState('');
  const [recentLogs, setRecentLogs] = useState(null);
  const [recentLogsLoading, setRecentLogsLoading] = useState(false);
  const [recentLogsAutoRefresh, setRecentLogsAutoRefresh] = useState(false);
  const recentLogsTimerRef = useRef(null);
  const [logStats, setLogStats] = useState(null);
  const [logAdminTokenInput, setLogAdminTokenInput] = useState('');
  const [nodesLogSummary, setNodesLogSummary] = useState(null);

  const loadLogFiles = useCallback(async () => {
    try {
      const { fetchLogFiles } = await import('../api/client');
      const data = await fetchLogFiles();
      setLogFiles(data.files || []);
    } catch (_) {
      setLogFiles([]);
    }
  }, []);

  const viewLogContent = useCallback(async (name) => {
    try {
      const { fetchLogContent } = await import('../api/client');
      const data = await fetchLogContent(name);
      setViewingLog({ name: data.name, content: data.content ?? '', truncated: data.truncated || false });
    } catch (e) {
      onToast?.({ type: 'error', msg: `读取日志失败: ${e.message}` });
    }
  }, [onToast]);

  const confirmDeleteLog = useCallback(async (name) => {
    if (!window.confirm(`确定删除日志文件 ${name}？\n如果删除当前日志，系统会立即重新生成新的日志文件。`)) return;
    try {
      const { deleteLogFile } = await import('../api/client');
      await deleteLogFile(name);
      await loadLogFiles();
      setViewingLog(prev => prev?.name === name ? null : prev);
      onToast?.({ type: 'success', msg: `已删除: ${name}；当前日志会自动重新生成` });
    } catch (e) {
      onToast?.({ type: 'error', msg: `删除失败: ${e.message}` });
    }
  }, [loadLogFiles, onToast]);

  const confirmClearAllLogs = useCallback(async () => {
    if (!window.confirm('确定删除所有日志文件？此操作不可撤销。\n如果包含当前日志，系统会立即重新生成新的日志文件。')) return;
    try {
      const { deleteAllLogFiles } = await import('../api/client');
      const result = await deleteAllLogFiles();
      await loadLogFiles();
      setViewingLog(null);
      const failed = result.failed?.length || 0;
      onToast?.({
        type: failed ? 'warning' : 'success',
        msg: failed ? `已清理部分日志，${failed} 个文件删除失败` : '已清理所有日志；当前日志会自动重新生成',
      });
    } catch (e) {
      onToast?.({ type: 'error', msg: `清理失败: ${e.message}` });
    }
  }, [loadLogFiles, onToast]);

  const copyAllLogs = useCallback(async () => {
    try {
      const { fetchLogContent } = await import('../api/client');
      const contents = await Promise.all(
        (logFiles || []).map(f => fetchLogContent(f.name).then(d => d.content))
      );
      const full = (logFiles || []).map((f, i) =>
        `===== ${f.name} =====\n${contents[i]}`
      ).join('\n\n');
      await navigator.clipboard.writeText(full);
      onToast?.({ type: 'success', msg: '已复制全部日志到剪贴板' });
    } catch (e) {
      onToast?.({ type: 'error', msg: `复制失败: ${e.message}` });
    }
  }, [logFiles, onToast]);

  // L3: 最近日志
  const loadRecentLogs = useCallback(async (params = {}) => {
    setRecentLogsLoading(true);
    try {
      const { fetchRecentLogs } = await import('../api/client');
      const data = await fetchRecentLogs({
        limit: params.limit || 200,
        level: params.level || logLevelFilter,
        name: params.name || '',
        node_id: params.node_id || '',
        request_id: params.request_id || '',
      });
      setRecentLogs(data);
    } catch (_) {
      setRecentLogs(null);
    } finally {
      setRecentLogsLoading(false);
    }
  }, [logLevelFilter]);

  // L3: 日志统计
  const loadLogStats = useCallback(async () => {
    try {
      const { fetchLogStats } = await import('../api/client');
      const data = await fetchLogStats();
      setLogStats(data);
    } catch (_) {
      setLogStats(null);
    }
  }, []);

  const loadNodesLogSummary = useCallback(async () => {
    if (!myRole?.is_master) {
      setNodesLogSummary(null);
      return;
    }
    try {
      const { fetchNodesLogSummary } = await import('../api/client');
      const data = await fetchNodesLogSummary();
      setNodesLogSummary(data);
    } catch (_) {
      setNodesLogSummary(null);
    }
  }, [myRole?.is_master]);

  const viewNodeRecentLogs = useCallback(async (nodeId) => {
    try {
      const { fetchNodeRecentLogs } = await import('../api/client');
      const data = await fetchNodeRecentLogs(nodeId, {
        limit: 200,
        level: logLevelFilter,
        timeout: 5,
      });
      setRecentLogs(data);
      onToast?.({ type: 'success', msg: `已加载节点 ${nodeId} 的最近日志` });
    } catch (e) {
      onToast?.({ type: 'error', msg: `加载节点日志失败: ${e.message}` });
    }
  }, [logLevelFilter, onToast]);

  const saveLogAdminToken = useCallback(async () => {
    try {
      const { setLogAdminToken } = await import('../api/client');
      setLogAdminToken(logAdminTokenInput);
      await Promise.all([
        loadLogFiles(),
        loadRecentLogs(),
        loadLogStats(),
        loadNodesLogSummary(),
      ]);
      onToast?.({
        type: 'success',
        msg: logAdminTokenInput.trim() ? '日志访问令牌已保存' : '日志访问令牌已清除',
      });
    } catch (e) {
      onToast?.({ type: 'error', msg: `保存日志令牌失败: ${e.message}` });
    }
  }, [
    logAdminTokenInput,
    loadLogFiles,
    loadRecentLogs,
    loadLogStats,
    loadNodesLogSummary,
    onToast,
  ]);

  // L3: 下载日志文件
  const downloadLog = useCallback(async (filename) => {
    try {
      const { downloadLogFileBlob } = await import('../api/client');
      const { blob } = await downloadLogFileBlob(filename);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) {
      onToast?.({ type: 'error', msg: `下载失败: ${e.message}` });
    }
  }, [onToast]);

  // L3: 最近日志自动刷新
  const toggleRecentLogsAutoRefresh = useCallback(() => {
    setRecentLogsAutoRefresh(prev => !prev);
  }, []);

  useEffect(() => {
    if (recentLogsAutoRefresh && open) {
      loadRecentLogs();
      recentLogsTimerRef.current = setInterval(() => loadRecentLogs(), 3000);
    } else {
      if (recentLogsTimerRef.current) {
        clearInterval(recentLogsTimerRef.current);
        recentLogsTimerRef.current = null;
      }
    }
    return () => {
      if (recentLogsTimerRef.current) {
        clearInterval(recentLogsTimerRef.current);
        recentLogsTimerRef.current = null;
      }
    };
  }, [recentLogsAutoRefresh, open, loadRecentLogs]);

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  // ================================================================
  // P3: 实验模型注册表单
  // ================================================================
  const [showAddModel, setShowAddModel] = useState(false);
  const [newModelForm, setNewModelForm] = useState({
    model_id: '', name: '', model_type: 'safetensors',
    model_path: '', gguf_path: '', recommended_vram_gb: 8.0,
    max_context: 4096, huggingface_id: '', description: '',
  });

  const handleAddModel = useCallback(async () => {
    if (!newModelForm.model_id || !newModelForm.name) {
      onToast?.({ type: 'error', msg: '模型 ID 和名称为必填项' });
      return;
    }
    try {
      await onRegisterModel?.(newModelForm);
      setShowAddModel(false);
      setNewModelForm({
        model_id: '', name: '', model_type: 'safetensors',
        model_path: '', gguf_path: '', recommended_vram_gb: 8.0,
        max_context: 4096, huggingface_id: '', description: '',
      });
    } catch (_) {
      // Parent handler owns the toast; keep the form open so the user can fix it.
    }
  }, [newModelForm, onRegisterModel, onToast]);

  const handleRemoveModel = useCallback(async (modelId) => {
    if (!window.confirm(`确定移除模型 "${modelId}" 的注册？\n不会删除磁盘上的模型文件。`)) return;
    await onUnregisterModel?.(modelId);
  }, [onUnregisterModel]);

  // 模态框打开时自动加载日志列表 + 模型列表 + 最近日志 + 统计
  useEffect(() => {
    if (open) {
      import('../api/client').then(({ getLogAdminToken }) => {
        setLogAdminTokenInput(getLogAdminToken());
      }).catch(() => {
        setLogAdminTokenInput('');
      });
      loadLogFiles();
      loadRecentLogs();
      loadLogStats();
      loadNodesLogSummary();
      onLoadModels?.();
    }
  }, [open, loadLogFiles, loadRecentLogs, loadLogStats, loadNodesLogSummary, onLoadModels]);

  // 判断档位是否匹配当前设备
  const isCurrentTier = (tier) => deviceTier === tier;

  // 获取某个设置项针对当前设备档位的推荐值
  const getTierRecommendation = (settingKey) => {
    if (!deviceTier || !TIER_PRESETS[deviceTier]) return null;
    return TIER_PRESETS[deviceTier][settingKey];
  };

  if (!open) return null;

  return (
    <div className="settings-overlay" ref={overlayRef} onClick={handleOverlayClick}>
      <div className="settings-modal">
        <div className="settings-header">
          <h2>⚙️ 系统设置</h2>
          <button className="settings-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="settings-body">
          <DevicePanel
            key={deviceRefreshKey}
            onToast={onToast}
            onProfileLoaded={handleDeviceProfile}
          />

          {/* ======== 推理参数设置 ======== */}
          <div className="sidebar-section">
            <h3>🎛️ 推理参数</h3>

            {/* ---- Token 输出限制 ---- */}
            <div className="setting-group">
              <div className="setting-label-row">
                <span className="setting-label">最大输出 Token</span>
                {deviceTier && (
                  <span className="setting-device-badge">
                    当前设备: {TIER_LABELS[deviceTier] || deviceTier}
                  </span>
                )}
              </div>
              <div className="setting-chip-row">
                {TOKEN_OPTIONS.map((opt) => {
                  const isTierRec = opt.tiers.includes(deviceTier);
                  const isSelected = settings.maxNewTokens === opt.value;
                  // 找出推荐值：当前设备档位对应的最佳值
                  const tierRecValue = getTierRecommendation('maxNewTokens');
                  const isRecommended = tierRecValue === opt.value && isTierRec;
                  return (
                    <button
                      key={opt.value}
                      className={`setting-chip${isSelected ? ' active' : ''}${isRecommended ? ' recommended' : ''}`}
                      onClick={() => onSettingsChange({ maxNewTokens: opt.value })}
                      title={
                        isRecommended
                          ? `推荐: 适合${TIER_LABELS[deviceTier] || deviceTier}设备`
                          : isTierRec
                            ? `兼容${TIER_LABELS[deviceTier] || deviceTier}设备`
                            : ''
                      }
                    >
                      {opt.label}
                      {isRecommended && <span className="chip-badge">推荐</span>}
                    </button>
                  );
                })}
              </div>
              {/* 自定义输入 */}
              <div className="setting-custom-row">
                <span className="setting-custom-label">自定义:</span>
                <input
                  type="number"
                  className="setting-number-input"
                  value={settings.maxNewTokens}
                  min={1}
                  max={16384}
                  step={1}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (v > 0 && v <= 16384) onSettingsChange({ maxNewTokens: v });
                  }}
                />
                <span className="setting-unit">tokens</span>
              </div>
            </div>

            {/* ---- Temperature ---- */}
            <div className="setting-group">
              <div className="setting-label-row">
                <span className="setting-label">Temperature 温度</span>
                <span className="setting-current-value">{settings.temperature}</span>
              </div>
              <div className="setting-chip-row">
                {TEMP_OPTIONS.map((opt) => {
                  const isSelected = settings.temperature === opt.value;
                  const isRecommended = opt.value === 0.7;
                  return (
                    <button
                      key={opt.value}
                      className={`setting-chip temp-chip${isSelected ? ' active' : ''}${isRecommended ? ' recommended' : ''}`}
                      onClick={() => onSettingsChange({ temperature: opt.value })}
                    >
                      {opt.label}
                      {isRecommended && <span className="chip-badge">推荐</span>}
                    </button>
                  );
                })}
              </div>
              {/* 滑块微调 */}
              <div className="setting-slider-row">
                <input
                  type="range"
                  className="setting-slider"
                  min={0}
                  max={2.0}
                  step={0.05}
                  value={settings.temperature}
                  onChange={(e) => onSettingsChange({ temperature: parseFloat(e.target.value) })}
                />
              </div>
            </div>

            {/* ---- Top-P ---- */}
            <div className="setting-group">
              <div className="setting-label-row">
                <span className="setting-label">Top-P 核采样</span>
                <span className="setting-current-value">{settings.topP}</span>
              </div>
              <div className="setting-chip-row">
                {TOPP_OPTIONS.map((opt) => {
                  const isSelected = settings.topP === opt.value;
                  const isRecommended = opt.value === 0.9;
                  return (
                    <button
                      key={opt.value}
                      className={`setting-chip${isSelected ? ' active' : ''}${isRecommended ? ' recommended' : ''}`}
                      onClick={() => onSettingsChange({ topP: opt.value })}
                    >
                      {opt.label}
                      {isRecommended && <span className="chip-badge">推荐</span>}
                    </button>
                  );
                })}
              </div>
              <div className="setting-slider-row">
                <input
                  type="range"
                  className="setting-slider"
                  min={0}
                  max={1.0}
                  step={0.05}
                  value={settings.topP}
                  onChange={(e) => onSettingsChange({ topP: parseFloat(e.target.value) })}
                />
              </div>
            </div>
          </div>

          {/* ======== 深度思考（仅独显设备可用） ======== */}
          <div className="sidebar-section">
            <h3>🧠 深度思考</h3>
            {hasDedicatedGpu ? (
              <>
                <div className="setting-toggle-row">
                  <div>
                    <div className="setting-label">深度思考展示</div>
                    <div className="setting-desc">
                      启用后，模型在回答前会展示推理过程。你可以在对话中展开「🧠 深度思考」折叠面板查看模型的思考逻辑。
                      注意：思考过程会消耗额外的 Token 配额（约+256 tokens）。
                    </div>
                  </div>
                  <button
                    className={`setting-toggle-btn${settings.showThinking ? ' on' : ''}`}
                    onClick={() => onSettingsChange({ showThinking: !settings.showThinking })}
                    title={settings.showThinking ? '已启用 — 点击关闭' : '已关闭 — 点击启用'}
                  >
                    <span className="setting-toggle-track">
                      <span
                        className="setting-toggle-thumb"
                        style={{
                          transform: settings.showThinking ? 'translateX(22px)' : 'translateX(0)',
                        }}
                      />
                    </span>
                  </button>
                </div>
                {settings.showThinking && (
                  <div className="setting-hint">
                    ✅ 深度思考已启用。模型将在回答前展示推理过程，思考内容仅用于展示，不会进入对话历史。
                  </div>
                )}
              </>
            ) : (
              <div className="setting-disabled-hint">
                <p>
                  ⚠️ 深度思考功能需要<strong>独立显卡</strong>（NVIDIA CUDA 或等效 GPU）。
                </p>
                <p>
                  当前设备为集显或 CPU-only 模式，推理计算资源不足以支持深度思考的额外开销。
                </p>
                <p style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                  💡 如连接了外置独显或 eGPU，请重新检测设备画像后重试。
                </p>
              </div>
            )}
          </div>

          {/* ======== 实验性模型 ======== */}
          {
            <div className="sidebar-section">
              <h3>🧪 模型管理</h3>
              <div className="setting-desc" style={{ marginBottom: 12 }}>
                模型需手动下载，不包含在安装包内。未落盘或当前设备不支持的模型会标注为不可选。
              </div>

              {/* 当前模型 */}
              <div className="setting-group">
                <div className="setting-label">当前模型</div>
                <span className="model-badge-chip">
                  {activeModelId || '未加载'}
                </span>
              </div>

              {/* 模型列表 */}
              {availableModels.length > 0 && (
                <div className="experimental-model-list">
                  {availableModels.map(m => (
                    <div key={m.model_id} className={`experimental-model-card${m.is_available ? '' : ' unavailable'}`}>
                      <div className="model-card-header">
                        <strong>{m.name}</strong>
                        {m.is_experimental && (
                          <span className="chip-badge exp">实验性</span>
                        )}
                        {!m.is_experimental && (
                          <span className="chip-badge default">默认</span>
                        )}
                        {m.is_available ? (
                          <span className="chip-badge available">可用</span>
                        ) : (
                          <span className="chip-badge unavailable">未下载</span>
                        )}
                      </div>
                      {m.description && (
                        <div className="model-card-desc">{m.description}</div>
                      )}
                      <div className="model-card-specs">
                        <span>显存 ≥ {m.recommended_vram_gb} GB</span>
                        <span>上下文 {m.max_context}</span>
                        <span>类型 {m.model_type}</span>
                        {m.available_formats?.length > 0 && (
                          <span>{m.available_formats.join(' + ')}</span>
                        )}
                      </div>
                      {!m.is_available && (
                        <div className="model-card-desc unavailable-reason">
                          {m.unavailable_reason || '模型文件未落盘，暂不可加载。'}
                        </div>
                      )}
                      <div className="model-card-actions">
                        <button
                          className="setting-btn secondary"
                          onClick={() => onSwitchModel?.(m.model_id, m.default_quant_type || 'int4', m.preferred_engine || 'auto')}
                          disabled={switchingModel || activeModelId === m.model_id || !m.is_available}
                        >
                          {activeModelId === m.model_id ? '当前模型' : switchingModel ? '切换中…' : '加载此模型'}
                        </button>
                        {m.is_experimental && !m.is_builtin && (
                          <button
                            className="setting-btn danger-ghost"
                            onClick={() => handleRemoveModel(m.model_id)}
                          >
                            移除
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* 添加实验模型 */}
              {!showAddModel ? (
                <button
                  className="setting-btn secondary"
                  style={{ marginTop: 8, width: '100%' }}
                  onClick={() => setShowAddModel(true)}
                >
                  ＋ 添加实验性模型
                </button>
              ) : (
                <div className="add-model-form">
                  <h4>注册新模型</h4>
                  <div className="form-row">
                    <label>模型 ID <span className="required">*</span></label>
                    <input
                      placeholder="如 qwen2.5-7b"
                      value={newModelForm.model_id}
                      onChange={e => setNewModelForm(f => ({ ...f, model_id: e.target.value }))}
                    />
                  </div>
                  <div className="form-row">
                    <label>显示名称 <span className="required">*</span></label>
                    <input
                      placeholder="如 Qwen2.5-7B-Instruct"
                      value={newModelForm.name}
                      onChange={e => setNewModelForm(f => ({ ...f, name: e.target.value }))}
                    />
                  </div>
                  <div className="form-row">
                    <label>模型类型</label>
                    <select
                      value={newModelForm.model_type}
                      onChange={e => setNewModelForm(f => ({ ...f, model_type: e.target.value }))}
                    >
                      <option value="safetensors">Safetensors (PyTorch)</option>
                      <option value="gguf">GGUF (llama.cpp)</option>
                      <option value="both">双格式</option>
                    </select>
                  </div>
                  {(newModelForm.model_type === 'safetensors' || newModelForm.model_type === 'both') && (
                    <div className="form-row">
                      <label>Safetensors 路径</label>
                      <input
                        placeholder="models/qwen2.5-7b-instruct"
                        value={newModelForm.model_path}
                        onChange={e => setNewModelForm(f => ({ ...f, model_path: e.target.value }))}
                      />
                    </div>
                  )}
                  {(newModelForm.model_type === 'gguf' || newModelForm.model_type === 'both') && (
                    <div className="form-row">
                      <label>GGUF 路径</label>
                      <input
                        placeholder="models/qwen2.5-7b-Q4_K_M.gguf"
                        value={newModelForm.gguf_path}
                        onChange={e => setNewModelForm(f => ({ ...f, gguf_path: e.target.value }))}
                      />
                    </div>
                  )}
                  <div className="form-row">
                    <label>推荐显存 (GB)</label>
                    <input
                      type="number" step="0.5" min="1"
                      value={newModelForm.recommended_vram_gb}
                      onChange={e => setNewModelForm(f => ({ ...f, recommended_vram_gb: parseFloat(e.target.value) || 0 }))}
                    />
                  </div>
                  <div className="form-row">
                    <label>最大上下文</label>
                    <input
                      type="number" step="512" min="512"
                      value={newModelForm.max_context}
                      onChange={e => setNewModelForm(f => ({ ...f, max_context: parseInt(e.target.value, 10) || 4096 }))}
                    />
                  </div>
                  <div className="form-row">
                    <label>HuggingFace ID</label>
                    <input
                      placeholder="Qwen/Qwen2.5-7B-Instruct"
                      value={newModelForm.huggingface_id}
                      onChange={e => setNewModelForm(f => ({ ...f, huggingface_id: e.target.value }))}
                    />
                  </div>
                  <div className="form-row">
                    <label>说明</label>
                    <input
                      placeholder="简短描述"
                      value={newModelForm.description}
                      onChange={e => setNewModelForm(f => ({ ...f, description: e.target.value }))}
                    />
                  </div>
                  <div className="form-actions">
                    <button className="setting-btn primary" onClick={handleAddModel}>
                      注册
                    </button>
                    <button className="setting-btn secondary" onClick={() => setShowAddModel(false)}>
                      取消
                    </button>
                  </div>
                </div>
              )}

              {/* 提示：模型需手动下载 */}
              <p className="setting-desc" style={{ marginTop: 10, fontSize: 11 }}>
                💡 实验性模型文件不会随安装包分发，需从 HuggingFace 手动下载到指定目录。
              </p>
            </div>
          }

          {/* ======== 对话历史 ======== */}
          <div className="sidebar-section">
            <h3>💾 对话记录</h3>
            <div className="setting-toggle-row">
              <div>
                <div className="setting-label">保存对话历史</div>
                <div className="setting-desc">
                  启用后对话内容将保存到浏览器本地存储{myRole?.is_master && '并同步到云端数据库'}。关闭后对话仅保留在内存中，刷新页面后丢失。
                </div>
              </div>
              <button
                className={`setting-toggle-btn${settings.saveHistory ? ' on' : ''}`}
                onClick={() => onSettingsChange({ saveHistory: !settings.saveHistory })}
                title={settings.saveHistory ? '已启用 — 点击关闭' : '已关闭 — 点击启用'}
              >
                <span className="setting-toggle-track">
                  <span
                    className="setting-toggle-thumb"
                    style={{
                      transform: settings.saveHistory ? 'translateX(22px)' : 'translateX(0)',
                    }}
                  />
                </span>
              </button>
            </div>
            {settings.saveHistory && (
              <div className="setting-hint">
                ✅ 对话历史将保存到本地浏览器{myRole?.is_master ? '和云端数据库，' : '。'}清除浏览器数据会导致本地历史丢失。
              </div>
            )}
          </div>

          {/* ======== 云同步设置偏好 ======== */}
          <div className="sidebar-section">
            <h3>☁️ 云同步</h3>
            <div className="setting-toggle-row">
              <div>
                <div className="setting-label">同步设置到云端</div>
                <div className="setting-desc">
                  启用后，推理参数、对话历史开关、分布式推理开关等偏好设置将自动同步到云数据库。
                  关闭后设置仅保存在浏览器本地存储。在新设备或清除缓存后可恢复设置。
                </div>
              </div>
              <button
                className={`setting-toggle-btn${settings.cloudSync ? ' on' : ''}`}
                onClick={() => onSettingsChange({ cloudSync: !settings.cloudSync })}
                title={settings.cloudSync ? '已启用 — 点击关闭' : '已关闭 — 点击启用'}
              >
                <span className="setting-toggle-track">
                  <span
                    className="setting-toggle-thumb"
                    style={{
                      transform: settings.cloudSync ? 'translateX(22px)' : 'translateX(0)',
                    }}
                  />
                </span>
              </button>
            </div>
            {settings.cloudSync && (
              <div className="setting-hint">
                ✅ 设置将自动同步到云数据库。在新设备登录或清除浏览器缓存后可自动恢复偏好。
              </div>
            )}
          </div>

          {/* ======== 流式输出模式 ======== */}
          <div className="sidebar-section">
            <h3>📡 流式输出</h3>
            <div className="setting-toggle-row">
              <div>
                <div className="setting-label">
                  流式输出模式：
                  <strong style={{
                    color: settings.streamingMode === 'fast' ? 'var(--accent)' : 'var(--text-primary)',
                  }}>
                    {settings.streamingMode === 'fast' ? '快速模式（真流式）' : '完整模式（假流式）'}
                  </strong>
                </div>
                <div className="setting-desc">
                  {settings.streamingMode === 'fast'
                    ? '⚡ 真流式：逐 token 实时推送，首 token 延迟 &lt; 200ms，打字机效果。跳过对话历史、追问生成和 DB 持久化，专注低延迟体验。适合快速问答和弱网场景。'
                    : '📋 假流式：先完整生成再一次性返回，保留全部功能——对话历史维护、追问建议、云端持久化。功能与普通对话完全一致，仅以 SSE 格式传输。适合需要完整上下文的场景。'}
                </div>
              </div>
              <button
                className={`setting-toggle-btn${settings.streamingMode === 'fast' ? ' on' : ''}`}
                onClick={() => onSettingsChange({
                  streamingMode: settings.streamingMode === 'fast' ? 'full' : 'fast',
                })}
                title={settings.streamingMode === 'fast' ? '快速模式 — 点击切换完整模式' : '完整模式 — 点击切换快速模式'}
              >
                <span className="setting-toggle-track">
                  <span
                    className="setting-toggle-thumb"
                    style={{
                      transform: settings.streamingMode === 'fast' ? 'translateX(22px)' : 'translateX(0)',
                    }}
                  />
                </span>
              </button>
            </div>
            <div className="setting-hint" style={{ marginTop: 8 }}>
              {settings.streamingMode === 'fast'
                ? '⚡ 快速模式已启用：对话不会保存到历史，刷新页面后丢失。如需完整功能（历史/追问/持久化），请切换到完整模式。'
                : '✅ 完整模式已启用：所有功能正常工作，对话数据完整保存。'}
            </div>
          </div>

          {/* ======== 分布式推理优化（所有节点可见） ======== */}
          <div className="sidebar-section">
            <h3>🌐 分布式推理优化</h3>
            <div className="setting-toggle-row">
              <div>
                <div className="setting-label">启用分布式推理优化</div>
                <div className="setting-desc">
                  {myRole?.is_master
                    ? '主节点：开启后将协调所有从节点进行分布式推理，从节点对话请求通过 TCP 转发至主节点统一调度。关闭后本节点独立运行，不接受从节点连接。'
                    : '从节点：开启后将参与分布式推理集群，对话请求自动转发至主节点执行。关闭后仅使用本地模型推理，不连接主节点。'}
                </div>
              </div>
              <button
                className={`setting-toggle-btn${settings.distributedInference ? ' on' : ''}`}
                onClick={async () => {
                  const next = !settings.distributedInference;
                  onSettingsChange({ distributedInference: next });
                  try {
                    const result = await updateDistributedInferenceConfig(next);
                    if (result.status === 'ok') {
                      onToast?.({ type: 'success', msg: `分布式推理已${result.enabled ? '启用' : '禁用'}` });
                    }
                  } catch (err) {
                    onToast?.({ type: 'error', msg: `后端同步失败: ${err.message}` });
                    // 回滚设置
                    onSettingsChange({ distributedInference: !next });
                  }
                }}
                title={settings.distributedInference ? '已启用 — 点击关闭' : '已关闭 — 点击启用'}
              >
                <span className="setting-toggle-track">
                  <span
                    className="setting-toggle-thumb"
                    style={{
                      transform: settings.distributedInference ? 'translateX(22px)' : 'translateX(0)',
                    }}
                  />
                </span>
              </button>
            </div>
            {settings.distributedInference && (
              <>
                <div className="setting-hint" style={{ marginTop: 8 }}>
                  {myRole?.is_master
                    ? '✅ 分布式推理已启用。后台管理中将显示集群状态、分层配置与节点管理。'
                    : '✅ 已启用分布式推理优化。后台管理中将显示本节点的运行状态与性能指标。'}
                </div>
                {/* ---- 流水线模式指示 ---- */}
                {myRole?.is_master && (
                  <div className="pipeline-mode-indicator" style={{
                    marginTop: 12,
                    padding: '10px 14px',
                    borderRadius: 8,
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                  }}>
                    <div className="setting-label" style={{ marginBottom: 6 }}>
                      🔗 流水线模式（层拆分推理）
                    </div>
                    <div className="setting-desc" style={{ marginBottom: 8 }}>
                      将 Qwen-1.8B 的 24 层 Transformer 按算力比例拆分到主节点和从节点，
                      各节点仅加载分配的层范围，通过 TCP 传递隐藏状态完成协作推理。
                      <strong>仅 PyTorch 引擎可用</strong>（llama.cpp 不支持层间拆分）。
                    </div>
                    <div className="pipeline-mode-status">
                      <span className="pipeline-mode-dot" style={{
                        display: 'inline-block',
                        width: 8, height: 8,
                        borderRadius: '50%',
                        background: 'var(--accent)',
                        marginRight: 6,
                      }} />
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        流水线模式在推理时自动启用 — 当从节点在线且模型为 PyTorch 引擎时自动生效，
                        否则自动回退到主节点全模型推理。
                      </span>
                    </div>
                  </div>
                )}
              </>
            )}
            {myRole?.node_id && (
              <div className="setting-node-id-row">
                <span className="setting-label">节点标识</span>
                <span className="setting-mono-badge">{myRole.node_id}</span>
              </div>
            )}
          </div>

          {/* ======== 设备档位快捷应用 ======== */}
          {deviceTier && (
            <div className="sidebar-section">
              <h3>📱 设备档位预设</h3>
              <p className="setting-desc" style={{ marginBottom: 10 }}>
                当前检测为 <strong>{TIER_LABELS[deviceTier] || deviceTier}</strong>，
                可一键应用该档位的推荐参数，也可手动切换其他档位查看对应配置:
              </p>
              <div className="tier-preset-grid">
                {Object.entries(TIER_PRESETS).map(([tier, preset]) => {
                  const isActive = tier === deviceTier;
                  return (
                    <button
                      key={tier}
                      className={`tier-preset-card${isActive ? ' active' : ''}`}
                      onClick={() => onApplyTierPreset(tier)}
                    >
                      <div className="tier-preset-header">
                        <span className="tier-preset-name">{TIER_LABELS[tier] || tier}</span>
                        {isActive && <span className="chip-badge current">当前</span>}
                      </div>
                      <div className="tier-preset-specs">
                        <span>Token: {preset.maxNewTokens}</span>
                        <span>Temp: {preset.temperature}</span>
                        <span>TopP: {preset.topP}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* ---- 外观 ---- */}
          <div className="sidebar-section">
            <h3>🎨 外观</h3>
            <div className="theme-toggle-row">
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                当前: {themeMode === 'system' ? `跟随系统 · ${theme === 'dark' ? '深色' : '浅色'}` : (theme === 'dark' ? '深色模式' : '浅色模式')}
              </span>
              <button
                className="theme-toggle-btn"
                onClick={onToggleTheme}
                title={theme === 'dark' ? '切换浅色模式' : '切换暗色模式'}
              >
                <span className="theme-toggle-track">
                  <span
                    className="theme-toggle-thumb"
                    style={{ transform: theme === 'dark' ? 'translateX(0)' : 'translateX(22px)' }}
                  />
                </span>
              </button>
            </div>
            <div className="theme-mode-segment" role="group" aria-label="主题模式">
              {[
                ['system', '跟随系统'],
                ['light', '浅色'],
                ['dark', '深色'],
              ].map(([mode, label]) => (
                <button
                  key={mode}
                  className={themeMode === mode ? 'active' : ''}
                  onClick={() => onThemeModeChange?.(mode)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* ---- 日志管理 ---- */}
          <div className="sidebar-section" style={{ borderBottom: 'none' }}>
            <h3>📋 日志管理</h3>

            <div className="log-subsection">
              <h4 style={{ margin: '8px 0 4px' }}>🔑 远程访问</h4>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input
                  type="password"
                  value={logAdminTokenInput}
                  onChange={e => setLogAdminTokenInput(e.target.value)}
                  placeholder="QLH_LOG_ADMIN_TOKEN"
                  autoComplete="off"
                  style={{
                    flex: 1,
                    minWidth: 0,
                    padding: '6px 8px',
                    fontSize: 12,
                    borderRadius: 4,
                    border: '1px solid var(--border)',
                    background: 'var(--bg-input)',
                    color: 'var(--text-primary)',
                  }}
                />
                <button className="sidebar-btn" onClick={saveLogAdminToken} style={{ fontSize: 11 }}>
                  保存
                </button>
              </div>
            </div>

            {/* L3: 日志统计摘要 */}
            {logStats && (
              <div className="log-stats-bar">
                <span title="日志文件数">📁 {logStats.files_count} 个文件</span>
                <span title="文件总大小">{formatFileSize(logStats.files_total_bytes)}</span>
                <span title="内存缓冲 / 容量">🔄 {logStats.buffer_size}/{logStats.buffer_capacity}</span>
                {logStats.node_id && <span title="节点 ID">🖥 {logStats.node_id}</span>}
              </div>
            )}

            {myRole?.is_master && (
              <div className="log-subsection">
                <div className="log-subsection-header">
                  <h4>🖧 节点日志</h4>
                  <button className="sidebar-btn" onClick={loadNodesLogSummary} style={{ fontSize: 11 }}>
                    刷新
                  </button>
                </div>
                {nodesLogSummary?.workers?.length > 0 ? (
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 2 }}>
                    {nodesLogSummary.workers.map(worker => (
                      <div
                        key={worker.node_id}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          gap: 8,
                          padding: '4px 0',
                          borderBottom: '1px solid var(--border)',
                        }}
                      >
                        <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {worker.node_id}
                          <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--text-muted)' }}>
                            buffer {worker.buffer_size || 0}
                          </span>
                          {worker.error && (
                            <span style={{ marginLeft: 8, fontSize: 10, color: '#e74c3c' }}>
                              {worker.error}
                            </span>
                          )}
                        </span>
                        <button
                          className="sidebar-btn"
                          onClick={() => viewNodeRecentLogs(worker.node_id)}
                          disabled={Boolean(worker.error)}
                          style={{ fontSize: 11 }}
                        >
                          查看
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    暂无在线从节点日志摘要
                  </div>
                )}
              </div>
            )}

            {/* L3: 最近日志（内存实时缓冲） */}
            <div className="log-subsection">
              <div className="log-subsection-header">
                <h4>📡 最近日志</h4>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {recentLogsAutoRefresh ? '⏱ 每 3s 刷新' : '已暂停刷新'}
                  </span>
                  <button
                    className={`sidebar-btn${recentLogsAutoRefresh ? ' active' : ''}`}
                    onClick={toggleRecentLogsAutoRefresh}
                    title={recentLogsAutoRefresh ? '暂停自动刷新' : '开启自动刷新 (3s)'}
                    style={{ fontSize: 11 }}
                  >
                    {recentLogsAutoRefresh ? '⏸ 停止' : '▶ 实时'}
                  </button>
                  <button
                    className="sidebar-btn"
                    onClick={() => loadRecentLogs()}
                    disabled={recentLogsLoading}
                    style={{ fontSize: 11 }}
                  >
                    🔄
                  </button>
                </div>
              </div>

              {/* Level 快捷过滤 */}
              <div className="log-level-filters">
                {['', 'ERROR', 'WARNING', 'INFO', 'DEBUG'].map(lv => (
                  <button
                    key={lv || 'ALL'}
                    className={`log-level-chip${(logLevelFilter || '') === lv ? ' active' : ''}${lv ? ` level-${lv.toLowerCase()}` : ''}`}
                    onClick={() => {
                      setLogLevelFilter(lv);
                      loadRecentLogs({ level: lv });
                    }}
                  >
                    {lv || '全部'}
                  </button>
                ))}
              </div>

              {/* 最近日志列表 */}
              {recentLogsLoading && !recentLogs ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 0' }}>
                  ⏳ 加载最近日志...
                </div>
              ) : recentLogs ? (
                <>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>
                    {recentLogs.truncated && '⚠ 结果已截断 · '}
                    匹配 {recentLogs.matched} 条 · 显示最近 {recentLogs.count} 条
                    {recentLogs.filters?.level && ` · 级别: ${recentLogs.filters.level}`}
                  </div>
                  <div className="recent-logs-list">
                    {recentLogs.logs?.length > 0 ? recentLogs.logs.map((entry, i) => (
                      <div
                        key={i}
                        className={`recent-log-entry level-${(entry.level || 'info').toLowerCase()}`}
                        title={`${entry.name || ''}:${entry.lineno || ''} ${entry.funcName || ''}`}
                        onClick={() => {
                          // 点击条目显示详情（复制到查看弹窗）
                          const detail = [
                            `时间: ${entry.timestamp || '-'}`,
                            `级别: ${entry.level || '-'}`,
                            `模块: ${entry.name || '-'}`,
                            `节点: ${entry.node_id || '-'}`,
                            `位置: ${entry.filename || '-'}:${entry.lineno || '-'} ${entry.funcName || ''}`,
                            `消息: ${entry.message || '-'}`,
                          ].join('\n');
                          setLogSearch('');
                          setViewingLog({ name: `recent #${i + 1}`, content: detail, truncated: false });
                        }}
                      >
                        <span className={`recent-log-level ${(entry.level || 'INFO').toLowerCase()}`}>
                          {entry.level || 'INFO'}
                        </span>
                        <span className="recent-log-ts">
                          {entry.timestamp ? entry.timestamp.slice(-8) : ''}
                        </span>
                        <span className="recent-log-msg">
                          {entry.message || ''}
                        </span>
                      </div>
                    )) : (
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: 4 }}>
                        暂无匹配的日志条目
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '4px 0' }}>
                  <button className="sidebar-btn" onClick={() => loadRecentLogs()}>
                    加载最近日志
                  </button>
                </div>
              )}
            </div>

            {/* 日志文件列表 */}
            <div className="log-subsection">
              <h4 style={{ margin: '8px 0 4px' }}>📁 日志文件</h4>

              {logFiles === null ? (
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  <button className="sidebar-btn" onClick={loadLogFiles}>加载日志列表</button>
                </div>
              ) : logFiles.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>暂无日志文件</div>
              ) : (
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 2 }}>
                  {logFiles.map(f => (
                    <div key={f.name} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '4px 0', borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{
                        flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {f.name}
                        <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--text-muted)' }}>
                          {formatFileSize(f.size)}
                        </span>
                      </span>
                      <span style={{ display: 'flex', gap: 4 }}>
                        <button className="sidebar-btn" onClick={() => viewLogContent(f.name)} title="查看">👁</button>
                        <button className="sidebar-btn" onClick={() => downloadLog(f.name)} title="下载">⬇</button>
                        <button className="sidebar-btn" onClick={() => confirmDeleteLog(f.name)} title="删除">🗑</button>
                      </span>
                    </div>
                  ))}
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <button className="sidebar-btn" onClick={copyAllLogs} style={{ flex: 1 }}>📋 复制全部</button>
                    <button className="sidebar-btn" onClick={confirmClearAllLogs}
                            style={{ flex: 1, color: '#e74c3c' }}>🗑 清理所有</button>
                  </div>
                </div>
              )}
            </div>

            {/* 日志内容查看弹窗（L3增强：搜索 + truncated 标记） */}
            {viewingLog !== null && (
              <div className="log-viewer-overlay" onClick={() => { setViewingLog(null); setLogSearch(''); }}
                   style={{
                     position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                     background: 'rgba(0,0,0,0.5)', zIndex: 9999,
                     display: 'flex', justifyContent: 'center', alignItems: 'center',
                   }}>
                <div onClick={e => e.stopPropagation()} style={{
                  background: 'var(--bg-secondary)', borderRadius: 8, maxWidth: '85vw', width: '85vw',
                  maxHeight: '85vh', display: 'flex', flexDirection: 'column',
                }}>
                  <div style={{
                    padding: '12px 16px', borderBottom: '1px solid var(--border)',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12,
                  }}>
                    <strong style={{ color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>
                      {viewingLog.name}
                    </strong>
                    <div style={{ display: 'flex', gap: 8, flex: 1, justifyContent: 'flex-end', alignItems: 'center' }}>
                      {/* L3: 日志内容搜索 */}
                      <input
                        className="log-search-input"
                        type="text"
                        placeholder="🔍 搜索日志内容..."
                        value={logSearch}
                        onChange={e => setLogSearch(e.target.value)}
                        onClick={e => e.stopPropagation()}
                        style={{
                          padding: '4px 8px', fontSize: 11, borderRadius: 4,
                          border: '1px solid var(--border)', background: 'var(--bg-input)',
                          color: 'var(--text-primary)', width: 180,
                        }}
                      />
                      <button onClick={() => { setViewingLog(null); setLogSearch(''); }}
                              style={{ cursor: 'pointer', background: 'none', border: 'none',
                                       color: 'var(--text-muted)', fontSize: 16 }}>✕</button>
                    </div>
                  </div>
                  {/* L3: truncated 标记 */}
                  {viewingLog.truncated && (
                    <div style={{
                      padding: '6px 16px', fontSize: 11,
                      background: '#fff3cd', color: '#856404',
                      borderBottom: '1px solid var(--border)',
                    }}>
                      ⚠️ 日志文件过大（&gt; 1 MB），仅显示末尾 1 MB 内容。请下载完整文件查看全部日志。
                    </div>
                  )}
                  {logSearch && (
                    <div style={{
                      padding: '4px 16px', fontSize: 10, color: 'var(--text-muted)',
                      borderBottom: '1px solid var(--border)',
                    }}>
                      搜索: "{logSearch}" — {
                        viewingLog.content.split('\n').filter(l => l.toLowerCase().includes(logSearch.toLowerCase())).length
                      } 行匹配
                    </div>
                  )}
                  <pre style={{
                    margin: 0, padding: 16, overflow: 'auto', flex: 1,
                    fontSize: 11, lineHeight: 1.5, fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                    color: 'var(--text-primary)',
                  }}>{
                    logSearch
                      ? viewingLog.content.split('\n').filter(l =>
                          l.toLowerCase().includes(logSearch.toLowerCase())
                        ).join('\n')
                      : viewingLog.content
                  }</pre>
                </div>
              </div>
            )}
          </div>

          <div className="sidebar-section" style={{ borderBottom: 'none' }}>
            <h3>📖 快捷键</h3>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 2.2 }}>
              <div><kbd style={kbdStyle}>Enter</kbd> 发送消息</div>
              <div><kbd style={kbdStyle}>Shift+Enter</kbd> 换行</div>
              <div><kbd style={kbdStyle}>Ctrl+B</kbd> 折叠侧边栏</div>
              <div><kbd style={kbdStyle}>Esc</kbd> 关闭设置</div>
            </div>
          </div>

          <div className="sidebar-section" style={{ borderBottom: 'none' }}>
            <h3>💡 关于</h3>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.8 }}>
              轻量化大模型分布式边缘推理优化系统<br />
              北京交通大学 · 大学生创新创业训练计划<br />
              <span style={{ color: 'var(--text-secondary)' }}>
                项目团队 | 指导: 高博
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const kbdStyle = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  borderRadius: 3,
  padding: '1px 6px',
  fontSize: 10,
  fontFamily: 'monospace',
  marginRight: 4,
};
