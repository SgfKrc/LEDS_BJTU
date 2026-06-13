import { useEffect, useRef, useCallback } from 'react';
import DevicePanel from './DevicePanel';
import { TIER_PRESETS, TIER_LABELS } from '../App';

// Token 限制档位选项
const TOKEN_OPTIONS = [
  { value: 64,   label: '64',   tiers: [] },
  { value: 128,  label: '128',  tiers: ['mobile'] },
  { value: 256,  label: '256',  tiers: ['mobile', 'edge'] },
  { value: 512,  label: '512',  tiers: ['mobile', 'edge', 'ultrabook'] },
  { value: 1024, label: '1024', tiers: ['edge', 'ultrabook', 'laptop'] },
  { value: 2048, label: '2048', tiers: ['ultrabook', 'laptop', 'workstation'] },
  { value: 4096, label: '4096', tiers: ['laptop', 'workstation'] },
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
  settings, onSettingsChange, deviceTier, hasDedicatedGpu,
  onDeviceProfileLoaded, onApplyTierPreset,
  myRole,
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
                  max={8192}
                  step={1}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (v > 0 && v <= 8192) onSettingsChange({ maxNewTokens: v });
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
                onClick={() => onSettingsChange({ distributedInference: !settings.distributedInference })}
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
              <div className="setting-hint" style={{ marginTop: 8 }}>
                {myRole?.is_master
                  ? '✅ 分布式推理已启用。后台管理中将显示集群状态、分层配置与节点管理。'
                  : '✅ 已启用分布式推理优化。后台管理中将显示本节点的运行状态与性能指标。'}
              </div>
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
                当前: {theme === 'dark' ? '🌙 暗色模式' : '☀️ 浅色模式'}
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
                杨睿涵 · 张禄政 · 王泽远 | 指导: 高博
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
