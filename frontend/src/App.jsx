import { useState, useCallback, useEffect } from 'react';
import DevicePanel from './components/DevicePanel';
import ModelSelector from './components/ModelSelector';
import MetricsPanel from './components/MetricsPanel';
import ChatPanel from './components/ChatPanel';
import SettingsModal from './components/SettingsModal';

// ---- 设备档位预设 ----
export const TIER_PRESETS = {
  workstation: { maxNewTokens: 2048, temperature: 0.7, topP: 0.9 },
  laptop:      { maxNewTokens: 1024, temperature: 0.7, topP: 0.9 },
  ultrabook:   { maxNewTokens: 512,  temperature: 0.7, topP: 0.9 },
  edge:        { maxNewTokens: 256,  temperature: 0.7, topP: 0.9 },
  mobile:      { maxNewTokens: 128,  temperature: 0.7, topP: 0.9 },
};

export const TIER_LABELS = {
  workstation: '工作站',
  laptop: '笔记本',
  ultrabook: '超极本',
  edge: '边缘设备',
  mobile: '移动端',
};

// 默认设置（无设备档位时使用）
const DEFAULT_SETTINGS = {
  saveHistory: false,
  maxNewTokens: 512,
  temperature: 0.7,
  topP: 0.9,
};

// 从 localStorage 读取主题，默认暗色
function getInitialTheme() {
  try {
    const stored = localStorage.getItem('qlh-theme');
    if (stored === 'light' || stored === 'dark') return stored;
  } catch (_) {}
  // 跟随系统偏好
  if (window.matchMedia?.('(prefers-color-scheme: light)').matches) return 'light';
  return 'dark';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('qlh-theme', theme); } catch (_) {}
}

// 从 localStorage 读取设置
function getInitialSettings() {
  try {
    const stored = localStorage.getItem('qlh-settings');
    if (stored) {
      const parsed = JSON.parse(stored);
      // 合并默认值以兼容新增字段
      return { ...DEFAULT_SETTINGS, ...parsed };
    }
  } catch (_) {}
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(settings) {
  try { localStorage.setItem('qlh-settings', JSON.stringify(settings)); } catch (_) {}
}

export default function App() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [currentQuant, setCurrentQuant] = useState(null);
  const [toast, setToast] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastInferMetrics, setLastInferMetrics] = useState(null);
  const [deviceRefreshKey, setDeviceRefreshKey] = useState(0);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [theme, setTheme] = useState(getInitialTheme);
  const [settings, setSettings] = useState(getInitialSettings);
  const [deviceTier, setDeviceTier] = useState(null);  // 由 DevicePanel 检测后回传

  // 初始化主题
  useEffect(() => { applyTheme(theme); }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      return next;
    });
  }, []);

  // 更新设置（自动持久化到 localStorage）
  const updateSettings = useCallback((partial) => {
    setSettings((prev) => {
      const next = { ...prev, ...partial };
      saveSettings(next);
      return next;
    });
  }, []);

  // 按设备档位应用推荐设置
  const applyTierPreset = useCallback((tier) => {
    const preset = TIER_PRESETS[tier];
    if (preset) {
      updateSettings(preset);
    }
  }, [updateSettings]);

  // Ctrl+B to toggle sidebar
  useEffect(() => {
    const handler = (e) => {
      if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        setSidebarCollapsed((prev) => !prev);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const showToast = useCallback(({ type, msg }) => {
    setToast({ type, msg, id: Date.now() });
    setTimeout(() => setToast(null), 4000);
  }, []);

  const handleModelChange = useCallback((quant, status) => {
    setModelLoaded(true);
    setCurrentQuant(quant);
    setRefreshKey((k) => k + 1);
    setDeviceRefreshKey((k) => k + 1);
    // 从模型加载结果中提取设备档位信息
    if (status?.device_tier) {
      setDeviceTier(status.device_tier);
    }
  }, []);

  const handleInferMetrics = useCallback((m) => {
    setLastInferMetrics(m);
    setRefreshKey((k) => k + 1);
  }, []);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar${sidebarCollapsed ? ' collapsed' : ''}`}>
        <div className="sidebar-header">
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarCollapsed((prev) => !prev)}
            title={sidebarCollapsed ? '展开侧边栏 (Ctrl+B)' : '折叠侧边栏 (Ctrl+B)'}
          >
            {sidebarCollapsed ? '☰' : '◀'}
          </button>
          {!sidebarCollapsed && (
            <>
              <h1>边缘推理优化系统</h1>
              <div className="subtitle">
                北京交通大学 · 大创项目 | Qwen-1.8B-Chat
              </div>
            </>
          )}
          {sidebarCollapsed && (
            <div className="sidebar-collapsed-icon">🧠</div>
          )}
        </div>

        {!sidebarCollapsed ? (
          <>
            <ModelSelector onModelChange={handleModelChange} onToast={showToast} />

            <MetricsPanel
              refreshTrigger={refreshKey}
              lastInferMetrics={lastInferMetrics}
            />

            <div style={{ padding: '16px 24px', marginTop: 'auto' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textAlign: 'center' }}>
                © 2026 杨睿涵 · 张禄政 · 王泽远
              </div>
            </div>
          </>
        ) : (
          /* Collapsed sidebar — icon strip */
          <div className="sidebar-collapsed-icons">
            <button
              className="sidebar-icon-btn"
              onClick={() => setSidebarCollapsed(false)}
              title="模型选择"
            >
              🧩
            </button>
            <button
              className="sidebar-icon-btn"
              onClick={() => setSidebarCollapsed(false)}
              title="系统状态"
            >
              📊
            </button>
            <button
              className="sidebar-icon-btn"
              onClick={() => setSettingsOpen(true)}
              title="系统设置"
            >
              ⚙️
            </button>
          </div>
        )}
      </aside>

      {/* Main Chat Area */}
      <main className="main-area">
        <ChatPanel
          modelLoaded={modelLoaded}
          currentQuant={currentQuant}
          onToast={showToast}
          metricsTrigger={handleInferMetrics}
          onOpenSettings={() => setSettingsOpen(true)}
          settings={settings}
        />
      </main>

      {/* Settings Modal */}
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        deviceRefreshKey={deviceRefreshKey}
        onToast={showToast}
        theme={theme}
        onToggleTheme={toggleTheme}
        settings={settings}
        onSettingsChange={updateSettings}
        deviceTier={deviceTier}
        onApplyTierPreset={applyTierPreset}
      />

      {/* Toast */}
      {toast && (
        <div key={toast.id} className={`toast ${toast.type}`}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}
