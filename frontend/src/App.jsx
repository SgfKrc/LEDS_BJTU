import { useState, useCallback, useEffect } from 'react';
import DevicePanel from './components/DevicePanel';
import ModelSelector from './components/ModelSelector';
import MetricsPanel from './components/MetricsPanel';
import ChatPanel from './components/ChatPanel';
import AdminPanel from './components/AdminPanel';
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
  distributedInference: true,  // 分布式推理：主节点默认开启，从节点从服务端同步
  cloudSync: false,            // 云同步设置偏好：默认关闭，需手动开启
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
  const [activeView, setActiveView] = useState('chat'); // 'chat' | 'admin'
  const [myRole, setMyRole] = useState(null);  // { node_role, node_id, is_master, is_client, ... }

  // 初始化主题
  useEffect(() => { applyTheme(theme); }, [theme]);

  // 获取当前节点角色 + 同步分布式推理开关 + 云端设置恢复
  useEffect(() => {
    import('./api/client').then(({ fetchMyRole, fetchDistributedInferenceConfig, fetchUserSettings }) => {
      // 获取角色
      fetchMyRole()
        .then(setMyRole)
        .catch(() => setMyRole({ node_role: 'master', node_id: 'master', is_master: true, is_client: false }));
      // 从服务端同步分布式推理开关状态
      fetchDistributedInferenceConfig()
        .then((config) => {
          if (config && typeof config.enabled === 'boolean') {
            updateSettings({ distributedInference: config.enabled });
          }
        })
        .catch(() => {});  // 服务端不可用时保持本地设置
      // 从云端恢复用户偏好设置（仅在用户已开启云同步时）
      const localSettings = getInitialSettings();
      if (localSettings.cloudSync) {
        fetchUserSettings()
          .then((res) => {
            if (res && res.settings && Object.keys(res.settings).length > 0) {
              setSettings((prev) => {
                // localStorage 优先（最新用户意图），云端补充缺失字段
                const merged = { ...res.settings, ...prev };
                saveSettings(merged);
                return merged;
              });
            }
          })
          .catch(() => {});  // 服务端不可用时保持本地设置
      }
    });
  }, []);

  // 后台管理 Tab 是否可见
  const showAdminTab = !myRole                     // 加载中：显示（兜底）
    || myRole.is_master                           // 主节点：始终显示
    || (myRole.is_client && settings.distributedInference);  // 从节点：需开启分布式推理

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      return next;
    });
  }, []);

  // 更新设置（自动持久化到 localStorage，云同步需手动开启）
  const updateSettings = useCallback((partial) => {
    setSettings((prev) => {
      const next = { ...prev, ...partial };
      saveSettings(next);
      // 仅当用户手动开启「云同步设置」时才推送到云数据库
      if (next.cloudSync) {
        import('./api/client').then(({ updateUserSettings }) => {
          updateUserSettings(next).catch(() => {});
        });
      }
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

              {/* 导航切换 */}
              <div className="nav-tabs">
                <button
                  className={`nav-tab ${activeView === 'chat' ? 'active' : ''}`}
                  onClick={() => setActiveView('chat')}
                >
                  💬 对话
                </button>
                {showAdminTab && (
                  <button
                    className={`nav-tab ${activeView === 'admin' ? 'active' : ''}`}
                    onClick={() => setActiveView('admin')}
                  >
                    ⚙️ 后台管理
                  </button>
                )}
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
              onClick={() => { setSidebarCollapsed(false); setActiveView('chat'); }}
              title="对话"
            >
              💬
            </button>
            {showAdminTab && (
              <button
                className="sidebar-icon-btn"
                onClick={() => { setSidebarCollapsed(false); setActiveView('admin'); }}
                title="后台管理"
              >
                ⚙️
              </button>
            )}
            <button
              className="sidebar-icon-btn"
              onClick={() => setSettingsOpen(true)}
              title="系统设置"
            >
              🔧
            </button>
          </div>
        )}
      </aside>

      {/* Main Area */}
      <main className="main-area">
        {activeView === 'chat' || !showAdminTab ? (
          <ChatPanel
            modelLoaded={modelLoaded}
            currentQuant={currentQuant}
            onToast={showToast}
            metricsTrigger={handleInferMetrics}
            onOpenSettings={() => setSettingsOpen(true)}
            settings={settings}
          />
        ) : (
          <AdminPanel onToast={showToast} myRole={myRole} />
        )}
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
        myRole={myRole}
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
