import { useState, useEffect, useCallback } from 'react';
import { fetchDeviceProfile, autoConfigure, selectGpu } from '../api/client';

const TIER_COLORS = {
  workstation: '#f0a500',
  laptop: '#10b981',
  ultrabook: '#3b82f6',
  edge: '#f97316',
  mobile: '#ef4444',
};

const TIER_ICONS = {
  workstation: '🖥️',
  laptop: '💻',
  ultrabook: '📔',
  edge: '📟',
  mobile: '📱',
};

export default function DevicePanel({ onToast, onTierDetected }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [applying, setApplying] = useState(false);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchDeviceProfile()
      .then((data) => {
        setProfile(data);
        // 回传设备档位给父组件
        if (data?.tier && onTierDetected) {
          onTierDetected(data.tier);
        }
      })
      .catch(() => onToast?.({ type: 'error', msg: '设备检测失败' }))
      .finally(() => setLoading(false));
  }, [onToast, onTierDetected]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleAutoConfig = async () => {
    setApplying(true);
    try {
      const result = await autoConfigure();
      onToast?.({ type: 'success', msg: `已应用 ${result.tier_label || result.tier} 档配置` });
      refresh();
    } catch (err) {
      onToast?.({ type: 'error', msg: `配置失败: ${err.message}` });
    } finally {
      setApplying(false);
    }
  };

  const handleSelectGpu = async (gpuIndex) => {
    try {
      const result = await selectGpu(gpuIndex);
      const gpuLabel = result.selected_gpu?.gpu_type === 'integrated' ? '集显' : '独显';
      onToast?.({
        type: 'success',
        msg: `已切换至${gpuLabel}: ${result.selected_gpu?.name}${result.warning ? '（需重新加载模型）' : ''}`,
      });
      refresh();
    } catch (err) {
      onToast?.({ type: 'error', msg: `GPU 切换失败: ${err.message}` });
    }
  };

  if (loading) {
    return (
      <div className="sidebar-section">
        <h3>📊 设备检测</h3>
        <div className="device-skeleton">
          <div className="skeleton-bar" style={{ width: '60%' }} />
          <div className="skeleton-bar" style={{ width: '80%' }} />
          <div className="skeleton-bar" style={{ width: '40%' }} />
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="sidebar-section">
        <h3>📊 设备检测</h3>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          无法获取设备信息
          <button className="btn-ghost" onClick={refresh} style={{ marginLeft: 8, fontSize: 11 }}>
            重试
          </button>
        </div>
      </div>
    );
  }

  const { tier, tier_label, tier_icon, score_total, score_breakdown, cpu, ram, gpu,
          gpus, selected_gpu_index, recommendations, warnings } = profile;
  const hasMultipleGpus = gpus && gpus.length >= 2;
  const tierColor = TIER_COLORS[tier] || '#888';

  const renderScoreBar = (label, score, max) => (
    <div className="device-score-row">
      <span className="score-label">{label}</span>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${(score / max) * 100}%`, background: tierColor }}
        />
      </div>
      <span className="score-value">{score}</span>
    </div>
  );

  return (
    <div className="sidebar-section">
      <div className="device-header">
        <h3>
          {tier_icon || '📊'} 设备画像
        </h3>
        <button className="btn-ghost" onClick={refresh} title="重新检测" style={{ fontSize: 14 }}>
          🔄
        </button>
      </div>

      {/* Tier Badge */}
      <div className="device-tier-badge" style={{ background: tierColor }}>
        <span>{tier_icon || '🖥️'}</span>
        <span>{tier_label || tier}</span>
        <span className="tier-score">{score_total || 0}/100</span>
      </div>

      {/* Score Breakdown */}
      <div className="device-scores">
        {renderScoreBar('GPU', score_breakdown?.gpu || 0, 50)}
        {renderScoreBar('RAM', score_breakdown?.ram || 0, 30)}
        {renderScoreBar('CPU', score_breakdown?.cpu || 0, 20)}
      </div>

      {/* Core Metrics */}
      <div className="device-metrics-grid">
        {cpu && (
          <div className="device-metric">
            <div className="metric-icon">🖥</div>
            <div className="metric-info">
              <div className="metric-value">{cpu.physical_cores}核 @ {cpu.freq_max_mhz?.toFixed(0) || '?'}MHz</div>
              <div className="metric-label">{cpu.model_name?.slice(0, 24) || 'CPU'}</div>
            </div>
          </div>
        )}
        {ram && (
          <div className="device-metric">
            <div className="metric-icon">🧠</div>
            <div className="metric-info">
              <div className="metric-value">{ram.total_gb} GB</div>
              <div className="metric-label">可用 {ram.available_gb} GB</div>
            </div>
          </div>
        )}
        {gpu && (
          <div className="device-metric">
            <div className="metric-icon">🎮</div>
            <div className="metric-info">
              <div className="metric-value">{gpu.vram_total_gb > 0 ? `${gpu.vram_total_gb} GB` : '共享'}</div>
              <div className="metric-label">{gpu.name?.slice(0, 20) || 'GPU'}
                {gpu.is_integrated && ' (集显)'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ---- GPU 选择器（多 GPU 时显示） ---- */}
      {hasMultipleGpus && (
        <div className="gpu-selector" style={{ marginTop: 10 }}>
          <div className="gpu-selector-label">
            🎮 推理 GPU
            {selected_gpu_index !== undefined && gpus && gpus[selected_gpu_index]?.is_integrated && (
              <span style={{ fontSize: 10, color: 'var(--warning)', marginLeft: 6 }}>
                ⚡ 低功耗模式
              </span>
            )}
            {selected_gpu_index !== undefined && gpus && !gpus[selected_gpu_index]?.is_integrated && (
              <span style={{ fontSize: 10, color: 'var(--danger)', marginLeft: 6 }}>
                🔥 高性能模式
              </span>
            )}
          </div>
          <div className="gpu-toggle-group">
            {gpus.map((g, i) => {
              const isActive = i === selected_gpu_index;
              const icon = g.gpu_type === 'integrated'
                ? (g.mps_available ? '🍎' : '💧')
                : '🔥';
              const label = g.gpu_type === 'integrated'
                ? (g.mps_available ? 'Apple MPS' : '集显 (CPU)')
                : g.cuda_available
                  ? '独显 (CUDA)'
                  : '独显';
              const shortName = g.name?.length > 20
                ? g.name.slice(0, 18) + '…'
                : g.name;
              return (
                <button
                  key={i}
                  className={`gpu-toggle-btn ${isActive ? 'active' : ''}`}
                  onClick={() => handleSelectGpu(i)}
                  disabled={isActive}
                  title={g.name}
                >
                  <span className="gpu-toggle-icon">{icon}</span>
                  <span className="gpu-toggle-name">{shortName}</span>
                  <span className="gpu-toggle-type">{label}</span>
                  {g.vram_total_gb > 0 && (
                    <span className="gpu-toggle-vram">{g.vram_total_gb} GB</span>
                  )}
                  {isActive && <span className="gpu-toggle-check">✓</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings?.length > 0 && (
        <div className="device-warnings">
          {warnings.slice(0, 2).map((w, i) => (
            <div key={i} className="device-warning">{w}</div>
          ))}
        </div>
      )}

      {/* Recommendations */}
      {recommendations?.length > 0 && (
        <div className="device-recommendations">
          {recommendations.slice(0, 3).map((r, i) => (
            <div key={i} className="device-recommendation">{r}</div>
          ))}
        </div>
      )}

      {/* Auto-Configure Button */}
      <button
        className="btn-primary"
        onClick={handleAutoConfig}
        disabled={applying}
        style={{ width: '100%', marginTop: 8, fontSize: 12 }}
      >
        {applying ? '应用配置中...' : '🎯 应用推荐配置'}
      </button>

      {/* Android Note */}
      {tier === 'mobile' && (
        <div className="android-note">
          📱 当前 PyTorch 栈无法直接在 Android 运行。<br />
          建议导出模型为 ONNX/GGUF 格式，搭配 ONNX Runtime Mobile 或 llama.cpp。
        </div>
      )}
    </div>
  );
}
