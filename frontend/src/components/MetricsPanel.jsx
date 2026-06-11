import { useState, useEffect, useCallback } from 'react';
import { fetchStatus } from '../api/client';

export default function MetricsPanel({ refreshTrigger }) {
  const [status, setStatus] = useState(null);

  const refresh = useCallback(() => {
    fetchStatus()
      .then(setStatus)
      .catch(() => {});
  }, []);

  useEffect(() => { refresh(); }, [refresh, refreshTrigger]);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  if (!status) return null;

  const gpu = status.gpu || {};
  const kv = status.kv_cache || {};

  return (
    <>
      <div className="sidebar-section">
        <h3>📊 系统状态</h3>
        <div className="status-grid">
          <div className="status-item">
            <div className="label">模型</div>
            <div className={`value ${status.model_loaded ? 'green' : 'red'}`}>
              {status.model_loaded ? '就绪' : '未加载'}
            </div>
          </div>
          <div className="status-item">
            <div className="label">量化</div>
            <div className="value blue">
              {status.current_quant ? status.current_quant.toUpperCase() : '—'}
            </div>
          </div>
          <div className="status-item">
            <div className="label">GPU 显存</div>
            <div className={`value ${(gpu.utilization || 0) > 90 ? 'red' : 'yellow'}`}>
              {gpu.allocated_mb || 0} MB
            </div>
          </div>
          <div className="status-item">
            <div className="label">GPU 利用率</div>
            <div className="value blue">
              {gpu.utilization || 0}%
            </div>
          </div>
          <div className="status-item">
            <div className="label">KV 缓存</div>
            <div className="value blue">
              {kv.total_tokens || 0}
            </div>
          </div>
          <div className="status-item">
            <div className="label">对话轮次</div>
            <div className="value blue">
              {(status.conversation_turns || 0) / 2}
            </div>
          </div>
        </div>
      </div>

      <div className="sidebar-section">
        <h3>💻 GPU 信息</h3>
        {status.device && (
          <div className="device-tier-mini" style={{ marginBottom: 8 }}>
            <span>{status.device.tier_icon || '🖥️'}</span>
            <span>{status.device.tier_label || status.device.tier}</span>
            <span className="tier-score-sm">{status.device.score || '?'}/100</span>
          </div>
        )}
        {gpu.name ? (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            <div>设备: {gpu.name}</div>
            <div>总显存: {gpu.total_mb} MB</div>
            <div>已分配: {gpu.allocated_mb} MB</div>
            <div>已保留: {gpu.reserved_mb} MB</div>
            {status.load_time_seconds && (
              <div>加载耗时: {status.load_time_seconds}s</div>
            )}
          </div>
        ) : (
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>未检测到 GPU</div>
        )}
      </div>

      {status.model_loaded && (
        <div className="sidebar-section">
          <h3>📈 KV 缓存详情</h3>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            <div>累计 Token: {kv.total_tokens || 0} / {kv.max_tokens || '—'}</div>
            <div>已分配页: {kv.allocated_pages || 0} / {kv.max_pages || '—'}</div>
            <div>空闲页: {kv.free_pages ?? '—'}</div>
            <div>利用率: {((kv.utilization || 0) * 100).toFixed(1)}%</div>
            {kv.estimated_memory_mb > 0 && (
              <div>预估显存: {kv.estimated_memory_mb} MB</div>
            )}
            {kv.rounds > 0 && (
              <>
                <div>对话轮次: {kv.rounds}</div>
                <div>累计耗时: {kv.total_time_s}s</div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
