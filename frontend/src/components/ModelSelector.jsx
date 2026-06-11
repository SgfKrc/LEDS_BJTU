import { useState, useEffect } from 'react';
import { fetchAvailableModels, loadModel } from '../api/client';

export default function ModelSelector({ onModelChange, onToast }) {
  const [models, setModels] = useState([]);
  const [current, setCurrent] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAvailableModels()
      .then((data) => {
        setModels(data.models);
        setCurrent(data.current);
      })
      .catch(() => {});
  }, []);

  const handleLoad = async (quantType) => {
    setLoading(true);
    try {
      const result = await loadModel(quantType, false);
      setCurrent(quantType);
      onModelChange?.(quantType, result);
      onToast?.({ type: 'success', msg: `模型加载成功 (${quantType}) — ${result.load_time_seconds}s` });
    } catch (err) {
      onToast?.({ type: 'error', msg: `加载失败: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  // Map quant type to display icon
  const icons = { fp16: '⚡', int8: '📦', int4: '💎' };

  return (
    <div className="sidebar-section">
      <h3>🧪 模型选择</h3>
      {models.map((m) => (
        <div
          key={m.id}
          className={`model-card ${current === m.id ? 'active' : ''}`}
          onClick={() => !loading && handleLoad(m.id)}
        >
          {current === m.id && <span className="badge">已加载</span>}
          <div className="model-name">{icons[m.id]} {m.name}</div>
          <div className="model-desc">{m.description}</div>
          <div className="model-meta">
            <span>💾 {m.memory_gb} GB</span>
            <span>⚡ ~{m.speed_tok_s} tok/s</span>
            {m.compile_support && <span>🔧 融合</span>}
          </div>
        </div>
      ))}

      <button
        className={`load-btn btn-primary ${loading ? 'loading' : ''}`}
        disabled={loading || !current}
        onClick={() => current && handleLoad(current)}
      >
        {loading ? (
          <><span className="spinner" />加载中...</>
        ) : current ? (
          `重新加载 ${current.toUpperCase()}`
        ) : (
          '选择一个模型'
        )}
      </button>
    </div>
  );
}
