import { useState, useEffect, useCallback } from 'react';
import { fetchAvailableModels, loadModel } from '../api/client';

export default function ModelSelector({ onModelChange, onToast }) {
  const [models, setModels] = useState([]);
  const [engines, setEngines] = useState([]);
  const [current, setCurrent] = useState(null);
  const [currentEngine, setCurrentEngine] = useState(null);
  const [selectedEngine, setSelectedEngine] = useState('auto');
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(() => {
    fetchAvailableModels()
      .then((data) => {
        setModels(data.models || []);
        setEngines(data.available_engines || []);
        setCurrent(data.current);
        setCurrentEngine(data.current_engine);
        // 默认选中当前引擎或 auto
        if (data.current_engine) {
          setSelectedEngine(data.current_engine);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleLoad = async (quantType) => {
    setLoading(true);
    try {
      const result = await loadModel(selectedEngine, quantType, false);
      setCurrent(quantType);
      setCurrentEngine(selectedEngine);
      onModelChange?.(quantType, result);
      const engineLabel = selectedEngine === 'llama_cpp' ? 'llama.cpp' : 'PyTorch';
      onToast?.({ type: 'success', msg: `模型加载成功 (${engineLabel} / ${quantType}) — ${result.load_time_seconds}s` });
    } catch (err) {
      onToast?.({ type: 'error', msg: `加载失败: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  const icons = { fp16: '⚡', int8: '📦', int4: '💎' };
  const engineIcon = { llama_cpp: '🐛', pytorch: '🔥', auto: '🎯' };

  return (
    <div className="sidebar-section">
      <h3>🧪 模型选择</h3>

      {/* ---- 引擎切换 ---- */}
      {engines.length > 1 && (
        <div className="engine-selector">
          <span className="engine-label">引擎</span>
          <div className="engine-tabs">
            <button
              className={`engine-tab ${selectedEngine === 'auto' ? 'active' : ''}`}
              onClick={() => setSelectedEngine('auto')}
              title="自动检测最优引擎"
            >
              🎯 自动
            </button>
            {engines.map((eng) => (
              <button
                key={eng.id}
                className={`engine-tab ${selectedEngine === eng.id ? 'active' : ''}`}
                onClick={() => setSelectedEngine(eng.id)}
                title={eng.description}
              >
                {engineIcon[eng.id] || '⚙'} {eng.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ---- 量化选项（仅 PyTorch） ---- */}
      {selectedEngine !== 'llama_cpp' && models.map((m) => (
        <div
          key={m.id}
          className={`model-card ${current === m.id && currentEngine === 'pytorch' ? 'active' : ''}`}
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

      {/* ---- llama.cpp 一键加载 ---- */}
      {selectedEngine === 'llama_cpp' && (
        <div
          className={`model-card ${currentEngine === 'llama_cpp' ? 'active' : ''}`}
          onClick={() => !loading && handleLoad('gguf')}
        >
          {currentEngine === 'llama_cpp' && <span className="badge">已加载</span>}
          <div className="model-name">🐛 llama.cpp + GGUF</div>
          <div className="model-desc">
            {engines.find(e => e.id === 'llama_cpp')?.description || 'CPU/集显推理，Q4_K_M ~1.16 GB'}
          </div>
          <div className="model-meta">
            <span>💾 ~1.2 GB</span>
            <span>⚡ ~10-15 tok/s</span>
          </div>
        </div>
      )}

      {/* ---- 加载按钮 ---- */}
      <button
        className={`load-btn btn-primary ${loading ? 'loading' : ''}`}
        disabled={loading}
        onClick={() => {
          const quant = selectedEngine === 'llama_cpp' ? 'gguf' : (current || 'int4');
          handleLoad(quant);
        }}
      >
        {loading ? (
          <><span className="spinner" />加载中...</>
        ) : (
          `加载 ${selectedEngine === 'llama_cpp' ? 'llama.cpp' : (current || 'INT4').toUpperCase()}`
        )}
      </button>
    </div>
  );
}
