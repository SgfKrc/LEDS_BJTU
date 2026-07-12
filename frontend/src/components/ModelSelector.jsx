import { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchAvailableModels, fetchModels, loadModel } from '../api/client';

const ENGINE_ICON = { llama_cpp: 'GGUF', pytorch: 'PT', auto: 'AUTO' };

function formatFormats(model) {
  const formats = model?.available_formats || [];
  if (!formats.length) return '未下载';
  return formats.map(f => f === 'gguf' ? 'GGUF' : 'Safetensors').join(' + ');
}

function quantLabel(q) {
  const value = String(q || '').toUpperCase();
  if (value === 'FP16') return 'FP16';
  if (value === 'INT8') return 'INT8';
  if (value === 'INT4') return 'INT4';
  return value || 'GGUF';
}

function normalizePytorchQuant(value, options) {
  const wanted = String(value || '').toLowerCase();
  return options.find(q => String(q).toLowerCase() === wanted) || options[0] || 'int4';
}

export default function ModelSelector({ onModelChange, onToast }) {
  const [modelCatalog, setModelCatalog] = useState([]);
  const [engines, setEngines] = useState([]);
  const [currentQuant, setCurrentQuant] = useState(null);
  const [currentEngine, setCurrentEngine] = useState(null);
  const [currentModelId, setCurrentModelId] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [selectedEngine, setSelectedEngine] = useState('auto');
  const [selectedQuant, setSelectedQuant] = useState('int4');
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [runtimeData, modelData] = await Promise.all([
        fetchAvailableModels(),
        fetchModels(),
      ]);
      const models = modelData.models || [];
      setModelCatalog(models);
      setEngines(runtimeData.available_engines || []);
      setCurrentQuant(runtimeData.current);
      setCurrentEngine(runtimeData.current_engine);
      setCurrentModelId(modelData.active_model_id || null);

      setSelectedModelId(prev => {
        if (prev && models.some(m => m.model_id === prev)) return prev;
        return modelData.active_model_id || models.find(m => m.is_available)?.model_id || models[0]?.model_id || '';
      });
      if (runtimeData.current_engine) {
        setSelectedEngine(runtimeData.current_engine);
      }
    } catch (_) {
      setModelCatalog([]);
      setEngines([]);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const selectedModel = useMemo(
    () => modelCatalog.find(m => m.model_id === selectedModelId) || null,
    [modelCatalog, selectedModelId],
  );

  const supportedEngines = selectedModel?.supported_engines || [];
  const effectiveEngine = selectedEngine === 'auto'
    ? (selectedModel?.preferred_engine || 'auto')
    : selectedEngine;

  const pytorchQuants = useMemo(() => {
    const values = (selectedModel?.quant_types || ['fp16', 'int8', 'int4'])
      .map(q => String(q))
      .filter(q => ['fp16', 'int8', 'int4'].includes(q.toLowerCase()));
    return values.length ? values : ['int4'];
  }, [selectedModel]);

  useEffect(() => {
    if (effectiveEngine !== 'llama_cpp' && !pytorchQuants.includes(selectedQuant)) {
      setSelectedQuant(normalizePytorchQuant(selectedQuant, pytorchQuants));
    }
  }, [effectiveEngine, pytorchQuants, selectedQuant]);

  const handleModelSelect = (modelId) => {
    const model = modelCatalog.find(m => m.model_id === modelId);
    setSelectedModelId(modelId);
    const nextEngine = model?.preferred_engine || 'auto';
    if (model?.preferred_engine) {
      setSelectedEngine(nextEngine);
    }
    const nextPytorchQuants = (model?.quant_types || ['fp16', 'int8', 'int4'])
      .map(q => String(q))
      .filter(q => ['fp16', 'int8', 'int4'].includes(q.toLowerCase()));
    if (nextEngine === 'llama_cpp') {
      setSelectedQuant(model?.default_quant_type || 'gguf');
    } else {
      setSelectedQuant(normalizePytorchQuant(model?.default_quant_type || selectedQuant, nextPytorchQuants.length ? nextPytorchQuants : ['int4']));
    }
  };

  const handleLoad = async () => {
    if (!selectedModel) return;
    if (!selectedModel.is_available) {
      onToast?.({ type: 'warning', msg: selectedModel.unavailable_reason || '模型文件未下载，无法加载' });
      return;
    }
    if (effectiveEngine !== 'auto' && supportedEngines.length > 0 && !supportedEngines.includes(effectiveEngine)) {
      onToast?.({ type: 'warning', msg: `当前模型不支持 ${effectiveEngine} 引擎` });
      return;
    }

    const quant = effectiveEngine === 'llama_cpp'
      ? (selectedModel.default_quant_type || 'gguf')
      : normalizePytorchQuant(selectedQuant, pytorchQuants);

    setLoading(true);
    try {
      const result = await loadModel(effectiveEngine, quant, false, selectedModel.model_id);
      setCurrentQuant(quant);
      setCurrentEngine(effectiveEngine);
      setCurrentModelId(result.active_model_id || selectedModel.model_id);
      onModelChange?.(quant, result);
      onToast?.({
        type: 'success',
        msg: `已加载模型: ${result.model_name || selectedModel.name}`,
      });
      fetchData();
    } catch (err) {
      onToast?.({ type: 'error', msg: `加载失败: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="sidebar-section">
      <h3>模型选择</h3>

      <div className="model-select-row">
        <label className="engine-label" htmlFor="sidebar-model-select">模型</label>
        <select
          id="sidebar-model-select"
          className="model-select"
          value={selectedModelId}
          onChange={e => handleModelSelect(e.target.value)}
        >
          {modelCatalog.map(model => (
            <option key={model.model_id} value={model.model_id} disabled={!model.is_available}>
              {model.name}{model.is_available ? '' : '（未下载）'}
            </option>
          ))}
        </select>
      </div>

      {selectedModel && (
        <div className={`model-card compact ${selectedModel.is_available ? '' : 'disabled'}`}>
          {currentModelId === selectedModel.model_id && <span className="badge">已加载</span>}
          <div className="model-name">{selectedModel.name}</div>
          <div className="model-desc">{selectedModel.description}</div>
          <div className="model-meta">
            <span>{formatFormats(selectedModel)}</span>
            <span>{selectedModel.model_type}</span>
            <span>ctx {selectedModel.max_context}</span>
          </div>
          {!selectedModel.is_available && (
            <div className="model-unavailable-note">
              {selectedModel.unavailable_reason || '模型文件未下载，暂不可加载'}
            </div>
          )}
        </div>
      )}

      {engines.length > 0 && (
        <div className="engine-selector">
          <span className="engine-label">引擎</span>
          <div className="engine-tabs">
            <button
              className={`engine-tab ${selectedEngine === 'auto' ? 'active' : ''}`}
              onClick={() => setSelectedEngine('auto')}
              type="button"
              title="按模型可用格式自动选择"
            >
              {ENGINE_ICON.auto} 自动
            </button>
            {engines.map((eng) => {
              const unsupported = supportedEngines.length > 0 && !supportedEngines.includes(eng.id);
              return (
                <button
                  key={eng.id}
                  className={`engine-tab ${selectedEngine === eng.id ? 'active' : ''}`}
                  onClick={() => !unsupported && setSelectedEngine(eng.id)}
                  disabled={unsupported}
                  type="button"
                  title={unsupported ? '当前模型不支持该引擎' : eng.description}
                >
                  {ENGINE_ICON[eng.id] || eng.id} {eng.name}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {effectiveEngine !== 'llama_cpp' && (
        <div className="quant-selector">
          <span className="engine-label">量化</span>
          <div className="engine-tabs">
            {pytorchQuants.map(q => (
              <button
                key={q}
                className={`engine-tab ${selectedQuant === q ? 'active' : ''}`}
                onClick={() => setSelectedQuant(q)}
                type="button"
              >
                {quantLabel(q)}
              </button>
            ))}
          </div>
        </div>
      )}

      {effectiveEngine === 'llama_cpp' && (
        <div className="model-card compact">
          <div className="model-name">llama.cpp + GGUF</div>
          <div className="model-desc">量化由 GGUF 文件决定：{quantLabel(selectedModel?.default_quant_type || 'gguf')}</div>
        </div>
      )}

      <button
        className={`load-btn btn-primary ${loading ? 'loading' : ''}`}
        disabled={loading || !selectedModel?.is_available}
        onClick={handleLoad}
        type="button"
      >
        {loading ? (
          <><span className="spinner" />加载中...</>
        ) : (
          `加载 ${selectedModel?.name || '模型'}`
        )}
      </button>
    </div>
  );
}
