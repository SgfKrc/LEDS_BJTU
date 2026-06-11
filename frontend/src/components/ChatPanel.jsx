import { useState, useRef, useEffect, useCallback } from 'react';
import { sendMessage, clearChat, fetchPresets, uploadFile } from '../api/client';

const CHAT_HISTORY_KEY = 'qlh-chat-history';

export default function ChatPanel({ modelLoaded, currentQuant, onToast, metricsTrigger, onOpenSettings, settings }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [lastMetrics, setLastMetrics] = useState(null);
  const [presets, setPresets] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const messagesEnd = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(scrollToBottom, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // ---- 对话历史持久化: 加载 ----
  useEffect(() => {
    if (settings?.saveHistory) {
      try {
        const saved = localStorage.getItem(CHAT_HISTORY_KEY);
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed) && parsed.length > 0) {
            setMessages(parsed);
            // 恢复最后一条 assistant 消息的 metrics
            const lastAssistant = [...parsed].reverse().find(m => m.role === 'assistant');
            if (lastAssistant?.metrics) {
              setLastMetrics(lastAssistant.metrics);
            }
          }
        }
      } catch (_) {}
    }
  }, []); // 仅首次挂载

  // ---- 对话历史持久化: 保存 ----
  useEffect(() => {
    if (settings?.saveHistory && messages.length > 0) {
      try {
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(messages));
      } catch (_) {}
    }
  }, [messages, settings?.saveHistory]);

  // ---- 关闭保存时清除本地历史 ----
  useEffect(() => {
    if (!settings?.saveHistory) {
      try { localStorage.removeItem(CHAT_HISTORY_KEY); } catch (_) {}
    }
  }, [settings?.saveHistory]);

  // Fetch preset questions when model changes or chat is cleared
  useEffect(() => {
    if (modelLoaded) {
      fetchPresets()
        .then(setPresets)
        .catch(() => setPresets(null));
    } else {
      setPresets(null);
    }
  }, [modelLoaded, messages.length === 0]);

  const handlePresetClick = useCallback((question) => {
    setInput(question);
    inputRef.current?.focus();
  }, []);

  // ---- 文件上传 ----
  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadFile(file);
      setUploadedFile(result);
      onToast?.({ type: 'success', msg: `已上传: ${result.filename} (${result.line_count} 行)` });
    } catch (err) {
      onToast?.({ type: 'error', msg: `上传失败: ${err.message}` });
    } finally {
      setUploading(false);
      // 重置 input 以便重复选择同一文件
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

    // 构建完整消息：有文件时用 markdown 代码块包裹文件内容
    let fullMessage = text;
    let fileContext = null;
    if (uploadedFile) {
      fileContext = uploadedFile;
      const lang = uploadedFile.language || 'plaintext';
      fullMessage = [
        `📄 **${uploadedFile.filename}** (${uploadedFile.line_count} 行):`,
        '',
        '```' + lang,
        uploadedFile.content,
        '```',
        '',
        '---',
        '',
        text,
      ].join('\n');
    }

    const userMsg = {
      role: 'user',
      content: fullMessage,
      displayContent: text,
      fileContext: fileContext,
      id: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setUploadedFile(null);
    setSending(true);

    try {
      const res = await sendMessage(fullMessage, {
        maxNewTokens: settings?.maxNewTokens ?? 512,
        temperature: settings?.temperature ?? 0.7,
        topP: settings?.topP ?? 0.9,
      });
      const assistantMsg = {
        role: 'assistant',
        content: res.content,
        metrics: res.metrics,
        followups: res.followups || [],
        id: Date.now() + 1,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setLastMetrics(res.metrics);
      metricsTrigger?.(res.metrics);
    } catch (err) {
      const errMsg = {
        role: 'system',
        content: `错误: ${err.message}`,
        id: Date.now() + 1,
      };
      setMessages((prev) => [...prev, errMsg]);
      onToast?.({ type: 'error', msg: `推理失败: ${err.message}` });
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const handleClear = async () => {
    try {
      await clearChat();
      setMessages([]);
      setLastMetrics(null);
      setUploadedFile(null);
      // 同时清除本地保存的历史
      try { localStorage.removeItem(CHAT_HISTORY_KEY); } catch (_) {}
      onToast?.({ type: 'success', msg: '对话历史已清空' });
    } catch (err) {
      onToast?.({ type: 'error', msg: `清空失败: ${err.message}` });
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // format metrics for display
  const formatMetrics = (m) => {
    if (!m) return '';
    const parts = [];
    if (m.new_tokens) parts.push(`${m.new_tokens} tokens`);
    if (m.tokens_per_second) parts.push(`${m.tokens_per_second} tok/s`);
    if (m.elapsed_seconds) parts.push(`${m.elapsed_seconds.toFixed(1)}s`);
    if (m.gpu_memory_mb) parts.push(`${m.gpu_memory_mb} MB`);
    return parts.join(' · ');
  };

  return (
    <>
      {/* Header */}
      <div className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <h2>💬 对话测试</h2>
          {currentQuant && (
            <span className="model-badge">
              {currentQuant.toUpperCase()}
            </span>
          )}
          {lastMetrics && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 12 }}>
              {formatMetrics(lastMetrics)}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn-ghost" onClick={onOpenSettings} title="系统设置">
            ⚙️
          </button>
          <button className="btn-ghost" onClick={handleClear} disabled={messages.length === 0}>
            清空对话
          </button>
        </div>
      </div>

      {/* Messages */}
      {messages.length === 0 ? (
        <div className="empty-state">
          {modelLoaded && presets?.presets?.length > 0 ? (
            <>
              <div className="icon">🧠</div>
              <p>模型已就绪，选择一个预设问题或直接输入</p>
              {presets.current_speed_tok_s && (
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 16 }}>
                  当前 {presets.current_quant?.toUpperCase() || 'INT4'} 模式
                  · 约 {presets.current_speed_tok_s} tok/s
                  · 最大 {presets.max_new_tokens} tokens
                </p>
              )}
              <div className="preset-grid">
                {presets.presets.map((p) => (
                  <button
                    key={p.id}
                    className="preset-card"
                    onClick={() => handlePresetClick(p.question)}
                  >
                    <div className="preset-header">
                      <span className="preset-icon">{p.icon}</span>
                      <span className="preset-label">{p.label}</span>
                    </div>
                    <div className="preset-question">{p.question}</div>
                    <div className="preset-meta">
                      <span title="预估输入+输出 Token">🪙 ~{p.estimated_prompt_tokens + p.estimated_response_tokens} tokens</span>
                      <span title="预估 KV 缓存显存">💾 ~{p.estimated_memory_mb} MB</span>
                      <span title="预估耗时">⏱ ~{p.estimated_seconds}s</span>
                    </div>
                  </button>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="icon">🧠</div>
              <p>
                {modelLoaded
                  ? '模型已就绪，输入消息开始对话测试'
                  : '请先在左侧选择并加载模型'}
              </p>
              {!modelLoaded && (
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  加载 INT4 量化版仅需 ~1.8 GB 显存，适合边缘设备
                </p>
              )}
            </>
          )}
        </div>
      ) : (
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="avatar">
                {msg.role === 'user' ? '👤' : msg.role === 'assistant' ? '🤖' : '⚠️'}
              </div>
              <div>
                {msg.fileContext && (
                  <div className="file-attachment-badge">
                    📎 {msg.fileContext.filename} ({msg.fileContext.line_count} 行)
                  </div>
                )}
                <div className="bubble">
                  {msg.displayContent || msg.content}
                </div>
                {msg.metrics && (
                  <div className="meta">{formatMetrics(msg.metrics)}</div>
                )}
                {msg.role === 'assistant' && msg.followups?.length > 0 && (
                  <div className="followup-chips">
                    {msg.followups.map((q, i) => (
                      <button
                        key={i}
                        className="followup-chip"
                        onClick={() => handlePresetClick(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {sending && (
            <div className="message assistant">
              <div className="avatar">🤖</div>
              <div className="bubble">
                <div className="typing-indicator">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEnd} />
        </div>
      )}

      {/* Input */}
      <div className="chat-input-area">
        {/* 已上传文件标签 */}
        {uploadedFile && (
          <div className="file-chip-row">
            <div className="file-chip">
              <span className="file-chip-icon">📎</span>
              <span className="file-chip-name">{uploadedFile.filename}</span>
              <span className="file-chip-meta">
                {uploadedFile.language} · {uploadedFile.line_count} 行 · {Math.round(uploadedFile.size_bytes / 1024)} KB
              </span>
              <button className="file-chip-remove" onClick={handleRemoveFile} title="移除文件">
                ✕
              </button>
            </div>
          </div>
        )}
        <div className="chat-input-row">
          {/* 上传按钮 */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.csv,.py,.json,.log,.xml,.yaml,.yml,.ini,.cfg,.conf,.js,.ts,.jsx,.tsx,.html,.css,.sh,.bash,.zsh,.ps1,.cpp,.c,.h,.java,.go,.rs,.rb,.sql,.r,.m,.swift,.kt,.toml,.properties,.env"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          <button
            className="btn-ghost file-upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={!modelLoaded || sending || uploading}
            title="上传文本文件 (txt/md/csv/py/json 等)"
          >
            {uploading ? '⏳' : '📎'}
          </button>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              modelLoaded
                ? '输入消息... (Enter 发送, Shift+Enter 换行)'
                : '请先加载模型...'
            }
            disabled={!modelLoaded || sending}
            rows={1}
          />
          <button
            className="btn-primary"
            onClick={handleSend}
            disabled={!modelLoaded || sending || (!input.trim() && !uploadedFile)}
          >
            {sending ? '生成中...' : '发送'}
          </button>
        </div>
      </div>
    </>
  );
}
