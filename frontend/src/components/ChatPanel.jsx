import { useState, useRef, useEffect, useCallback } from 'react';
import { sendMessage, sendMessageStream, clearChat, fetchPresets, uploadFile, fetchConversations, deleteConversations, deleteTurn, deleteSession, createSession, activateSession } from '../api/client';

const CHAT_HISTORY_KEY_PREFIX = 'qlh-chat-history-';

// L3: 生产构建下关闭聊天诊断日志
const DEBUG_CHAT = import.meta.env.DEV;
const debugLog = (...args) => { if (DEBUG_CHAT) console.log(...args); };

export default function ChatPanel({ modelLoaded, currentQuant, onToast, metricsTrigger, onOpenSettings, settings, sessionId, onCreateSession, onRenameSession }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [lastMetrics, setLastMetrics] = useState(null);
  const [presets, setPresets] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [expandedThinking, setExpandedThinking] = useState(new Set());
  const [deletingTurn, setDeletingTurn] = useState(null);  // { turnIndex, msgId } 或 null
  const [historyLoading, setHistoryLoading] = useState(false);  // 会话切换时历史加载中
  const [clearing, setClearing] = useState(false);             // 清空操作进行中
  const messagesEnd = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const creatingSessionRef = useRef(false);  // 防止并发创建会话
  const clearingRef = useRef(false);         // 防止清空期间发送消息
  const currentSessionIdRef = useRef(sessionId);  // 跟踪当前活跃会话ID，跨渲染同步
  const abortControllerRef = useRef(null);       // SSE 流式请求取消控制器
  const streamTimerRef = useRef(null);           // P1-3: full 模式打字动画 timer，用于会话切换时清理
  currentSessionIdRef.current = sessionId;

  const scrollToBottom = () => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleThinking = useCallback((msgId) => {
    setExpandedThinking((prev) => {
      const next = new Set(prev);
      if (next.has(msgId)) {
        next.delete(msgId);
      } else {
        next.add(msgId);
      }
      return next;
    });
  }, []);

  // ---- 单轮删除 ----
  const handleCopyMessage = useCallback(async (text) => {
    const fallbackCopy = () => {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      try {
        const ok = document.execCommand('copy');
        if (!ok) throw new Error('浏览器拒绝复制操作');
      } finally {
        document.body.removeChild(textarea);
      }
    };

    try {
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable');
        await navigator.clipboard.writeText(text);
      } catch (_) {
        fallbackCopy();
      }
      onToast?.({ type: 'success', msg: '回答已复制' });
    } catch (err) {
      onToast?.({ type: 'error', msg: `复制失败: ${err.message}` });
    }
  }, [onToast]);

  const handleDeleteTurn = useCallback(async (turnIndex, msgId) => {
    setDeletingTurn(null);  // 关闭确认弹窗
    try {
      await deleteTurn(sessionId, turnIndex);
      // 从本地 messages 中移除该轮的两条消息
      setMessages((prev) => {
        // 计算要删除的绝对索引
        let userCount = 0;
        const indicesToRemove = [];
        for (let i = 0; i < prev.length; i++) {
          if (prev[i].role === 'user') {
            if (userCount === turnIndex) {
              // 找到该轮 user
              indicesToRemove.push(i);
              // 下一个 assistant 也是这一轮的
              if (i + 1 < prev.length && prev[i + 1].role === 'assistant') {
                indicesToRemove.push(i + 1);
              }
              break;
            }
            userCount++;
          }
        }
        return prev.filter((_, i) => !indicesToRemove.includes(i));
      });
      onToast?.({ type: 'success', msg: '该轮对话已删除' });
    } catch (err) {
      onToast?.({ type: 'error', msg: `删除失败: ${err.message}` });
    }
  }, [sessionId, onToast]);

  // 计算消息对应的 turnIndex
  const getTurnIndex = useCallback((messages, msgIndex) => {
    let turnCount = 0;
    for (let i = 0; i <= msgIndex; i++) {
      if (messages[i]?.role === 'assistant') {
        turnCount++;
      }
    }
    return turnCount - 1;  // 0-based
  }, []);

  useEffect(scrollToBottom, [messages]);

  // 诊断日志：追踪 messages 数组变化
  useEffect(() => {
    const summary = messages.map((m) => `${m.role}(${m.content.slice(0, 20)}...)`).join(', ');
    debugLog(`[ChatPanel] messages updated: count=${messages.length}, roles=[${summary}]`);
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // ---- 对话历史持久化: 加载（服务器优先，localStorage 降级） ----
  useEffect(() => {
    if (!sessionId) {
      setHistoryLoading(false);
      return;
    }

    let cancelled = false;
    const storageKey = CHAT_HISTORY_KEY_PREFIX + sessionId;
    setHistoryLoading(true);

    const loadHistory = async () => {
      debugLog('[ChatPanel] history-loading: fetch start', { sessionId, storageKey });
      // ★ 先同步后端活跃会话状态（确保 session_histories 中已加载该会话）
      try {
        await activateSession(sessionId);
      } catch (_) {
        // 非致命：后端不可用时仍可从 localStorage 加载
      }
      // 再从服务器（数据库 / 本地文件 / 后端内存）加载完整历史（含 metrics/followups）
      try {
        const res = await fetchConversations(sessionId);
        debugLog('[ChatPanel] history-loading: server response', {
          sessionId,
          cancelled,
          messagesCount: res?.messages?.length,
          source: res?.source,
        });
        if (!cancelled && res?.messages?.length > 0) {
          // 从服务器加载的消息中恢复 followups（保存在最后一条 assistant 的 metrics.followups 中）
          let savedFollowups = null;
          const lastAssistantMsg = [...res.messages].reverse().find(m => m.role === 'assistant');
          if (lastAssistantMsg?.metrics?.followups) {
            savedFollowups = lastAssistantMsg.metrics.followups;
          }
          const msgs = res.messages.map((m, i) => {
            const isLastAssistant = m.role === 'assistant' && i === res.messages.length - 1;
            return {
              role: m.role,
              content: m.content,
              id: Date.now() - res.messages.length + i,
              metrics: isLastAssistant ? (m.metrics || null) : null,
              followups: isLastAssistant ? (savedFollowups || []) : undefined,
            };
          });
          if (!cancelled) {
            debugLog('[ChatPanel] history-loading: SET messages from server', { count: msgs.length, source: res?.source });
            setMessages(msgs);
            setHistoryLoading(false);
          }
          // 更新 localStorage 缓存（始终同步）
          try { localStorage.setItem(storageKey, JSON.stringify(msgs)); } catch (_) {}
          return;
        }
        // 服务器返回空 → 尝试 localStorage 缓存
      } catch (err) {
        debugLog('[ChatPanel] history-loading: server fetch failed, trying localStorage', { sessionId, error: err.message });
      }

      // 降级：从 localStorage 加载（本地持久化主存储，或服务器不可用时的回退）
      try {
        const saved = localStorage.getItem(storageKey);
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed) && parsed.length > 0) {
            if (!cancelled) {
              debugLog('[ChatPanel] history-loading: SET messages from localStorage', { count: parsed.length });
              setMessages(parsed);
              const lastAssistant = [...parsed].reverse().find(m => m.role === 'assistant');
              if (lastAssistant?.metrics) {
                setLastMetrics(lastAssistant.metrics);
              }
              setHistoryLoading(false);
              return;
            }
          }
        }
      } catch (_) {}

      // 无历史（新会话或全部清空）
      if (!cancelled) {
        debugLog('[ChatPanel] history-loading: NO history (empty session)', { sessionId });
        setHistoryLoading(false);
      }
    };

    loadHistory();
    return () => { cancelled = true; };
  }, [sessionId]); // 会话切换时重新加载

  // 跟踪上次保存时对应的 sessionId，防止会话切换瞬间把旧消息存入新会话的 key
  const lastSavedSessionRef = useRef(sessionId);

  // ---- 对话历史持久化: 保存（localStorage 始终启用，云同步由 saveHistory 控制） ----
  useEffect(() => {
    // ★ 关键守卫：如果 sessionId 刚刚改变，messages 仍属于旧会话，不能保存
    if (lastSavedSessionRef.current !== sessionId) {
      lastSavedSessionRef.current = sessionId;
      return;
    }
    // ★ localStorage 始终保存（本地缓存，不依赖云连接）
    if (messages.length > 0 && sessionId) {
      try {
        const storageKey = CHAT_HISTORY_KEY_PREFIX + sessionId;
        localStorage.setItem(storageKey, JSON.stringify(messages));
      } catch (_) {}
    }
  }, [messages, sessionId]);

  const autoCreatedSessionId = useRef(null);  // handleSend 自动创建的会话 ID，仅此 ID 跳过清空
  const prevSessionIdRef = useRef(sessionId);   // 跟踪上一次 sessionId，用于日志

  // ---- 组件卸载时清理所有异步资源（timer + 进行中的请求） ----
  // Phase 1.3: 防止从 chat 切到 admin 时打字动画 interval 泄漏 + SSE 请求残留
  useEffect(() => {
    return () => {
      if (streamTimerRef.current) {
        clearInterval(streamTimerRef.current);
        streamTimerRef.current = null;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  // ---- 会话切换时清空当前消息（由历史加载 effect 重新填充） ----
  useEffect(() => {
    const prevSid = prevSessionIdRef.current;
    debugLog('[ChatPanel] session-switch effect:', {
      prevSessionId: prevSid,
      newSessionId: sessionId,
      autoCreatedSessionId: autoCreatedSessionId.current,
      currentMessagesCount: 'pending (via setMessages)',
    });
    prevSessionIdRef.current = sessionId;

    // P1-1修复: 取消进行中的 SSE/HTTP 请求，避免旧会话推理继续消耗服务器资源
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    // P1-3修复: 清除 full 模式打字动画 timer，防止 interval 泄漏
    if (streamTimerRef.current) {
      clearInterval(streamTimerRef.current);
      streamTimerRef.current = null;
    }

    if (sessionId && autoCreatedSessionId.current === sessionId) {
      // handleSend 刚自动创建此会话，消息已由 handleSend 添加，不清空
      debugLog('[ChatPanel] session-switch: SKIP clear (auto-created session match)', { sessionId });
      autoCreatedSessionId.current = null;
      return;
    }
    // 无论何种切换，始终重置此标记（防止因渲染批处理导致残留）
    debugLog('[ChatPanel] session-switch: CLEAR messages', { prevAutoCreated: autoCreatedSessionId.current, newSid: sessionId });
    autoCreatedSessionId.current = null;
    setMessages([]);
    setSending(false);  // P2-2修复: 重置发送状态，解锁新会话的发送按钮
    setLastMetrics(null);
    setUploadedFile(null);
    setExpandedThinking(new Set());
  }, [sessionId]);

  // Fetch preset questions when model changes or chat is cleared
  useEffect(() => {
    if (modelLoaded) {
      fetchPresets()
        .then(setPresets)
        .catch(() => setPresets(null));
    } else {
      setPresets(null);
    }
  }, [modelLoaded]);

  const handleSendRef = useRef(null);

  const handlePresetClick = useCallback((question) => {
    handleSendRef.current?.(question);
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

  const handleSend = async (presetText) => {
    const text = (presetText || input).trim();
    if (!text || sending) return;

    // 清空操作进行中时禁止发送，避免状态竞态
    if (clearingRef.current) {
      onToast?.({ type: 'info', msg: '正在清空对话历史，请稍候...' });
      return;
    }

    // 无活跃会话时自动从首条消息创建第一个对话
    let effectiveSessionId = sessionId;
    let isAutoCreated = false;
    if (!effectiveSessionId) {
      debugLog('[ChatPanel] handleSend: no active session, auto-creating...');
      if (creatingSessionRef.current) {
        // 已有创建任务在进行中，避免并发创建重复会话
        debugLog('[ChatPanel] handleSend: BLOCKED by creatingSessionRef');
        onToast?.({ type: 'info', msg: '正在创建对话，请稍候...' });
        return;
      }
      creatingSessionRef.current = true;
      try {
        const title = text.slice(0, 30);
        const session = await createSession(title);
        effectiveSessionId = session.id;
        isAutoCreated = true;
        autoCreatedSessionId.current = session.id;  // 抑制此 ID 的 session-switch effect
        debugLog('[ChatPanel] handleSend: auto-created session', { sessionId: session.id, title });
        onCreateSession?.(session);  // 通知父组件更新会话列表
      } catch (err) {
        autoCreatedSessionId.current = null;
        onToast?.({ type: 'error', msg: `创建对话失败: ${err.message}` });
        return;
      } finally {
        creatingSessionRef.current = false;
      }
    }

    // [+] 按钮创建的会话标题为"新对话"，首条消息时自动重命名
    if (!isAutoCreated && messages.length === 0 && effectiveSessionId) {
      const newTitle = text.slice(0, 30);
      onRenameSession?.(effectiveSessionId, newTitle);
    }

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

    // 取消上一次未完成的 SSE 请求（会话切换等场景）
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      debugLog('[ChatPanel] handleSend: sending message...', {
        effectiveSessionId,
        currentSessionId: currentSessionIdRef.current,
        textPreview: text.slice(0, 30),
        streamingMode: settings?.streamingMode || 'full',
      });

      const useFastStream = settings?.streamingMode === 'fast';
      const msgId = Date.now() + 1;
      let res;

      if (useFastStream) {
        // ================================================================
        // ★ fast 模式：SSE 真流式，逐 token 实时推送
        // ================================================================
        // 先插入空的助手消息占位
        setMessages((prev) => [...prev, {
          role: 'assistant', content: '',
          thinkingContent: null, metrics: {}, followups: [],
          id: msgId,
        }]);

        res = await sendMessageStream(fullMessage, {
          sessionId: effectiveSessionId,
          signal: abortController.signal,
          maxNewTokens: settings?.maxNewTokens ?? 512,
          temperature: settings?.temperature ?? 0.7,
          topP: settings?.topP ?? 0.9,
          showThinking: settings?.showThinking ?? false,
          streamingMode: 'fast',
          onToken: (_token, fullText) => {
            // 会话切换检测：丢弃已推送的 token
            if (currentSessionIdRef.current !== effectiveSessionId) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === msgId ? { ...m, content: fullText } : m
              )
            );
          },
        });

        // 推理期间用户可能切换了会话——丢弃结果
        if (currentSessionIdRef.current !== effectiveSessionId) {
          debugLog('[ChatPanel] handleSend: ABORT — session changed during inference, discarding result');
          return;
        }

        // 最终更新：填充完整内容 + 元数据
        setMessages((prev) =>
          prev.map((m) =>
            m.id === msgId
              ? { ...m, content: res.content, thinkingContent: res.thinking_content || '', metrics: res.metrics, followups: res.followups || [] }
              : m
          )
        );
      } else {
        // ================================================================
        // full 模式（默认）：使用 /api/chat，完整功能 + 客户端逐字动画
        // ================================================================
        res = await sendMessage(fullMessage, {
          sessionId: effectiveSessionId,
          signal: abortController.signal,
          maxNewTokens: settings?.maxNewTokens ?? 512,
          temperature: settings?.temperature ?? 0.7,
          topP: settings?.topP ?? 0.9,
          showThinking: settings?.showThinking ?? false,
        });

        // 推理期间用户可能切换了会话——丢弃结果，不污染新会话的消息列表
        debugLog('[ChatPanel] handleSend: response received, checking session match...', {
          currentSessionId: currentSessionIdRef.current,
          effectiveSessionId,
          match: currentSessionIdRef.current === effectiveSessionId,
        });
        if (currentSessionIdRef.current !== effectiveSessionId) {
          debugLog('[ChatPanel] handleSend: ABORT — session changed during inference, discarding result');
          return;
        }

        const fullContent = res.content;

        // 先插入一个 content 为空的助手消息（显示 typing 动画 → 立即开始逐字输出）
        const assistantMsg = {
          role: 'assistant',
          content: '',
          thinkingContent: res.thinking_content || '',
          metrics: res.metrics,
          followups: res.followups || [],
          id: msgId,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        // ---- 假流式：逐字符输出 ----
        let charIndex = 0;
        const totalLen = fullContent.length;
        const baseInterval = 18;

        // P1-3: 保存 timer 引用，会话切换时清理（见 session-switch useEffect）
        streamTimerRef.current = setInterval(() => {
          const burst = 1 + Math.floor(Math.random() * 3);
          charIndex = Math.min(charIndex + burst, totalLen);

          setMessages((prev) =>
            prev.map((m) =>
              m.id === msgId
                ? { ...m, content: fullContent.slice(0, charIndex) }
                : m
            )
          );

          if (charIndex >= totalLen) {
            clearInterval(streamTimerRef.current);
            streamTimerRef.current = null;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === msgId ? { ...m, content: fullContent } : m
              )
            );
          }
        }, baseInterval + Math.random() * 12);
      }

      setLastMetrics(res.metrics);
      metricsTrigger?.(res.metrics);
    } catch (err) {
      // AbortError = 用户切换会话导致请求被取消，静默忽略
      if (err.name === 'AbortError') {
        return;
      }
      // Phase 3.1: 移除非 AbortError 时残留的空白助手占位消息
      setMessages((prev) => prev.filter(m => m.id !== msgId));
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
  handleSendRef.current = handleSend;  // 每轮渲染同步，确保 handlePresetClick 始终拿到最新引用

  const handleClear = async () => {
    clearingRef.current = true;
    setClearing(true);
    try {
      await clearChat();
      // 同时清除服务器端（数据库）和本地保存的历史
      if (sessionId) {
        try { await deleteConversations(sessionId); } catch (_) {}
        try {
          const storageKey = CHAT_HISTORY_KEY_PREFIX + sessionId;
          localStorage.removeItem(storageKey);
        } catch (_) {}
      }
      setMessages([]);
      setLastMetrics(null);
      setUploadedFile(null);
      onToast?.({ type: 'success', msg: '对话历史已清空（服务器 + 本地）' });
    } catch (err) {
      onToast?.({ type: 'error', msg: `清空失败: ${err.message}` });
    } finally {
      clearingRef.current = false;
      setClearing(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // format metrics for display
  const MODE_LABELS = {
    distributed_pipeline: 'Pipeline 分布式',
    forwarded_to_master: '转发→主节点',
    local_pytorch: 'PyTorch 本地',
    local_llama_cpp: 'llama.cpp 本地',
    distributed_forward: '分布式转发',
    master_pipeline: '主节点 Pipeline',
    pipeline_fallback: '流水线回退',
  };
  const formatMetrics = (m) => {
    if (!m) return '';
    const parts = [];
    const engine = m.engine || m.execution_mode;
    if (engine) parts.push(MODE_LABELS[engine] || String(engine));
    if (typeof m.distributed_used === 'boolean') {
      parts.push(m.distributed_used ? '分布式: 是' : '分布式: 否');
    }
    const tokens = m.generated_tokens ?? m.new_tokens ?? m.completion_tokens ?? m.tokens_generated;
    if (tokens) parts.push(`${tokens} tokens`);
    const tps = m.tokens_per_second ?? m.tokens_per_sec;
    if (tps) parts.push(`${Number(tps).toFixed(1)} tok/s`);
    const elapsed = m.elapsed_seconds ?? (m.total_time_ms ? m.total_time_ms / 1000 : null);
    if (elapsed) parts.push(`${Number(elapsed).toFixed(1)}s`);
    const workers = m.workers_used || m.worker_nodes || [];
    if (workers.length) parts.push(`workers: ${workers.join('→')}`);
    if (m.fallback_reason) parts.push(`回退: ${m.fallback_reason}`);
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
          <button className="btn-ghost" onClick={handleClear} disabled={messages.length === 0 || clearing}>
            {clearing ? '清空中...' : '清空对话'}
          </button>
        </div>
      </div>

      {/* Messages */}
      {(() => { debugLog('[ChatPanel] render decision:', { messagesLen: messages.length, historyLoading, sessionId, sending }); return null; })()}
      {messages.length === 0 ? (
        historyLoading ? (
          <div className="empty-state">
            <div className="icon">⏳</div>
            <p>加载对话历史中...</p>
          </div>
        ) : (
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
        )
      ) : (
        <div className="chat-messages">
          {messages.map((msg, msgIndex) => {
            const turnIndex = msg.role === 'assistant'
              ? getTurnIndex(messages, msgIndex)
              : -1;
            const isConfirmingDelete = deletingTurn?.msgId === msg.id;

            return (
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
                    {/* assistant 消息快捷操作 */}
                    {msg.role === 'assistant' && msg.content && (
                      <>
                        {isConfirmingDelete ? (
                          <div className="delete-confirm-overlay">
                            <span className="delete-confirm-text">确定删除这轮对话？</span>
                            <button
                              className="delete-confirm-btn confirm"
                              onClick={() => handleDeleteTurn(turnIndex, msg.id)}
                            >
                              删除
                            </button>
                            <button
                              className="delete-confirm-btn cancel"
                              onClick={() => setDeletingTurn(null)}
                            >
                              取消
                            </button>
                          </div>
                        ) : (
                          <div className="message-actions">
                            <button
                              className="message-copy-btn"
                              onClick={() => handleCopyMessage(msg.content)}
                              title="复制回答"
                            >
                              ⧉
                            </button>
                            {!sending && (
                              <button
                                className="message-delete-btn"
                                onClick={() => setDeletingTurn({ turnIndex, msgId: msg.id })}
                                title="删除这轮对话"
                              >
                                ✕
                              </button>
                            )}
                          </div>
                        )}
                      </>
                    )}
                    {msg.thinkingContent && (
                      <div className="thinking-section">
                        <button
                          className="thinking-toggle"
                          onClick={() => toggleThinking(msg.id)}
                        >
                          <span className="thinking-toggle-icon">
                            {expandedThinking.has(msg.id) ? '▼' : '▶'}
                          </span>
                          <span className="thinking-toggle-label">🧠 深度思考</span>
                          <span className="thinking-toggle-hint">
                            {expandedThinking.has(msg.id) ? '点击收起' : '点击展开'}
                          </span>
                        </button>
                        {expandedThinking.has(msg.id) && (
                          <div className="thinking-content">
                            {msg.thinkingContent}
                          </div>
                        )}
                      </div>
                    )}
                    <div className="bubble-content">
                      {msg.displayContent || msg.content}
                    </div>
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
            );
          })}
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
            disabled={!modelLoaded || sending || clearing || (!input.trim() && !uploadedFile)}
          >
            {clearing ? '清空中...' : sending ? '生成中...' : '发送'}
          </button>
        </div>
      </div>
    </>
  );
}
