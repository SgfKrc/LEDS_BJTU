/**
 * 对话栏 — 工作台右侧，哥特一侧。
 *
 * 与控制台的关系：共用设计令牌体系，但走 --gothic-* 一组变量和衬线标题，
 * 尖拱、描边金和首字下沉都在 CSS 里，这里只负责状态机。
 *
 * 数据：GET /api/conversations 取历史，POST /api/chat/stream 逐 token 流式追加。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Eraser, Loader2, Paperclip, RefreshCw, SendHorizontal, Square, Trash2, X } from 'lucide-react';
import * as api from '../data/api';
import { fixturesEnabled } from '../data/fixtures';
import { conversationFixture } from '../data/fixtures';
import type { ApiErrorKind, ChatAttachment, ChatTurn } from '../data/types';
import { CommandButton } from './CommandButton';
import { EmptyState } from './EmptyState';
import { pushToast } from './Toast';
import { useReducedMotion } from '../motion/useReducedMotion';
import { GothicWorksCanvas } from '../visual/GothicWorksCanvas';

/**
 * 可选扩展名 —— 与后端 ALLOWED_TEXT_EXTENSIONS 对齐（api_server.py）。
 * 写进 accept 只是给系统文件框做默认筛选，真正的把关在后端；
 * 这里同步一份是为了让用户在选之前就看不到会被 400 的类型。
 */
const ACCEPT = [
  '.txt', '.md', '.csv', '.py', '.json', '.log', '.xml', '.yaml', '.yml',
  '.ini', '.cfg', '.conf', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
  '.sh', '.bash', '.zsh', '.ps1', '.cpp', '.c', '.h', '.java', '.go',
  '.rs', '.rb', '.sql', '.r', '.m', '.swift', '.kt', '.toml', '.properties', '.env',
].join(',');

/** 后端上限 5 MB；本地先挡一道，省一次必然失败的往返。 */
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * 把附件正文拼到消息前面。
 *
 * 用围栏代码块包起来并标注语言：模型能区分「这是附件」和「这是我的问题」，
 * 不然几千行日志会和提问糊成一段。
 */
function composePayload(text: string, attachments: ChatAttachment[]): string {
  if (attachments.length === 0) return text;
  const blocks = attachments.map((a) => {
    const note = a.truncated ? `（仅前 ${a.lineCount} 行，共 ${a.lineCount + a.truncatedLines} 行）` : '';
    return `文件：${a.filename}${note}\n\`\`\`${a.language || ''}\n${a.content}\n\`\`\``;
  });
  return `${blocks.join('\n\n')}\n\n${text}`;
}

/** React key 用的本地轮次 id，够唯一即可，不需要 uuid。 */
let seq = 0;
function nextId(prefix: string): string {
  seq += 1;
  return `${prefix}-${Date.now().toString(36)}-${seq}`;
}

function removeTurnPair(items: ChatTurn[], turnIndex: number): ChatTurn[] {
  let currentTurn = 0;
  let start = -1;
  for (let index = 0; index < items.length; index += 1) {
    if (items[index]?.role !== 'user') continue;
    if (currentTurn === turnIndex) {
      start = index;
      break;
    }
    currentTurn += 1;
  }
  if (start < 0) return items;
  const end = items[start + 1]?.role === 'assistant' ? start + 2 : start + 1;
  return items.filter((_item, index) => index < start || index >= end);
}

/**
 * 提交给后端的 generation_id。
 *
 * 格式必须匹配后端的 `^gen_[A-Za-z0-9_-]{8,96}$`（api_server.py 的
 * _GENERATION_ID_PATTERN）：前缀是下划线，且后面至少 8 个字符。
 * 用连字符前缀会被 400 拒掉，整轮对话发不出去。
 */
function nextGenerationId(): string {
  seq += 1;
  const rand = Math.random().toString(36).slice(2, 10);
  return `gen_${Date.now().toString(36)}_${seq}_${rand}`;
}

/**
 * 指标行：只挑用户真正会看的几项，字段名对齐后端 done 事件。
 * 分布式与回落是这个项目最关心的信息，所以单独标出来。
 */
function formatMetrics(turn: ChatTurn): string {
  const m = turn.metrics;
  if (!m) return '';
  const parts: string[] = [];
  if (typeof m.tokens_per_second === 'number') parts.push(`${m.tokens_per_second.toFixed(1)} tok/s`);

  const tokens =
    typeof m.generated_tokens === 'number'
      ? m.generated_tokens
      : typeof m.completion_tokens === 'number'
        ? m.completion_tokens
        : m.usage?.total_tokens;
  if (typeof tokens === 'number') parts.push(`${tokens} tokens`);

  // 走没走分布式，是这套集群界面最该说清的一件事
  if (m.distributed_used === true) {
    const workers = Array.isArray(m.workers_used) ? m.workers_used.length : 0;
    parts.push(workers > 0 ? `分布式 · ${workers} 节点` : '分布式');
  } else if (m.fallback === true) {
    parts.push('单机回落');
  }

  // 回落时 execution_mode 就叫 fallback_*，和上面那句「单机回落」重复，省掉
  const mode = m.execution_mode || m.mode;
  if (typeof mode === 'string' && mode && !(m.fallback === true && mode.startsWith('fallback'))) {
    parts.push(mode);
  }
  return parts.join(' · ');
}

interface ChatPaneProps {
  sessionId?: string;
  /** 对话产生新一轮后通知外层（控制台侧顺带刷新队列/日志）。 */
  onTurnComplete?: () => void;
}

export function ChatPane({ sessionId = 'default', onTurnComplete }: ChatPaneProps) {
  const usingFixtures = fixturesEnabled();
  const reduced = useReducedMotion();

  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [loadErrorKind, setLoadErrorKind] = useState<ApiErrorKind | null>(null);
  const [loadErrorStatus, setLoadErrorStatus] = useState<number | null>(null);
  const [sending, setSending] = useState(false);
  const [deletingTurn, setDeletingTurn] = useState<number | null>(null);
  const [confirmDeleteTurn, setConfirmDeleteTurn] = useState<number | null>(null);
  const [historyNonce, setHistoryNonce] = useState(0);
  const [showThinking, setShowThinking] = useState(false);
  const [openThinking, setOpenThinking] = useState<Record<string, boolean>>({});
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [uploading, setUploading] = useState(false);

  const listRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generationRef = useRef('');
  /** 用户手动向上翻看历史时不要抢滚动条。 */
  const pinnedRef = useRef(true);

  // ---- 历史加载 ----
  useEffect(() => {
    const controller = new AbortController();
    let alive = true;
    setLoading(true);
    setLoadError('');
    setLoadErrorKind(null);
    setLoadErrorStatus(null);

    const load = async () => {
      try {
        const data = usingFixtures
          ? conversationFixture
          : await api.fetchConversation(sessionId, 200, controller.signal);
        if (!alive) return;
        setTurns(
          data.messages
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .map((m) => ({
              id: nextId('hist'),
              role: m.role as 'user' | 'assistant',
              content: m.content,
              ...(m.metrics ? { metrics: m.metrics } : {}),
              ...(m.created_at ? { createdAt: m.created_at } : {}),
            })),
        );
      } catch (err) {
        if (!alive || controller.signal.aborted) return;
        const message = api.describeError(err);
        if (message) {
          setLoadError(message);
          setLoadErrorKind(api.getErrorKind(err));
          setLoadErrorStatus(
            typeof err === 'object' && err !== null && 'status' in err && typeof err.status === 'number'
              ? err.status
              : null,
          );
        }
      } finally {
        if (alive) {
          setLoading(false);
          setRefreshing(false);
        }
      }
    };

    void load();
    return () => {
      alive = false;
      controller.abort();
    };
  }, [sessionId, usingFixtures, historyNonce]);

  // ---- 自动滚到底 ----
  useEffect(() => {
    const el = listRef.current;
    if (!el || !pinnedRef.current) return;
    el.scrollTo({ top: el.scrollHeight, behavior: reduced ? 'auto' : 'smooth' });
  }, [turns, reduced]);

  const onScroll = useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    // 距底 48px 内算「贴底」，继续跟随流式输出。
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);

  // ---- 卸载时中止在途请求 ----
  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const patchTurn = useCallback((id: string, patch: Partial<ChatTurn>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  }, []);

  // ---- 附件 ----

  const pickFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      if (usingFixtures) {
        pushToast('演示数据模式下不上传附件。请在设置页关闭演示数据。', 'warn');
        return;
      }

      setUploading(true);
      try {
        for (const file of Array.from(files)) {
          if (file.size > MAX_UPLOAD_BYTES) {
            pushToast(`${file.name} 超过 5 MB 上限，已跳过。`, 'warn');
            continue;
          }
          try {
            const res = await api.uploadChatFile(file);
            setAttachments((prev) => [
              ...prev,
              {
                id: nextId('att'),
                filename: res.filename,
                language: res.language,
                lineCount: res.line_count,
                sizeBytes: res.size_bytes,
                truncated: res.truncated,
                truncatedLines: res.truncated_lines,
                content: res.content,
              },
            ]);
            if (res.truncated) {
              pushToast(
                `${res.filename} 超过 5000 行，仅取前 ${res.line_count} 行（省略 ${res.truncated_lines} 行）。`,
                'warn',
              );
            }
          } catch (err) {
            pushToast(`${file.name} 上传失败：${api.describeError(err)}`, 'danger');
          }
        }
      } finally {
        setUploading(false);
        // 清空 input，否则同一个文件再选一次不会触发 change
        if (fileRef.current) fileRef.current.value = '';
      }
    },
    [usingFixtures],
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;

    if (usingFixtures) {
      pushToast('演示数据模式下不发起真实推理。请在设置页关闭演示数据。', 'warn');
      return;
    }

    const sent = attachments;
    const payload = composePayload(text, sent);
    const userTurn: ChatTurn = {
      id: nextId('u'),
      role: 'user',
      content: text,
      ...(sent.length > 0
        ? { attachments: sent.map((a) => ({ filename: a.filename, lineCount: a.lineCount })) }
        : {}),
    };
    const replyId = nextId('a');
    setTurns((prev) => [
      ...prev,
      userTurn,
      { id: replyId, role: 'assistant', content: '', streaming: true },
    ]);
    setDraft('');
    setAttachments([]);
    setSending(true);
    pinnedRef.current = true;

    const controller = new AbortController();
    abortRef.current = controller;
    const generationId = nextGenerationId();
    generationRef.current = generationId;

    try {
      // 发给模型的是「附件正文 + 提问」，气泡里只显示提问本身
      const result = await api.streamChat(payload, {
        sessionId,
        streamingMode: 'interactive',
        showThinking,
        generationId,
        signal: controller.signal,
        onToken: (_token, full) => patchTurn(replyId, { content: full }),
      });
      patchTurn(replyId, {
        content: result.content,
        streaming: false,
        ...(result.thinkingContent ? { thinking: result.thinkingContent } : {}),
        metrics: result.metrics,
      });
      onTurnComplete?.();
    } catch (err) {
      if (controller.signal.aborted) {
        // 用户主动停止：保留已经流出来的片段，标注中断而不是报错。
        setTurns((prev) =>
          prev.map((t) =>
            t.id === replyId
              ? { ...t, streaming: false, error: t.content ? '已停止生成' : '已取消' }
              : t,
          ),
        );
      } else {
        const message = api.describeError(err) || '推理失败';
        patchTurn(replyId, { streaming: false, error: message });
        pushToast(`对话失败：${message}`, 'danger');
      }
    } finally {
      abortRef.current = null;
      generationRef.current = '';
      setSending(false);
      textareaRef.current?.focus();
    }
  }, [
    draft,
    sending,
    usingFixtures,
    sessionId,
    showThinking,
    patchTurn,
    onTurnComplete,
    attachments,
  ]);

  const stop = useCallback(() => {
    const generationId = generationRef.current;
    abortRef.current?.abort();
    // 同时通知后端取消，否则服务端会继续算完这一轮。
    if (generationId) void api.cancelGeneration(generationId).catch(() => undefined);
  }, []);

  const clear = useCallback(async () => {
    if (usingFixtures) {
      pushToast('演示数据模式下不执行清空。', 'warn');
      return;
    }
    try {
      await api.clearChat(sessionId);
      setTurns([]);
      pushToast('已清空当前会话历史。', 'ok');
    } catch (err) {
      pushToast(`清空失败：${api.describeError(err)}`, 'danger');
    }
  }, [usingFixtures, sessionId]);

  const refreshHistory = useCallback(() => {
    if (sending || loading || refreshing) return;
    setRefreshing(true);
    setHistoryNonce((value) => value + 1);
  }, [loading, refreshing, sending]);

  const getTurnIndex = useCallback(
    (messageIndex: number) =>
      turns.slice(0, messageIndex + 1).filter((turn) => turn.role === 'assistant').length - 1,
    [turns],
  );

  const deleteTurn = useCallback(
    async (turnIndex: number) => {
      if (sending || deletingTurn !== null) return;
      setConfirmDeleteTurn(null);
      setDeletingTurn(turnIndex);
      try {
        if (!usingFixtures) {
          await api.deleteConversationTurn(sessionId, turnIndex);
        }
        setTurns((previous) => removeTurnPair(previous, turnIndex));
        pushToast(
          usingFixtures ? '演示会话已移除这一轮。' : '已删除这一轮会话历史。',
          'ok',
        );
      } catch (err) {
        pushToast(`删除失败：${api.describeError(err)}`, 'danger');
      } finally {
        setDeletingTurn(null);
      }
    },
    [deletingTurn, sending, sessionId, usingFixtures],
  );

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Enter 发送，Shift+Enter 换行；输入法组合中的 Enter 不能拦。
      if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
        event.preventDefault();
        void send();
      }
    },
    [send],
  );

  const canSend = draft.trim().length > 0 && !sending;
  const empty = !loading && !loadError && turns.length === 0;

  const composerHint = useMemo(() => {
    if (sending) return '生成中，可按停止中断';
    if (uploading) return '正在上传附件…';
    if (attachments.length > 0) return `已附 ${attachments.length} 个文件 · Enter 发送`;
    return 'Enter 发送 · Shift+Enter 换行 · 附件仅支持文本类文件';
  }, [sending, uploading, attachments.length]);

  return (
    <div className="chat">
      {/* 装饰背景：哥特建筑 + 啮合齿轮钟。纯装饰，离开视口即停（§8） */}
      <GothicWorksCanvas className="chat__bg" />

      <header className="chat__head">
        <span className="chat__arch" aria-hidden="true" />
        <div className="chat__headtext">
          <p className="chat__eyebrow mono-label">SESSION / {sessionId}</p>
          <h2 className="chat__title">对话</h2>
        </div>
        <div className="chat__headtools">
          <label className="chat__toggle">
            <input
              type="checkbox"
              checked={showThinking}
              onChange={(e) => setShowThinking(e.target.checked)}
            />
            <span>思考过程</span>
          </label>
          <CommandButton
            variant="ghost"
            size="sm"
            icon={Eraser}
            onClick={() => void clear()}
            disabled={sending || turns.length === 0}
            ariaLabel="清空当前会话历史"
          >
            清空
          </CommandButton>
          <CommandButton
            variant="ghost"
            size="sm"
            icon={RefreshCw}
            busy={refreshing}
            onClick={refreshHistory}
            disabled={sending || loading || refreshing}
            ariaLabel="刷新会话历史"
          >
            刷新
          </CommandButton>
        </div>
      </header>

      <div className="chat__list" ref={listRef} onScroll={onScroll}>
        {loading ? (
          <EmptyState kind="loading" title="正在读取会话历史" description="从主节点 SQLite 加载。" compact />
        ) : null}

        {loadError ? (
          <EmptyState
            kind="error"
            title="会话历史加载失败"
            description="后端可能未启动，或当前节点不是主节点。"
            detail={loadError}
            errorKind={loadErrorKind}
            errorStatus={loadErrorStatus}
            compact
          />
        ) : null}

        {empty ? (
          <div className="chat__opening">
            <p className="chat__dropcap">在此开始一轮推理。</p>
            <p className="chat__openingnote">
              左侧是集群实况：发出请求后，队列深度、流水线准入与运行日志会同步变化。
            </p>
          </div>
        ) : null}

        <ol className="chat__turns">
          {turns.map((turn, messageIndex) => {
            const metricsText = turn.role === 'assistant' ? formatMetrics(turn) : '';
            const thinkingOpen = openThinking[turn.id] === true;
            const turnIndex = turn.role === 'assistant' ? getTurnIndex(messageIndex) : -1;
            const canDelete = turn.role === 'assistant' && !turn.streaming && turnIndex >= 0;
            const confirmingDelete = confirmDeleteTurn === turnIndex;
            return (
              <li key={turn.id} className={`msg msg--${turn.role}`}>
                <div className="msg__gutter" aria-hidden="true">
                  <span className="msg__rule" />
                </div>
                <div className="msg__body">
                  <p className="msg__who mono-label">
                    {turn.role === 'user' ? '你' : '模型'}
                    {turn.streaming ? <span className="msg__live"> 生成中</span> : null}
                  </p>

                  {turn.thinking ? (
                    <div className="msg__thinking">
                      <button
                        type="button"
                        className="msg__thinkingtoggle"
                        aria-expanded={thinkingOpen}
                        onClick={() =>
                          setOpenThinking((prev) => ({ ...prev, [turn.id]: !thinkingOpen }))
                        }
                      >
                        {thinkingOpen ? '收起思考过程' : '展开思考过程'}
                      </button>
                      {thinkingOpen ? <pre className="msg__thinkingbody">{turn.thinking}</pre> : null}
                    </div>
                  ) : null}

                  <div className="msg__text">
                    {turn.content}
                    {turn.streaming ? <span className="msg__caret" aria-hidden="true" /> : null}
                  </div>

                  {turn.attachments && turn.attachments.length > 0 ? (
                    <ul className="msg__atts">
                      {turn.attachments.map((a) => (
                        <li key={a.filename} className="chip chip--static">
                          <Paperclip size={12} aria-hidden="true" />
                          <span className="chip__name">{a.filename}</span>
                          <span className="chip__meta mono-label">{a.lineCount} 行</span>
                        </li>
                      ))}
                    </ul>
                  ) : null}

                  {turn.error ? <p className="msg__error">{turn.error}</p> : null}
                  {metricsText ? <p className="msg__metrics mono-label">{metricsText}</p> : null}
                  {canDelete ? (
                    <div className="msg__actions">
                      {confirmingDelete ? (
                        <>
                          <span className="msg__confirm-label">删除这一轮？</span>
                          <button
                            type="button"
                            className="msg__action msg__action--danger"
                            onClick={() => void deleteTurn(turnIndex)}
                            disabled={deletingTurn !== null}
                          >
                            {deletingTurn === turnIndex ? '删除中…' : '确认'}
                          </button>
                          <button
                            type="button"
                            className="msg__action"
                            onClick={() => setConfirmDeleteTurn(null)}
                            disabled={deletingTurn !== null}
                          >
                            取消
                          </button>
                        </>
                      ) : (
                        <button
                          type="button"
                          className="msg__delete"
                          onClick={() => setConfirmDeleteTurn(turnIndex)}
                          disabled={deletingTurn !== null || sending}
                          title="删除这一轮对话"
                          aria-label="删除这一轮对话"
                        >
                          <Trash2 size={13} strokeWidth={2.1} aria-hidden="true" />
                        </button>
                      )}
                    </div>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>

        {/* 流式追加靠这里播报，不用把整个列表设成 live 区 */}
        <p className="sr-only" aria-live="polite">
          {sending ? '正在生成回复' : ''}
        </p>
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <label className="sr-only" htmlFor="chat-input">
          输入消息
        </label>

        {attachments.length > 0 ? (
          <ul className="composer__atts">
            {attachments.map((a) => (
              <li key={a.id} className="chip">
                <Paperclip size={12} aria-hidden="true" />
                <span className="chip__name">{a.filename}</span>
                <span className="chip__meta mono-label">
                  {a.lineCount} 行 · {formatBytes(a.sizeBytes)}
                  {a.truncated ? ' · 已截断' : ''}
                </span>
                <button
                  type="button"
                  className="chip__x"
                  onClick={() => removeAttachment(a.id)}
                  aria-label={`移除附件 ${a.filename}`}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        ) : null}

        <textarea
          id="chat-input"
          ref={textareaRef}
          className="composer__input"
          value={draft}
          rows={3}
          placeholder={usingFixtures ? '演示数据模式下不可发送' : '写下你的问题…'}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer__bar">
          {/* 真实 file input，视觉上隐藏但仍可 Tab 聚焦（§5.5：不用 div 假冒控件） */}
          <input
            ref={fileRef}
            id="chat-file"
            className="sr-only"
            type="file"
            multiple
            accept={ACCEPT}
            onChange={(e) => void pickFiles(e.target.files)}
          />
          <CommandButton
            variant="ghost"
            size="sm"
            icon={uploading ? Loader2 : Paperclip}
            busy={uploading}
            disabled={uploading || sending}
            onClick={() => fileRef.current?.click()}
            ariaLabel="添加文本附件"
          >
            附件
          </CommandButton>
          <span className="composer__hint">{composerHint}</span>
          <div className="composer__actions">
            {sending ? (
              <CommandButton variant="danger" size="sm" icon={Square} onClick={stop}>
                停止
              </CommandButton>
            ) : null}
            <CommandButton
              type="submit"
              size="sm"
              icon={sending ? Loader2 : SendHorizontal}
              disabled={!canSend}
              busy={sending}
            >
              发送
            </CommandButton>
          </div>
        </div>
      </form>
    </div>
  );
}
