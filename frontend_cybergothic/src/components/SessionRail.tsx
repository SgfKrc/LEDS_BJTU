import { Check, MessageSquare, Pencil, Plus, Trash2, X } from 'lucide-react';
import { useState } from 'react';
import type { SessionSummary } from '../data/types';
import { CommandButton } from './CommandButton';

interface SessionRailProps {
  sessions: SessionSummary[];
  activeId: string;
  loading?: boolean;
  busy?: string;
  onCreate: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

function formatSessionDate(value: string): string {
  if (!value) return '尚无消息';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function SessionRail({
  sessions,
  activeId,
  loading = false,
  busy = '',
  onCreate,
  onSelect,
  onRename,
  onDelete,
}: SessionRailProps) {
  const [editingId, setEditingId] = useState('');
  const [draftTitle, setDraftTitle] = useState('');

  const beginRename = (session: SessionSummary) => {
    setEditingId(session.id);
    setDraftTitle(session.title || '新对话');
  };

  const cancelRename = () => {
    setEditingId('');
    setDraftTitle('');
  };

  const commitRename = async () => {
    const title = draftTitle.trim();
    if (!editingId || !title) return;
    await onRename(editingId, title);
    cancelRename();
  };

  const confirmDelete = async (session: SessionSummary) => {
    if (typeof window !== 'undefined' && !window.confirm(`确认删除会话“${session.title || '新对话'}”？`)) {
      return;
    }
    await onDelete(session.id);
  };

  return (
    <aside className="chat-sessions" aria-label="对话会话">
      <div className="chat-sessions__head">
        <div>
          <p className="chat-sessions__tag mono-label">ARCHIVE</p>
          <h2 className="chat-sessions__title">会话</h2>
        </div>
        <CommandButton
          variant="ghost"
          size="sm"
          icon={Plus}
          onClick={onCreate}
          busy={busy === 'create'}
          ariaLabel="新建会话"
        >
          新建
        </CommandButton>
      </div>

      {loading && sessions.length === 0 ? (
        <p className="chat-sessions__empty">正在读取会话…</p>
      ) : sessions.length === 0 ? (
        <div className="chat-sessions__empty">
          <MessageSquare size={18} aria-hidden="true" />
          <span>还没有会话</span>
        </div>
      ) : (
        <ul className="chat-sessions__list">
          {sessions.map((session) => {
            const isActive = session.id === activeId;
            const isEditing = session.id === editingId;
            const rowBusy = busy === session.id;
            return (
              <li className="chat-sessions__row" key={session.id} data-active={isActive ? 'true' : undefined}>
                {isEditing ? (
                  <div className="chat-sessions__edit">
                    <input
                      className="chat-sessions__input"
                      value={draftTitle}
                      maxLength={256}
                      autoFocus
                      aria-label="会话名称"
                      onChange={(event) => setDraftTitle(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault();
                          void commitRename();
                        }
                        if (event.key === 'Escape') cancelRename();
                      }}
                    />
                    <button
                      type="button"
                      className="chat-sessions__icon"
                      aria-label="保存会话名称"
                      disabled={rowBusy || !draftTitle.trim()}
                      onClick={() => void commitRename()}
                    >
                      <Check size={14} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="chat-sessions__icon"
                      aria-label="取消重命名"
                      onClick={cancelRename}
                    >
                      <X size={14} aria-hidden="true" />
                    </button>
                  </div>
                ) : (
                  <>
                    <button
                      type="button"
                      className="chat-sessions__select"
                      aria-current={isActive ? 'page' : undefined}
                      disabled={Boolean(busy)}
                      onClick={() => onSelect(session.id)}
                    >
                      <span className="chat-sessions__name">{session.title || '新对话'}</span>
                      <span className="chat-sessions__meta mono-label">
                        {session.message_count} 条 · {formatSessionDate(session.updated_at)}
                      </span>
                    </button>
                    <div className="chat-sessions__actions">
                      <button
                        type="button"
                        className="chat-sessions__icon"
                        aria-label={`重命名 ${session.title || '新对话'}`}
                        disabled={Boolean(busy)}
                        onClick={() => beginRename(session)}
                      >
                        <Pencil size={13} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        className="chat-sessions__icon chat-sessions__icon--danger"
                        aria-label={`删除 ${session.title || '新对话'}`}
                        disabled={Boolean(busy)}
                        onClick={() => void confirmDelete(session)}
                      >
                        <Trash2 size={13} aria-hidden="true" />
                      </button>
                    </div>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
