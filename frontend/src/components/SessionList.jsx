import { useState, useRef, useEffect, useCallback } from 'react';

export default function SessionList({
  sessions, activeSessionId, onSelectSession,
  onCreateSession, onDeleteSession, onRenameSession,
}) {
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState('');
  const renameInputRef = useRef(null);

  // Focus rename input when entering rename mode
  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  // Close confirm on Escape
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') {
        setConfirmDeleteId(null);
        setRenamingId(null);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const handleDoubleClick = useCallback((session) => {
    setRenamingId(session.id);
    setRenameValue(session.title || '新对话');
  }, []);

  const handleRenameSubmit = useCallback((sessionId) => {
    const trimmed = renameValue.trim();
    if (trimmed && onRenameSession) {
      onRenameSession(sessionId, trimmed);
    }
    setRenamingId(null);
    setRenameValue('');
  }, [renameValue, onRenameSession]);

  const handleRenameKeyDown = useCallback((e, sessionId) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleRenameSubmit(sessionId);
    }
  }, [handleRenameSubmit]);

  const handleDeleteClick = useCallback((e, sessionId) => {
    e.stopPropagation();
    setConfirmDeleteId(sessionId);
  }, []);

  const handleDeleteConfirm = useCallback((e, sessionId) => {
    e.stopPropagation();
    if (onDeleteSession) onDeleteSession(sessionId);
    setConfirmDeleteId(null);
  }, [onDeleteSession]);

  const handleDeleteCancel = useCallback((e) => {
    e.stopPropagation();
    setConfirmDeleteId(null);
  }, []);

  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    try {
      const d = new Date(dateStr);
      const now = new Date();
      const diffMs = now - d;
      const diffHrs = diffMs / (1000 * 60 * 60);
      if (diffHrs < 1) return '刚刚';
      if (diffHrs < 24) return `${Math.floor(diffHrs)}小时前`;
      if (diffHrs < 48) return '昨天';
      if (diffHrs < 168) return `${Math.floor(diffHrs / 24)}天前`;
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    } catch {
      return '';
    }
  };

  return (
    <div className="session-list">
      <div className="session-list-header">
        <span className="session-list-title">对话列表</span>
        <button
          className="session-new-btn"
          onClick={() => onCreateSession()}
          title="新建对话"
          disabled={sessions.length === 0}
        >
          +
        </button>
      </div>
      <div className="session-list-items">
        {sessions.length === 0 ? (
          <div className="session-list-empty">
            暂无对话，发送第一条消息自动创建
          </div>
        ) : (
          sessions.map((s) => {
            const isActive = s.id === activeSessionId;
            const isConfirming = s.id === confirmDeleteId;
            const isRenaming = s.id === renamingId;

            return (
              <div
                key={s.id}
                className={`session-item${isActive ? ' active' : ''}`}
                onClick={() => onSelectSession?.(s.id)}
              >
                {isRenaming ? (
                  <input
                    ref={renameInputRef}
                    className="session-item-rename-input"
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onBlur={() => handleRenameSubmit(s.id)}
                    onKeyDown={(e) => handleRenameKeyDown(e, s.id)}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <span
                    className="session-item-title"
                    title={s.title}
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      handleDoubleClick(s);
                    }}
                  >
                    {s.title || '新对话'}
                  </span>
                )}
                <span className="session-item-meta">
                  {s.message_count != null ? `${s.message_count} 条` : ''}
                  {s.message_count != null && s.updated_at ? ' · ' : ''}
                  {formatDate(s.updated_at || s.created_at)}
                </span>

                {isConfirming ? (
                  <div className="session-delete-confirm">
                    <button
                      className="session-delete-btn confirm"
                      onClick={(e) => handleDeleteConfirm(e, s.id)}
                    >
                      删除
                    </button>
                    <button
                      className="session-delete-btn cancel"
                      onClick={handleDeleteCancel}
                    >
                      取消
                    </button>
                  </div>
                ) : (
                  <button
                    className="session-item-delete"
                    onClick={(e) => handleDeleteClick(e, s.id)}
                    title="删除对话"
                  >
                    ✕
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
