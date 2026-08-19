import { useCallback, useEffect, useState } from 'react';
import { Database, RefreshCw, Search } from 'lucide-react';
import { fetchRagHealth, rebuildRagIndex, searchRag } from '../api/client';

function formatCount(value) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : '—';
}

export default function RagPanel({ onToast }) {
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const loadHealth = useCallback(async () => {
    try {
      const value = await fetchRagHealth();
      setHealth(value);
      setError('');
    } catch (reason) {
      setError(reason.message || '知识库状态不可用');
    }
  }, []);

  useEffect(() => { loadHealth(); }, [loadHealth]);

  const runSearch = async (event) => {
    event?.preventDefault();
    const text = query.trim();
    if (!text) return;
    setBusy(true);
    try {
      const response = await searchRag(text, { mode: 'fts', access_scope: 'owner', limit: 20 });
      setResults(response.results || []);
      setError('');
    } catch (reason) {
      setError(reason.message || '检索失败');
      onToast?.({ type: 'error', msg: `知识库检索失败: ${reason.message}` });
    } finally {
      setBusy(false);
    }
  };

  const rebuild = async () => {
    setBusy(true);
    try {
      await rebuildRagIndex();
      await loadHealth();
      onToast?.({ type: 'success', msg: '知识库 FTS5 索引已重建。' });
    } catch (reason) {
      setError(reason.message || '索引重建失败');
      onToast?.({ type: 'error', msg: `索引重建失败: ${reason.message}` });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rag-panel" aria-label="本地知识库">
      <div className="rag-panel-heading">
        <div>
          <span className="workspace-kicker">LOCAL RAG</span>
          <h3>本地知识库</h3>
        </div>
        <div className="rag-panel-actions">
          <button type="button" className="setting-btn secondary" onClick={loadHealth} title="刷新知识库状态" aria-label="刷新知识库状态">
            <RefreshCw size={15} aria-hidden="true" />
            刷新
          </button>
          <button type="button" className="setting-btn secondary" onClick={rebuild} disabled={busy} title="重建 FTS5 索引">
            <Database size={15} aria-hidden="true" />
            {busy ? '处理中…' : '重建索引'}
          </button>
        </div>
      </div>
      <div className="rag-health-grid" role="status">
        <span>来源 {formatCount(health?.source_count)}</span>
        <span>文档 {formatCount(health?.document_count)}</span>
        <span>片段 {formatCount(health?.chunk_count)}</span>
        <span>向量 {formatCount(health?.embedding_count)}</span>
        <span>{health?.journal_mode === 'wal' ? 'WAL' : '存储检查中'}</span>
      </div>
      <form className="rag-search-form" onSubmit={runSearch}>
        <Search size={16} aria-hidden="true" />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="检索本机资料、模型卡和日志摘要" maxLength={512} aria-label="知识库检索" />
        <button type="submit" className="setting-btn" disabled={busy || !query.trim()}>检索</button>
      </form>
      {error && <div className="rag-panel-error" role="alert">{error}</div>}
      <div className="rag-results" aria-live="polite">
        {results.map((item) => (
          <article className="rag-result" key={item.chunk_id}>
            <div className="rag-result-meta">
              <strong>{item.source_id}</strong>
              <span>{item.relative_ref} · r{item.revision} · #{item.ordinal}</span>
            </div>
            <p>{item.snippet}</p>
          </article>
        ))}
        {query.trim() && !busy && results.length === 0 && !error && <p className="rag-empty">没有匹配的本地资料。</p>}
      </div>
    </section>
  );
}
