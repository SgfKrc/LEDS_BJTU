import { useEffect, useMemo, useState, type FormEvent } from 'react';
import {
  Archive,
  Bot,
  BrainCircuit,
  ChevronRight,
  DatabaseSearch,
  Image,
  Menu,
  MessageSquareText,
  Moon,
  Network,
  Paperclip,
  PanelLeft,
  Plus,
  RefreshCw,
  Search,
  Send,
  Settings2,
  Sun,
  TerminalSquare,
  UserRound,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import { checkHealth, completeChat, searchRag, type ChatMessage, type ConnectionState, type RagHit } from './data';

type ViewId = 'chat' | 'rag' | 'assets' | 'runtime';
type Theme = 'dark' | 'light';

const fixtureMessages: ChatMessage[] = [
  { id: 'fixture-1', role: 'assistant', content: '工作台已就绪。连接一个本地或 QLH 适配器后，我会把能力、会话和检索状态放在同一条工作流里。', meta: 'fixture / ready' },
];

const navItems: { id: ViewId; label: string; icon: typeof MessageSquareText }[] = [
  { id: 'chat', label: '对话', icon: MessageSquareText },
  { id: 'rag', label: '知识库', icon: DatabaseSearch },
  { id: 'assets', label: '资产', icon: Image },
  { id: 'runtime', label: '运行时', icon: BrainCircuit },
];

function App() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('harness-theme') as Theme) || 'dark');
  const [view, setView] = useState<ViewId>('chat');
  const [connection, setConnection] = useState<ConnectionState>('checking');
  const [backend, setBackend] = useState('checking');
  const [messages, setMessages] = useState<ChatMessage[]>(fixtureMessages);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [ragQuery, setRagQuery] = useState('');
  const [ragHits, setRagHits] = useState<RagHit[]>([]);
  const [ragNotice, setRagNotice] = useState('输入查询，结果会保留来源与 chunk 引用。');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('harness-theme', theme);
  }, [theme]);

  const refresh = async () => {
    setConnection('checking');
    try {
      const health = await checkHealth();
      setConnection('online');
      setBackend(health.backend || 'harness api');
    } catch {
      setConnection('fixture');
      setBackend('offline fixture');
    }
  };

  useEffect(() => { void refresh(); }, []);

  const statusLabel = connection === 'online' ? 'ONLINE' : connection === 'fixture' ? 'FIXTURE' : connection === 'offline' ? 'OFFLINE' : 'CHECKING';
  const statusIcon = connection === 'online' ? <Wifi size={14} /> : <WifiOff size={14} />;

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    const userMessage: ChatMessage = { id: `user-${Date.now()}`, role: 'user', content: text };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput('');
    setSending(true);
    try {
      if (connection !== 'online') {
        setMessages([...nextMessages, { id: `fixture-${Date.now()}`, role: 'assistant', content: `这是离线 fixture 回应：已接收“${text}”。连接 API 后会替换为真实模型输出。`, meta: 'fixture / no network' }]);
      } else {
        const response = await completeChat('harness-default', nextMessages);
        setMessages([...nextMessages, { id: `assistant-${Date.now()}`, role: 'assistant', content: response.content, meta: backend }]);
      }
    } catch (error) {
      setMessages([...nextMessages, { id: `error-${Date.now()}`, role: 'system', content: `请求失败：${error instanceof Error ? error.message : 'unknown error'}`, meta: 'request failed' }]);
      setConnection('offline');
    } finally {
      setSending(false);
    }
  };

  const runSearch = async (event: FormEvent) => {
    event.preventDefault();
    const query = ragQuery.trim();
    if (!query) return;
    setRagNotice('检索中…');
    try {
      const response = await searchRag(query);
      setRagHits(response.hits);
      setRagNotice(response.context.omitted_count ? `已返回 ${response.hits.length} 条，另有 ${response.context.omitted_count} 条因预算省略。` : `已返回 ${response.hits.length} 条引用。`);
    } catch {
      setRagHits([]);
      setRagNotice('RAG API 不可用；当前只显示离线状态，不伪造检索结果。');
    }
  };

  const viewTitle = useMemo(() => navItems.find((item) => item.id === view)?.label || '工作台', [view]);

  return (
    <div className="app-shell" data-connection={connection}>
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><Bot size={22} /></div>
          <div><strong>QLH</strong><span>HARNESS WORKBENCH</span></div>
        </div>
        <div className="topbar-context"><span className="eyebrow">S6 / UI-01</span><span>{viewTitle}</span></div>
        <div className="topbar-actions">
          <span className={`connection connection--${connection}`}><span className="connection-dot" />{statusIcon}{statusLabel}</span>
          <button className="icon-button" type="button" onClick={() => void refresh()} title="刷新连接状态" aria-label="刷新连接状态"><RefreshCw size={17} /></button>
          <button className="icon-button" type="button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title={theme === 'dark' ? '切换浅色模式' : '切换深色模式'} aria-label="切换主题">{theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}</button>
          <button className="icon-button mobile-menu" type="button" onClick={() => setSidebarOpen(!sidebarOpen)} title="打开导航" aria-label="打开导航"><Menu size={18} /></button>
        </div>
      </header>

      <div className="workspace-grid">
        <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
          <div className="sidebar-head"><span className="eyebrow">WORKSPACE</span><button className="icon-button sidebar-close" type="button" onClick={() => setSidebarOpen(false)} aria-label="关闭导航"><X size={16} /></button></div>
          <nav className="main-nav" aria-label="工作台导航">
            {navItems.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${view === id ? 'nav-item--active' : ''}`} type="button" onClick={() => { setView(id); setSidebarOpen(false); }}><Icon size={17} /><span>{label}</span><ChevronRight size={14} className="nav-chevron" /></button>)}
          </nav>
          <div className="sidebar-rule" />
          <div className="session-head"><span className="eyebrow">SESSIONS</span><button className="icon-button" type="button" title="新建会话" aria-label="新建会话" onClick={() => setMessages(fixtureMessages)}><Plus size={17} /></button></div>
          <button className="session-item session-item--active" type="button"><span className="session-pulse" /><span><strong>默认工作区</strong><small>刚刚活动</small></span></button>
          <button className="session-item" type="button"><span className="session-pulse session-pulse--dim" /><span><strong>未命名会话</strong><small>fixture 示例</small></span></button>
          <div className="sidebar-foot"><span className="eyebrow">BACKEND</span><span className="backend-name">{backend}</span></div>
        </aside>

        <main className="main-stage">
          <div className="stage-line" />
          {view === 'chat' && <ChatView messages={messages} input={input} setInput={setInput} sending={sending} onSubmit={sendMessage} />}
          {view === 'rag' && <RagView query={ragQuery} setQuery={setRagQuery} hits={ragHits} notice={ragNotice} onSubmit={runSearch} />}
          {view === 'assets' && <AssetView />}
          {view === 'runtime' && <RuntimeView connection={connection} backend={backend} onRefresh={() => void refresh()} />}
        </main>
      </div>
    </div>
  );
}

function ChatView({ messages, input, setInput, sending, onSubmit }: { messages: ChatMessage[]; input: string; setInput: (value: string) => void; sending: boolean; onSubmit: (event: FormEvent) => void }) {
  return <section className="chat-view" aria-labelledby="chat-title">
    <div className="view-heading"><div><span className="eyebrow">CONVERSATION / DEFAULT</span><h1 id="chat-title">小模型工作台</h1><p>把上下文、资产和后端能力放在同一条可审计的工作流里。</p></div><div className="view-heading-mark"><MessageSquareText size={28} /></div></div>
    <div className="message-list" aria-live="polite">
      {messages.map((message) => <article className={`message message--${message.role}`} key={message.id}><div className="message-avatar" aria-hidden="true">{message.role === 'user' ? <UserRound size={16} /> : message.role === 'assistant' ? <Bot size={16} /> : <TerminalSquare size={16} />}</div><div className="message-body"><div className="message-meta"><span>{message.role === 'user' ? 'YOU' : message.role === 'assistant' ? 'HARNESS' : 'SYSTEM'}</span>{message.meta && <small>{message.meta}</small>}</div><p>{message.content}</p></div></article>)}
      {sending && <div className="typing"><span /><span /><span /> adapter 正在响应</div>}
    </div>
    <form className="composer" onSubmit={onSubmit}><div className="composer-tools"><button className="icon-button" type="button" title="添加资产" aria-label="添加资产"><Paperclip size={17} /></button><span className="composer-hint">Markdown / fixture-safe</span></div><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入一条消息…" rows={3} /><div className="composer-submit"><span>ENTER 发送</span><button className="send-button" type="submit" disabled={sending || !input.trim()} title="发送消息" aria-label="发送消息"><Send size={18} /></button></div></form>
  </section>;
}

function RagView({ query, setQuery, hits, notice, onSubmit }: { query: string; setQuery: (value: string) => void; hits: RagHit[]; notice: string; onSubmit: (event: FormEvent) => void }) {
  return <section className="utility-view" aria-labelledby="rag-title"><div className="view-heading"><div><span className="eyebrow">RETRIEVAL / FTS5 FIRST</span><h1 id="rag-title">知识库检索</h1><p>引用保持来源、chunk 与预算信息；没有 provider 时安全回退到 FTS。</p></div><DatabaseSearch size={30} className="heading-icon" /></div><form className="search-form" onSubmit={onSubmit}><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索本地资料…" /><button className="action-button" type="submit">检索</button></form><p className="utility-notice">{notice}</p><div className="result-list">{hits.length === 0 ? <EmptyUtility icon={<Archive size={24} />} title="尚无检索结果" detail="输入查询，或先通过 API 写入用户-owned 资料。" /> : hits.map((hit) => <article className="result-row" key={hit.chunk_id}><div><span className="eyebrow">{hit.title} / CHUNK {hit.ordinal}</span><p>{hit.text}</p></div><code>{hit.chunk_id}</code></article>)}</div></section>;
}

function AssetView() { return <section className="utility-view" aria-labelledby="assets-title"><div className="view-heading"><div><span className="eyebrow">ASSETS / USER OWNED</span><h1 id="assets-title">会话资产</h1><p>图片、文档与模型输出只归用户指定的资产目录管理。</p></div><Image size={30} className="heading-icon" /></div><EmptyUtility icon={<PanelLeft size={24} />} title="资产抽屉已就绪" detail="完成一次生图或附件上传后，资产引用会出现在这里。" /><div className="asset-policy"><Settings2 size={17} /><span>当前票只提供入口与状态合同；真实 blob、缩略图和编辑动作进入 HARNESS-UI-03。</span></div></section>; }

function RuntimeView({ connection, backend, onRefresh }: { connection: ConnectionState; backend: string; onRefresh: () => void }) { return <section className="utility-view" aria-labelledby="runtime-title"><div className="view-heading"><div><span className="eyebrow">RUNTIME / CAPABILITIES</span><h1 id="runtime-title">运行时能力</h1><p>能力来源于 adapter 探测，不根据 UI 假设补齐。</p></div><Network size={30} className="heading-icon" /></div><div className="runtime-grid"><div className="runtime-cell"><span className="eyebrow">CONNECTION</span><strong>{connection.toUpperCase()}</strong><small>API healthz</small></div><div className="runtime-cell"><span className="eyebrow">ADAPTER</span><strong>{backend}</strong><small>当前后端</small></div><div className="runtime-cell"><span className="eyebrow">IMAGES</span><strong>PROBE</strong><small>由 image adapter 决定</small></div><div className="runtime-cell"><span className="eyebrow">RAG</span><strong>FTS5</strong><small>embedding 可选</small></div></div><button className="action-button action-button--quiet" type="button" onClick={onRefresh}><RefreshCw size={16} />重新探测</button></section>; }

function EmptyUtility({ icon, title, detail }: { icon: React.ReactNode; title: string; detail: string }) { return <div className="empty-utility"><div className="empty-icon">{icon}</div><strong>{title}</strong><p>{detail}</p></div>; }

export default App;
