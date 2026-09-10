import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  Archive,
  AlertTriangle,
  Braces,
  Bot,
  BrainCircuit,
  Cable,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  DatabaseSearch,
  ExternalLink,
  Image,
  Menu,
  MessageSquareText,
  Moon,
  Network,
  Paperclip,
  PanelLeft,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  Square,
  Sun,
  TerminalSquare,
  UserRound,
  Wifi,
  WifiOff,
  Workflow,
  X,
} from 'lucide-react';
import { apiUrl, appendSessionMessage, attachSessionAsset, callMcpTool, checkHealth, createSession, generateImage, getImageCapabilities, getMcpManifest, getRagHealth, getSession, HarnessApiError, listModelDownloads, listModelPresets, listModelProfiles, listModels, listSessions, loadModelAsset, queueModelDownload, searchRag, streamChat, type ChatMessage, type ConnectionState, type HarnessModel, type HarnessModelDownload, type HarnessModelPreset, type ImageCapabilities, type ImageResult, type MCPCallResponse, type MCPManifest, type MCPTool, type ModelProfileSummary, type RagContext, type RagHit, type SessionSummary } from './data';

type ViewId = 'chat' | 'rag' | 'assets' | 'mcp' | 'runtime';
type Theme = 'dark' | 'light';

type ImageAsset = ImageResult & { prompt: string; createdAt: number };

const fixtureMessages: ChatMessage[] = [
  { id: 'fixture-1', role: 'assistant', content: '工作台已就绪。连接一个本地或 QLH 适配器后，我会把能力、会话和检索状态放在同一条工作流里。', meta: 'fixture / ready' },
];

const navItems: { id: ViewId; label: string; icon: typeof MessageSquareText }[] = [
  { id: 'chat', label: '对话', icon: MessageSquareText },
  { id: 'rag', label: '知识库', icon: DatabaseSearch },
  { id: 'assets', label: '资产', icon: Image },
  { id: 'mcp', label: 'MCP 控制面', icon: Cable },
  { id: 'runtime', label: '运行时', icon: BrainCircuit },
];

function describeHarnessError(error: unknown, fallback: string): string {
  if (error instanceof HarnessApiError) {
    const code = error.code ? ` [${error.code}]` : '';
    const retry = error.retryable ? ' · 可重试' : '';
    return `${error.message}${code} · HTTP ${error.status}${retry}`;
  }
  return error instanceof Error ? error.message : fallback;
}

function readMcpError(response: MCPCallResponse): { code?: number | string; message: string; retryable?: boolean } | null {
  if (response.error) return response.error;
  if (!response.result?.isError) return null;
  const text = response.result.content?.find((item) => item.type === 'text')?.text;
  if (!text) return { message: 'MCP 工具返回受控错误' };
  try {
    const payload = JSON.parse(text) as { error?: { code?: string; message?: string; retryable?: boolean } };
    if (payload.error?.message) {
      return {
        code: payload.error.code,
        message: payload.error.message,
        retryable: payload.error.retryable,
      };
    }
  } catch {
    // Keep the full raw response available below when a non-JSON MCP server fails.
  }
  return { message: 'MCP 工具返回受控错误' };
}

function App() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('harness-theme') as Theme) || 'dark');
  const [view, setView] = useState<ViewId>('chat');
  const [connection, setConnection] = useState<ConnectionState>('checking');
  const [backend, setBackend] = useState('checking');
  const [modelCatalog, setModelCatalog] = useState<HarnessModel[]>([]);
  const [modelProfiles, setModelProfiles] = useState<ModelProfileSummary[]>([]);
  const [modelPresets, setModelPresets] = useState<HarnessModelPreset[]>([]);
  const [modelDownloads, setModelDownloads] = useState<HarnessModelDownload[]>([]);
  const [modelBusy, setModelBusy] = useState('');
  const [modelNotice, setModelNotice] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>(fixtureMessages);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [ragQuery, setRagQuery] = useState('');
  const [ragHits, setRagHits] = useState<RagHit[]>([]);
  const [ragContext, setRagContext] = useState<RagContext | null>(null);
  const [ragNotice, setRagNotice] = useState('输入查询，结果会保留来源与 chunk 引用。');
  const [ragHealth, setRagHealth] = useState('checking');
  const [imageCapabilities, setImageCapabilities] = useState<ImageCapabilities | null>(null);
  const [imageAssets, setImageAssets] = useState<ImageAsset[]>([]);
  const [imagePrompt, setImagePrompt] = useState('');
  const [imageSize, setImageSize] = useState('512x512');
  const [imageSteps, setImageSteps] = useState('28');
  const [imageSeed, setImageSeed] = useState('');
  const [imageBusy, setImageBusy] = useState(false);
  const [imageNotice, setImageNotice] = useState('先探测图像 adapter，再决定是否允许生成。');
  const [mcpManifest, setMcpManifest] = useState<MCPManifest | null>(null);
  const [mcpToolName, setMcpToolName] = useState('rag_search');
  const [mcpArguments, setMcpArguments] = useState('{\n  "query": "registry"\n}');
  const [mcpResult, setMcpResult] = useState<MCPCallResponse | null>(null);
  const [mcpBusy, setMcpBusy] = useState(false);
  const [mcpNotice, setMcpNotice] = useState('等待 MCP manifest。');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('harness-theme', theme);
  }, [theme]);

  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true });
  }, [view]);

  const loadSession = async (sessionId: string) => {
    const detail = await getSession(sessionId);
    setActiveSessionId(sessionId);
    setMessages(detail.messages.map((message) => ({
      id: message.message_id,
      role: message.role === 'tool' ? 'system' : message.role,
      content: message.content,
      meta: message.metadata?.source ? String(message.metadata.source) : undefined,
    })));
  };

  const refreshUtilityCapabilities = async () => {
    try {
      const health = await getRagHealth();
      setRagHealth(`${String(health.backend || 'sqlite_fts5')} · ${String(health.chunks || 0)} chunks`);
    } catch (error) {
      setRagHealth(`unavailable · ${error instanceof Error ? error.message : 'RAG API unavailable'}`);
    }
    try {
      const capabilities = await getImageCapabilities();
      setImageCapabilities(capabilities);
      setImageNotice(capabilities.runtime_available && capabilities.supports_txt2img ? `${capabilities.backend} · txt2img ready` : `${capabilities.backend} 已配置，但运行时或 txt2img 未准入`);
    } catch (error) {
      setImageCapabilities(null);
      setImageNotice(`图像能力不可用：${error instanceof Error ? error.message : 'image API unavailable'}`);
    }
  };

  const refreshMcp = async () => {
    try {
      const manifest = await getMcpManifest();
      setMcpManifest(manifest);
      setMcpToolName((current) => manifest.tools.some((tool) => tool.name === current) ? current : manifest.tools.find((tool) => tool.name === 'rag_search')?.name || manifest.tools[0]?.name || '');
      setMcpNotice(`${manifest.tools.length} tools · ${manifest.transports.sse.mode} SSE · external MCP ${manifest.external_mcp.configuration_only ? 'configuration only' : 'enabled'}`);
    } catch (error) {
      setMcpManifest(null);
      setMcpResult(null);
      setMcpNotice(`MCP API 不可用：${error instanceof Error ? error.message : 'manifest unavailable'}`);
    }
  };

  const refresh = async () => {
    setConnection('checking');
    try {
      const health = await checkHealth();
      setConnection('online');
      setBackend(health.backend || 'harness api');
    } catch {
      setConnection('fixture');
      setBackend('offline fixture');
      setSessions([]);
      setActiveSessionId(null);
      setMessages(fixtureMessages);
      setRagHealth('offline');
      setModelCatalog([{ id: 'harness-default', owned_by: 'offline-fixture' }]);
      setSelectedModel('harness-default');
      setModelProfiles([]);
      setModelPresets([]);
      setModelDownloads([]);
      setImageCapabilities(null);
      setImageNotice('图像 API 未连接，不生成 fixture 图片。');
      setMcpManifest(null);
      setMcpResult(null);
      setMcpNotice('MCP API 未连接；不伪造工具目录或调用结果。');
      return;
    }
    try {
      const [modelsResponse, profilesResponse] = await Promise.all([listModels(), listModelProfiles()]);
      const nextModels = Array.isArray(modelsResponse.data) ? modelsResponse.data : [];
      setModelCatalog(nextModels);
      setSelectedModel((current) => current && nextModels.some((model) => model.id === current) ? current : nextModels[0]?.id || '');
      setModelProfiles(Array.isArray(profilesResponse.profiles) ? profilesResponse.profiles : []);
      const [presetResponse, downloadResponse] = await Promise.allSettled([listModelPresets(), listModelDownloads()]);
      if (presetResponse.status === 'fulfilled') setModelPresets(presetResponse.value.presets || []);
      if (downloadResponse.status === 'fulfilled') setModelDownloads(downloadResponse.value.jobs || []);
    } catch (error) {
      setModelCatalog([]);
      setSelectedModel('');
      setModelProfiles([]);
      setMcpNotice(`模型画像 API 不可用：${error instanceof Error ? error.message : 'model catalog unavailable'}`);
    }
    try {
      const response = await listSessions();
      setSessions(response.sessions);
      if (response.sessions.length > 0) {
        const selected = response.sessions.find((session) => session.session_id === activeSessionId) || response.sessions[0];
        await loadSession(selected.session_id);
      } else {
        setActiveSessionId(null);
        setMessages([]);
      }
    } catch (error) {
      setSessions([]);
      setActiveSessionId(null);
      setMessages(fixtureMessages);
      setMcpNotice(`会话 API 不可用；MCP 仍继续探测：${error instanceof Error ? error.message : 'session store unavailable'}`);
    }
    await refreshUtilityCapabilities();
    await refreshMcp();
  };

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!modelDownloads.some((job) => ['queued', 'downloading', 'verifying', 'registering'].includes(job.status))) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const [modelsResponse, downloadResponse] = await Promise.all([listModels(), listModelDownloads()]);
        setModelCatalog(Array.isArray(modelsResponse.data) ? modelsResponse.data : []);
        setModelDownloads(downloadResponse.jobs || []);
      } catch {
        // The main refresh path retains the last known asset state.
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [modelDownloads]);

  const startModelDownload = async (preset: HarnessModelPreset) => {
    if (modelBusy || !preset.installable) return;
    setModelBusy(`download:${preset.id}`);
    try {
      await queueModelDownload(preset.id);
      setModelNotice(`已开始下载 ${preset.display}`);
      const response = await listModelDownloads();
      setModelDownloads(response.jobs || []);
    } catch (error) {
      setModelNotice(`下载启动失败：${error instanceof Error ? error.message : 'unknown error'}`);
    } finally {
      setModelBusy('');
    }
  };

  const loadSelectedModel = async (model: HarnessModel) => {
    if (modelBusy || !model.available) return;
    setModelBusy(`load:${model.id}`);
    try {
      await loadModelAsset(model.id);
      setModelNotice(`已请求加载 ${model.id}`);
      await refresh();
    } catch (error) {
      setModelNotice(`模型加载失败：${error instanceof Error ? error.message : 'unknown error'}`);
    } finally {
      setModelBusy('');
    }
  };

  const selectedMcpTool = useMemo(() => mcpManifest?.tools.find((tool) => tool.name === mcpToolName) || null, [mcpManifest, mcpToolName]);

  useEffect(() => {
    if (!selectedMcpTool) return;
    const properties = selectedMcpTool.inputSchema.properties || {};
    const defaults: Record<string, unknown> = {};
    for (const field of selectedMcpTool.inputSchema.required || []) {
      if (field === 'query') defaults[field] = 'registry';
      else if (field === 'title') defaults[field] = 'MCP session';
      else defaults[field] = properties[field]?.type === 'boolean' ? false : '';
    }
    setMcpArguments(JSON.stringify(defaults, null, 2));
    setMcpResult(null);
  }, [selectedMcpTool]);

  const statusLabel = connection === 'online' ? 'ONLINE' : connection === 'fixture' ? 'FIXTURE' : connection === 'offline' ? 'OFFLINE' : 'CHECKING';
  const statusIcon = connection === 'online' ? <Wifi size={14} /> : <WifiOff size={14} />;

  const startSession = async () => {
    if (connection !== 'online') {
      setActiveSessionId(null);
      setMessages(fixtureMessages);
      return;
    }
    try {
      const session = await createSession(`会话 ${new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`);
      setSessions((current) => [session, ...current.filter((item) => item.session_id !== session.session_id)]);
      setActiveSessionId(session.session_id);
      setMessages([]);
      setView('chat');
      setSidebarOpen(false);
    } catch (error) {
      setMessages([{ id: `error-${Date.now()}`, role: 'system', content: `新建会话失败：${error instanceof Error ? error.message : 'unknown error'}`, meta: 'session request failed' }]);
    }
  };

  const selectSession = async (sessionId: string) => {
    if (sessionId === activeSessionId || sending) return;
    try {
      await loadSession(sessionId);
      setView('chat');
      setSidebarOpen(false);
    } catch (error) {
      setMessages([{ id: `error-${Date.now()}`, role: 'system', content: `会话加载失败：${error instanceof Error ? error.message : 'unknown error'}`, meta: 'session request failed' }]);
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
  };

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
        const activeModel = selectedModel.trim();
        if (!activeModel) throw new Error('后端未返回可用模型，停止发送请求');
        const selectedCatalogModel = modelCatalog.find((model) => model.id === activeModel);
        if (selectedCatalogModel?.available === false) {
          throw new Error('模型尚未下载，请打开运行时页面排队下载后再发送');
        }
        let sessionId = activeSessionId;
        if (!sessionId) {
          const session = await createSession('默认工作区');
          sessionId = session.session_id;
          setActiveSessionId(sessionId);
          setSessions((current) => [session, ...current]);
        }
        await appendSessionMessage(sessionId, userMessage, { source: 'harness-ui-02' });
        const assistantId = `assistant-${Date.now()}`;
        let assistantContent = '';
        setMessages([...nextMessages, { id: assistantId, role: 'assistant', content: '', meta: `${backend} / streaming` }]);
        const controller = new AbortController();
        abortRef.current = controller;
        await streamChat(activeModel, nextMessages, controller.signal, (delta) => {
          assistantContent += delta;
          setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: assistantContent } : message));
        });
        if (assistantContent.trim()) await appendSessionMessage(sessionId, { role: 'assistant', content: assistantContent }, { source: 'harness-ui-02', stream: true });
      }
    } catch (error) {
      const message = error instanceof DOMException && error.name === 'AbortError' ? '已停止生成。' : `请求失败：${error instanceof Error ? error.message : 'unknown error'}`;
      setMessages((current) => [...current, { id: `error-${Date.now()}`, role: 'system', content: message, meta: error instanceof DOMException && error.name === 'AbortError' ? 'generation stopped' : 'request failed' }]);
      if (!(error instanceof DOMException && error.name === 'AbortError') && !(error instanceof HarnessApiError)) setConnection('offline');
    } finally {
      abortRef.current = null;
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
      setRagContext(response.context);
      setRagNotice(response.context.omitted_count ? `已返回 ${response.context.included_count} 条，另有 ${response.context.omitted_count} 条因预算省略。` : `已返回 ${response.context.included_count} 条引用，未发生预算省略。`);
    } catch (error) {
      setRagHits([]);
      setRagContext(null);
      setRagNotice(`RAG API 不可用：${error instanceof Error ? error.message : 'unknown error'}；不伪造检索结果。`);
    }
  };

  const runImageGeneration = async (event: FormEvent) => {
    event.preventDefault();
    const prompt = imagePrompt.trim();
    if (!prompt || imageBusy) return;
    if (connection !== 'online') {
      setImageNotice('当前为离线/fixture 状态，不伪造图片结果。');
      return;
    }
    if (!imageCapabilities?.runtime_available || !imageCapabilities.supports_txt2img) {
      setImageNotice('当前 image adapter 未准入 txt2img；请先配置真实本地或远端图像运行时。');
      return;
    }
    const [widthText, heightText] = imageSize.split('x');
    setImageBusy(true);
    setImageNotice('图像请求已提交，等待 adapter 返回用户-owned URL…');
    try {
      const response = await generateImage({
        prompt,
        model: imageCapabilities.model_ids[0],
        width: Number(widthText),
        height: Number(heightText),
        steps: Number(imageSteps),
        ...(imageSeed.trim() ? { seed: Number(imageSeed) } : {}),
      });
      const result = response.data?.[0];
      if (!result?.url || !result.asset_id) throw new Error('图像响应缺少可验证的 asset URL 或 asset_id');
      const asset: ImageAsset = { ...result, prompt, createdAt: Date.now() };
      const assetId = result.asset_id;
      const assetUrl = result.url;
      setImageAssets((current) => [asset, ...current.filter((item) => item.asset_id !== asset.asset_id)].slice(0, 8));
      if (activeSessionId) await attachSessionAsset(activeSessionId, assetId, { url: assetUrl, prompt, source: 'harness-ui-03' });
      setImageNotice(`生成完成：${asset.asset_id}，资产已归入 local scope。`);
      setImagePrompt('');
    } catch (error) {
      setImageNotice(`生成失败：${error instanceof Error ? error.message : 'unknown image error'}`);
    } finally {
      setImageBusy(false);
    }
  };

  const runMcpCall = async (event: FormEvent) => {
    event.preventDefault();
    if (!mcpToolName || mcpBusy) return;
    let argumentsValue: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(mcpArguments || '{}');
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') throw new Error('参数必须是 JSON 对象');
      argumentsValue = parsed as Record<string, unknown>;
    } catch (error) {
      setMcpNotice(`参数 JSON 无效：${error instanceof Error ? error.message : 'invalid JSON'}`);
      setMcpResult(null);
      return;
    }
    setMcpBusy(true);
    setMcpNotice(`调用 ${mcpToolName}…`);
    try {
      const response = await callMcpTool(mcpToolName, argumentsValue);
      setMcpResult(response);
      const failure = readMcpError(response);
      if (failure) {
        const code = failure.code !== undefined ? ` [${String(failure.code)}]` : '';
        const retry = failure.retryable ? ' · 可重试' : '';
        setMcpNotice(`${mcpToolName} 失败：${failure.message}${code}${retry}`);
      }
      else setMcpNotice(`${mcpToolName} 完成。`);
    } catch (error) {
      setMcpResult(null);
      setMcpNotice(`MCP 调用失败：${describeHarnessError(error, 'unknown MCP error')}`);
    } finally {
      setMcpBusy(false);
    }
  };

  const viewTitle = useMemo(() => navItems.find((item) => item.id === view)?.label || '工作台', [view]);

  return (
    <div className="app-shell" data-connection={connection}>
      <a className="skip-link" href="#main-content">跳转到主要内容</a>
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><Bot size={22} /></div>
          <div><strong>QLH</strong><span>HARNESS WORKBENCH</span></div>
        </div>
        <div className="topbar-context"><span className="eyebrow">S6 / UI-04</span><span>{viewTitle}</span></div>
        <div className="topbar-actions">
          <span className={`connection connection--${connection}`} role="status" aria-live="polite"><span className="connection-dot" />{statusIcon}{statusLabel}</span>
          <button className="icon-button" type="button" onClick={() => void refresh()} title="刷新连接状态" aria-label="刷新连接状态"><RefreshCw size={17} /></button>
          <button className="icon-button" type="button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title={theme === 'dark' ? '切换浅色模式' : '切换深色模式'} aria-label="切换主题" aria-pressed={theme === 'light'}>{theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}</button>
          <button className="icon-button mobile-menu" type="button" onClick={() => setSidebarOpen(!sidebarOpen)} title="打开导航" aria-label="打开导航"><Menu size={18} /></button>
        </div>
      </header>

      <div className="workspace-grid">
        <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
          <div className="sidebar-head"><span className="eyebrow">WORKSPACE</span><button className="icon-button sidebar-close" type="button" onClick={() => setSidebarOpen(false)} aria-label="关闭导航"><X size={16} /></button></div>
          <nav className="main-nav" aria-label="工作台导航">
            {navItems.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${view === id ? 'nav-item--active' : ''}`} type="button" aria-current={view === id ? 'page' : undefined} onClick={() => { setView(id); setSidebarOpen(false); }}><Icon size={17} /><span>{label}</span><ChevronRight size={14} className="nav-chevron" /></button>)}
          </nav>
          <div className="sidebar-rule" />
          <div className="session-head"><span className="eyebrow">SESSIONS</span><button className="icon-button" type="button" title="新建会话" aria-label="新建会话" onClick={() => void startSession()}><Plus size={17} /></button></div>
          {sessions.length === 0 ? <div className="session-empty">{connection === 'online' ? '暂无会话' : '离线 fixture'}</div> : sessions.map((session) => <button key={session.session_id} className={`session-item ${activeSessionId === session.session_id ? 'session-item--active' : ''}`} type="button" onClick={() => void selectSession(session.session_id)}><span className={`session-pulse ${activeSessionId === session.session_id ? '' : 'session-pulse--dim'}`} /><span><strong>{session.title}</strong><small>{new Date(session.updated_at * 1000).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</small></span></button>)}
          <div className="sidebar-foot"><span className="eyebrow">BACKEND</span><span className="backend-name">{backend}</span></div>
        </aside>

        <main ref={mainRef} id="main-content" className="main-stage" tabIndex={-1}>
          <div className="stage-line" />
          {view === 'chat' && <ChatView messages={messages} input={input} setInput={setInput} sending={sending} modelCatalog={modelCatalog} selectedModel={selectedModel} onModelChange={setSelectedModel} onSubmit={sendMessage} onStop={stopGeneration} />}
          {view === 'rag' && <RagView query={ragQuery} setQuery={setRagQuery} hits={ragHits} context={ragContext} health={ragHealth} notice={ragNotice} onSubmit={runSearch} />}
          {view === 'assets' && <AssetView capabilities={imageCapabilities} assets={imageAssets} prompt={imagePrompt} setPrompt={setImagePrompt} size={imageSize} setSize={setImageSize} steps={imageSteps} setSteps={setImageSteps} seed={imageSeed} setSeed={setImageSeed} busy={imageBusy} notice={imageNotice} onSubmit={runImageGeneration} />}
          {view === 'mcp' && <McpView manifest={mcpManifest} selectedTool={selectedMcpTool} selectedName={mcpToolName} setSelectedName={setMcpToolName} argumentsValue={mcpArguments} setArgumentsValue={setMcpArguments} result={mcpResult} busy={mcpBusy} notice={mcpNotice} onSubmit={runMcpCall} onRefresh={() => void refreshMcp()} />}
          {view === 'runtime' && <RuntimeView connection={connection} backend={backend} models={modelCatalog} profiles={modelProfiles} presets={modelPresets} downloads={modelDownloads} busy={modelBusy} notice={modelNotice} onDownload={startModelDownload} onLoad={loadSelectedModel} onRefresh={() => void refresh()} />}
        </main>
      </div>
    </div>
  );
}

function ChatView({ messages, input, setInput, sending, modelCatalog, selectedModel, onModelChange, onSubmit, onStop }: { messages: ChatMessage[]; input: string; setInput: (value: string) => void; sending: boolean; modelCatalog: HarnessModel[]; selectedModel: string; onModelChange: (value: string) => void; onSubmit: (event: FormEvent) => void; onStop: () => void }) {
  return <section className="chat-view" aria-labelledby="chat-title">
    <div className="view-heading"><div><span className="eyebrow">CONVERSATION / LIVE MODEL</span><h1 id="chat-title">小模型工作台</h1><p>把上下文、资产和后端能力放在同一条可审计的工作流里。</p><label className="model-picker">模型<select aria-label="选择对话模型" value={selectedModel} onChange={(event) => onModelChange(event.target.value)} disabled={modelCatalog.length === 0}>{modelCatalog.length === 0 ? <option value="">暂无在线模型</option> : modelCatalog.map((model) => <option key={model.id} value={model.id}>{model.id}</option>)}</select></label></div><div className="view-heading-mark"><MessageSquareText size={28} /></div></div>
    <div className="message-list" role="log" aria-live="polite" aria-busy={sending} aria-label="对话记录">
      {messages.map((message) => <article className={`message message--${message.role}`} key={message.id}><div className="message-avatar" aria-hidden="true">{message.role === 'user' ? <UserRound size={16} /> : message.role === 'assistant' ? <Bot size={16} /> : <TerminalSquare size={16} />}</div><div className="message-body"><div className="message-meta"><span>{message.role === 'user' ? 'YOU' : message.role === 'assistant' ? 'HARNESS' : 'SYSTEM'}</span>{message.meta && <small>{message.meta}</small>}</div><p>{message.content}</p></div></article>)}
      {sending && <div className="typing" role="status"><span /><span /><span /> adapter 正在响应</div>}
    </div>
    <form className="composer" onSubmit={onSubmit}><div className="composer-tools"><button className="icon-button" type="button" title="添加资产" aria-label="添加资产"><Paperclip size={17} /></button><span className="composer-hint">Markdown / fixture-safe</span></div><textarea aria-label="消息输入" value={input} onChange={(event) => setInput(event.target.value)} placeholder="输入一条消息…" rows={3} /><div className="composer-submit"><span>{sending ? 'STREAMING' : 'ENTER 发送'}</span>{sending ? <button className="send-button" type="button" onClick={onStop} title="停止生成" aria-label="停止生成"><Square size={16} /></button> : <button className="send-button" type="submit" disabled={!input.trim()} title="发送消息" aria-label="发送消息"><Send size={18} /></button>}</div></form>
  </section>;
}

function RagView({ query, setQuery, hits, context, health, notice, onSubmit }: { query: string; setQuery: (value: string) => void; hits: RagHit[]; context: RagContext | null; health: string; notice: string; onSubmit: (event: FormEvent) => void }) {
  return <section className="utility-view" aria-labelledby="rag-title"><div className="view-heading"><div><span className="eyebrow">RETRIEVAL / FTS5 FIRST</span><h1 id="rag-title">知识库检索</h1><p>引用保持来源、chunk 与预算信息；没有 provider 时安全回退到 FTS。</p></div><DatabaseSearch size={30} className="heading-icon" /></div><div className="utility-strip"><span>BACKEND <strong>{health}</strong></span><span>OWNER SCOPE <strong>local</strong></span></div><form className="search-form" onSubmit={onSubmit}><Search size={17} /><input aria-label="知识库查询" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索本地资料…" /><button className="action-button" type="submit">检索</button></form><p className="utility-notice" role="status" aria-live="polite">{notice}</p>{context && <details className="citation-context" open><summary><span>引用上下文</span><span>{context.included_count} included · {context.omitted_count} omitted</span></summary><p>{context.text || '没有可组装的上下文。'}</p>{context.citations.length > 0 && <div className="citation-list">{context.citations.map((citation) => <code key={citation.chunk_id}>{citation.title} / {citation.chunk_id}</code>)}</div>}</details>}<div className="result-list">{hits.length === 0 ? <EmptyUtility icon={<Archive size={24} />} title="尚无检索结果" detail="输入查询，或先通过 API 写入用户-owned 资料。" /> : hits.map((hit) => <article className="result-row" key={hit.chunk_id}><div><span className="eyebrow">{hit.title} / CHUNK {hit.ordinal}</span><p>{hit.text}</p></div><code>score {hit.score.toFixed(3)}<br />{hit.chunk_id}</code></article>)}</div></section>;
}

function AssetView({ capabilities, assets, prompt, setPrompt, size, setSize, steps, setSteps, seed, setSeed, busy, notice, onSubmit }: { capabilities: ImageCapabilities | null; assets: ImageAsset[]; prompt: string; setPrompt: (value: string) => void; size: string; setSize: (value: string) => void; steps: string; setSteps: (value: string) => void; seed: string; setSeed: (value: string) => void; busy: boolean; notice: string; onSubmit: (event: FormEvent) => void }) {
  const [failedAssets, setFailedAssets] = useState<Record<string, boolean>>({});
  const ready = Boolean(capabilities?.runtime_available && capabilities.supports_txt2img);
  return <section className="utility-view" aria-labelledby="assets-title"><div className="view-heading"><div><span className="eyebrow">ASSETS / USER OWNED</span><h1 id="assets-title">图像与会话资产</h1><p>生成请求必须经过能力准入，输出只通过用户-owned asset URL 回读。</p></div><Image size={30} className="heading-icon" /></div><div className={`capability-banner ${ready ? 'capability-banner--ready' : 'capability-banner--blocked'}`}><span>{ready ? 'TXT2IMG READY' : 'TXT2IMG BLOCKED'}</span><small>{capabilities ? `${capabilities.backend} · ${capabilities.model_ids.join(', ') || 'no model declared'}` : 'image adapter unavailable'}</small></div><form className="asset-generator" onSubmit={onSubmit}><div className="asset-form-heading"><span className="eyebrow">GENERATE / URL ONLY</span><span className="composer-hint">{busy ? 'adapter working…' : 'no fixture image'}</span></div><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="描述要生成的图像…" rows={3} /><div className="asset-controls"><label>尺寸<select value={size} onChange={(event) => setSize(event.target.value)}><option value="512x512">512 × 512</option><option value="768x512">768 × 512</option><option value="512x768">512 × 768</option></select></label><label>步数<input type="number" min="1" max="100" value={steps} onChange={(event) => setSteps(event.target.value)} /></label><label>Seed<input inputMode="numeric" value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="随机" /></label><button className="action-button" type="submit" disabled={busy || !prompt.trim() || !ready}><Image size={16} />{busy ? '生成中' : '生成'}</button></div></form><p className="utility-notice">{notice}</p>{assets.length === 0 ? <EmptyUtility icon={<PanelLeft size={24} />} title="暂无用户-owned 图像资产" detail="图像 adapter 未准入时不会生成或伪造缩略图；成功后资产 URL 会出现在这里。" /> : <div className="asset-grid">{assets.map((asset) => <article className="asset-card" key={asset.asset_id}><div className="asset-preview">{asset.url && !failedAssets[asset.asset_id || ''] ? <img src={apiUrl(asset.url)} alt={asset.prompt} onError={() => setFailedAssets((current) => ({ ...current, [asset.asset_id || 'unknown']: true }))} /> : <div className="asset-error"><AlertTriangle size={22} /><span>资产 URL 不可用</span></div>}</div><div className="asset-card-body"><span className="eyebrow">{asset.asset_id || 'unknown asset'}</span><p>{asset.prompt}</p>{asset.url && <a href={apiUrl(asset.url)} target="_blank" rel="noreferrer"><ExternalLink size={14} />打开原始资产</a>}</div></article>)}</div>}<div className="asset-policy"><Settings2 size={17} /><span>资产服务返回错误、缺少 URL 或 URL 校验失败时，界面保持失败状态，不降级成虚假图片。</span></div></section>; }

function McpView({ manifest, selectedTool, selectedName, setSelectedName, argumentsValue, setArgumentsValue, result, busy, notice, onSubmit, onRefresh }: { manifest: MCPManifest | null; selectedTool: MCPTool | null; selectedName: string; setSelectedName: (value: string) => void; argumentsValue: string; setArgumentsValue: (value: string) => void; result: MCPCallResponse | null; busy: boolean; notice: string; onSubmit: (event: FormEvent) => void; onRefresh: () => void }) {
  const tools = manifest?.tools || [];
  const configuredCount = tools.filter((tool) => tool._meta?.qlh?.configured).length;
  const selectedMeta = selectedTool?._meta?.qlh;
  return <section className="utility-view mcp-view" aria-labelledby="mcp-title">
    <div className="view-heading"><div><span className="eyebrow">PROTOCOL / JSON-RPC 2.0</span><h1 id="mcp-title">MCP 控制面</h1><p>工具目录、输入合同和执行结果来自同一份 Harness registry。</p></div><Workflow size={30} className="heading-icon" /></div>
    <div className="mcp-statusbar"><div><span className="eyebrow">SERVER</span><strong>{manifest?.server.serverInfo?.name || 'unavailable'}</strong></div><div><span className="eyebrow">TOOLS</span><strong>{tools.length} / {configuredCount} ready</strong></div><div><span className="eyebrow">STDIO</span><strong>{manifest?.transports.stdio.available ? 'READY' : 'BLOCKED'}</strong></div><div><span className="eyebrow">SSE</span><strong>{manifest?.transports.sse.mode || 'offline'}</strong></div><button className="icon-button" type="button" onClick={onRefresh} title="刷新 MCP manifest" aria-label="刷新 MCP manifest"><RefreshCw size={16} /></button></div>
    <div className="mcp-notice" data-testid="mcp-notice" role="status" aria-live="polite"><ShieldCheck size={16} />{notice}</div>
    <div className="mcp-layout">
      <aside className="mcp-catalog" aria-label="MCP 工具目录"><div className="mcp-panel-head"><span className="eyebrow">TOOL REGISTRY</span><span>{tools.length.toString().padStart(2, '0')}</span></div>{tools.length === 0 ? <div className="mcp-empty"><CircleAlert size={20} /><span>工具目录不可用</span></div> : <div className="mcp-tool-list">{tools.map((tool) => { const meta = tool._meta?.qlh; const annotations = tool.annotations; return <button className={`mcp-tool-item ${selectedName === tool.name ? 'mcp-tool-item--active' : ''}`} type="button" key={tool.name} onClick={() => setSelectedName(tool.name)}><span className="mcp-tool-icon">{meta?.source === 'external' ? <Cable size={15} /> : <Braces size={15} />}</span><span className="mcp-tool-copy"><strong>{tool.name}</strong><small>{meta?.configured ? 'configured' : meta?.source === 'external' ? 'external declaration' : 'unconfigured'} · {annotations?.readOnlyHint ? 'read' : annotations?.destructiveHint ? 'destructive write' : 'write'}</small></span><ChevronRight size={14} /></button>; })}</div>}</aside>
      <div className="mcp-inspector">{selectedTool ? <><div className="mcp-inspector-head"><div><span className="eyebrow">SELECTED TOOL</span><h2>{selectedTool.name}</h2></div><span className={`mcp-state ${selectedMeta?.configured ? 'mcp-state--ready' : ''}`}>{selectedMeta?.configured ? <CircleCheck size={14} /> : <CircleAlert size={14} />}{selectedMeta?.capability || 'unconfigured'}</span></div><p className="mcp-description">{selectedTool.description}</p><div className="mcp-badges"><span>{selectedTool.annotations?.readOnlyHint ? 'READ ONLY' : 'WRITE'}</span>{selectedTool.annotations?.destructiveHint && <span className="mcp-badge--danger">CONFIRMATION SENSITIVE</span>}{selectedTool.annotations?.openWorldHint && <span className="mcp-badge--gold">OPEN WORLD</span>}</div><details className="mcp-schema" open><summary><span><Braces size={14} />输入 schema</span><span>{(selectedTool.inputSchema.required || []).length} required</span></summary><pre>{JSON.stringify(selectedTool.inputSchema, null, 2)}</pre></details><form className="mcp-call-form" onSubmit={onSubmit}><label htmlFor="mcp-arguments">调用参数 <small>JSON object</small></label><textarea id="mcp-arguments" value={argumentsValue} onChange={(event) => setArgumentsValue(event.target.value)} spellCheck={false} rows={7} /><button className="action-button" type="submit" disabled={busy || !selectedMeta?.configured}><Play size={15} />{busy ? '调用中' : '调用工具'}</button></form>{result && <div className={`mcp-result ${result.result?.isError || result.error ? 'mcp-result--error' : ''}`} data-testid="mcp-result"><div className="mcp-panel-head"><span className="eyebrow">LAST RESPONSE</span>{result.result?.isError || result.error ? <CircleAlert size={15} /> : <CircleCheck size={15} />}</div><pre>{JSON.stringify(result, null, 2)}</pre></div>}</> : <div className="mcp-empty mcp-empty--large"><Workflow size={28} /><strong>等待 MCP 工具目录</strong><span>连接 Harness API 后可浏览并调用已注册工具。</span></div>}</div>
    </div>
    {manifest && <div className="mcp-footnote"><span><strong>External MCP</strong> {manifest.external_mcp.configuration_only ? `${manifest.external_mcp.configurations.length} 个配置声明，真实连接未启用。` : '已启用。'}</span><span><strong>SSE</strong> {manifest.transports.sse.mode} · <strong>stdio</strong> {manifest.transports.stdio.command}</span></div>}
  </section>;
}

function RuntimeView({ connection, backend, models, profiles, presets, downloads, busy, notice, onDownload, onLoad, onRefresh }: {
  connection: ConnectionState;
  backend: string;
  models: HarnessModel[];
  profiles: ModelProfileSummary[];
  presets: HarnessModelPreset[];
  downloads: HarnessModelDownload[];
  busy: string;
  notice: string;
  onDownload: (preset: HarnessModelPreset) => void;
  onLoad: (model: HarnessModel) => void;
  onRefresh: () => void;
}) {
  const activeDownloads = downloads.filter((job) => ['queued', 'downloading', 'verifying', 'registering'].includes(job.status));
  return <section className="utility-view" aria-labelledby="runtime-title">
    <div className="view-heading"><div><span className="eyebrow">RUNTIME / MODEL ASSETS</span><h1 id="runtime-title">运行时能力</h1><p>Harness 直接消费 QLH 模型目录、预设和下载任务；候选画像仍与已安装运行时分开显示。</p></div><Network size={30} className="heading-icon" /></div>
    <div className="runtime-grid"><div className="runtime-cell"><span className="eyebrow">CONNECTION</span><strong>{connection.toUpperCase()}</strong><small>API healthz</small></div><div className="runtime-cell"><span className="eyebrow">ADAPTER</span><strong>{backend}</strong><small>当前后端</small></div><div className="runtime-cell"><span className="eyebrow">MODEL CATALOG</span><strong>{models.length}</strong><small>可见资产</small></div><div className="runtime-cell"><span className="eyebrow">DOWNLOADS</span><strong>{activeDownloads.length}</strong><small>进行中</small></div></div>
    {notice ? <p className="utility-notice" role="status">{notice}</p> : null}
    <div className="profile-list" aria-label="模型资产列表">
      {models.length === 0 ? <EmptyUtility icon={<CircleAlert size={22} />} title="暂无模型资产" detail="请先确认 QLH API 在线，然后从预设列表安装模型。" /> : models.map((model) => <article className="profile-row" key={model.id}>
        <div><strong>{model.id}</strong><small>{model.owned_by || 'qlh'} · {model.available === false ? '未下载' : '可加载'}</small>{model.unavailable_reason ? <small>{model.unavailable_reason}</small> : null}</div>
        <button className="action-button action-button--quiet" type="button" disabled={busy !== '' || model.available === false} onClick={() => onLoad(model)}>{busy === `load:${model.id}` ? '加载中…' : '加载'}</button>
      </article>)}
    </div>
    <div className="profile-list" aria-label="模型预设列表">
      {presets.length === 0 ? <EmptyUtility icon={<CircleAlert size={22} />} title="暂无可用预设" detail="当前连接的 adapter 没有暴露 QLH 下载服务。" /> : presets.map((preset) => <article className="profile-row" key={preset.id}>
        <div><strong>{preset.display}</strong><small>{preset.kind} · {preset.default_engine || 'auto'} · {preset.default_model_id || preset.id}</small>{preset.description ? <small>{preset.description}</small> : null}</div>
        <button className="action-button action-button--quiet" type="button" disabled={busy !== '' || !preset.installable} onClick={() => onDownload(preset)}>{busy === `download:${preset.id}` ? '排队中…' : preset.installable ? '下载' : '资源不足'}</button>
      </article>)}
    </div>
    {downloads.length > 0 ? <div className="profile-list" aria-label="模型下载任务">{downloads.map((job) => <article className="profile-row" key={job.job_id}><div><strong>{job.model_id || job.preset_id || job.job_id}</strong><small>{job.status} · {Math.round((job.progress || 0) * 100)}%</small>{job.error ? <small>{job.error}</small> : null}</div><span className={`profile-status profile-status--${job.status}`}>{job.status}</span></article>)}</div> : null}
    <div className="profile-list" aria-label="模型画像列表">{profiles.length === 0 ? <EmptyUtility icon={<CircleAlert size={22} />} title="暂无模型画像" detail="连接 Harness API 后读取候选画像。" /> : profiles.map((profile) => <article className="profile-row" key={`${profile.model_id}:${profile.backend}:${profile.revision}`}><div><strong>{profile.model_id}</strong><small>{profile.backend} · {profile.roles.join(' / ')}</small>{profile.aliases?.length ? <small>资产 ID：{profile.aliases.join(' / ')}</small> : null}</div><span className={`profile-status profile-status--${profile.status}`}>{profile.status}</span><small>{profile.production_eligible ? 'production eligible' : 'runtime gate pending'}</small></article>)}</div>
    <button className="action-button action-button--quiet" type="button" onClick={onRefresh}><RefreshCw size={16} />重新探测</button>
  </section>;
}

function ProfileRuntimeView({ connection, backend, models, profiles, onRefresh }: { connection: ConnectionState; backend: string; models: HarnessModel[]; profiles: ModelProfileSummary[]; onRefresh: () => void }) {
  return <section className="utility-view" aria-labelledby="runtime-title"><div className="view-heading"><div><span className="eyebrow">RUNTIME / MODEL PROFILES</span><h1 id="runtime-title">运行时能力</h1><p>在线模型来自 adapter，候选画像来自 Harness registry；候选不会被 UI 自动视为生产模型。</p></div><Network size={30} className="heading-icon" /></div><div className="runtime-grid"><div className="runtime-cell"><span className="eyebrow">CONNECTION</span><strong>{connection.toUpperCase()}</strong><small>API healthz</small></div><div className="runtime-cell"><span className="eyebrow">ADAPTER</span><strong>{backend}</strong><small>当前后端</small></div><div className="runtime-cell"><span className="eyebrow">LIVE MODELS</span><strong>{models.length}</strong><small>可用于对话选择</small></div><div className="runtime-cell"><span className="eyebrow">CANDIDATES</span><strong>{profiles.length}</strong><small>画像待运行时验证</small></div></div><div className="profile-list" aria-label="模型画像列表">{profiles.length === 0 ? <EmptyUtility icon={<CircleAlert size={22} />} title="暂无模型画像" detail="连接 Harness API 后读取候选画像。" /> : profiles.map((profile) => <article className="profile-row" key={`${profile.model_id}:${profile.backend}:${profile.revision}`}><div><strong>{profile.model_id}</strong><small>{profile.backend} · {profile.roles.join(' / ')}</small>{profile.aliases?.length ? <small>资产 ID：{profile.aliases.join(' / ')}</small> : null}</div><span className={`profile-status profile-status--${profile.status}`}>{profile.status}</span><small>{profile.production_eligible ? 'production eligible' : 'runtime gate pending'}</small></article>)}</div><button className="action-button action-button--quiet" type="button" onClick={onRefresh}><RefreshCw size={16} />重新探测</button></section>;
}

function EmptyUtility({ icon, title, detail }: { icon: React.ReactNode; title: string; detail: string }) { return <div className="empty-utility"><div className="empty-icon">{icon}</div><strong>{title}</strong><p>{detail}</p></div>; }

export default App;
