/** Help workspace: indexed setup notes, endpoint reference, and focused troubleshooting detail. */

import { useMemo, useState } from 'react';
import { BookOpen, ChevronDown, Code2, FileText, Rocket } from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { routeHref } from '../app/routes';
import { APP_VERSION, DESIGN_DOC } from '../app/version';
import { PageBackdrop } from '../visual/PageBackdrop';

type HelpSection = 'start' | 'api' | 'faq' | 'about';

interface Faq {
  q: string;
  a: string;
}

interface Endpoint {
  path: string;
  use: string;
  scope: string;
}

const HELP_SECTIONS: Array<{ id: HelpSection; label: string; code: string; icon: typeof Rocket }> = [
  { id: 'start', label: '启动', code: 'START', icon: Rocket },
  { id: 'api', label: '接口清单', code: 'API', icon: Code2 },
  { id: 'faq', label: '常见问题', code: 'FAQ', icon: BookOpen },
  { id: 'about', label: '关于', code: 'INFO', icon: FileText },
];

const FAQS: Faq[] = [
  { q: '工作台的左右比例怎么调？', a: '拖动中间的分隔条即可，比例会记在本机 localStorage。焦点位于分隔条时，左右方向键每次调整 2%，按住 Shift 调整 8%；Home / End 到两端，双击恢复默认。窗口窄于 860px 时改为上下堆叠。' },
  { q: '页面显示“无法连接后端”怎么办？', a: '控制台通过 /api 前缀访问 FastAPI。开发模式下由 Vite 代理到 http://localhost:8000，先确认后端已启动；若后端不在默认端口，用环境变量 QLH_VITE_API_TARGET 指定。也可以在设置页启用演示数据。' },
  { q: '任务页的队列区块提示“无权限”？', a: '/api/cluster/queue 仅主节点开放，从节点会返回 403。请在主节点打开控制台，或在设置页确认本机角色。' },
  { q: '活动页读不到日志？', a: '后端可以为日志接口开启令牌保护。在设置页填入管理令牌后，请求会带上 X-QLH-Log-Token 请求头。' },
  { q: '工作流列表一直为空？', a: '任务图需要后端启用 TASK_GRAPH_ENABLED。未启用时接口会返回 enabled=false，页面会显示状态说明而不是空表格。' },
  { q: '动效太多影响使用？', a: '设置页的动效偏好有跟随系统、完整、减少三档。选择“减少动效”后位移与脉冲关闭，背景 Canvas 只绘制一帧静态图。' },
];

const ENDPOINTS: Endpoint[] = [
  { path: 'GET /api/status', use: '模型、显存、KV 缓存与本机设备等级', scope: '所有节点' },
  { path: 'GET /api/cluster/nodes', use: '节点列表与在线数', scope: '所有节点' },
  { path: 'GET /api/cluster/my-role', use: '本机角色（主 / 从）', scope: '所有节点' },
  { path: 'GET /api/cluster/queue', use: '三级队列深度与排队任务', scope: '仅主节点' },
  { path: 'GET /api/cluster/pipeline-capacity', use: '流水线准入判定与层分配', scope: '仅主节点' },
  { path: 'GET /api/workflows', use: '任务图工作流与执行提供者', scope: '所有节点' },
  { path: 'GET /api/logs/recent', use: '内存环形缓冲中的日志', scope: '可能需要令牌' },
  { path: 'GET /api/sessions', use: '已保存的对话会话', scope: '所有节点' },
  { path: 'GET /api/conversations', use: '工作台右栏的历史消息', scope: '所有节点' },
  { path: 'POST /api/chat/stream', use: '流式对话（SSE，逐 token 返回）', scope: '所有节点' },
  { path: 'POST /api/chat/generations/{id}/cancel', use: '中止当前生成', scope: '当前会话' },
  { path: 'POST /api/chat/clear', use: '清空当前会话历史', scope: '当前会话' },
  { path: 'GET /api/rag/health', use: '本地知识库索引规模', scope: '所有节点' },
  { path: 'GET /api/health', use: '连接探针与顶栏状态灯', scope: '所有节点' },
];

function StartGuide() {
  return (
    <ol className="help-steps">
      <li><span className="mono-label">01</span><div><strong>安装依赖</strong><pre className="codeblock">cd frontend_cybergothic{`\n`}npm install</pre></div></li>
      <li><span className="mono-label">02</span><div><strong>启动后端</strong><p>在仓库根目录按既有方式启动 FastAPI，默认监听 <code>http://localhost:8000</code>。</p></div></li>
      <li><span className="mono-label">03</span><div><strong>启动开发服务</strong><pre className="codeblock">npm run dev{`\n`}# 后端不在默认端口时：{`\n`}QLH_VITE_API_TARGET=http://192.168.1.10:8000 npm run dev</pre></div></li>
      <li><span className="mono-label">04</span><div><strong>构建产物</strong><pre className="codeblock">npm run build{`\n`}npm run preview</pre><p>产物是静态文件，路由使用 hash，可直接部署而无需 rewrite 规则。</p></div></li>
    </ol>
  );
}

export function HelpPage() {
  const [section, setSection] = useState<HelpSection>('start');
  const [selectedFaq, setSelectedFaq] = useState(FAQS[0]?.q ?? '');
  const [selectedEndpoint, setSelectedEndpoint] = useState(ENDPOINTS[0]?.path ?? '');

  const current = HELP_SECTIONS.find((item) => item.id === section) ?? HELP_SECTIONS[0];
  const activeFaq = useMemo(() => FAQS.find((item) => item.q === selectedFaq) ?? FAQS[0], [selectedFaq]);
  const activeEndpoint = useMemo(() => ENDPOINTS.find((item) => item.path === selectedEndpoint) ?? ENDPOINTS[0], [selectedEndpoint]);

  return (
    <div className="help-page" data-testid="help-page">
      <PageBackdrop scene="help" className="help-page__bg" />
      <PageHeader
        tag="HELP"
        title="帮助"
        description="启动方式、后端契约和故障排查集中在本地文档工作区，不发出额外请求。"
        actions={<CommandButton variant="ghost" size="sm" href={routeHref('overview')}>回到概览</CommandButton>}
      />

      <div className="help-layout">
        <aside className="help-rail" aria-label="帮助目录">
          <section className="help-panel help-identity">
            <span className="mono-label">LOCAL REFERENCE</span>
            <strong>QLH CONTROL CONSOLE</strong>
            <span>v{APP_VERSION} · static handbook</span>
          </section>
          <nav className="help-nav" aria-label="帮助章节">
            {HELP_SECTIONS.map((item) => {
              const Icon = item.icon;
              return <button key={item.id} type="button" data-active={section === item.id ? 'true' : undefined} aria-pressed={section === item.id} onClick={() => setSection(item.id)}><Icon size={15} aria-hidden="true" /><span>{item.label}</span><em>{item.code}</em></button>;
            })}
          </nav>
          <section className="help-panel help-rail-note"><span className="mono-label">SAFE MODE</span><p>本文档只描述接口与本地操作，不会修改集群配置。</p></section>
        </aside>

        <main className="help-main">
          <section className="help-panel help-workspace" data-section={section} aria-labelledby="help-workspace-title">
            <div className="help-workspace__head"><SectionHead id="help-workspace-title" title={current.label} hint={section === 'start' ? '前端、后端和静态构建的本地启动路径。' : section === 'api' ? '选择接口以查看用途与适用范围。' : section === 'faq' ? '选择问题以在右栏固定阅读答案。' : '版本、设计文档与旧界面关系。'} /></div>
            <div className="help-workspace__scroll">
              {section === 'start' ? <StartGuide /> : null}
              {section === 'api' ? <div className="help-endpoint-list" role="list">{ENDPOINTS.map((endpoint) => <div role="listitem" key={endpoint.path}><button type="button" data-active={selectedEndpoint === endpoint.path ? 'true' : undefined} onClick={() => setSelectedEndpoint(endpoint.path)}><code>{endpoint.path}</code><span>{endpoint.use}</span><em>{endpoint.scope}</em></button></div>)}</div> : null}
              {section === 'faq' ? <div className="help-faq-list">{FAQS.map((faq, index) => <button key={faq.q} type="button" data-active={selectedFaq === faq.q ? 'true' : undefined} aria-pressed={selectedFaq === faq.q} onClick={() => setSelectedFaq(faq.q)}><span className="mono-label">{String(index + 1).padStart(2, '0')}</span><strong>{faq.q}</strong><ChevronDown size={16} aria-hidden="true" /></button>)}</div> : null}
              {section === 'about' ? <dl className="kvgrid help-about"><div><dt>版本</dt><dd className="cell-mono">v{APP_VERSION}</dd></div><div><dt>设计依据</dt><dd className="cell-mono">{DESIGN_DOC}</dd></div><div><dt>技术栈</dt><dd>React 18 + TypeScript + Vite</dd></div><div><dt>与旧界面的关系</dt><dd>独立目录与端口，共用后端和登录令牌存储。</dd></div></dl> : null}
            </div>
          </section>
        </main>

        <aside className="help-details" aria-label="当前帮助详情">
          <section className="help-panel help-detail-panel">
            <SectionHead title="当前条目" hint={current.code} />
            {section === 'start' ? <><p className="help-detail-title">本地启动</p><dl className="kvlist"><div><dt>前端目录</dt><dd className="cell-mono">frontend_cybergothic</dd></div><div><dt>开发端口</dt><dd className="cell-mono">5174</dd></div><div><dt>后端默认端口</dt><dd className="cell-mono">8000</dd></div><div><dt>路由模式</dt><dd>Hash SPA</dd></div></dl></> : null}
            {section === 'api' && activeEndpoint ? <><p className="help-detail-title cell-mono">{activeEndpoint.path}</p><dl className="kvlist"><div><dt>用途</dt><dd>{activeEndpoint.use}</dd></div><div><dt>适用范围</dt><dd>{activeEndpoint.scope}</dd></div><div><dt>数据类型</dt><dd>{activeEndpoint.path.startsWith('GET') ? '只读查询' : '状态变更'}</dd></div></dl></> : null}
            {section === 'faq' && activeFaq ? <><p className="help-detail-title">{activeFaq.q}</p><p className="help-answer">{activeFaq.a}</p></> : null}
            {section === 'about' ? <><p className="help-detail-title">控制台边界</p><p className="help-answer">前端遵循后端已有契约；演示数据只在显式开启时替代读接口，写操作会被拦截。</p><CommandButton href={routeHref('settings')} variant="ghost" size="sm">打开设置</CommandButton></> : null}
          </section>
        </aside>
      </div>
    </div>
  );
}
