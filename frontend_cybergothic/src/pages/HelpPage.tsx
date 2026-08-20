/**
 * 帮助 — 使用说明、接口契约与排障清单。
 *
 * 纯静态内容，不发请求；作为验收标准 §8 中「另一位开发者能跑起来」的入口。
 */

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { CommandButton } from '../components/CommandButton';
import { useReveal } from '../motion/useReveal';
import { routeHref } from '../app/routes';
import { APP_VERSION, DESIGN_DOC } from '../app/version';

interface Faq {
  q: string;
  a: string;
}

const FAQS: Faq[] = [
  {
    q: '工作台的左右比例怎么调？',
    a: '拖动中间的分隔条即可，比例会记在本机（localStorage），下次打开保持原样。也可以用键盘：聚焦分隔条后左右方向键每次 2%，按住 Shift 是 8%，Home / End 直接到两端，双击分隔条恢复默认。窗口窄于 860px 时改为上下堆叠，分隔条隐藏。',
  },
  {
    q: '页面显示「无法连接后端」怎么办？',
    a: '控制台通过 /api 前缀访问 FastAPI。开发模式下由 Vite 代理到 http://localhost:8000，先确认后端已启动；若后端不在默认端口，用环境变量 QLH_VITE_API_TARGET 指定。也可以在设置页打开演示数据，先看界面。',
  },
  {
    q: '任务页的队列区块提示「无权限」？',
    a: '/api/cluster/queue 只在主节点开放，从节点会返回 403。请在主节点打开控制台，或在设置页确认本机角色。',
  },
  {
    q: '活动页读不到日志？',
    a: '后端可以为日志接口开启令牌保护。在设置页填入管理令牌后，请求会带上 X-QLH-Log-Token 请求头。',
  },
  {
    q: '工作流列表一直为空？',
    a: '任务图需要后端启用（TASK_GRAPH_ENABLED）。未启用时接口会返回 enabled=false，页面会直接说明而不是显示空表格。',
  },
  {
    q: '动画太多影响使用？',
    a: '设置页的动效偏好有三档：跟随系统、完整、减少。选择「减少动效」后位移与脉冲会关闭，背景 Canvas 只画一帧静态图。',
  },
];

const ENDPOINTS: Array<{ path: string; use: string }> = [
  { path: 'GET /api/status', use: '模型、显存、KV 缓存与本机设备等级' },
  { path: 'GET /api/cluster/nodes', use: '节点列表与在线数' },
  { path: 'GET /api/cluster/my-role', use: '本机角色（主 / 从）' },
  { path: 'GET /api/cluster/queue', use: '三级队列深度与排队任务（仅主节点）' },
  { path: 'GET /api/cluster/pipeline-capacity', use: '流水线准入判定与层分配' },
  { path: 'GET /api/workflows', use: '任务图工作流与执行提供者' },
  { path: 'GET /api/logs/recent', use: '内存环形缓冲中的日志（可能需要令牌）' },
  { path: 'GET /api/sessions', use: '已保存的对话会话' },
  { path: 'GET /api/conversations', use: '工作台右栏的历史消息' },
  { path: 'POST /api/chat/stream', use: '流式对话（SSE，逐 token 返回）' },
  { path: 'POST /api/chat/generations/{id}/cancel', use: '中止本轮生成，服务端同时停算' },
  { path: 'POST /api/chat/clear', use: '清空当前会话历史' },
  { path: 'GET /api/rag/health', use: '知识库索引规模' },
  { path: 'GET /api/health', use: '连接探针，顶栏状态灯使用' },
];

export function HelpPage() {
  const [open, setOpen] = useState<string>(FAQS[0]?.q ?? '');
  useReveal([]);

  return (
    <>
      <PageHeader
        tag="HELP"
        title="帮助"
        description="这套控制台只读取后端已有接口，不改变集群配置。以下是启动方式、接口清单与常见问题。"
        actions={
          <CommandButton variant="ghost" size="sm" href={routeHref('overview')}>
            回到概览
          </CommandButton>
        }
      />

      <section className="band" data-reveal>
        <SectionHead title="启动" hint="独立于既有 frontend 目录，端口 5174，可与旧界面同时运行。" />
        <ol className="steplist">
          <li>
            <span className="steplist__num mono-label">01</span>
            <div>
              <p className="steplist__title">安装依赖</p>
              <pre className="codeblock">cd frontend_cybergothic
npm install</pre>
            </div>
          </li>
          <li>
            <span className="steplist__num mono-label">02</span>
            <div>
              <p className="steplist__title">启动后端</p>
              <p className="steplist__desc">
                在仓库根目录按既有方式启动 FastAPI，默认监听 <code>http://localhost:8000</code>。
              </p>
            </div>
          </li>
          <li>
            <span className="steplist__num mono-label">03</span>
            <div>
              <p className="steplist__title">启动开发服务器</p>
              <pre className="codeblock">npm run dev
# 后端不在默认端口时：
QLH_VITE_API_TARGET=http://192.168.1.10:8000 npm run dev</pre>
            </div>
          </li>
          <li>
            <span className="steplist__num mono-label">04</span>
            <div>
              <p className="steplist__title">构建产物</p>
              <pre className="codeblock">npm run build   # 先 tsc --noEmit 再 vite build
npm run preview</pre>
              <p className="steplist__desc">
                产物是纯静态文件，路由使用 hash，直接放到任意静态目录即可，不需要 rewrite 规则。
              </p>
            </div>
          </li>
        </ol>
      </section>

      <section className="band band--alt" data-reveal>
        <SectionHead title="接口清单" hint="控制台读取的全部后端接口；写操作仅限队列与工作流控制。" />
        <div className="ttable__scroll">
          <table className="ttable">
            <caption className="sr-only">控制台使用的后端接口</caption>
            <thead>
              <tr>
                <th scope="col">接口</th>
                <th scope="col">用途</th>
              </tr>
            </thead>
            <tbody>
              {ENDPOINTS.map((ep) => (
                <tr key={ep.path}>
                  <td data-label="接口">
                    <span className="cell-mono">{ep.path}</span>
                  </td>
                  <td data-label="用途">{ep.use}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="inline-note">
          写操作：<code>POST /api/cluster/queue/pause</code>、<code>/resume</code>、
          <code>/strategy</code>，<code>DELETE /api/cluster/queue/task/&#123;id&#125;</code>，
          <code>POST /api/workflows/&#123;id&#125;/cancel</code>。演示数据模式下这些操作会被拦截。
        </p>
      </section>

      <section className="band" data-reveal>
        <SectionHead title="常见问题" />
        <div className="faq">
          {FAQS.map((item) => {
            const isOpen = open === item.q;
            return (
              <div className="faq__item" key={item.q} data-open={isOpen ? 'true' : undefined}>
                <h3 className="faq__heading">
                  <button
                    type="button"
                    className="faq__trigger"
                    aria-expanded={isOpen}
                    onClick={() => setOpen(isOpen ? '' : item.q)}
                  >
                    <span>{item.q}</span>
                    <ChevronDown className="faq__chevron" size={16} strokeWidth={2.25} aria-hidden />
                  </button>
                </h3>
                {isOpen ? <p className="faq__answer">{item.a}</p> : null}
              </div>
            );
          })}
        </div>
      </section>

      <section className="band band--alt" data-reveal>
        <SectionHead title="关于" />
        <dl className="kvgrid">
          <div>
            <dt>版本</dt>
            <dd className="cell-mono">v{APP_VERSION}</dd>
          </div>
          <div>
            <dt>设计依据</dt>
            <dd className="cell-mono">{DESIGN_DOC}</dd>
          </div>
          <div>
            <dt>技术栈</dt>
            <dd>React 18 + TypeScript + Vite（与既有 frontend 同栈）</dd>
          </div>
          <div>
            <dt>与旧界面的关系</dt>
            <dd>独立目录、独立端口，共用后端与登录令牌存储，互不影响。</dd>
          </div>
        </dl>
      </section>
    </>
  );
}
