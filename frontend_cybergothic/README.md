# QLH 控制台 · Cybergothic

按 `docs/StudyPact风格分析与frontend_cybergothic前端计划.md` 实现的第二套前端。与
`frontend/` 完全独立：独立的 `package.json`、独立的 `dist`、独立的端口，共用同一个
FastAPI 后端。两者可以同时开着跑，互不影响。

技术栈沿用仓库既有选择，没有引入第二套构建体系：React 18 + TypeScript 5（strict）+ Vite 5，
图标用 `lucide-react`，动效用 CSS transition / keyframes，没有动画库。

## 快速开始

```bash
cd frontend_cybergothic
npm install

# 另开一个终端启动后端（默认 http://localhost:8000）
# python -m src.api_server   等价的仓库启动方式见项目根 README

npm run dev            # http://localhost:5174
```

后端不在默认地址时用环境变量指过去，`/api` 会被代理到那里：

```bash
QLH_VITE_API_TARGET=http://192.168.1.20:8000 npm run dev
```

不想启动后端也能看全部界面 —— 追加 `?fixtures=1` 走本地演示数据：

```
http://localhost:5174/?fixtures=1
http://localhost:5174/#/tasks?fixtures=1      # 写在 hash 后面也认
```

## 页面

| 路由 | 内容 |
| --- | --- |
| `#/workbench` | **落地页**。左集群实况 / 右对话的分屏，比例可拖动 |
| `#/overview` | 集群概览：节点卡片、关键数字、近期事件 |
| `#/tasks` | 三级队列与任务图工作流 |
| `#/activity` | 运行日志（可按级别筛选）与会话记录 |
| `#/settings` | 后端连接、本机角色、动效偏好、演示数据开关 |
| `#/help` | 接口契约、FAQ、排障清单 |

### 工作台的分屏

左右比例由用户决定，记在 `localStorage.qlh_cg_split_ratio`，刷新后保持。

- **鼠标**：拖中间分隔条；双击复位到默认（42%，接近 5:7）。
- **键盘**：聚焦分隔条后 ←/→ 每次 2%，`Shift` 加大到 8%，`Home`/`End`
  直接到两端（22% / 78%）。分隔条是真 `role="separator"`，带完整
  `aria-valuenow` / `aria-valuetext`，读屏能听到比例变化。
- **窄屏**：窗口窄于 860px 自动改为上下堆叠，分隔条隐藏且不可聚焦。

左右两侧是两套色板：控制台用 `--accent` 酸绿 + `--skew` 平行四边形，对话用
`--gothic-*` 苍紫/描边金 + `--arch` 尖拱。`.chat` 会把共享组件的取色变量
（`--accent`、`--text`、`--line` 等）在子树内改绑到哥特色板，所以放进对话栏的
组件自动跟着变色，不会漏出一颗酸绿按钮。

对话产生新一轮后会立刻刷新左栏的队列/日志/状态 —— 这是分屏的主要价值：
能直接看到自己那句话对集群的影响，不用等轮询。

演示数据模式下顶部会出现常驻提示条，所有写操作被拦截，不会打到真实集群。也可以在
设置页里开关（存 `localStorage.qlh_cg_use_fixtures`）；URL 参数优先级更高。

## 构建与部署

```bash
npm run build          # tsc --noEmit && vite build，产物在 dist/
npm run preview        # 本地预览构建产物
```

路由是自己实现的 hash 路由（`#/overview`、`#/tasks` …），不依赖 react-router。
好处是 `dist/` 可以直接丢到任意静态服务器或后端的静态目录下，**不需要配置
history fallback / rewrite 规则**。

参考体积（gzip）：JS 约 79 kB，CSS 约 10 kB。

## 桌面窗口（pywebview，与主应用同栈）

构建产物也可以放进**桌面原生窗口**，复用主应用的 pywebview（Windows 用 Edge WebView2，
Linux 回退系统浏览器），不引入 Electron。

```bash
npm run build                                     # 先产出 dist/
# 后端在默认地址跑（127.0.0.1:8000）；随后：
python packaging/launcher_cybergothic.py          # 弹原生窗口加载 http://127.0.0.1:9851/
```

用法：

| 参数/环境变量 | 作用 |
| --- | --- |
| `--port <n>` | 本地窗口服务端口（默认 9851；占用自动向后探测，避开 5174/8000/9090） |
| `--api-target <url>` | `/api/*` 反向代理目标（默认 `http://127.0.0.1:8000`；可设 `QLH_API_TARGET`） |
| `--dist <dir>` | 指定 dist 目录（可设 `QLH_CG_DIST`）；`--dist` 指向缺失目录会直接报错 |
| `--no-window` | 不开窗口，只打印访问地址（供调试/无人环境；不会误开浏览器） |
| `--debug` | 打印请求级日志 |

要点：

- 生产构建下前端 `BASE='/api'` 是相对同源，桌面壳会把 `/api/*` 反向代理到
  `--api-target`，否则窗口里接口全 404。
- 静态服务带 SPA history 回退：无扩展名的路由回到 `index.html`，带扩展名的缺失
  资源（如 `.png`）返回 404，不给前端吞错误。
- 服务只监听 `127.0.0.1`，非回环来源一律 403。
- 需要 Python 环境具备 `pywebview`（主应用 requirements 已含 `pywebview>=6.0`；
  开发机 `.venv-test` 未安装时用主应用运行时 venv 或 `pip install pywebview`）。

测试：`pytest tests/test_launcher_cybergothic.py`（不开真窗口/不连外网，回环 fake 后端驱动）。

## 验证

```bash
npm run verify         # 类型检查 + 对比度自检 + 构建 + E2E，一条命令跑完
```

拆开跑：

| 命令 | 覆盖 |
| --- | --- |
| `npm run typecheck` | TS strict，含 `noUnusedLocals` / `noUnusedParameters` |
| `npm run check:contrast` | 从 `tokens.css` 解析颜色，校验 27 组前景/背景达到 WCAG AA（含哥特色板） |
| `npm run test:e2e` | Playwright：10 项功能冒烟 + 8 项工作台用例 + 4 个断点的响应式检查 |

E2E 用系统安装的 Edge（`channel: 'msedge'`），不额外下载浏览器内核，与仓库既有
e2e 配置一致。全部用例都跑在 fixture 模式下，不需要后端在线。

响应式用例会在 1440 / 1024 / 768 / 390px 下逐页断言「无横向溢出」「无区块滚动后
仍不显现」，并把截图写到 `../build/cybergothic-shots/`，方便人工过一眼。

## 目录结构

```
src/
  data/         接口封装、类型、fixture、useResource 状态机
  app/          hash 路由、AppShell 外壳
  components/   通用展示组件（表格、时间线、空状态、抽屉、Toast…）
  pages/        六个页面：Workbench / Overview / Tasks / Activity / Settings / Help
  motion/       useReveal、useReducedMotion
  visual/       AccentCanvas、GrainOverlay（纯装饰，aria-hidden）
  styles/       tokens.css（设计变量）、global.css、components.css、workbench.css
public/assets/  静态资源占位，替换说明见该目录 README
scripts/        contrast.mjs 对比度自检
tests/          smoke.spec.js、responsive.spec.js、workbench.spec.js
```

样式按这个顺序加载（后者可覆盖前者）：
`tokens.css` → `global.css` → `components.css` → `workbench.css`。
分屏/平行四边形/哥特对话的规则单独放在 `workbench.css`，没有追加到已经
2400 行的 `components.css` —— 那份文件结尾是响应式与 `@media print` 覆盖块，
追加在后面的新类会绕过那些覆盖。

分层约束：页面只声明「要什么数据」，请求细节在 `data/`；演示数据只存在于
`data/fixtures.ts`，不散落在组件里。

## 后端接口

只读接口（全部 GET，都在 `/api` 下）：

`/api/status`、`/api/health`、`/api/cluster/nodes`、`/api/cluster/queue`、
`/api/cluster/my-role`、`/api/cluster/pipeline-capacity`、`/api/workflows?limit=`、
`/api/rag/health`、`/api/logs/recent?limit=&level=`、`/api/sessions?limit=`、
`/api/conversations?session_id=&limit=`

写操作（任务页与工作台对话触发，演示数据模式下被拦截）：

`POST /api/cluster/queue/pause`、`POST /api/cluster/queue/resume`、
`POST /api/cluster/queue/strategy`、`DELETE /api/cluster/queue/task/{id}`、
`POST /api/workflows/{id}/cancel`、`POST /api/chat/stream`、
`POST /api/chat/generations/{id}/cancel`、`POST /api/chat/clear`

执行提供者的状态不是独立接口，来自 `/api/workflows` 响应里的
`provider_status` / `provider_error` 字段。

两个约定容易踩：

- `/api/cluster/queue` 在非 master 节点返回 403。界面把它当「无权限」状态展示，
  不是报错。
- `/api/logs/recent?level=WARNING` 是**最低级别**过滤（`levelno >= 阈值`），不是精确
  匹配。所以筛选项写成 `INFO+` / `WARN+`。
- `POST /api/chat/stream` 是 SSE，必须边读边解析，不能像其他接口那样把响应体一次
  读成 JSON（`data/api.ts` 里的 `streamChat` 因此绕开了统一的 `request()` helper）。
  用 `streaming_mode: 'interactive'` 才有逐 token 返回。经 nginx 反代时要关掉
  该路径的 `proxy_buffering`，否则 token 会被攒到最后一次吐出来。
- 点「停止」除了 `AbortController` 掉本地连接，还会调
  `POST /api/chat/generations/{id}/cancel`，否则服务端会继续把这一轮算完。

鉴权沿用既有前端的约定，不另起一套：会话令牌读 `sessionStorage` 的
`qlh-auth-session-token`（`Authorization: Bearer`），日志令牌读 `localStorage` 的
`qlh_log_admin_token`（`X-QLH-Log-Token`，可在设置页填）。

## 换皮

`src/styles/tokens.css` 是唯一的颜色来源，组件里没有硬编码色值。控制台侧改
`--accent`，对话侧改 `--gothic-accent` / `--gothic-gold`，两套互不影响；改完跑
`npm run check:contrast` 确认对比度仍达标（脚本直接解析 token 文件，所以样式和
校验不会脱节）。

形状也走 token：`--skew` / `--slant-sm` / `--slant-md` 控制控制台侧平行四边形的
倾角，`--arch` 控制对话侧尖拱连券的跨度。斜切元素内部的文字要反向 `skewX`
（`--skew-back`）抵消回来，否则字会跟着歪。

字体用系统字体栈，未打包字体文件；替换字体和 logo 的说明见
[public/assets/README.md](public/assets/README.md)。

## 已知事项

- `npm audit` 会报一条 esbuild 开发服务器的 moderate 告警（GHSA-67mh-4wv8-2f99）。
  彻底修需要升到 vite 6+，会和 `frontend/` 的技术栈分叉，所以暂时保持 vite 5.4.21。
  该问题只影响本地 dev server，不进生产产物。
- 日志超过 100 条时列表按 60 条一屏窗口化渲染，点「加载更多」增量展开。
