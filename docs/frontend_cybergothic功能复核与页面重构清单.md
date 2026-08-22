# frontend_cybergothic 功能复核与页面重构清单

> 目的：以现有 `frontend` 为历史行为基线，盘点 `frontend_cybergothic` 的缺口，记录单机模式下 Overview 的故障路径，并给出后续分屏布局与逐页背景动效方案。
>
> 范围：本文件是实现清单和验收标准，不在本轮直接修改前端代码。已有的 StudyPact 风格分析与首期视觉计划见 [`StudyPact风格分析与frontend_cybergothic前端计划.md`](./StudyPact风格分析与frontend_cybergothic前端计划.md)。

## 1. 当前基线

### 1.1 两套前端的边界

`frontend_cybergothic` 当前通过 hash 路由提供八页：

| 路由 | 当前内容 | 当前状态 |
| --- | --- | --- |
| `#/workbench` | 左侧控制台、右侧流式对话 | 已有分屏，是当前交互基准 |
| `#/overview` | 状态、指标、准入、节点、活动、本机资源 | 页面存在，但角色受限接口会以错误态出现 |
| `#/tasks` | 队列、工作流、执行提供者 | 有基础读写，仍是纵向长页 |
| `#/activity` | 日志时间线、会话表 | 有基础筛选，仍是纵向长页 |
| `#/image` | 生图任务、资产目录、生成参数和任务详情 | P1 基础工作区已接入；编辑/局部重绘待补 |
| `#/models` | 模型目录、运行时加载/卸载、本地资产和只读预检 | P1.2 基础工作区已接入；导入/拉取/许可证管理待补 |
| `#/settings` | 后端连接、设备/GPU、RAG、动效、fixture、日志令牌 | P1.3 已接入设备与 RAG 工作区；模型能力位于 `#/models`，其余运行时选项待补 |
| `#/help` | 启动、接口清单、FAQ、版本 | P3 三栏文档工作区已接入；目录、正文与当前条目详情独立组织，主区局部滚动 |

旧 `frontend` 的入口在 `frontend/src/App.jsx`，视图为 `chat / image / admin / account`，并在设置弹窗中挂载设备、模型、RAG 等工作区。自 2026-08-22 起旧前端冻结，仅用于行为/接口对照，不再开新开发票。`frontend_cybergothic` 的路由表在 `frontend_cybergothic/src/app/routes.tsx`，数据入口在 `frontend_cybergothic/src/data/api.ts` 和 `src/data/hooks.ts`；两者不是功能等价实现，差异以《前端安卓后端接口与功能缺口审查-2026-08-22》为准。

视觉现状已在 P3 阶段统一：`ChatPane` 使用 `GothicWorksCanvas`，Overview、Tasks、Activity、Settings、Help 分别接入独立页面场景，并迁移为固定视口内的侧栏、主区与详情布局。三栏外层壳层使用约 56% 至 66% 的半透明底色，Canvas 提升至约 52% 至 58%；正文卡片、表格和表单维持更实的底色，背景动效以古铜远景与金色近景出现，不干扰阅读。

### 1.2 差异清单

状态定义：

- **已覆盖**：新前端已有可使用的页面或接口，但仍可能需要补齐交互细节。
- **部分覆盖**：有展示或只读子集，和旧前端完整能力不等价。
- **缺失**：没有路由、组件或对应 API 接线。
- **阻断风险**：在特定角色、单机或鉴权条件下会把可选能力误报成整页故障。

| 功能域 | `frontend` 基线 | `frontend_cybergothic` 现状 | 状态 | 后续验收点 |
| --- | --- | --- | --- | --- |
| 流式对话 | `ChatPanel`、预设、工作流详情、取消生成、附件 | `ChatPane` 已支持流式、取消、文本上传、清空、刷新和单轮删除 | 部分覆盖 | 保留预设、工作流详情、失败重试、生成状态和附件列表 |
| 会话管理 | `SessionList`：新建、切换、重命名、删除、激活、删除单轮 | 工作台左侧会话栏已支持新建、切换、重命名、删除和当前会话同步 | 已覆盖（P0） | 后续补服务端分页、最近会话排序和跨设备恢复 |
| 对话历史 | `fetchConversations`、分页/清空 | `ChatPane` 支持历史读取、清空、刷新和确认后删除单轮；fixture 与真实接口路径一致 | 已覆盖（P1.4） | 后续补服务端分页/增量加载、失败重试按钮和持久化结果校验 |
| 生图工作区 | `DiffusionPanel`，生成/编辑/局部重绘/图生图、模型 artifact、资产目录、任务状态 | `#/image` 已有生成参数、任务轮询、取消、结果预览、下载和本地历史 | 部分覆盖 | 编辑/局部重绘/图生图模式、真实结果历史持久化和资产下载待补 |
| 生图列表 | 旧组件维护 artifacts、asset catalog、运行中 job、结果 blob | 已有任务/资产侧栏、状态、prompt、结果和下载；fixture/真实 job 查询均接线 | 已覆盖（P1） | 状态筛选、重试、删除、详情抽屉和真实 blob 生命周期待补 |
| 模型选择与加载 | `ModelSelector`、当前模型、可用模型、加载/卸载、量化 | `#/models` 已有目录、引擎/量化、加载/卸载、运行时状态和预检 | 已覆盖（P1.2） | 导入/拉取、加载进度、失败恢复和 sidecar 控制待补 |
| 设备/GPU | `DevicePanel`，档位检测、GPU 选择、自动配置 | `#/settings` 已有设备画像、GPU 选择和推荐配置操作 | 已覆盖（P1.3） | 显存预算、CPU 回退策略和配置持久化待补 |
| 模型资产/模型舰队 | `ModelFleetPanel`：本地资产、artifact、导入/拉取、预检、运行时 sidecar、许可证与来源 | `#/models` 已有本地资产、来源/状态、只读预检和运行时详情 | 部分覆盖 | 导入/拉取、许可证与 sidecar 控制待补 |
| RAG | `RagPanel`：健康、容量、检索、重建、ANN、embedding job | `#/settings` 已有健康、FTS 检索和重建操作 | 部分覆盖 | 容量、ANN 和 embedding job 详情待补 |
| 集群/管理员 | `AdminPanel`：节点注册/注销、邀请、角色、层分配、分布式配置、邮件、主节点转移、备用主、审查工单、队列控制 | `#/cluster` 已覆盖节点拓扑、状态、邀请端点和角色门控；`#/audit` 覆盖审查工单 | 部分覆盖（P2） | 层分配、邮件与完整转移流程待补；所有写操作有确认、审计反馈和失败回滚提示 |
| 账户/鉴权 | `AuthGate`、用户管理、登录会话、头像、Tailscale | `#/account` 已接入能力探针、登录/退出、会话撤销、管理员用户管理、Tailscale 绑定、Auth App bootstrap/QR/字符串、TOTP 首次确认和恢复码轮换；真实接口缺失时显示明确错误态 | 已覆盖（P0/P2） | 真实 control/gateway、OS credential store、跨设备恢复与首次安装联调后置 |
| 日志 | 最近日志、会话记录 | `#/activity` 保留实时日志与会话；`#/audit` 接入日志文件预览/删除、统计、节点汇总/聚合和审查工单 | 已覆盖（P2） | 日志导出、会话同步状态和完整审查邮件流待补 |
| 设置 | 主题、设备、模型、RAG、分布式推理、日志和运行时选项 | 连接信息、动效、fixture、日志令牌 | 部分覆盖 | 采用设置侧栏，按领域拆分并保留未保存/保存成功状态 |
| 任务图 | 工作流列表和阶段 Drawer | 已有工作流列表、筛选、取消和提供者状态 | 部分覆盖 | 迁移完整阶段详情、输入输出、错误堆栈、重试/取消和能力开关 |
| API/错误语义 | 旧 client 覆盖完整 API 和鉴权 | 新 API 已统一请求超时、网络异常、401/403/404/409/429/5xx 分类，并保留 request id | 已覆盖（P0） | 后续继续扩展缺失业务接口；页面可按 `errorKind` 细化权限引导 |

## 2. 生图工作区优先清单

用户已明确指出“生图列表”缺失，这不是装饰性差异，而是旧前端的主要工作区之一。建议新增 `#/image`（或命名为 `#/studio`）并拆成三块：

1. **左侧任务/资产侧栏**：生图任务列表、状态筛选（排队中/生成中/完成/失败/取消）、搜索、日期和模型过滤；每行显示缩略图、提示词摘要、耗时和进度。
2. **中央画布**：生成、编辑、局部重绘、图生图四种模式使用 segmented control；参数区固定在画布下方或右侧，不把全部表单堆成长页。
3. **右侧详情面板**：当前任务的 prompt、负面 prompt、seed、尺寸、采样器、步数、模型 artifact、输入/输出 blob、错误详情、下载/重试/取消/删除操作。

需要从 `frontend/src/api/client.js` 对齐的接口族包括：

- capabilities、artifact inspect/register/load/unload、asset catalog/status/download/import；
- `generateDiffusionImage`、`editDiffusionImage`、blob 上传/读取/删除；
- job 查询、进度、取消和结果预览；
- 本地单机无 diffusion 能力时的 capability 空态，而不是白屏或无限加载。

验收要求：刷新页面后任务列表仍可恢复；运行中任务可取消；失败任务显示服务端原因并可重试；生成结果可预览和下载；后端不可用时表单仍能打开并给出可操作的连接提示；窄屏时侧栏变为抽屉，详情变为底部抽屉。

## 3. Overview 单机问题复核

### 3.1 已确认的代码路径

- `frontend_cybergothic/src/pages/OverviewPage.tsx` 无条件调用 `useQueue(8_000)` 和 `usePipelineCapacity()`，并将队列深度用于指标、将准入请求用于整块“流水线准入”区域。
- `frontend_cybergothic/src/data/hooks.ts` 的 `useQueue` 默认启用，直接调用 `GET /api/cluster/queue`；`usePipelineCapacity` 也没有按角色跳过或将“不可适用”单独建模。
- `src/api_server.py` 的 `GET /api/cluster/queue` 明确检查 `scheduler._effective_role() == "master"`，非主节点返回 403“仅主节点可查看请求队列”。
- `src/api_server.py` 的 `GET /api/cluster/pipeline-capacity` 没有显式的主节点 403，而是直接返回 `scheduler.get_pipeline_capacity_plan()`；单机时可能得到未准入/空计划，不能把它与队列的 403 混为一谈。
- `src/config.py` 在缺少显式环境变量时，非 frozen 进程的角色回退值是 `client`，而 frozen 进程回退为 `master`。因此本地源码启动若未设置 `QLH_NODE_ROLE=master`，很容易命中队列的 403 路径。

### 3.2 需要补的实现规则

1. 启动阶段先请求 `GET /api/cluster/my-role`，把角色、运行模式和是否单机作为页面上下文；不要等队列失败后猜角色。
2. `useQueue` 增加 `enabled` 或 `role` 条件：确认不是主节点时不轮询队列，返回 `not_applicable`，而不是 `error`。
3. Overview 的指标改为独立卡片状态：本机状态、节点、日志、队列、容量互不阻塞。建议使用 `Promise.allSettled` 或等价的独立资源状态。
4. 单机客户端显示“单机模式：请求队列由本机执行器管理，主节点队列不适用”，同时继续展示 `/status`、本机资源和对话入口。
5. 队列和容量的 403/空计划都应有“查看角色/前往设置”的操作；网络错误才显示“重试”。
6. 追加启动矩阵测试：`QLH_NODE_ROLE=master`、`client`、未设置三种环境，覆盖 Overview 首屏、刷新、离线和 fixture 模式。

### 3.3 验收标准

- 单机源码启动（默认或显式 `QLH_NODE_ROLE=client`）访问 `#/overview` 时，页面能完整渲染，不出现整页错误、不产生无限重试。
- 主节点仍能看到队列深度、暂停/恢复和流水线准入。
- 客户端能看到本机状态、节点信息、最近活动和“不可适用”说明。
- 任何一个可选接口失败，不影响导航、状态摘要、模型/对话入口和其他已成功卡片。

当前实现补充：`fetchPipelineCapacity` 已在数据层归一化后端的精简 `unavailable` 响应（缺少 `control_only_nodes` 等字段），Overview 不再因读取 `undefined.length` 崩溃；对应回归覆盖在 `frontend_cybergothic/tests/p0.spec.js`。

## 4. 页面分屏与侧栏重构

目标不是把长页简单切成两个大卡片，而是给“选择 -> 列表 -> 详情/操作”建立稳定的工作区关系。所有列表都应在自己的 pane 内滚动，页面外层不再因为一个详情区把整页高度拉长。

### 4.1 通用布局断点

| 宽度 | 布局 | 交互规则 |
| --- | --- | --- |
| `>= 1200px` | `侧栏 / 主列表 / 详情` 三列，或 `导航侧栏 / 主内容` 两列 | 侧栏固定，主列表和详情各自 `overflow: auto`；详情可折叠 |
| `860–1199px` | `可收起侧栏 / 主内容`，详情作为右侧 Drawer | 默认保留主列表；选择条目后打开详情 Drawer |
| `<= 859px` | 单列主内容，侧栏和详情均为 Drawer/Bottom Sheet | 只保留一个滚动容器；返回操作有明确焦点回收 |

所有工作区需定义 `min-height: 0`、局部 `overflow: auto`、固定工具栏高度和 `overscroll-behavior: contain`，避免再次出现“对话框很长但无法分开滚动”的问题。现有 Workbench 的左右分屏继续作为基准：左侧控制台和右侧 ChatPane 各自滚动，顶栏不参与内容滚动。

### 4.2 逐页布局建议

| 页面 | 建议结构 | 侧栏内容 | 主内容/详情 |
| --- | --- | --- | --- |
| Workbench | 保留左右分屏 | 控制台、会话入口可折叠 | 左侧集群实况；右侧对话消息和输入区 |
| Overview | `运行上下文侧栏 / 可切换状态主区 / 当前详情` | 角色、运行模式、模型与快捷入口 | 已迁移为 `#/overview` 三栏局部滚动；总览、准入、节点、活动按 pane 切换，节点/日志固定详情，平板与窄屏使用 Drawer |
| Tasks | `队列/筛选侧栏 / 任务表 / 任务详情` | 队列层级、工作流状态、提供者、暂停状态 | 已迁移为 `#/tasks` 队列/工作流切换工作区；桌面三栏局部滚动，平板与窄屏使用详情 Drawer |
| Activity | `筛选与会话侧栏 / 日志时间线 / 事件详情` | 日志级别、节点、时间范围、会话 | 已迁移为 `#/activity` 桌面三栏局部滚动；点击日志/会话固定详情，平板与窄屏切换 Drawer |
| Settings | `设置导航侧栏 / 设置工作区 / 运行上下文` | 设备、RAG、连接、动效、数据源、日志令牌 | 已迁移为 `#/settings` 三栏局部滚动；设备/RAG 受控嵌入，右栏固定显示数据源、节点角色与令牌脏状态 |
| Help | `目录侧栏 / 文档内容 / 当前条目详情` | 启动、接口、FAQ、故障排查、版本 | 已迁移为 `#/help` 三栏局部滚动；目录切换启动、API、FAQ 与关于，API/FAQ 当前条目固定显示在右栏 |
| Image Studio | `生图任务/资产 / 生成画布 / 任务详情` | 见第 2 节 | 画布和参数稳定，历史列表不随画布高度增长 |
| Models & Assets | `模型/资产筛选 / 资产表 / 预检与运行时详情` | 状态、来源、许可证、设备兼容性 | 导入、拉取、加载和卸载放在详情 pane |
| Cluster Admin | `管理导航 / 操作台 / 审计详情` | 节点、加入、角色、分层、分布式、邮件、审查 | 高风险操作必须有确认、影响范围和结果日志 |
| Account | `账户导航 / 用户与会话 / 安全详情` | 个人资料、登录会话、用户管理、Tailscale | 普通用户和管理员显示不同域；无权限功能不渲染空白页 |

## 5. 多层视差背景动效规范

### 5.1 统一实现方式

当前 `frontend_cybergothic/src/visual/PageBackdrop.tsx` 和 `sceneRegistry.ts` 已将页面 Canvas 统一注册；各场景均具备 DPR 上限、ResizeObserver、IntersectionObserver、减少动效与后台暂停。场景保持独立绘制循环，以便每页拥有不重复的建筑和机械语汇。

每个场景包含：`sceneId`、调色板、远景建筑类型、主机械部件、次级装饰、视差系数、粒子密度、速度、随机种子。Canvas 仅作 `aria-hidden` 装饰，任何状态、文本和操作都必须在 DOM 中可用。

### 5.2 视差层级

建议固定四层，鼠标/触摸位移使用平滑插值，不直接跟随指针跳动：

| 层 | 内容 | 视差 | 动效 |
| --- | --- | ---: | --- |
| L0 远景 | 哥特建筑轮廓、尖拱、塔楼、玫瑰窗 | `0.05x` | 几乎静止，低对比度 |
| L1 中景 | 场景主结构与大型机械部件 | `0.12x` | 缓慢旋转、摆动或扫描 |
| L2 近景 | 框架、齿条、链条、钟摆、管线 | `0.22x` | 与主结构有相位差的运动 |
| L3 氛围 | 少量尘埃、火花、雨线、微光 | `0.35x` | 低密度、有限生命周期 |

工程约束：每页只有一个 canvas；静态建筑进离屏缓存；动画只更新可见层；页面不可见或标签页后台时暂停 `requestAnimationFrame`；DPR 上限 2；粒子数量随面积封顶；`prefers-reduced-motion` 或用户选择“减少动效”时只画一帧静态场景；避免大面积 blur 和持续高亮，以免影响文本对比度。

### 5.3 页面场景目录

每页必须有自己的主图案和配色重点。齿轮与时钟不应复制到每页；它们是 Workbench/某一场景的主元素，其他页面使用不同的机械语汇。

| 页面 | 场景名 | 主元素 | 色彩/动效重点 |
| --- | --- | --- | --- |
| Workbench / Chat | Cathedral Works | 尖拱、互相啮合的巨型齿轮、玫瑰窗时钟 | 暗紫石材 + 金色/淡青线条；齿轮反向旋转，时钟慢速走针 |
| Overview | Observatory Nave | 天文台拱顶、星盘、轨道环、远处塔楼 | 已接入独立 Canvas；古铜远景 + 金色星盘，轨道、反向齿轮与彗星低速进动 |
| Tasks | Gearworks Queue | 分层齿条、输送链、闸门和排队脉冲 | 已接入独立 Canvas；古铜齿条 + 金色传动轮，齿条与啮合齿轮反向移动且只影响背景层 |
| Activity | Bell Tower Rain | 钟楼、垂直雨线、摆锤、检修栈桥 | 古铜钟楼 + 金色雨线与传动轮；雨线有深度差，摆锤和齿轮相位独立 |
| Settings | Clockwork Archive | 档案柜、锁孔、刻度盘、机械书脊 | 已接入独立 Canvas；古铜档案柜 + 金色锁件，档案柜、书脊、表盘和啮合齿轮分层视差 |
| Help | Stained-glass Scriptorium | 柳叶窗、彩色玻璃分格、翻页框架、导管 | 已接入独立 Canvas；古铜窗格 + 金色玫瑰盘，彩窗光带、手稿与窗格按层缓慢移动 |
| Image Studio | Alchemical Foundry | 炼金炉、转台、光圈环、图像版片、蒸汽管 | 品红/琥珀作为少量强调；转台和光圈与生成状态联动 |
| Models & Assets | Reliquary Engine | 资产匣、堆叠金属板、轴承、铭牌 | 青绿 + 金色；资产卡槽有序点亮，下载时显示单向流光 |
| Cluster Admin | Gargoyle Relay | 城堡塔楼、连线、继电器、节点信标 | 铁红 + 青色；节点信标按在线状态呼吸，连线不做强光束 |
| Account | Iron Gate | 门闩、锁环、钥匙、护盾和链条 | 金色 + 暗红；锁环慢旋，安全状态用低频脉冲 |
| Audit Ledger | Archive Clock | 钟楼、档案卷轴、反向齿轮、刻度盘 | 金色 + 苍紫；卷轴分层视差，齿轮低速反向旋转，指针缓慢走动 |

### 5.4 动效与交互验收

- 鼠标、触摸或设备方向变化只影响 L1–L3 的偏移，内容布局不移动；没有输入时通过 `lerp` 回到中位。
- 动画速度、线宽、透明度和粒子密度均可由场景配置控制；不得把“页面滚动”作为唯一驱动，否则局部滚动 pane 时会失去连续性。
- Canvas 与正文之间保持足够对比度，背景装饰不得覆盖按钮、表格、输入框或焦点环。
- 减少动效模式保留建筑和机械轮廓，但停止旋转、雨线、火花和脉冲；测试中应检查首帧稳定、无布局跳动。
- 关闭 Canvas 或浏览器不支持 Canvas 时，使用纯色/静态 CSS 纹理兜底，功能不能降级。

## 6. 实施顺序与交付物

### P0：先恢复功能可用性

- [x] 增加 `my-role` 上下文和单机模式判定。
- [x] 修复 Overview 队列 403 的 `not_applicable` 状态；拆分独立资源错误边界。
- [x] 将 ChatPane 从固定 `default` 会话接入会话列表的创建、切换、重命名、删除。
- [x] 补齐 API client 的错误分类、超时、离线和鉴权状态（请求层已覆盖 401/403/404/409/429/5xx、网络、超时与流式中断；资源 hook 暴露 `errorKind/errorStatus/retryable`）。
- [x] 为单机 master/client/未设置角色补启动矩阵测试（`tests/p0_role_matrix.spec.js` 覆盖首屏、队列门控、容量状态、全局刷新和页面错误；client/未设置角色不会请求主节点队列）。

### P1：补齐旧前端的核心工作区

- [x] 新增 Image Studio，优先完成生图列表、任务详情、取消/重试/下载（P1 已完成 `#/image` 基础任务列表、生成、轮询、取消和本地历史；编辑模式与真实结果历史持久化待补）。
- [x] 新增 Models & Assets，接入模型选择、加载/卸载和本地资产（P1.2 已完成 `#/models` 的模型目录、引擎/量化选择、运行时加载/卸载、资产详情和只读预检；导入、拉取、许可证与 sidecar 控制待补）。
- [x] 将 Device、RAG、运行时设置从旧弹窗迁移为设置侧栏页面（P1.3 已完成设备画像、GPU 选择、推荐配置、RAG 健康、FTS 检索与重建；容量、ANN 和 embedding job 待补）。
- [x] 新增完整会话工作区和对话历史操作（P1.4 已完成会话栏与对话历史刷新、清空、单轮删除；服务端分页、失败重试和跨设备恢复留后续）。

### P2：集群与账户能力

- [x] 新增 Cluster Admin，并按角色控制可见性和写操作（`#/cluster` 已接入节点拓扑、状态、邀请端点、主节点容量、手动预注册、注销/删除；client 只读且不请求邀请数据，覆盖 `tests/p2_cluster.spec.js`）。
- [x] 新增 Account/Auth、登录会话、用户管理和 Tailscale（`#/account` 已接入 owner/admin 权限域、fixture/真实 API、会话撤销、Tailscale inspect/prepare/confirm/revoke 与服务缺失错误态，覆盖 `tests/p2_account.spec.js`）。
- [x] 补日志文件、节点日志聚合、统计和审查工单等管理能力（`#/audit` 已接入受保护日志文件预览/删除、buffer 统计、master-only 节点汇总与聚合、审查工单创建/投票/过期检查/清理，fixture/真实 API 与 client 只读门控覆盖 `tests/p2_audit.spec.js`）。

### P3：布局与视觉系统

- [x] 抽象 `GothicSceneCanvas/PageBackdrop` 和 scene registry（`src/visual/PageBackdrop.tsx`、`sceneRegistry.ts` 已提供统一场景注册；Overview、Activity、Tasks、Settings、Help 已接入独立场景）。
- [x] 将 Overview、Tasks、Activity、Settings、Help 改成侧栏 + 主区 + 详情的局部滚动布局；桌面保持三栏，窄屏退化为单列或 Drawer，并统一调整外层面板透明度。
- [x] 为每个页面接入第 5.3 节的独特场景，并做桌面、平板、窄屏截图检查（分别使用 Observatory Nave / Bell Tower Rain / Gearworks Queue / Clockwork Archive / Stained-glass Scriptorium）。
- [x] 修复工作区壳层过实的问题：所有已接入背景的侧栏、主区和详情壳层降低透明底色，内部表单、表格和可操作卡片保持实底；背景统一使用古铜远景与金色近景，并补充反向齿轮、钟面、玫瑰盘和扫光等低速层次。
- [ ] 检查 reduced motion、Canvas 不可用、后台暂停和高 DPI 下的 CPU/GPU 占用。

## 7. 验收矩阵

| 类别 | 最低验收 |
| --- | --- |
| 功能完整性 | 新前端路由和旧前端主要工作区一一对应；每个“缺失”项有页面、API、加载/空/错误/成功状态 |
| 单机 | 未设置角色、`master`、`client` 均可打开 Overview；角色不适用的接口显示说明，不显示整页错误 |
| 分屏滚动 | 桌面上侧栏、主列表、详情各自可滚动；窗口高度变化不把对话框或页面撑成长不可操作列表 |
| 响应式 | `>=1200px` 三列、`860–1199px` Drawer、`<=859px` 单列/Bottom Sheet 均可完成选择和返回 |
| 视觉 | 每页场景主元素不重复；L0–L3 有明显深度差；正文对比度、焦点环和按钮可读 |
| 动效 | 正常模式连续且低干扰；不可见时暂停；减少动效只保留静态首帧；Canvas 兜底不影响功能 |
| 性能 | 每页单 Canvas、DPR <= 2、有限粒子；切换路由后旧动画循环被清理；长列表使用局部窗口化/虚拟化 |
| 安全与权限 | 403、过期 token、普通用户和管理员均有明确反馈；高风险集群写操作显示范围、确认和结果 |

## 8. 参考代码索引

- 新前端路由：`frontend_cybergothic/src/app/routes.tsx`
- 新前端请求与鉴权：`frontend_cybergothic/src/data/api.ts`、`frontend_cybergothic/src/data/hooks.ts`
- Overview 角色敏感调用：`frontend_cybergothic/src/pages/OverviewPage.tsx`
- 当前对话背景：`frontend_cybergothic/src/visual/GothicWorksCanvas.tsx`、`frontend_cybergothic/src/components/ChatPane.tsx`
- 旧前端视图入口：`frontend/src/App.jsx`
- 生图工作区：`frontend/src/components/DiffusionPanel.jsx`
- 设置中的设备/模型/RAG：`frontend/src/components/DevicePanel.jsx`、`ModelFleetPanel.jsx`、`RagPanel.jsx`
- 集群与账户：`frontend/src/components/AdminPanel.jsx`、`UserManagementPanel.jsx`、`TailscaleBindingPanel.jsx`
- 队列权限：`src/api_server.py` 的 `/api/cluster/queue`
- 单机默认角色：`src/config.py` 的 `QLH_NODE_ROLE` 回退逻辑
