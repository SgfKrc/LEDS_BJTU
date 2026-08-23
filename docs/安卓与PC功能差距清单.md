# 安卓版与 PC 版功能差距清单

> 状态：现行（随两端开发持续更新）
>
> 更新日期：2026-08-23
> 适用范围：Android Full/Lite 相对 PC 版（Windows/Linux 主节点）缺失功能的**全面排查清单**，供排期与任务分配使用；能力现状依据 [Android 版本远期计划](Android版本远期计划.md)（2026-07-28 基线 + C5 真机验收）、README 与源码

> 本次接口复核补充：Android HTTP 控制面已覆盖登录/会话、聊天、远程 SD、主节点 GGUF Range/SHA 下载、bootstrap 和带 lease 的 presence；Android Full Worker 已进入 PC scheduler 准入门，并已用可注入 executor 接入任务图 Stage contract；Full 原生构建已接入 Gemma4 `mtmd` projector/图像路径（fake/journal/JVM/交叉编译开发回归通过）。旧 `frontend/` 已冻结；PC 管理面缺口以 `frontend_cybergothic` 为唯一目标，详见《前端安卓后端接口与功能缺口审查-2026-08-22》。
>
> **2026-08-23 界面复核补充**：Android 继续采用原生 Material 3 极简风（深色黑底白字、浅色白底黑字、四个底部导航页），不迁移 PC 赛博哥特 canvas/三栏工作台；这是触控、小屏、耗电与可读性优先的产品决定。PC 的新赛博哥特前端已有 pywebview 独立启动器，`CY-PKG-01` 已将默认后端静态页和标准安装包输入统一为 `frontend_cybergothic/dist`；旧 `frontend/dist` 只保留显式兼容覆盖，真实包首启归 E8。完整证据与页面映射见《桌面赛博哥特与安卓原生界面复核-2026-08-23》。
>
> 结论一句话：**安卓版当前 = 聊天客户端（远程转发）+ Full 本地单模型 GGUF 推理，并已具备 Worker/Stage 与 Gemma4 mmproj 的本机开发门；PC 版的多模态真实质量、任务链真机运行、实验判题、模型管理、运维工具仍未形成 Android 产品闭环**

## 1. 差距总览（按类别）

| 类别 | PC 版能力 | 安卓现状 | 差距等级 |
|---|---|---|---|
| 多模态（图像理解） | Gemma 4 12B 图生文（原生 MTMD + Ollama 双轨） | Full 已有相册/远程图片与本地 `mtmd`/mmproj 开发路径；真实设备 RAM、语义、多图与长时未验收，Lite fail-closed | 🔴 大 |
| 多模态（图像生成） | SD 1.5 全套（文生图/图生图/IP-Adapter/inpaint/指令编辑） | 本地不做；远程 SD 请求、图片/结果和轮询取消已有开发路径，真实 PC SD/设备验收未完成 | 🟡 中（远程子集） |
| 分布式参与 | 层流水线 Worker、任务链/任务图、TP 孤岛、外部服务、推测解码 | ⚠️ Android Full Worker 准入与 Stage executor 开发门已完成；真实设备/认证/长时仍未验收 | 🔴 大（需设备验收与运行时质量门） |
| 模型管理 | MODEL-TOOLS 全套（导入向导/GGUF 转换/受管下载/注册表/扫描） | SAF 本地目录 + 主节点 GGUF Range/SHA/续传下载开发路径；无完整模型注册表/来源许可/多工件管理 | 🟠 中 |
| 模型能力梯度 | Qwen 1.8B / Qwen3-4B / Gemma4 12B / SD 多模型并存 | Full 单次一个 GGUF（llama.cpp CPU） | 🟠 中 |
| 实验与判题 | EX-N3 判题（LLM/Gemma/SD 自动门）、标定、质量总 gate | ❌ 无实验入口 | 🟠 中 |
| 用户/凭据管理 | control auth（bootstrap/恢复码/DPAPI token/多用户） | Keystore 登录/会话开发路径已具备；无 Auth App 引导、恢复码、成员/Tailscale 管理产品页 | 🟠 中 |
| 抗弱网 | TCP 直连专线/WSS/ICE-TCP（计划）、重试与续传 | 基础 HTTP + IPv6 地址选择 | 🟡 小-中 |
| 运维工具 | TUI 管理、跨节点日志聚合（qlh log）、健康门 | 设置中已有有界脱敏日志、连接诊断与更新开发路径；无 TUI/跨节点运营工作台 | 🟡 中 |
| 更新与数据 | Launcher A/B 自更新、Ed25519 签名、跨卷数据保留 | APK 更新清单、下载/校验/安装权限开发路径已具备；真实更新源、商店/安装和长时日志未验收 | 🟡 中 |
| Web 管理面 | 模型舰队、SD 图像工作区、任务图、用户管理、设置 API | 四个原生页（对话/图像/会话/设置）；无桌面运营控制面 | 🟡 中 |
| 前端工程 | 双语 README、E2E 测试套件 | Full/Lite JVM、AndroidTest 与原生交叉编译开发门；设备视觉/可用性验收后置 | 🟡 小（验收覆盖） |

## 2. 逐项差距明细

### 🔴 A. 多模态（差距最大，两子项）

**A1 图像理解（图生文）**
- PC：Gemma 4 12B 原生绑定（MTMD，`gemma4_native_image_smoke` 已验）+ Ollama 双轨；EX-N3 Gemma 判题依赖此能力。
- 安卓：已具备相册选择、压缩与远程图片发送；Full `qlh_llama_jni.cpp` 已接入 `mtmd` projector 生命周期、受限内存图片解码和 chunk forward，并由 Gemma4 资产/vision capability 闸门控制；Lite 仍保持 fail-closed。
- 差距点：① Gemma4 原生 ~7.3GB GGUF+mmproj 在 Android RAM 上的加载与采样验收；② 真机图片语义、多图和长时温控验收；③ 发布包/设备资源门仍不得因交叉编译通过而自动放开。

**A2 图像生成（SD 1.5）——走远程推理**
- PC：完整图像工作区（五资产 15GB，全部自动质量门通过）。
- 安卓：**本地不做**——图像生成全丢给 PC 主节点：安卓 UI 发起生成请求（提示词/参考图上传）→ 主节点 SD 工作区执行（质量门照常）→ 结果图回传显示。
- 差距点：① 已有远程生成/图片上传/结果回传与轮询取消开发路径，仍需真实 PC SD、设备网络和长任务验收；② 缺桌面资产目录、许可证/导入、Inpaint/指令编辑、四宫格工作流与分布式遥测操作面；③ 不要求安卓本地 SD 任何能力（省资源）。

### 🔴 B. 分布式参与（需协议，路线 A/B 未实施）

| 子项 | PC | 安卓 | 备注 |
|---|---|---|---|
| 完整推理 Worker（主节点派发任务） | ✅ 多 Worker 并行 | ⚠️ v2 长连接/租约/幂等/结果回传和节点准入已有开发门；真实 Android offer→result 未验收 | 路线 A；AND-API-03 已把 executor 绑定本地 `InferenceService` |
| 任务链/任务图拆分 | ✅ G0-G5.2 全套（fencing/journal/优化器/故障注入验证） | ⚠️ 已具备 `full_inference` Stage executor 与 fake/journal 回归；真实设备任务图仍后置 | 路线 B |
| 层流水线 | ✅ PyTorch 层间（gemma4 P1 adapter） | ❌ `pipeline_worker=false` 明确排除 | 路线 C 仅预研，**不接入** |

### 🟠 C. 模型管理与能力梯度

- PC：MODEL-TOOLS P0-P8（导入向导支持 HF/ModelScope 真实下载、GGUF 转换器、资产扫描、注册表、原子发布）。
- 安卓：SAF 选目录与主节点 GGUF Range/SHA/续传下载均有开发路径；仍以单模型本地运行和有限 Settings 入口为主。
- 差距点：① 完整模型注册表、来源/许可证、资产状态与多工件展示；② 多模型切换（当前单模型加载）；③ 真实大文件、断网和 SAF 提供器验收。

### 🟠 D. 实验与判题

- PC：EX-N3 质量总 gate（required=true 三侧）、LLM 判题（Qwen3-4B v2 正确率判据已恢复）、Gemma 判题、SD 自动门、双人目视流程。
- 安卓：无。
- 差距点：Android 端跑判题（本地小模型判题理论上可行——Qwen3-4B GGUF 安卓可跑）——**可选实验线**，优先级低。

### 🟠 E. 用户/凭据管理

- PC：control 面 bootstrap 注册、TOTP、恢复码、锁定期、DPAPI token 保护、多用户角色（owner/admin/member）。
- 安卓：Keystore 登录/会话、轮换和登出清理已有客户端开发门；未形成账户页和 Auth App 产品引导。
- 差距点：① Auth App QR/字符串配置与恢复码展示；② 多用户成员和 Tailscale 绑定管理；③ 真实 auth-required、真机密钥库与 token 生命周期验收。

### 🟡 F. 运维与可靠性

- PC：TUI 管理（27 命令）、跨节点日志聚合（P7）、启动预检、Launcher 更新（A/B/签名）、数据保留工具。
- 安卓：Settings 已接更新清单、下载/签名校验/安装权限、有界脱敏日志和主节点/认证连接诊断开发路径；无 PC TUI 或跨节点运营台。
- 差距点：① 真实 APK/商店、外部更新源、真机网络切换和长时日志验收；② 只读移动端集群摘要是否有真实使用价值；③ 不把 PC 运维台机械复制到手机。

## 3. 历史开发排期（2026-08-17，保留票据沿革）

下表中的 AND-A1、AND-A2、AND-B、AND-C、AND-E、AND-F 与 `AND-API-01..05` 已完成本机开发门，不能再被当作未开发票；当前仅保留对应真机、主节点、真实模型与发布链路验收。最新页面取舍和桌面交付票以《桌面赛博哥特与安卓原生界面复核-2026-08-23》及《待完成工作清单与推进顺序-2026-08-23》为准。

| 优先级 | 项 | 理由 | 阻塞 |
|---|---|---|---|
| **P1** | A1 图像理解（UI + JNI mmproj 接线） | 多模态是最醒目差距；llama.cpp 子模块已含 mmproj 支持 | 需确认安卓端 Gemma4 GGUF+mmproj 内存可行性（16GB RAM 设备可试） |
| **P1** | B 完整推理 Worker（路线 A 最小协议） | 远期计划主线；基础已具备（前台 Service/取消） | 需设计长连接协议（可复用 PC 任务链租约语义的简化版） |
| **P2** | A2 图像生成远程推理（请求 API + 图片上传/回传 UI） | 生成全丢 PC 主节点，安卓零本地负担 | 需生成请求端点（PC 已有 SD 工作区 API）+ 任务轮询 |
| **P2** | C 应用内模型下载 + SHA 校验 | 免人工放文件；配合整合包安卓版（纯 GGUF）分发 | 需 serve 分发端点（已有） |
| **P2** | E 凭据管理（Keystore + 连接 token） | 主节点开 auth 后安卓无法直连（现状缺口） | 需 control auth 的安卓客户端流程 |
| **P3** | F 应用内更新/日志 | 运维体验 | 无 |
| **不做** | ~~本地判题~~ / 安卓本地图像生成 | 判题与 SD 生成都留在 PC 侧，安卓不承担 | — |

## 3.1 Android 分票排期（2026-08-17）

排期以“先收口协议与失败边界，再接入设备能力”为原则；未完成真机、RAM 或 CUDA 验收的票可以先开发，但不能在清单中标记为已验收。

| 票号 | 优先级 | 范围 | 依赖 | 状态 |
|---|---|---|---|---|
| **AND-A1-01** | P1 | 远程图像请求契约：`image_data_urls` DTO、最多 4 张、Full 本地模式 fail-closed、失败重试保留载荷 | PC `/api/chat` 已有字段 | **开发完成；设备验收后置** |
| AND-A1-02 | P1 | SAF/系统图片选择器、尺寸/格式压缩、data URL 预览与上传 | AND-A1-01 | **开发完成；设备验收后置** |
| **AND-A1-03** | P1 | Gemma4 GGUF + mmproj 资源登记、JNI MTMD 能力闸门、本地模式接线 | AND-A1-02；需 Android 真机验收 | **开发完成；JNI/设备验收后置** |
| **AND-B-01** | P1 | Worker A 最小协议：hello、lease、幂等键、结果/错误 envelope | PC 任务链契约 | **开发完成；传输/PC准入后置** |
| **AND-B-02** | P1 | Android 前台 Service 承载 Worker 客户端、断线/取消状态机 | AND-B-01 | **开发完成；真实连接/设备验收后置** |
| **AND-B-03** | P1 | ServerSocket/Mock Worker 契约测试与跨平台构建检查 | AND-B-02 | **开发完成；真实认证/准入/设备验收后置** |
| **AND-API-02** | P0 | PC scheduler Android Full Worker 准入、精确模型身份、单并发和资源门 | AND-B-01/02/03 | **开发完成；真实认证、温控/电量和设备验收后置** |
| **AND-API-03** | P1 | Android Full Worker 统一 Stage executor、模型门、有界输入和 journal provenance | AND-B-03 | **开发完成；真实 Android 推理、网络和长时验收后置** |
| **AND-API-04** | P1 | Full `mtmd`/Gemma4 mmproj projector 生命周期、内存图片解码、MTMD chunk forward 与 bounded generation | AND-A1-03；无需真机开发 | **开发完成；真实设备 RAM、图片质量和长时验收后置；Lite 不提供本地多模态** |
| **AND-API-05** | P2 | Android 更新检查/下载/安装权限、脱敏日志导出/上报和连接健康诊断 | AND-C-01、AND-E-01 | **开发完成；真实 APK/商店、更新源和真机网络验收后置** |
| **AND-A2-01** | P2 | 远程 SD 生成 DTO、上传/任务状态轮询与取消 | PC SD API | **开发完成；真实 PC SD/设备验收后置** |
| **AND-A2-02** | P2 | SD 结果下载、缩略图/失败状态 UI | AND-A2-01 | **开发完成；真实 PC SD/设备验收后置** |
| **AND-C-01** | P2 | 主节点模型清单、下载进度、SHA-256 校验与断点续传 | 主节点分发 API | **开发完成；真实大文件/断网/SAF 提供器验收后置** |
| AND-E-01 | P2 | Auth/Keystore token 保存、轮换与登出清理 | 主节点 auth API | **客户端开发完成；真实 auth API/真机验收后置** |
| AND-F-01 | P3 | 应用更新、日志上传与连接健康诊断 | Launcher/日志 API | **开发完成（f05bcf3）；真实安装/更新/日志链路验收后置** |

本阶段不排本地 SD、生成本地判题和分布式真机性能标定；这些属于后续验收项，不作为 Android 开发阻塞条件。

## 3.2 Android 移动控制面排期（2026-08-23）

此次将“缺完整集群、模型舰队、审计与账户控制面”拆为渐进式移动端能力，而非复刻 PC 管理台：

| 票号 | 优先级 | 目标 | 权限边界 | 状态 |
|---|---:|---|---|---|
| `AND-CTRL-01` | P0 | 在设置页增加只读集群概览，显示运行模式、就绪状态、当前任务与最多 8 个节点。 | 只能刷新；不提供节点删除/注销、容量/层配置、切主、队列或任务操作。 | **本机开发完成**：修正 `/api/cluster/status` 节点映射 DTO；Full JVM `128 passed`，`compileFullDebugAndroidTestKotlin` 通过；真机/主节点验收后置。 |
| `AND-CTRL-02` | P1 | 模型舰队只读统一视图：已选本地模型、主节点当前模型、注册表、资产状态和已验签 GGUF 清单，最多展开 8 项。既有下载进度保留在原 SAF 下载面板。 | 禁止远程删除、注册、代理/源调整与强制切换；保留用户主动下载的 SHA/SAF 边界。 | **本机开发完成**：Full JVM `132 passed`，`compileFullDebugAndroidTestKotlin` 通过；真机/主节点/大工件/SAF 验收后置。 |
| `AND-CTRL-03` | P1 | 有界审计/活动只读列表，投影工作流阶段、attempt 状态和 review 摘要。 | 禁止投票、重派、删除、过期或提交结果；不下发提示词、原始错误、路径、投票评论或投票人身份。 | **本机开发完成**：主节点 `summary=1` 安全投影与 `limit`、Android Settings 审计折叠组；Full JVM `135 passed`，后端任务图定向 `1 passed`，`compileFullDebugAndroidTestKotlin` 通过；真机/真实授权验收后置。 |
| `AND-CTRL-04` | P1 | 账户/Auth App 会话：能力、用户角色/到期、登录/TOTP/恢复码、Keystore 会话校验和登出。 | 不展示或持久化种子、恢复码明文、bearer、管理员密钥或 cluster secret。 | **本机开发完成**：Android Settings 新增账户/Auth App 折叠组，统一解析 gateway `required/enforced` 与单体 `available/reason_code` capability；接入 Keystore 登录/会话校验/登出，能力不可用、会话 401 和退出失败均 fail-closed；登录错误不回显响应体。Full JVM `138 passed`，AndroidTest Kotlin 编译通过；真实 Auth App、control/gateway 和真机验收后置。 |
| `AND-CTRL-05` | P2 | Owner/admin 的成员、Tailscale 绑定与入群审批受限入口。 | 高风险/批量/破坏性管理保持 PC 优先；Android 仅在明确授权、二次确认和审计均具备时开放，`review_admin` 未迁移时只读。 | **本机开发完成**：管理摘要、成员/绑定/审计投影和撤销确认 UI 已接入；真实权限/审计/真机联调后置。 |

`AND-CTRL-01/02/03/04/05` 与 `CY-PKG-01` 已完成本机开发门；下一步转入真实桌面包/E8 和 Android 管理面真机联调。`review_admin` 审批仍保持后端未迁移的只读边界。

`AND-B-01` 只冻结并实现跨语言的 `qlh.task_worker` v2 envelope、Android Full Worker 能力声明、严格 UTF-8/canonical JSON、租约/摘要校验和 1024 条 `message_id` replay cache。Android 的 `worker_kind=android_full_worker` 已被 PC v2 schema 接受；后续 `AND-API-02` 已开放注册节点类型/worker kind 对齐的 scheduler 准入，`AND-API-03` 又接入了统一 Stage executor。真实认证、设备资源门、Android offer→result 和长时验收仍后置。

`AND-B-02` 已新增独立 `TaskWorkerService` 前台生命周期壳、可注入的 length-prefixed transport、连接/hello/有界指数退避状态机，以及 offer/lease/result/error/cancel 的 attempt identity fencing。断线会将活动 attempt 标记为 `LOST`，迟到结果被丢弃；Android 主动取消使用 `stage_error`，只有协调器发起的 `stage_cancel` 才回 `stage_cancelled`。本票尚未开放 PC scheduler 准入、认证协议和真实设备验收，ServerSocket/Mock Worker 跨平台契约归 `AND-B-03`。

`AND-B-03` 已用本地 `ServerSocket` 验证 Android transport 与 PC 现有 4 字节大端 length-prefix、`task_worker/json` outer envelope 和 v2 inner envelope 的双向 hello/ack/UTF-8 result；非法 outer frame 会 fail-closed。PC 侧直接复用 `tcp_comm.build_message/parse_message` 做同一 framing 检查。Full/Lite JVM 与 PC wire/protocol 专项通过，`compileFullDebugAndroidTestKotlin` 通过；这仍不代表真实认证、PC scheduler 准入或物理设备链路已验收。

`AND-API-02`（2026-08-22）已完成本机准入门：scheduler 只接受注册节点类型与 `worker_kind` 一致的 PC/Android Full Worker；provider 对 Android 强制 `resource_gate.admitted=true`、单并发和 v2 健康连接，Stage 请求继续按完整模型身份精确匹配。Android `TaskWorkerService` 增加模型身份/资源门能力构建器，资源门默认关闭；fake worker Python 回归 `63 passed`，Android Worker 协议/能力 JVM 回归通过，真实认证、offer→result、温控/电量和设备验收后置。

`AND-API-03`（2026-08-22）已完成本机开发门：Android `TaskWorkerClient` 支持可注入 `AndroidFullWorkerStageExecutor`，Service 将其绑定本地 `InferenceService`；executor 只接受 `full_inference`，严格匹配完整模型身份并限制 prompt/context/sampling 参数，失败/取消沿用 lease fencing。Python fake Android provider + `SQLiteTaskJournal` 验证 offer→accept→result、attempt 的 `provider_kind/provider_node_id` 和显式 `commit_result` 终态持久化；Full/Lite JVM 单测通过。真实 Android 模型执行、资源门、认证、网络和长时验收后置。

`AND-API-04`（2026-08-22）已完成本机开发门：Full CMake 启用独立 `mtmd` 静态库，JNI 接入用户自持 Gemma4 mmproj 打开/释放、vision capability、单/多图内存解码、MTMD tokenize/chunk eval 和有界生成；`InferenceService` 在调用前校验 Gemma4 主模型、mmproj 文件身份/大小和 native capability，`ChatRepository` 的 Full 本地图片请求改走该路径，Lite 仍 thin/stub。Full/Lite JVM、Lite 原生构建和 Full arm64 交叉编译通过；真实 Android RAM、Gemma4 12B 图片语义、多图/长时/温控验收后置。

`AND-API-05`（2026-08-22）已完成本机开发门：Settings 接入更新清单检查、下载进度、APK 完整性/签名校验、FileProvider 安装和未知来源权限回到应用后的刷新；日志查看/复制/分享统一使用有界脱敏 bundle；连接诊断接主节点/认证会话探针，手动上报只发送脱敏摘要。Full/Lite Kotlin 编译、更新/脱敏/API client 定向回归通过；真实 APK/商店安装、外部更新源、真机网络切换和长时日志后置。

`AND-A2-01` 已新增 Android PC SD 客户端契约：生成/编辑请求 DTO、PNG/JPEG/WebP multipart 参考图/遮罩上传、job 快照解析、终态轮询和远程取消；轮询有界且遵守协程取消，job ID 使用 fail-closed 字符集校验。结果 blob 下载、缩略图和界面状态归 `AND-A2-02`，真实 PC SD 引擎、网络与设备验收后置。

`AND-A2-02` 已新增独立 Android 图像页：文生图与参考图 img2img 入口、任务状态/进度/取消反馈、结果 blob 有界下载、1024px 缩略图解码、空结果/失败/取消状态展示；Full/Lite 共用同一远程链路，不在 Android 本地加载 SD。

`AND-C-01` 已接通 PC `/api/models/gguf` 与带 `Range` 的模型分发端点；Android Full 模式可列出主节点 GGUF，显示容量和 SHA-256 摘要，并续传到用户授权的 SAF 目录。下载只在精确匹配 `206 Content-Range` 时追加，服务器返回完整 `200` 时安全重写；临时 `.part` 文件在大小和 SHA-256 均通过后才原子提升并选中。PC 清单不再泄露服务端绝对路径。真实大文件下载、断网重连、不同 DocumentsProvider 的重命名行为及物理设备空间不足仍后置验收。

## 4. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-23 | `AND-CTRL-02` 完成：Settings 新增只读“模型舰队”，聚合主节点当前模型、注册表、已发现本地资产和已验签 GGUF 清单，并显示 Android 已选模型；只显示运行中/可用/待验证/缺失及最多 8 项，不保留服务器路径、下载 URL、代理或注册表写入入口。现有 SAF 下载进度不变。Full JVM `132 passed`、AndroidTest Kotlin 编译通过，真实主节点、大工件和 SAF 验收后置。 |
| 2026-08-23 | `AND-CTRL-03` 完成：主节点工作流与复核票新增有界 `summary=1` 安全投影，Android Settings 新增只读“审计与活动”折叠组，最多 8 个工作流/复核票、每阶段 8 个阶段/4 个 attempt；不下发提示词、原始错误、路径、lease、输出元数据、投票评论或写操作。Full JVM `135 passed`、后端任务图定向 `1 passed`、AndroidTest Kotlin 编译通过，真实主节点与授权验收后置。 |
| 2026-08-23 | `AND-CTRL-04` 完成：Android Settings 新增账户/Auth App 会话折叠组，统一 gateway/单体 capability，显示脱敏账号、角色和到期时间；登录支持 Auth App 验证码/恢复码二选一，复用 Keystore 会话，退出始终清本地凭据；不展示 token、seed、恢复码明文或管理员密钥，认证错误不回显响应体。Full JVM `138 passed`、AndroidTest Kotlin 编译通过；真实 Auth App/control/gateway/真机验收后置。 |
| 2026-08-23 | 新增移动控制面分期。`AND-CTRL-01` 将 Settings 的旧 `/api/cluster/status` DTO 从过期的数组/计数假设改为服务端实际节点映射，并新增只读、最多 8 节点的集群概览；不接任何调度或节点管理写操作。Full JVM `128 passed`、AndroidTest Kotlin 编译通过，真实主节点和真机验收后置。 |
| 2026-08-18 | AND-E-01 客户端开发门完成：Android Keystore-backed session store 已接入 ApiClient，登录/会话校验/登出、Bearer 注入、401 清理和 Authorization 日志脱敏均已实现；JVM contract tests 覆盖登录、授权头、失效会话和离线登出。AND-F-01 核对确认已由 f05bcf3 交付。真实 auth API、安装更新与设备验收后置。 |
| 2026-08-17 | AND-C-01 已完成：Android Full 设置页新增主节点 GGUF 目录、进度与校验状态，下载落入用户授权 SAF 目录并支持严格 Range 续传、大小/SHA-256 校验和 `.part` 提升；PC 清单移除绝对目录泄露并增加 Range 契约测试；Full/Lite JVM、AndroidTest Kotlin 编译和 PC 安全边界测试通过，真实大文件/断网/SAF 提供器验收后置 |
| 2026-08-17 | AND-A2-02 已完成：新增 Android 图像导航页、提示词/反向提示词/步数表单、参考图选择与预览、远程生成/变体提交、任务取消、结果 blob 32 MiB 有界下载和 1024px 缩略图展示；补齐状态机、ApiClient 下载和 Compose 契约测试，Full/Lite JVM 与 Full AndroidTest Kotlin 编译通过，真实 PC SD/网络/设备验收后置 |
| 2026-08-17 | AND-A2-01 已完成：Android `ApiClient` 接入 PC `/api/diffusion/generate`、`/edit`、`/blobs`、`/jobs/{id}` 与取消端点，新增 snake_case DTO、16 MiB 上传限制、multipart 参考图/遮罩上传、有限终态轮询和协程取消传播；Full/Lite 全量 JVM 单测、远程 SD 契约测试与 Full AndroidTest Kotlin 编译通过，真实 PC SD/设备验收后置 |
| 2026-08-17 | AND-B-03 已完成：新增 Android `ServerSocket` 双向 task-worker framing/hello-ack/UTF-8 result 契约测试、非法 outer frame 拒绝测试和 PC `tcp_comm` length-prefix 对照测试；Full/Lite JVM、PC 23 项协议/wire 专项与 Full AndroidTest Kotlin 编译通过，真实认证、scheduler 准入和设备验收后置 |
| 2026-08-22 | AND-API-02 已完成本机开发：PC scheduler 开放注册 Android Full Worker，但要求节点类型/worker kind 对齐；provider 引入显式 resource gate、精确模型身份和单并发门；Android Service 增加模型身份/资源门能力构建器，fake worker `63 passed`，真实认证、温控/电量和设备验收后置 |
| 2026-08-22 | AND-API-03 已完成本机开发：Android Full Worker 接入可注入 Stage executor 并绑定本地 `InferenceService`；Python fake provider + SQLite journal 验证 offer/result 与 attempt provenance，Full/Lite JVM 单测通过；真实 Android 执行、网络和长时验收后置 |
| 2026-08-22 | AND-API-04 已完成本机开发：Full CMake/JNI 接入 `mtmd` projector 与受限内存图片路径，服务层保留 Gemma4 资产/vision capability 闸门；Full/Lite JVM、Lite 原生和 Full arm64 交叉编译通过；真实设备与图片质量验收后置 |
| 2026-08-22 | AND-API-05 已完成本机开发：Settings 接入更新/下载/安装权限、FileProvider、脱敏日志 bundle、连接健康探针和可选客户端错误上报；Full/Lite Kotlin、更新/脱敏/API client 定向回归通过；真实 APK/商店与真机验收后置 |
| 2026-08-17 | AND-B-02 已完成：新增 Android `TaskWorkerService` 前台 Worker 生命周期、socket length-prefixed transport、连接/hello/有界退避、断线丢弃迟到结果、lease 与取消状态机；增加 Full JVM 状态机回归，真实连接、认证、PC 准入和设备验收后置，ServerSocket 契约留待 AND-B-03 |
| 2026-08-17 | AND-B-01 已完成：新增 Android `qlh.task_worker` v2 Kotlin codec，覆盖 hello/ack、stage offer/accept、lease renew、result/error、cancel envelope；固定 canonical JSON + SHA-256、严格字段/ID/租约校验、UTF-8 拒绝和有界 message-id replay cache；PC `task_worker_protocol.py` v2 接受 `android_full_worker` 角色但不改变当前 scheduler/transport 准入门，Full/Lite JVM 与 PC 协议专项通过 |
| 2026-08-17 | AND-A1-03 已完成：登记 Gemma4 原生 GGUF/mmproj 固定文件名、大小与 SHA-256 常量，覆盖 SAF/internal 资源配对扫描（当前仅做文件身份/大小闸门，完整 SHA 校验仍由导入链路负责）；新增 JNI MTMD 能力查询并对当前 text-only APK fail-closed；运行时状态与 presence payload 已暴露资源/能力原因，真实 MTMD 接线与真机验收后置 |
| 2026-08-17 | AND-A1-02 已完成：系统图片选择器、源文件/输出大小上限、最长边缩放、JPEG data URL 编码、缩略图预览、清除与错误状态；Full 模式禁用图片入口，设备验收后置 |
| 2026-08-17 | 建立 AND-A1/B/A2/C/E/F 分票排期；AND-A1-01 已完成远程图像请求契约、Full 模式 fail-closed 和重试载荷保留，Full/Lite JVM 单测通过，图片选择器与 mmproj/JNI 留待后续票 |
| 2026-08-16 | 建立本清单（全面排查 PC vs 安卓差距：12 类 7 大项；P1-P3 排期建议） |
| 2026-08-16 | 修订：① **本地判题移除**（安卓不承担判题，留在 PC）；② **图像生成改远程推理**（安卓 UI 发起 → PC SD 工作区生成 → 回传，本地零负担）；排期表同步 |
