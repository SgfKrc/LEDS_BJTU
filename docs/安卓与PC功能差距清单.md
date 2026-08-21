# 安卓版与 PC 版功能差距清单

> 状态：现行（随两端开发持续更新）
>
> 更新日期：2026-08-22
> 适用范围：Android Full/Lite 相对 PC 版（Windows/Linux 主节点）缺失功能的**全面排查清单**，供排期与任务分配使用；能力现状依据 [Android 版本远期计划](Android版本远期计划.md)（2026-07-28 基线 + C5 真机验收）、README 与源码

> 本次接口复核补充：Android HTTP 控制面已覆盖登录/会话、聊天、远程 SD、主节点 GGUF Range/SHA 下载、bootstrap 和节点注册；但当前用 register 代替独立 heartbeat，Android Full Worker 尚未进入 PC scheduler 正式准入，任务图 Stage executor 尚未接通。旧 `frontend/` 已冻结；PC 管理面缺口以 `frontend_cybergothic` 为唯一目标，详见《前端安卓后端接口与功能缺口审查-2026-08-22》。
>
> 结论一句话：**安卓版当前 = 聊天客户端（远程转发）+ Full 本地单模型 GGUF 推理；PC 版的多模态、任务链、实验判题、模型管理、运维工具在安卓上全部缺失**

## 1. 差距总览（按类别）

| 类别 | PC 版能力 | 安卓现状 | 差距等级 |
|---|---|---|---|
| 多模态（图像理解） | Gemma 4 12B 图生文（原生 MTMD + Ollama 双轨） | ❌ 无图像输入/输出，聊天纯文本 | 🔴 大 |
| 多模态（图像生成） | SD 1.5 全套（文生图/图生图/IP-Adapter/inpaint/指令编辑） | ❌ 本地无；**走远程推理**——安卓 UI 发起请求，全丢给 PC 主节点 SD 工作区生成后回传 | 🟡 中（远程通道） |
| 分布式参与 | 层流水线 Worker、任务链/任务图、TP 孤岛、外部服务、推测解码 | ❌ 仅 presence 注册，`pipeline_worker=false` | 🔴 大（需协议） |
| 模型管理 | MODEL-TOOLS 全套（导入向导/GGUF 转换/受管下载/注册表/扫描） | 仅 SAF 本地目录选 GGUF | 🟠 中 |
| 模型能力梯度 | Qwen 1.8B / Qwen3-4B / Gemma4 12B / SD 多模型并存 | Full 单次一个 GGUF（llama.cpp CPU） | 🟠 中 |
| 实验与判题 | EX-N3 判题（LLM/Gemma/SD 自动门）、标定、质量总 gate | ❌ 无实验入口 | 🟠 中 |
| 用户/凭据管理 | control auth（bootstrap/恢复码/DPAPI token/多用户） | ❌ 无本地凭据管理 | 🟠 中 |
| 抗弱网 | TCP 直连专线/WSS/ICE-TCP（计划）、重试与续传 | 基础 HTTP + IPv6 地址选择 | 🟡 小-中 |
| 运维工具 | TUI 管理、跨节点日志聚合（qlh log）、健康门 | ❌ 无 | 🟡 小 |
| 更新与数据 | Launcher A/B 自更新、Ed25519 签名、跨卷数据保留 | ❌ APK 手动安装 | 🟡 小 |
| Web 管理面 | 模型舰队、SD 图像工作区、任务图、用户管理、设置 API | 仅聊天 + 设置 | 🟡 小 |
| 前端工程 | 双语 README、E2E 测试套件 | JVM 单测 17 例 + androidTest 2 例 | 🟡 小（测试覆盖） |

## 2. 逐项差距明细

### 🔴 A. 多模态（差距最大，两子项）

**A1 图像理解（图生文）**
- PC：Gemma 4 12B 原生绑定（MTMD，`gemma4_native_image_smoke` 已验）+ Ollama 双轨；EX-N3 Gemma 判题依赖此能力。
- 安卓：已具备相册选择、压缩与远程图片发送；`qlh_llama_jni.cpp` 仍是文本生成，并已提供 MTMD 能力闸门与 Gemma4 资源登记，但没有图像 token 处理（llama.cpp submodule 47e1de77 含 mmproj 支持但 JNI 尚未接入）。
- 差距点：① JNI 图像编码（mmproj）实际接线；② Gemma4 原生 ~7.3GB GGUF+mmproj 在 Android RAM 上的加载与采样验收；③ 本地模式在 MTMD 接线完成前保持 fail-closed。

**A2 图像生成（SD 1.5）——走远程推理**
- PC：完整图像工作区（五资产 15GB，全部自动质量门通过）。
- 安卓：**本地不做**——图像生成全丢给 PC 主节点：安卓 UI 发起生成请求（提示词/参考图上传）→ 主节点 SD 工作区执行（质量门照常）→ 结果图回传显示。
- 差距点：① 安卓端生成请求 API + 图片上传/回传 UI；② 任务状态轮询/推送（可复用 PC 任务链语义的简化版）；③ 不要求安卓本地 SD 任何能力（省资源）。

### 🔴 B. 分布式参与（需协议，路线 A/B 未实施）

| 子项 | PC | 安卓 | 备注 |
|---|---|---|---|
| 完整推理 Worker（主节点派发任务） | ✅ 多 Worker 并行 | ❌ 无长连接协议/租约/幂等/结果回传 | 远期计划路线 A，具备基础（前台 Service/生成/取消）只缺协议 |
| 任务链/任务图拆分 | ✅ G0-G5.2 全套（fencing/journal/优化器/故障注入验证） | ❌ 无 Stage 概念 | 路线 B |
| 层流水线 | ✅ PyTorch 层间（gemma4 P1 adapter） | ❌ `pipeline_worker=false` 明确排除 | 路线 C 仅预研，**不接入** |

### 🟠 C. 模型管理与能力梯度

- PC：MODEL-TOOLS P0-P8（导入向导支持 HF/ModelScope 真实下载、GGUF 转换器、资产扫描、注册表、原子发布）。
- 安卓：仅 SAF 选目录（人工放文件）；无受管下载（远期计划有"PC 分发/应用内下载"规划未实施）。
- 差距点：① 应用内从主节点/局域网下载 GGUF；② 模型版本/SHA 校验显示；③ 多模型切换（当前单模型加载）。

### 🟠 D. 实验与判题

- PC：EX-N3 质量总 gate（required=true 三侧）、LLM 判题（Qwen3-4B v2 正确率判据已恢复）、Gemma 判题、SD 自动门、双人目视流程。
- 安卓：无。
- 差距点：Android 端跑判题（本地小模型判题理论上可行——Qwen3-4B GGUF 安卓可跑）——**可选实验线**，优先级低。

### 🟠 E. 用户/凭据管理

- PC：control 面 bootstrap 注册、TOTP、恢复码、锁定期、DPAPI token 保护、多用户角色（owner/admin/member）。
- 安卓：无登录/凭据（直接连主节点 IP）。
- 差距点：① 连接主节点的凭据存储（Android Keystore）；② 多用户登录（若主节点启用 auth-required）；③ token 生命周期管理。

### 🟡 F. 运维与可靠性

- PC：TUI 管理（27 命令）、跨节点日志聚合（P7）、启动预检、Launcher 更新（A/B/签名）、数据保留工具。
- 安卓：无对应项；更新靠手动 APK。
- 差距点：① 应用内更新（校验签名后安装）；② 日志上报；③ 连接健康诊断（当前仅"测试连接"）。

## 3. 建议排期优先级（供任务分配）

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
| **AND-A2-01** | P2 | 远程 SD 生成 DTO、上传/任务状态轮询与取消 | PC SD API | **开发完成；真实 PC SD/设备验收后置** |
| **AND-A2-02** | P2 | SD 结果下载、缩略图/失败状态 UI | AND-A2-01 | **开发完成；真实 PC SD/设备验收后置** |
| **AND-C-01** | P2 | 主节点模型清单、下载进度、SHA-256 校验与断点续传 | 主节点分发 API | **开发完成；真实大文件/断网/SAF 提供器验收后置** |
| AND-E-01 | P2 | Auth/Keystore token 保存、轮换与登出清理 | 主节点 auth API | **客户端开发完成；真实 auth API/真机验收后置** |
| AND-F-01 | P3 | 应用更新、日志上传与连接健康诊断 | Launcher/日志 API | **开发完成（f05bcf3）；真实安装/更新/日志链路验收后置** |

本阶段不排本地 SD、生成本地判题和分布式真机性能标定；这些属于后续验收项，不作为 Android 开发阻塞条件。

`AND-B-01` 只冻结并实现跨语言的 `qlh.task_worker` v2 envelope、Android Full Worker 能力声明、严格 UTF-8/canonical JSON、租约/摘要校验和 1024 条 `message_id` replay cache。Android 的 `worker_kind=android_full_worker` 已被 PC v2 schema 接受，但 PC scheduler 当前仍只准入已注册 PC Full Worker；TCP framing、认证连接、Android 前台 Service、任务执行和取消状态机分别属于 `AND-B-02/03`，因此本票不代表 Android Worker 已可接单。

`AND-B-02` 已新增独立 `TaskWorkerService` 前台生命周期壳、可注入的 length-prefixed transport、连接/hello/有界指数退避状态机，以及 offer/lease/result/error/cancel 的 attempt identity fencing。断线会将活动 attempt 标记为 `LOST`，迟到结果被丢弃；Android 主动取消使用 `stage_error`，只有协调器发起的 `stage_cancel` 才回 `stage_cancelled`。本票尚未开放 PC scheduler 准入、认证协议和真实设备验收，ServerSocket/Mock Worker 跨平台契约归 `AND-B-03`。

`AND-B-03` 已用本地 `ServerSocket` 验证 Android transport 与 PC 现有 4 字节大端 length-prefix、`task_worker/json` outer envelope 和 v2 inner envelope 的双向 hello/ack/UTF-8 result；非法 outer frame 会 fail-closed。PC 侧直接复用 `tcp_comm.build_message/parse_message` 做同一 framing 检查。Full/Lite JVM 与 PC wire/protocol 专项通过，`compileFullDebugAndroidTestKotlin` 通过；这仍不代表真实认证、PC scheduler 准入或物理设备链路已验收。

`AND-A2-01` 已新增 Android PC SD 客户端契约：生成/编辑请求 DTO、PNG/JPEG/WebP multipart 参考图/遮罩上传、job 快照解析、终态轮询和远程取消；轮询有界且遵守协程取消，job ID 使用 fail-closed 字符集校验。结果 blob 下载、缩略图和界面状态归 `AND-A2-02`，真实 PC SD 引擎、网络与设备验收后置。

`AND-A2-02` 已新增独立 Android 图像页：文生图与参考图 img2img 入口、任务状态/进度/取消反馈、结果 blob 有界下载、1024px 缩略图解码、空结果/失败/取消状态展示；Full/Lite 共用同一远程链路，不在 Android 本地加载 SD。

`AND-C-01` 已接通 PC `/api/models/gguf` 与带 `Range` 的模型分发端点；Android Full 模式可列出主节点 GGUF，显示容量和 SHA-256 摘要，并续传到用户授权的 SAF 目录。下载只在精确匹配 `206 Content-Range` 时追加，服务器返回完整 `200` 时安全重写；临时 `.part` 文件在大小和 SHA-256 均通过后才原子提升并选中。PC 清单不再泄露服务端绝对路径。真实大文件下载、断网重连、不同 DocumentsProvider 的重命名行为及物理设备空间不足仍后置验收。

## 4. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-18 | AND-E-01 客户端开发门完成：Android Keystore-backed session store 已接入 ApiClient，登录/会话校验/登出、Bearer 注入、401 清理和 Authorization 日志脱敏均已实现；JVM contract tests 覆盖登录、授权头、失效会话和离线登出。AND-F-01 核对确认已由 f05bcf3 交付。真实 auth API、安装更新与设备验收后置。 |
| 2026-08-17 | AND-C-01 已完成：Android Full 设置页新增主节点 GGUF 目录、进度与校验状态，下载落入用户授权 SAF 目录并支持严格 Range 续传、大小/SHA-256 校验和 `.part` 提升；PC 清单移除绝对目录泄露并增加 Range 契约测试；Full/Lite JVM、AndroidTest Kotlin 编译和 PC 安全边界测试通过，真实大文件/断网/SAF 提供器验收后置 |
| 2026-08-17 | AND-A2-02 已完成：新增 Android 图像导航页、提示词/反向提示词/步数表单、参考图选择与预览、远程生成/变体提交、任务取消、结果 blob 32 MiB 有界下载和 1024px 缩略图展示；补齐状态机、ApiClient 下载和 Compose 契约测试，Full/Lite JVM 与 Full AndroidTest Kotlin 编译通过，真实 PC SD/网络/设备验收后置 |
| 2026-08-17 | AND-A2-01 已完成：Android `ApiClient` 接入 PC `/api/diffusion/generate`、`/edit`、`/blobs`、`/jobs/{id}` 与取消端点，新增 snake_case DTO、16 MiB 上传限制、multipart 参考图/遮罩上传、有限终态轮询和协程取消传播；Full/Lite 全量 JVM 单测、远程 SD 契约测试与 Full AndroidTest Kotlin 编译通过，真实 PC SD/设备验收后置 |
| 2026-08-17 | AND-B-03 已完成：新增 Android `ServerSocket` 双向 task-worker framing/hello-ack/UTF-8 result 契约测试、非法 outer frame 拒绝测试和 PC `tcp_comm` length-prefix 对照测试；Full/Lite JVM、PC 23 项协议/wire 专项与 Full AndroidTest Kotlin 编译通过，真实认证、scheduler 准入和设备验收后置 |
| 2026-08-17 | AND-B-02 已完成：新增 Android `TaskWorkerService` 前台 Worker 生命周期、socket length-prefixed transport、连接/hello/有界退避、断线丢弃迟到结果、lease 与取消状态机；增加 Full JVM 状态机回归，真实连接、认证、PC 准入和设备验收后置，ServerSocket 契约留待 AND-B-03 |
| 2026-08-17 | AND-B-01 已完成：新增 Android `qlh.task_worker` v2 Kotlin codec，覆盖 hello/ack、stage offer/accept、lease renew、result/error、cancel envelope；固定 canonical JSON + SHA-256、严格字段/ID/租约校验、UTF-8 拒绝和有界 message-id replay cache；PC `task_worker_protocol.py` v2 接受 `android_full_worker` 角色但不改变当前 scheduler/transport 准入门，Full/Lite JVM 与 PC 协议专项通过 |
| 2026-08-17 | AND-A1-03 已完成：登记 Gemma4 原生 GGUF/mmproj 固定文件名、大小与 SHA-256 常量，覆盖 SAF/internal 资源配对扫描（当前仅做文件身份/大小闸门，完整 SHA 校验仍由导入链路负责）；新增 JNI MTMD 能力查询并对当前 text-only APK fail-closed；运行时状态与 presence payload 已暴露资源/能力原因，真实 MTMD 接线与真机验收后置 |
| 2026-08-17 | AND-A1-02 已完成：系统图片选择器、源文件/输出大小上限、最长边缩放、JPEG data URL 编码、缩略图预览、清除与错误状态；Full 模式禁用图片入口，设备验收后置 |
| 2026-08-17 | 建立 AND-A1/B/A2/C/E/F 分票排期；AND-A1-01 已完成远程图像请求契约、Full 模式 fail-closed 和重试载荷保留，Full/Lite JVM 单测通过，图片选择器与 mmproj/JNI 留待后续票 |
| 2026-08-16 | 建立本清单（全面排查 PC vs 安卓差距：12 类 7 大项；P1-P3 排期建议） |
| 2026-08-16 | 修订：① **本地判题移除**（安卓不承担判题，留在 PC）；② **图像生成改远程推理**（安卓 UI 发起 → PC SD 工作区生成 → 回传，本地零负担）；排期表同步 |
