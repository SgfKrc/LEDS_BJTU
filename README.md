# 🧠 轻量化大模型分布式边缘推理优化系统

> **Language**: [English](docs/README.en.md) · [简体中文](README.md)
>
> 👀 **新人/评审快速入口**：[双语摘要速览 QLH at a Glance](docs/项目速览-QLH-at-a-Glance.md) —— 2 分钟了解项目是什么、已验证什么、如何上手；完整能力边界见本文档。

**面向异构边缘设备的 llama.cpp 主引擎分布式大模型推理系统**

模型量化 · 算子融合 · 分页KV缓存 · 图算法智能编排 · 多终端协同推理 · TUI 主入口 · 边缘优化

**v0.1.8.3**（更新日期：2026-09-15）

> 📌 当前主线：[主线开发计划：分布式推理与边缘优化](docs/主线开发计划-分布式推理与边缘优化-2026-09-14.md)；当前支线：[支线开发计划：外置迁移与 Koakumix](docs/支线开发计划-外置迁移与Koakumix-2026-09-14.md)；历史能力快照：**[项目进展与下一步计划](docs/archive/项目进展与下一步计划.md)**。
> 📦 **仓库边界已固化**：主仓只保留分布式推理核心、控制面、跨平台 TUI、模型合同和核心质量门；Android、Web/Desktop shell、发布和工具箱位于工作区兄弟仓库。详见[主仓拆分后仓库边界与环境合同](docs/主仓拆分后仓库边界与环境合同-2026-09-15.md)。
> 本 README 描述**已实现**的能力；标注 *PoC* 的部分默认关闭、能力边界见对应专项文档，不等同于生产能力。
> 适用范围：QLH 项目能力总览、快速上手与文档索引；能力边界与最新证据以专项文档、源码和测试为准。
> 🧰 **刚克隆仓库？先看 [克隆后资产获取清单](#-克隆后资产获取清单)。**

---

## 📋 项目简介

QLH 面向算力、内存和网络条件不同的异构边缘设备，包括 Windows/Linux 台式机、工作站、服务器、笔记本以及 Android 手机和平板。生产主线是 GGUF/llama.cpp：先完成单机基线，再攻克 host + RPC worker 模型分片，让单机装不下的模型由多个节点共同承载、每个节点只保有实际分配的部分。PyTorch/Safetensors 层流水线只保留为 PC 对照实验，任务图/整请求只作为临时完整模型回退；Web、Android UI、发布、工具和生图均在外置仓库或 Koakumix 支线维护。当前主线的硬门是裁切/低耦合 → llama.cpp 单机 → 同机双进程 RPC → PC 真机 → Android 真机，不能用旧双机证据替代新模型分片证据。

现已覆盖 **Windows PC + Linux PC + Android**。设备类型不是调度能力的充分条件：是否能参与某种分布式执行，还取决于运行引擎、模型格式、模型指纹、可用内存、加速器和网络拓扑。

### 主仓运行形态（四类边界）

项目按硬件能力和使用场景划分为四种软件版本：

| 级别 | 形态 | 目标设备 | 核心能力 | 不包含/不推荐 |
|------|----------|----------|----------|---------------|
| 1 | **Edge/TUI 单机** | 无 NVIDIA 的 Windows/Linux PC、弱设备 | llama.cpp + GGUF CPU/集显推理、跨平台 TUI、模型下载/选择 | torch、Web UI、发布工具、图像生成 |
| 2 | **PC llama.cpp 分片** | 普通 PC/工作站/异构节点 | llama host + `ggml-rpc-server` worker，目标是模型部分驻留和容量合并 | 未通过 G0-G2 前不得宣称生产 RPC |
| 3 | **PC Reference P** | NVIDIA/高内存 PC | PyTorch + Safetensors 层流水线，用于正确性、性能和容量对照 | 不进入边缘安装、默认路由或正式发行 |
| 4 | **Android worker/client** | Android 手机/平板 | 外置仓维护本地 GGUF、RPC worker 或 HTTP 客户端；按模型舰队选择能力 | Android UI、Gradle、JNI 不回流主仓 |

> Android 普通版和极简版的区别：普通版面向“完整移动客户端”，极简版面向“尽量小、尽量少设置、尽量低存储占用”的手机轻量入口。

### 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **模型分片攻关** | llama.cpp/GGUF 主线按节点能力研究模型部分驻留、RPC、租约和故障重分配；先通过单机与同机双进程门 → [详见主线计划](docs/主线开发计划-分布式推理与边缘优化-2026-09-14.md) |
| 🔗 **PyTorch 参考路径** | 兼容 Safetensors 的层流水线仅用于 PC 正确性/性能/容量对照；既有 QW1.8B 双机证据是历史参考，不代表 llama.cpp 模型分片已验收 |
| 🔄 **生产引擎边界** | llama.cpp + GGUF 是默认生产/边缘路径；PyTorch + bitsandbytes 只在 PC Reference P 环境存在，不自动切换到边缘 |
| 📋 **MLFQ 请求队列** | 三级反馈队列管理并发推理请求，短交互优先 + 老化防饥饿 + FIFO 兼容 → [详见调度文档](docs/分布式资源调度系统.md) |
| 🗄️ **多会话与本地事实源** | 会话/设置/模型注册等由主节点本地 SQLite 承载（远端 PostgreSQL 已退场，仅一次性迁移审计通道），旧数据一次性导入；断网本地不中断 |
| 🧩 **模型资产治理** | 模型注册、清单/SHA 校验、来源与许可证、Sidecar 契约、部署模拟及下载来源回退（HF 直连 → 用户代理 → ModelScope）均已接入本机产品面；真实大工件、CUDA 和跨机分发仍待验收 |
| 🖼️ **多模型与多模态** | 模型舰队按设备能力按需选择文本或多模态 GGUF；画像声明模板、thinking、vision、内存预算和工件摘要，缺少能力时降级文本；PyTorch sidecar 只作 PC 对照 |
| 🌐 **Tailscale 与双栈组网** | IPv4/IPv6 端点、手动入群和启动重连按“用户首选 → bootstrap → Tailnet”回退；显式连接的偏好会持久化。双机 IPv6 短任务已实测，IPv4-only/IPv6-only 安装包和真实 WSS/443 仍待环境验收 |
| 🔐 **本地 Auth App 控制面** | Owner bootstrap、Auth-App 字符串/二维码下发、TOTP、恢复码轮换、成员管理与一次性入群票据均有本机 UI/API 门；系统凭据和首次安装联调后置 |
| 📦 **安装、更新与离线整合包** | 独立 Launcher 的签名更新/回滚、下载进度与诊断已实现；离线整合包支持容量预检、SHA/manifest、原子 ZIP、7z/分卷和恢复校验。真实全量出包、空目录/Android SAF 导入和跨平台安装验收后置 |
| 🎛️ **控制面** | 节点注册/注销、分层覆盖、角色转让、备用主节点、TCP 连接状态监控；用户交互以 TUI 为主 |
| 🖥️ **TUI 主入口** | `qlh chat` 面向本地/远端引擎；管理 TUI 负责节点、模型舰队、任务和状态；`qlh_edge` 提供最小 HTTP → [使用指南](docs/TUI使用指南.md) |
| 🖼️ **图像能力边界** | 主项目保留图片上传、Gemma/Qwen 多模态理解和图生文，不提供图像生成或编辑，也不安装相关运行时与模型资产。生图唯一归属为 Koakumix `harness_workbench`，接口为 `/v1/images/generations`；旧 SD 计划只作历史记录。 |
| 📱 **Android 客户端（支线）** | Android Full/Lite、SAF、Full Worker/Stage 和真机证据由独立端侧仓库维护；主仓只冻结任务、模型和能力合同 |
| 🏝️ **TP 孤岛接入** *(PoC)* | 集群外的同构 GPU 张量并行子集群（vLLM/SGLang/llama.cpp rpc）封装为**单个逻辑高算力节点**接入，承担整请求推理 → [接入指南](docs/TP孤岛接入指南.md) |
| ☁️ **外部推理服务辅助** *(PoC)* | 整条请求按策略路由到集群外 OpenAI 兼容端点，**数据作用域门控默认不出集群** → [接入指南](docs/外部推理服务Provider接入指南.md) |
| 🎯 **投机解码辅助** *(实验)* | 本地小模型起草 + 外部大模型校验，跨慢网只传 token id；默认关闭，未接生产解码循环 → [实施说明](docs/投机解码外部辅助实施说明.md) |
| ⚙️ **任务链 Full Worker** | `dual_candidate` DAG、journal、lease-epoch fencing 和 Provider registry 保留为临时整模回退；不代表模型已分片，`task_dispatch` 生产准入门保持关闭 → [任务链专项](docs/任务链下一阶段实施计划.md) |
| 🗂️ **本地 RAG** | 主节点 SQLite FTS5 + 有界向量 embedding（Ollama `nomic-embed-text` / 原生 llama.cpp 双 provider）、可恢复 job、容量预算与 ANN 决策门（RAG-S0…S5D）；30 条人工查询的本机质量门已完成，真实长时、规模与 sqlite-vec benchmark 后置 → [集群接入与本地 RAG 计划](docs/集群接入稳定性与本地RAG实施计划.md) |
| 🔑 **手动入群（CLUSTER-JOIN）** | 目标节点生成一次性授权票据，主节点 Auth App 审批后签发 Ed25519 client-only grant（文本码 + 二维码，nonce ledger 原子消费），成功即降级为从节点；Web/TUI 已接线 → [集群接入计划](docs/集群接入稳定性与本地RAG实施计划.md) |
| 🌐 **抗弱网与 Transport v2** | `cluster_transport` 提供 `legacy_tcp`/`wss_443` 能力选择、有界 ACK 窗口、稳定故障矩阵与 circuit breaker；NW3.1 本地自签名 WSS loopback 门完成；真实 443/证书/流量对照后置 → [抗弱网专项](docs/抗弱网通信协议专项计划.md) |
| 🧪 **实验质量与文档治理** | EX-N3 以只读生产质量门复核文本与多模态理解的计划、样本、校准、性能、质量和人工复核；文档维护 Agent 已完成本机检索/语义质量门，只生成建议而不自动改写文档 |
| 🧩 **子项目：Koakumix harness 工作台** | 面向玩具/自用的小模型定制化推理工作台（S1-S8 本机/离线开发门完成）：上下文预算与 STATE 压缩、模型画像/能力门、OpenAI 兼容 `/v1`、定制化实验台（A/B + Pareto）、**唯一生图工作区**、SQLite 会话与 RAG、长期记忆（RAG+压缩+本地 memory）、联网搜索与轻量 MCP 服务、红队安全样本；**不 import 主项目代码、仅共享模型工件**，亚1B 与 DS3 模型画像已登记 → [harness 方案](harness_workbench/docs/小模型轻量推理harness工作台调研与方案.md) |
| 📡 **联网搜索与轻量 Fetch 工具** | WEB-TOOL G1-G6 本机开发门完成：离线能力探测、Tool Gateway fail-closed（HTTPS 强制/SSRF/DNS/重定向复检）、受限 Fetch/SearXNG adapter、TaskGraph `tool_request` Stage、显式 `persist` 工具缓存与 API、质量门与联合审计；`production_network_enabled=false`，真实网络验收后置 → [调研与分期计划](docs/archive/联网搜索与轻量Fetch工具调用可行性调研与分期计划.md) |
| 📝 **文档维护 Agent 子项目** | 独立包（独立仓库 [qlh-docagent](https://github.com/SgfKrc/qlh-docagent)，主项目以 submodule 引入）：规则数据化（`RULES.md` + `rules.yaml` 驱动扫描器）、规则变更机械扫描（增量矩阵 new/gone/changed + `--max-new/--max-gone` 门）、演进门控（agent 改规则 proposed→preflight→gates→released）与等价回归 → [专项计划](tools/docagent/docs/文档维护Agent工具子项目化与通用化专项计划.md) |
| 🔌 **Reasonix ↔ Codex 桥接子项目** | 独立包（独立仓库 [reasonix-codex-bridge](https://github.com/SgfKrc/reasonix-codex-bridge)，主项目以 submodule 引入）：把 Reasonix 子智能体以 stdio MCP 工具（`reasonix_run` / `reasonix_resume` / `reasonix_cancel` / `reasonix_rollback` / `reasonix_exec` / `reasonix_status`）接给 Codex；CLI 路径与模型 ref 全部按本机解析、零硬编码（`node src/configure.mjs list/use/codex/verify`）；默认只读，W1/W2/W3 已加入受控写入、变更证据、显式回滚和读写 profile 分工，审计修复票 `AUD-01`～`AUD-08` 已完成，`R2-EXT-01` 已放宽有限预算，`E2-EXT-01` 已落地任务级 checkpoint/续跑，`R3-EXT-01` 已落地显式只读并行与取消回收，G3 已完成 WSL 跨平台实跑，R4 已统一输出截断语义，`TOOL-RXB-EXEC-01` 已完成命名命令执行与真实 CLI 验收，`TOOL-RXB-NET-01` 已接入 Reasonix 原生 `web_fetch`（bridge 不自建网络栈），`TOOL-RXB-NET-02` 已加入 provider 搜索 fail-closed 门控，`TOOL-RXB-LOOP-01` 已加入显式 plan/implement/exec/review 阶段审计 → [完善方向](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge完善方向-2026-09-12.md) · [Harness 工具扩展排期](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge-Harness工具扩展与能力补齐排期-2026-09-13.md) |
| 🎯 **判题口径修复与 DS3 替代 R1** | `loose_contains` + 512 token（P5）实证可区分（Qwen3-4B 1/4 vs 1.8B 0/4）；DS3-0324-7B v2 全口径 **2/4×3、8/11×3**（仅预算 192→512 即 0/4→2/4，判题口径问题实证）→ 替代 R1 判题模型的**已批准候选** → [DS3 专项](docs/DistilQwen2.5-DS3-0324替代R1判题模型专项计划.md) |

### 项目设计理念

- **资产主权与本地自治**：模型工件、外部算力资产、API key、会话、任务 journal、知识库、认证材料、备份与迁移能力均属于用户。开发组只提供代码、导入/校验/登记/迁移工具，不代管、不代持，也不保留用户重置权；除源码托管与 release 发布外，运行时不以开发组第三方服务为前置。主节点 `local_only` SQLite、用户文件系统、加密备份和 `.qlhmigrate` 是这一原则的实现边界。
- **工程化、自动化与受控 Agent 协作**：测试、验收与维护工具及其测试/质量门代码是工程主体的一部分，与产品代码并行建设。自动化既证明质量，也消除重复劳动：隔离环境一键创建、测试通道、自动仿真/实验、契约/接口扫描、故障注入、质量门、离线资产包构建/校验/恢复、启动器清单生成、环境诊断与受控部署同步，均把可重复的机械步骤变成可复现流程；Agent 只在明确的数据、检索与权限范围内辅助归纳证据、生成建议和执行质量检查，不能绕过脚本门、人工复核或授权边界自动改写事实、发布资产和作出准入决定。
- **以测试质量约束自动化质量**：测试按风险、依赖和真实度分为单元、契约、浏览器/API、竞态/故障注入、仿真与真机/双机验收，而非只追求一次全量通过。针对环境污染、竞态和接口盲区持续重拆分类、隔离通道、审计 fixture/覆盖边界并补负例，以降低“同一批固定用例反复通过却漏掉新缺陷”的杀虫剂效应。
- **数据不出集群**：外部推理的默认数据作用域为 `deny`，显式授权才允许出集群；离线资产包与签名清单保证模型分发可审计、可重建。

### 架构演进：从功能优先到用户主权

立项阶段聚焦量化、缓存优化和多机流水线，尚未把“谁拥有数据、断网后谁能恢复系统”明确为产品约束。早期工程曾保留 PostgreSQL 兼容路径；随着主节点离线运行、跨设备迁移和用户自带资产成为刚性需求，系统已收敛为以下本地优先架构。这是对目标边界的补全，不是把远端依赖换成另一种远端依赖。

| 维度 | 早期工程路线 | 当前确定的边界 |
|------|--------------|----------------|
| 状态与身份 | PostgreSQL 兼容路径存在，主节点本地事实边界尚未完整定义 | 默认 `local_only`；会话、设置、模型登记、任务 journal、审计与本地用户均由主节点 SQLite 承载，WAL/FULL 与幂等恢复保证断网可继续运行 |
| 模型与文件资产 | 重点是加载与分发能力 | 权重仍由用户保存于本地文件系统；SQLite 仅保存受校验的引用、清单和摘要。导入、离线包、下载与部署均以 SHA/manifest、空间预检和原子发布为边界 |
| 知识与凭据 | 未形成统一的数据主权策略 | RAG 使用独立的用户主节点 SQLite（FTS5 + 有界向量）；embedding 可由本地 Ollama 或原生 llama.cpp 提供。认证密钥走操作系统凭据存储，恢复码只保存哈希；开发组不保留用户副本或重置权 |
| 迁移与恢复 | 远端服务容易成为运行时前置 | 加密备份、`.qlhmigrate` 流式迁移与本地恢复是主路径；PostgreSQL 只保留用户主动发起的一次性兼容导出/审计窗口，远端不可用不得影响核心功能 |

> 两条原则共同约束后续功能：新模型、RAG、Auth App、集群入群和外部 Provider 都必须先说明资产归属、数据作用域、离线恢复和撤销语义，也必须给出可复现的自动化检查、失败边界与人工验收路径；不能以“方便接入”重新引入开发组托管依赖，也不能以 Agent 或单次绿灯替代事实证据。

**应用场景**：智能终端 · 物联网 · 边缘计算 · 教育科研

---

## 🌐 Tailscale 组网（重要）

分布式推理模式依赖 **Tailscale** 实现跨子网设备互联。所有参与推理的节点（PC、Android）建议先安装 Tailscale 并加入同一网络。

### 安装 Tailscale

**PC 端**（Windows / macOS / Linux）：

> 🔗 https://tailscale.com/download

安装后用同一账号登录即可自动组网。

**Android 端**：

> 🔗 Google Play 搜索 "Tailscale" 安装，或从 APK Mirror 侧载

**验证组网**：

打开 Tailscale 控制台 https://login.tailscale.com/admin/machines ，确认所有节点均在线且分配了 `100.x.x.x` 地址。

### 为什么需要 Tailscale？

- 校园网 / 家庭网络通常不分配公网 IP，设备间无法直接互访
- Tailscale 基于 WireGuard 创建虚拟局域网，每个设备获得一个固定的 `100.x.x.x` 地址
- Windows 打包版启动器会自动检查 Tailscale 是否已安装并登录

> 当前校园网实测会阻断 UDP，Tailscale 可能退化为中继路径。`NET-DUALSTACK-PREF-01` 已完成本机门：显式连接成功后持久化用户选择的 IPv4/IPv6 端点，启动与自动重连按“首选 → bootstrap 原地址 → Tailnet”回退；双机 IPv6 短任务已验证。自建 DERP、路径观测、备用中继、主节点直连 WSS 数据面和分块续传仍需真实 443/证书/网络环境验收；边界见[抗弱网通信协议专项计划](docs/抗弱网通信协议专项计划.md)。

---

## 🏗️ 项目架构

```
项目根目录
├── docs/                          # 项目文档
│   ├── 项目技术说明.md              # 新人入口：KV、融合、量化、分布式、调度与协议
│   ├── 整体架构.md                 # 项目总览、设备范围、当前执行路径
│   ├── 核心技术原理.md              # 多引擎、量化、KV缓存与分布式方式边界
│   ├── 模块接口说明.md              # 当前主要模块职责（接口以源码为准）
│   ├── 测试与评判标准.md            # 单机与多种分布式执行的评判标准
│   ├── 文档状态与清理清单.md         # 文档状态定义与后续维护规则
│   ├── 图算法.md                   # PyTorch 层流水线的拓扑路径算法
│   ├── 分布式资源调度系统.md          # MLFQ 三级反馈队列 + 图算法层编排（原理与关系）
│   ├── 分布式推理流水线实施计划.md    # 链式拓扑、LAYER_FORWARD 协议、KV Cache 方案
│   ├── 混合分布式推理体系规划.md      # 层间、任务链、张量并行与 GGUF stage 多 Provider 体系
│   ├── 三种分布式拆分细化实施方案.md  # 层间待测试、任务链与张量并行实施方案
│   ├── Android版本远期计划.md       # Android 端方案评估与规划
│   ├── Android SAF模型存储方案.md   # Android SAF 外部模型目录方案
│   ├── 主线开发计划-分布式推理与边缘优化-2026-09-14.md # ★ 当前主线基线
│   ├── 支线开发计划-外置迁移与Koakumix-2026-09-14.md # ★ 当前支线基线
│   ├── 总体下一步计划.md             # 历史总排期与计划索引
│   ├── 项目进展与下一步计划.md       # ★ 能力、证据与原 P0/P1/P2 快照
│   ├── 张量并行外部辅助与混合拆分调研方案.md  # ★ mesh 内 TP 不可行的量化论证 + 三条外部辅助路线
│   ├── TP孤岛接入指南.md            # ★ 路线 A：孤岛=单逻辑高算力节点（PoC）
│   ├── 外部推理服务Provider接入指南.md # ★ 路线 B：整请求外部路由 + 数据作用域门控（PoC）
│   └── 投机解码外部辅助实施说明.md   # ★ 路线 C：draft-verify（默认关闭的实验路径）
├── src/                           # Python 源代码（PC 端）
│   ├── config.py                  # 全局配置（网络/模型/KV/分层/运行模式/图算法阈值）
│   ├── model_module.py            # 模型加载、量化、算子融合、层级拆分、前向推理
│   ├── llama_engine.py            # llama.cpp 引擎封装（CPU/集显 GGUF 推理）
│   ├── island_engine.py           # ★ TP 孤岛引擎（OpenAI 兼容端点 → 单逻辑节点，路线 A）
│   ├── external_provider.py       # ★ 外部推理服务 Provider + 数据作用域门控（路线 B）
│   ├── speculative.py             # ★ draft-verify 投机解码（默认关闭的实验路径，路线 C）
│   ├── tui_admin.py               # ★ 跨平台 TUI 管理菜单（纯标准库，零依赖）
│   ├── qlh.py                     # 主仓跨平台 TUI 命令入口
│   ├── qlh_edge.py                # Edge L 档最小 HTTP 服务
│   ├── tui_chat.py                # ★ T9 简化聊天页（Textual + httpx；安装包内置，源码可选）
│   ├── tui_sse.py / tui_shared.py # T9 SSE 增量解析器与共享层（端点/命令/metrics）
│   ├── paged_kv_cache.py          # 轻量化分页KV缓存（内存热页；可选磁盘冷页）
│   ├── tcp_comm.py                # TCP主从通信（长连接、心跳、封包解包、张量序列化）
│   ├── scheduler.py               # 任务调度（节点管理、层分配、流水线控制、请求队列）
│   ├── graph_orchestrator.py      # ★ 图算法智能编排（最大带宽生成树 + DFS 路径搜索）
│   ├── device_profiler.py         # 设备画像采集（CPU/GPU/RAM/网络）
│   ├── api_server.py              # FastAPI 服务端（REST API + WebSocket）
│   ├── local_store.py             # 主节点 SQLite 本地存储（旧 JSON 一次性只读导入）
│   ├── model_downloader.py        # 模型下载引导（HuggingFace/ModelScope/百度网盘）
│   ├── model_host.py              # 模型生命周期宿主（统一持有 LLM/多模态理解引擎）
│   ├── scheduler_svc_http.py      # scheduler-svc 微服务 HTTP 壳（透传契约）
│   ├── inference_service/         # ★ inference-svc 微服务（engine_host/协议/路由）
│   └── node_config.py             # 本机节点配置（集群密钥/档案等，非源码控制）
├── schemas/                       # ★ MODEL-FLEET 冻结契约（artifact/pull-job/deployment/profile JSON Schema）
├── fixtures/                      # 测试与走查 fixture（API 事件流、模型门样例）
├── (sibling) qlh-android/         # Android Full/Lite、JNI、Gradle、Android tests/resources
├── (sibling) qlh-shell/            # CyberGothic Web/Desktop shell、Node tests、可选 Textual
├── (sibling) qlh-release/          # Launcher、PyInstaller/Inno/Linux 发布与发布 venv
├── (sibling) qlh-toolbox/          # SSH/patch、演示、答辩和性能工具
├── .venv-test/                    # 隔离测试环境（setup_test_env.py 创建；全量测试专用，勿装系统 Python）
├── harness_workbench/             # ★ 小模型 harness 工作台（独立子项目；不 import 主项目代码）
│   ├── context_engine/            # 上下文预算与 STATE 压缩
│   ├── model_profiles/            # 模型画像与能力门
│   ├── rag/                       # 本地 RAG（分块/索引/改写/重排）与检索质量门
│   ├── api_layer/                 # OpenAI 兼容 /v1 与工作台后端
│   ├── image_workbench/           # Koakumix 独占生图工作区
│   ├── memory/ session/ mcp_server/ research/ adaptation/ eval/  # 长期记忆 / 会话 / 轻量 MCP / 实验与评测
│   ├── tools/                     # rag_baseline、Tool Gateway 等
│   ├── ui_react/                  # 工作台前端（React + Vite）
│   └── cli.py / tui.py            # 命令行与终端入口
├── tests/                         # 单元/契约/回归测试（全量基线仅作历史参考；当前证据见计划与专项文档）
├── scripts/                       # 工具脚本
│   ├── quantize_model.py          # 模型准备与量化验证
│   ├── benchmark_all.py           # 全量化档位基准测试
│   ├── benchmark_compile.py       # torch.compile 融合测试
│   ├── convert_to_gguf.py         # Safetensors → GGUF 转换
│   ├── build_offline_bundle.py    # 离线整合包容量预检、清单与原子发布
│   ├── edge_preflight.py          # Edge L 档体积/冷启动/依赖/路由预检
│   ├── experiment_quality_production_gate.py # EX-N3 只读质量复核
│   └── docagent_*_gate.py         # 文档检索/语义质量门
├── tools/                         # ★ 子模块与运维工具
│   ├── docagent/                  # 子模块（自研）：文档维护 Agent 独立仓库
│   ├── reasonix-codex-bridge/     # 子模块（自研）：Codex ↔ Reasonix MCP 桥接
│   ├── ssh_sync_*.py              # 从节点同步/补丁分发
│   └── modelscope_download.py     # ModelScope 下载辅助
├── models/                        # 模型文件存放目录（需自行下载）
│   ├── qwen-1_8b-chat/            # PC: Safetensors 格式
│   └── qwen-1_8b-chat-Q4_K_M.gguf # PC: GGUF 格式（llama.cpp 引擎）
├── logs/                          # 运行日志目录
├── requirements.txt               # Python 依赖清单
├── requirements-edge.txt          # Edge L 档最小依赖（无 torch）
└── README.md                      # 本文件
```

### 子模块（Git submodule）

仓库共引入 **3 个** Git 子模块；其中**只有 `llama.cpp` 是第三方依赖，另外两个都是本项目自研并独立维护的子项目**：

| 路径 | 仓库 | 性质 | 用途 |
|------|------|------|------|
| `tools/docagent` | [SgfKrc/qlh-docagent](https://github.com/SgfKrc/qlh-docagent) | **自研**（文档维护 Agent 独立化） | 规则数据化扫描器、规则变更机械扫描（new/gone/changed 增量矩阵）与演进门控 |
| `tools/reasonix-codex-bridge` | [SgfKrc/reasonix-codex-bridge](https://github.com/SgfKrc/reasonix-codex-bridge) | **自研**（Codex ↔ Reasonix 协作桥） | stdio MCP 桥接，供 Codex 调用只读 Reasonix 子智能体；CLI 路径与模型 ref 按本机解析，`configure verify` 自检 |
| `../qlh-android/app/src/main/cpp/llama.cpp` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | **第三方**（Android 外置仓库依赖） | Android Full 变体原生构建；固定 revision，PC 侧与 Python sidecar 都不需要 |

#### 自研子项目一：`qlh-docagent`（文档维护 Agent）

- **定位**：把主仓原本内嵌的文档维护工具（原 `docs/agent_tool` 套件）独立为通用包，规则不再写死在代码里，而是**数据化**为 `RULES.md` + `rules.yaml`。
- **能力**：规则驱动的机械扫描器；规则变更增量矩阵（`new/gone/changed`，配 `--max-new/--max-gone` 门）；演进门控（`proposed → preflight → gates → released`）与等价回归。
- **与主仓的关系**：主仓只保留 gitlink（`tools/docagent`）与专项文档；源码与规则在独立仓库演进，主仓不复制其源码。
- **入口**：[文档维护Agent工具子项目化与通用化专项计划](tools/docagent/docs/文档维护Agent工具子项目化与通用化专项计划.md) · [文档维护 Agent 工具设计](tools/docagent/docs/文档维护Agent工具设计.md)

#### 自研子项目二：`reasonix-codex-bridge`（Codex ↔ Reasonix 桥）

- **定位**：把 Reasonix（本地多模型编码 agent）接成 Codex 可调用的 stdio MCP 服务，让 Codex 调度一个**独立模型、独立配额**的子智能体。
- **能力**：6 个 MCP 工具（`reasonix_run` / `status` / `resume` / `rollback` / `cancel` / `exec`）；CLI 路径与模型 ref 按本机解析、零硬编码（`configure list/use/codex/verify`）；受控写入（`allowWrite` + `allowedPaths` 白名单 + 干净工作区 + git 快照回滚）；命名 argv-only 测试/构建执行（默认关闭、`shell:false`、clean-tree、超时/输出上限、变更检测与脱敏）；任务级 checkpoint 续跑；脱敏 JSONL 调用日志。
- **与主仓的关系**：主仓只保留 gitlink（`tools/reasonix-codex-bridge`）与规划/审计文档；子智能体 profile（`deepseek-worker` / `deepseek-worker-write`）由子项目的 `configure profile` 在 Reasonix 全局目录创建。
- **入口**：[完善方向](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge完善方向-2026-09-12.md) · [Harness 工具扩展排期](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge-Harness工具扩展与能力补齐排期-2026-09-13.md) · [工具面现状](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge工具面现状-2026-09-13.md) · [审计报告](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge审计报告-2026-09-12.md) · [ACP 会话级恢复专项计划](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge-ACP会话级恢复专项计划-2026-09-13.md)


拉取与更新：

```bash
git submodule update --init --recursive                              # 已有工作区
git clone --recurse-submodules https://github.com/SgfKrc/qlh       # 主仓首次克隆
# 按需在同级目录克隆 https://github.com/SgfKrc/qlh-android、qlh-shell、qlh-release、qlh-toolbox
```

> `harness_workbench/` 是独立子模块；Koakumix harness 的生图、RAG、MCP 和定制化实验不属于 QLH 主线，主仓只保留 gitlink 与边界说明。
### PyTorch 层流水线示例（PC 参考路径，历史对照）

```
用户输入 → 主节点(Master)  → TCP → 从节点1(Client) → TCP → 从节点2(Client) → 结果回传
          Embed + L0-3          L4-14              L15-23 + LM Head
          独显主节点参与首段计算，不再仅协调调度
```

### Android 当前两种运行方式

```
┌──────────────────────────────┬──────────────────────────────┐
│ 本地模式（现有 UI：全有模式）   │ 远程模式（现有 UI：全无模式）   │
│                              │                              │
│  Android 本地 llama.cpp      │  Android 聊天 UI             │
│  GGUF Q4_K_M (~1.16 GB)      │  HTTP → PC 主节点             │
│  离线可用，不依赖网络          │  PC 集群分布式推理            │
└──────────────────────────────┴──────────────────────────────┘
```

### 软件分层架构

| 层级 | 功能 | 技术 |
|------|------|------|
| 应用层 | 可视化交互 & 节点管理 & 性能监控 | React + TUI（标准库）+ Jetpack Compose (Android) |
| 调度层 | 任务调度、指令分发、状态管理、请求队列 | Python threading + 图算法 |
| 通信层 | TCP长连接、粘包处理、心跳、张量序列化 | Python socket + struct |
| 推理层 | llama.cpp/GGUF 生产主引擎；PyTorch 仅 PC 对照；模型分片/RPC 正在攻关 | llama.cpp (CPU / 集显 / Android worker) / PyTorch Reference P / island *(PoC)* |
| 外部辅助层 *(PoC)* | 整请求外发的路由与数据作用域门控、投机解码校验 | OpenAI 兼容 HTTP（vLLM / SGLang 等） |
| 存储层 | 对话持久化、节点注册、配置管理 | 主节点 SQLite（Python/Node 共库）+ Room (Android) |
| 基础层 | 运行环境 | Python / CUDA / bitsandbytes / llama.cpp |

---

## 📦 环境依赖

### 核心框架

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 开发环境 3.12.10；源码已核对可在 3.10 / 3.11 / 3.12 解析 |
| PyTorch | ≥ 2.2.0 | 仅 PC Reference P/独立 sidecar；不进入 Edge 或默认生产环境 |
| **transformers** | **≥ 4.45, < 5.0** | ⚠️ 必须保持 4.x！5.x 移除了 `load_in_4bit`/`load_in_8bit` |
| accelerate | ≥ 1.0.0 | 模型加载加速（bitsandbytes 依赖） |

### 模型量化

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| bitsandbytes | ≥ 0.45.0 | INT4/INT8 量化（独显必装，集显可选） |

### CPU/集显推理引擎

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| llama-cpp-python | ≥ 0.3.0 | 本地 GGUF 推理；分布式路径使用原生 llama.cpp host/RPC worker 合同 |

### Web 可视化

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| fastapi | ≥ 0.110.0 | API 后端框架 |
| uvicorn[standard] | ≥ 0.29.0 | ASGI 服务器 |
| pywebview | ≥ 5.0 | 打包版原生窗口（替代浏览器） |
| python-multipart | ≥ 0.0.12 | 文件上传支持 |

### 数据库

> 远端 PostgreSQL 已退场（M1.3，2026-08-10）：生产运行时不再连接或打包 PG 驱动，数据由主节点 SQLite 承载；仅历史迁移审计需要时按需安装 psycopg2。

### 网络（分布式模式必装）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| **Tailscale** | 最新版 | 跨子网虚拟组网，所有分布式节点必须安装 |

> 🔗 下载: https://tailscale.com/download

### 工具

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| tqdm | ≥ 4.65.0 | 进度条 |
| psutil | ≥ 5.9.0 | 系统资源监控 |

### 产品壳（支线，可选）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Node.js | ≥ 18 | 仅迁移/开发产品壳支线 |
| npm | — | 包管理器 |

### Android 客户端

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Android SDK | API 34+ | 编译目标 |
| Gradle | 8.11+ | Wrapper 已内置，无需单独安装 |
| Kotlin | 2.1.0 | 通过 Gradle 自动下载 |
| Java | JDK 17 | 编译必需 |

> Android 客户端**不需要 Android Studio**，有 JDK + Android SDK 命令行工具即可通过 `gradlew.bat` 构建。

### 一键安装

```bash
# Python 依赖（主节点 SQLite 自持，无需 PostgreSQL）
pip install -r requirements.txt

# 产品壳依赖（支线；主线运行不需要）
# 产品壳在支线仓库维护：cd ../qlh-shell/frontend_cybergothic && npm ci
```

### Edge L 档最小环境

边缘节点使用独立环境和 `requirements-edge.txt`，不安装主环境的 torch、Transformers 或其他重型依赖：

```bash
python -m venv .venv-edge
.venv-edge/Scripts/python.exe -m pip install -r requirements-edge.txt  # Windows
# Linux/macOS: .venv-edge/bin/python -m pip install -r requirements-edge.txt

.venv-edge/Scripts/python.exe scripts/edge_preflight.py \
  --python .venv-edge/Scripts/python.exe --json
.venv-edge/Scripts/python.exe qlh_edge.py --help
```

预检固定检查 venv 不超过 300 MB、导入冷启动不超过 15 秒、禁用模块未安装/未导入，以及
`/health`、`/status`、`/generate` 路由存在。启动服务时设置 `QLH_EDGE_MODEL` 指向本地 GGUF：

```bash
# Linux/macOS
QLH_EDGE_MODEL=models/Qwen-1_8B-Chat.Q4_K_M.gguf \
  .venv-edge/bin/python qlh_edge.py --port 8010
# Windows PowerShell: $env:QLH_EDGE_MODEL = "models/Qwen-1_8B-Chat.Q4_K_M.gguf"
# .venv-edge\\Scripts\\python.exe qlh_edge.py --port 8010
```

### 🚀 一键配置全部开发环境（克隆后推荐）

仓库包含**主运行时 + Python 虚拟环境**；Node 子项目属于支线迁移源，不是主线运行前置。
统一入口 `scripts/setup_envs.py`（或根目录 `setup_all_envs.bat` / `setup_all_envs.sh`）可一次配完：

```bash
# Windows
setup_all_envs.bat --all

# Linux / macOS
./setup_all_envs.sh --all

# 或直接调脚本（等价；另有 --only / --skip / --check / --snapshot / --list 等）
python scripts/setup_envs.py --all
python scripts/setup_envs.py --check      # 只校验现有环境，不安装
python scripts/setup_envs.py --list       # 查看环境清单
```

> ⚠️ **torch 等平台相关大件不自动进入主线 Edge 环境**：脚本自动过滤 `torch/torchvision/torchaudio`
> 并打印 Reference P/sidecar 环境的平台安装命令，避免 CPU/CUDA 版本互相污染。llama.cpp/GGUF
> 主线不要求 torch；需要 PC 对照时用
> `--torch-index-url URL` 让提示带好源，例如 `https://download.pytorch.org/whl/cu126`。
> llama-cpp-python 等需源码构建的包会在缺编译工具链时报错，按对应 requirements 头注释处理。

**环境 ↔ 依赖文件 ↔ 锁定快照**：`requirements-lock/*.lock.txt` 由 `--snapshot` 从现有
venv 的 `pip freeze` 自动生成，记录精确版本做复现参考（torch 系与 editable/本地安装不
锁定）；实际安装仍以各 `requirements-*.txt` 的版本窗口为准。

| 环境 | 用途 | 依赖来源 | lock 快照 |
|---|---|---|---|
| **主环境**（系统 Python） | Control/TUI、工具脚本与可选 PC Reference P 开发 | `requirements.txt` | `requirements-lock/main.lock.txt` |
| `.venv-edge` | Edge L 档 GGUF/CPU 节点 | `requirements-edge.txt` | 本票以预检 JSON 为证据 |
| `.venv-test` | 唯一测试环境（全量/定向 pytest） | `requirements-test.txt` | `requirements-lock/test.lock.txt` |
| `../qlh-shell/.venv-tui` | T9 终端聊天页（textual） | `../qlh-shell/requirements-tui.txt` | `../qlh-shell/requirements-lock/tui.lock.txt` |
| `.venv-gemma4-native` | 原生 Gemma 4 MTMD / llama.cpp | `requirements/requirements-gemma4-native.txt` | `requirements-lock/gemma4-native.lock.txt` |
| `.venv-gemma4-pipeline` | Gemma 4 PyTorch Transformers 5.10.1 sidecar | `requirements/requirements-gemma4-pipeline-sidecar.txt` | `requirements-lock/gemma4-pipeline.lock.txt` |
| `.venv-qwen3-sidecar` | Qwen3 PyTorch sidecar（含 pipeline 执行依赖） | `requirements/requirements-qwen3-sidecar.txt` + `requirements/requirements-qwen3-pipeline-sidecar.txt` | `requirements-lock/qwen3-sidecar.lock.txt` |
| `../qlh-release/.venv-packaging` | 集显版打包（torch CPU + PyInstaller） | `../qlh-release/packaging/requirements-cpu.txt` | `../qlh-release/requirements-lock/packaging.lock.txt` |
| `../qlh-release/.venv-packaging-cuda` | 独显版打包 | `../qlh-release/packaging/requirements-cpu.txt` | `../qlh-release/requirements-lock/packaging-cuda.lock.txt` |
| `../qlh-shell/frontend_cybergothic` | 产品壳（支线） | `package-lock.json`（`npm ci`） | — |

> `setup_all_envs.bat` 在 Windows 会自动 `chcp 65001`；直接跑脚本时若终端乱码，
> 手动 `chcp 65001` 或 `set PYTHONIOENCODING=utf-8` 即可。

### 🔒 环境分割（主环境 / 测试环境 / 打包环境）

**硬性规则：测试依赖只装进 `.venv-test`，严禁装入系统 Python（主环境）；严禁在 `.venv-test` 与主环境之间复制/同步 site-packages。**

| 环境 | 用途 | 依赖来源 | 禁止事项 |
|---|---|---|---|
| **主环境**（系统 Python） | 运行时（transformers 4.47.1 / torch / 推理服务）与工具脚本 | `requirements.txt` | 不装 pytest 系测试依赖；不跑全量测试 |
| **`.venv-test`** | **唯一测试环境**（全量/定向 pytest 都在这跑） | `scripts/setup_test_env.py` + `requirements-test.txt` | 不承载运行时推理；不被当作主环境使用 |
| `../qlh-release/.venv-packaging/` | 集显版打包（torch CPU） | `../qlh-release/packaging/requirements-cpu.txt` | — |
 | `../qlh-release/.venv-packaging-cuda/` | 独显版打包（torch CUDA） | 见 qlh-release 打包文档 | — |
| `.venv-gemma4-native/` | 原生 Gemma 4 MTMD/llama.cpp 运行时 | `requirements/requirements-gemma4-native.txt` | 不得复用给 Transformers pipeline |
| `.venv-gemma4-pipeline/` | Gemma 4 PyTorch Transformers 5.10.1 sidecar | `requirements/requirements-gemma4-pipeline-sidecar.txt` | 与 native/Qwen3 环境隔离 |
| `.venv-qwen3-sidecar/` | Qwen3 PyTorch sidecar | 各自 requirements | — |

**常用命令**：

```powershell
# 创建/校验测试环境（--check 只读健康检查）
python scripts/setup_test_env.py --check
# 在测试环境跑测试（通道脚本自带 venv 守卫，拒绝系统 Python）
.venv-test\Scripts\python.exe scripts/run_test_channels.py
# 定向测试
.venv-test\Scripts\python.exe -m pytest tests/test_xxx.py -q -n 1
```

> **事件记录（2026-08-14）**：曾发生主环境被灌入 pytest 系测试包、同时 `.venv-test` 的 pytest 被掏空（两个环境 site-packages 被复制混淆），导致测试被迫在主环境运行。已修复（主环境卸载测试包、`.venv-test` 重装 requirements-test.txt 恢复）。**请勿通过 `run_test_channels.py --allow-system-python` 绕过守卫**，该参数仅限一次性 CI 镜像。

---

## 🧰 克隆后资产获取清单

> 克隆仓库 ≠ 立即可用。以下清单列出**代码之外还需要获取的离线资产**（模型权重、子模块、密钥、环境文件）。标 ✅ 的项目随仓库已有或 clone 自动带出，其余需按表格获取。

### 0. 克隆后必做（一次性的环境步骤）

```bash
# 1. 主仓只拉取自己的自研子模块；Android Full 的 llama.cpp 在 qlh-android 仓内维护，主线可跳过
#    产品壳、Android、发布工具和 Toolbox 是兄弟仓库，按需在工作区另行 clone。
git submodule update --init --recursive

# 2. 安装 Python 依赖（主环境；联网）
pip install -r requirements.txt

# 3. 产品壳依赖（可选，仅开发 qlh-shell 时；主线可跳过）
# cd ../qlh-shell/frontend_cybergothic && npm ci

# 4. 环境文件（不进仓库，按需自建）
#    主环境 .env 至少含 QLH_CLUSTER_SECRET（分布式密钥）；判题/工具密钥见
#    docs/文档维护Agent工具设计.md §4.1 的 .env.docagent 说明

# 5. 验证
python -c "import src.api_server" && python -m pytest tests/ -q --collect-only | tail -1
```

`llama.cpp` 已锁定为外置仓库 `../qlh-android/app/src/main/cpp/llama.cpp` 的 Git submodule。要保留 Android Full
构建能力，需在兄弟目录 clone `https://github.com/SgfKrc/qlh-android`，进入该仓库后使用 `git submodule update --init --recursive`；
PC llama.cpp host/worker 与 Android RPC worker 需要对应平台的原生 llama.cpp 构建；本地 Python
单机入口使用 `llama-cpp-python`。只有运行 PC Reference P 的 Qwen3/Gemma 4 PyTorch sidecar
时，才另行准备 PyTorch 环境；sidecar 不属于 Edge 或默认生产路径。

无 CUDA/独显的 PC 主线节点无需安装 PyTorch，直接使用 llama.cpp/GGUF。若该节点被明确指定为
PC Reference P 对照节点，才显式安装 CPU PyTorch，且不要复制主节点 CUDA venv：

```bash
python scripts/setup_qwen3_sidecar_env.py --pipeline \
  --torch-index-url https://download.pytorch.org/whl/cpu
python scripts/setup_gemma4_pipeline_env.py \
  --torch-index-url https://download.pytorch.org/whl/cpu
python scripts/setup_envs.py --check --no-node
```

运行时使用 `execution_device=cpu`（或 `auto`，无 CUDA 时会回退 CPU）；资源门按可用 RAM 而非
显存判定。CPU 与 CUDA 环境必须在各节点本地创建，不能跨机器复制 `.venv-*`。

### 1. 离线资产清单（按需获取）

| 资产 | 大小 | 用途 | 获取方式 | 必需性 |
|---|---|---|---|---|
| **Qwen-1.8B-Chat（Safetensors）** | ~3.5 GB | PC Reference P 历史对照 | ModelScope `Qwen/Qwen-1.8B-Chat` 或 HF（见下文模型下载节） | 可选（参考环境） |
| **Qwen-1.8B-Chat（GGUF Q4_K_M）** | ~1.16 GB | llama.cpp CPU/集显单机、Android 本地推理和分片 PoC | `huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf ...` | ⭐ 推荐（主线基线） |
| **Qwen3-4B（GGUF Q4_K_M）** | ~2.5 GB | EX-N3 判题模型（v2 正确率判据）、实验 | 受管下载（MODEL-TOOLS）/ HF `Qwen/Qwen3-4B-GGUF` | 实验必需 |
| **Gemma 4 12B 原生绑定**（GGUF + mmproj） | ~7.3 GB | 图像理解（图生文）原生路径 | 受管工件清单 `models/gemma4-native/gemma4-native.lock.json` + 下载脚本 | 多模态实验 |
| **nomic-embed-text:latest**（Ollama） | 按需 | 主节点本地 RAG embedding provider | `ollama pull nomic-embed-text:latest` | RAG 本机质量/容量门 |
| **Ollama 模型**（`gemma4:12b` 等） | 按需 | EX-N3 Gemma 判题、外部路径验证 | `ollama pull gemma4:12b` | 判题实验 |

### 2. 随仓库已有 / 不需要获取的

| 项 | 说明 |
|---|---|
| ✅ 测试 fixture 与实验计划 | `fixtures/` 全部入库 |
| ✅ 签名源站 / serve 分发 | 代码在 `../qlh-release/`，无需主仓额外资产 |
| ⚠️ 发布签名密钥 | `../qlh-release/.signing-keys/` **不进仓库**；由发布者持有，克隆者无密钥只能验签不能签发 |
| ⚠️ `.env`（QLH_CLUSTER_SECRET 等） | 各节点自备，不入库 |
| ⚠️ `models/` 大文件 | 全部 gitignore；按上表获取，不随仓库分发 |

### 3. 安装包（不克隆也可用）

Windows CPU/CUDA Setup、Launcher、Android Full/Lite APK、Linux `.deb` 均从**发布渠道**获取（本项目内网：主节点在 `qlh-release` 运行 `python packaging/serve.py`）。不要求克隆主仓即可安装使用；克隆仓库主要用于开发与验收。

---

## 🤖 模型下载

> **默认源**：当前 control-svc 内置并启用 Hugging Face 官方源，同时登记 HF 镜像与 ModelScope 端点描述（后两者默认关闭，待对应 adapter/真实网络验收）；支持来源优先级、启停和 `credential_ref`。Windows token 由当前用户 DPAPI 保护；模型代理按 `QLH_HTTP_PROXY > 用户持久化配置 > 直连` 选择，可通过本机 `/models/network/proxy` API 设置或清除，不修改系统代理。gated 仓库必须先登记凭据并显式接受许可证；明文不进入 SQLite/job/manifest/响应。机制见 [专项计划](../qlh-release/docs/一键模型部署与自治集群远期计划.md) §4.2/§7.1。

项目默认示例模型是 **Qwen-1.8B-Chat**，并通过模型注册表提供其他 Qwen/DeepSeek 实验槽位。下面仅说明默认模型的两种格式，不代表系统只支持该模型：

| 格式 | 引擎 | 大小 | 适用场景 |
|------|------|------|---------|
| **Safetensors** | PyTorch (CUDA) | ~3.5 GB | 独显推理、分布式流水线 |
| **GGUF Q4_K_M** | llama.cpp (CPU / Android) | ~1.16 GB | 集显/CPU、单机推理、Android 本地推理 |

### Safetensors 格式（PyTorch / 分布式）

**方式一：ModelScope（推荐，国内更快）**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
```

**方式二：Hugging Face**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen-1.8B-Chat --local-dir models/qwen-1_8b-chat
```

**方式三：百度网盘**

> 🔗 https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3 | 提取码：vtp3

### GGUF 格式（llama.cpp / PC CPU 引擎）

```bash
# 下载推荐版本 Q4_K_M (~1.16 GB)
huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf \
  Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/
```

| 量化 | 大小 | 说明 |
|------|------|------|
| Q3_K_M | ~0.94 GB | 实验档；14B+ 容量验证或小模型链路 smoke，小模型日常不推荐 |
| **Q4_K_M** ⭐ | **~1.16 GB** | **推荐 — 速度/质量最佳平衡** |
| Q5_K_M | ~1.31 GB | 更高质量 |
| Q8_0 | ~1.82 GB | 近无损 |

### 图像生成

QLH 主项目不提供图像生成、图像编辑或相关模型下载。图片上传、Gemma/Qwen 图像理解与图生文仍受支持；需要生图时使用独立的 Koakumix `harness_workbench`，其唯一兼容入口为 `/v1/images/generations`。

### GGUF 格式（Android 本地推理）

Android 本地模式（现有 UI 中称“全有模式”）下，模型需放在**用户选择的外部目录**中（SAF `ACTION_OPEN_DOCUMENT_TREE`），**不放在应用内部存储**，这样卸载 APK 时模型会默认保留。

**Android 模型存放位置**：

| 推荐位置 | 说明 |
|----------|------|
| `Download/QLH/models/` | 手机内置的下载目录，卸载 APK 不会删除 |
| 用户自选的外部 SD 卡目录 | 通过 SAF 授权的任意目录 |

**获取方式**：

1. **PC 分发**：在 PC 上启动分发服务器，Android 浏览器下载后移动到 SAF 模型目录

   ```bash
   cd packaging
   python serve.py
   ```

2. **直接下载**：Android 浏览器访问 Hugging Face 或通过 USB 传文件

3. **后续**：应用内会提供从 PC 主节点直接下载到 SAF 目录的功能

**操作流程**：

```text
打开应用 → 设置 → 切换"全有模式" → 模型管理 → 选择目录
  → 选择包含 .gguf 的目录 → 扫描 → 选中模型 → 完成
```

> 详细方案参见 [Android SAF 模型存储方案](docs/Android SAF模型存储方案.md)

---

## 🚀 快速开始

### 开发模式（PC）

```bash
# 终端 1：启动 Python 后端（从项目根目录运行）
python src/api_server.py

# 终端 2：主线交互入口（标准库 TUI；产品壳属于支线）
python -m src.tui_admin --plain --host http://127.0.0.1:8000
```

后端就绪后：
- **后端 API**：`http://localhost:8000`
- **主线 TUI**：终端内多轮/流式对话和模型/节点控制
- **产品壳**：由支线 `qlh-shell` 迁移计划维护，不能作为主线启动前置

> 产品壳、pywebview Launcher、安装 spec 与 Linux `.deb` 属于支线迁移范围；它们的干净机/升级验收不构成主线 TUI 或 Edge L 档的启动前置。

### 单机模式（PC）

修改 `src/config.py`：`RUN_MODE = "single"`，然后：

```bash
python src/api_server.py
```

### 分布式模式（PC）

> ⚠️ 前提：所有参与节点已安装 Tailscale 并用同一账号登录。

**主节点**：

```bash
python src/api_server.py
# 在管理面板启用"分布式推理"，配置 Tailscale 组网
```

**从节点**：

```bash
python src/api_server.py
# 在管理面板输入主节点 Tailscale IP，点击"连接主节点"
```

> 系统会自动完成：节点注册 → 设备画像上报 → 层分配计算 → 分层配置推送。

### TUI 管理菜单（终端版，跨平台）

无浏览器环境（SSH、服务器、树莓派等）可用终端版管理菜单，功能对应 Web 管理面板（系统总览 / 节点管理 / 分布式与分层 / 请求队列 / 设备画像 / 日志 / 设置），纯 Python 标准库实现，支持 Windows 10+ / Linux / macOS。

**一键启动**（自动启动后端 + 等待就绪 + 进入 TUI，退出 TUI 后后端继续运行）：

```bash
bjtu                                        # 全局命令：任意终端输入即启动（安装见下）
./start_tui.sh                              # Linux / macOS（无需安装）
start_tui.bat                               # Windows（双击或命令行）
```

**安装全局 `bjtu` 命令**（推荐）：打包版 Windows 在安装向导中选择 PATH 注册（静默参数 `/ENVREG=0|1`）；Linux `.deb` 始终安装 `/usr/local/bin/bjtu`，可用 `QLH_ENVREG=1` 或 `qlh-env-register enable` 额外注册 `/opt` PATH。源码检出时，Windows 建议在图形环境变量界面添加项目根（避免 `setx` 重写过长 PATH）；Linux/macOS 可用 `sudo ln -s <项目根>/bjtu.sh /usr/local/bin/bjtu`。

**手动/高级用法**（后端未运行时先 `python src/api_server.py`）：

```bash
python src/tui_admin.py --host 100.x.x.x    # 直接管理远程 Tailscale 主节点
python src/tui_admin.py --plain             # 老终端/管道降级为纯文本编号菜单
python src/tui_admin.py --host 100.x.x.x --log-token xxx   # 远程模式带日志 token
bjtu --help                                 # 查看完整命令集与启动参数（不启动后端）
```

**TUI 命令集**（任意界面输入 `/` 开头命令后 Enter 执行，ESC 取消；`--plain` 模式同样可用）：模型/量化/引擎切换、GPU 选择、分布式开关、队列控制、日志、设置与优雅退出等常用操作无需进入菜单：

```bash
/help                     # 命令集帮助（TUI 内）
/status  /models  /model  # 状态与模型信息
/switch <模型ID> [--quant 精度] [--engine 引擎]   # 切换模型（失败自动回滚）
/quant  <int4|int8|fp16|gguf>                    # 量化切换（重载当前模型）
/engine <auto|llama_cpp|pytorch|island>          # 引擎切换（重载当前模型）
/gpu <序号>  /device auto                        # GPU 选择 / 设备自动配置
/dist on|off  /queue pause|resume|clear          # 分布式开关 / 队列控制
/logs  /host <主机> [端口]  /interval <秒>        # 日志 / 设置
/quit                     # 退出 TUI（后端保持运行）
/shutdown                 # 优雅退出：后端清理资源后退出，TUI 随后退出
```

完整参数表、`QLH_BACKEND_PORT` 覆盖、故障排查与自动化走查见 **[TUI 使用指南](docs/TUI使用指南.md)**；**27 条 `/` 命令的完整参考（别名/参数/选项/退出语义/菜单对应）见 [TUI 指令集](docs/TUI指令集.md)**；网关契约与测试见 [TUI 适配实施计划](docs/TUI适配实施计划.md)（T1-T8 现行·Active；T9.0-T9.5 已完成，终端走查 54/54；T9.6-R2 Windows 开发机实装门及 UP-N6.4W 跨卷保留门已通过，外部干净机/Linux/真实模型会话与默认入口仍待）。

### 外部算力辅助（三条路线，均默认关闭）

张量并行在本项目的异构 Tailscale mesh 内不可行（每 token 需 48 次 all-reduce，20ms RTT 下仅同步开销就 ≥960ms/token，量化论证见[调研方案](docs/张量并行外部辅助与混合拆分调研方案.md) §1）。因此 TP 只留在集群**之外**的快速互联内，通过三条路线借力：

| 路线 | 形态 | 开关 | 状态 |
|------|------|------|------|
| **A · TP 孤岛** | 集群外同构 GPU 子集群跑 TP，对集群呈现为**单个逻辑高算力节点**，承担整请求推理（不参与层拆分） | `QLH_ISLAND_ENABLED=1` + `QLH_ISLAND_BASE_URL` | 阶段 1 PoC，已验证 |
| **B · 外部推理服务** | 整条请求按策略路由到集群外 OpenAI 兼容端点；**默认不出集群** | `QLH_EXTERNAL_ENABLED=1` + `QLH_EXTERNAL_DATA_SCOPE` | 阶段 1 PoC，已验证 |
| **C · 投机解码** | 本地小模型起草 γ 个 token，外部大模型一次校验；跨慢网只传 token id | `QLH_SPEC_ENABLED=1`（默认关闭时实验端点 404） | 阶段 0-1 探索，**未接生产解码循环** |

```bash
# 路线 A：孤岛侧（多卡机/同 LAN 同构 GPU 组）
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
# 网关侧（跑 QLH，再照常连主节点即可）
set QLH_ISLAND_ENABLED=1 && set QLH_ISLAND_BASE_URL=http://10.0.0.2:8000
set QLH_ISLAND_GPU_COUNT=2 && set QLH_ISLAND_VRAM_GB=48 && set QLH_ISLAND_TP_SIZE=2
python src/api_server.py

# 路线 B：默认 opt_in —— 只有显式带 allow_external 的请求才可能出集群
set QLH_EXTERNAL_ENABLED=1 && set QLH_EXTERNAL_BASE_URL=https://gpu-box.example.com:8000
set QLH_EXTERNAL_DATA_SCOPE=opt_in
curl -X POST localhost:8000/api/chat -H "Content-Type: application/json" \
     -d "{\"message\":\"...\",\"allow_external\":true,\"prefer_external\":true}"
```

> ⚠️ **数据边界**：路线 B / C 会把用户内容（含投机解码的草稿 token）送出集群。作用域档位 `deny` / `opt_in`（默认）/ `allow_all` 是安全边界而非性能开关，取值写错会 fail-closed 回落 `deny`。放开前请确认合规要求。

### Windows 打包基线与构建脚本（发布支线，可选）

下表是现有完整包构建脚本的历史体积基线，不是 `PACK-SLIM` 的发布承诺。`PACK-SLIM` 已完成本机开发门，但真实 PyInstaller 构建与首次外置 runtime 引导仍待打包环境验收。

| 版本 | 安装包 | 典型大小 | 适用场景 |
|------|--------|---------|---------|
| **集显版** | `QLH-Edge-Inference-Setup-vX.X.X.exe` | ~180 MB | CPU / 集成显卡节点（从节点） |
| **独显版** | `QLH-Edge-Inference-Setup-vX.X.X-CUDA.exe` | ~1.7 GB | NVIDIA GPU 节点（主节点），无 GPU 时自动回退 CPU |

**集显版 (CPU) 构建**：

```bash
# 发布构建在同级 qlh-release 仓库执行
cd ..\qlh-release
# 0. 创建并激活集显版 venv（仅首次）
python -m venv .venv-packaging
.venv-packaging\Scripts\activate

# 1. 安装依赖（仅首次）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2. 构建发布支线的历史兼容静态资源（主线可跳过）
#    产品壳迁移由支线计划管理。
# cd ..\qlh-shell\frontend_cybergothic && npm install && npm run build && cd ..\..\qlh-release

# 3. PyInstaller 打包（★ 从项目根目录运行）
pyinstaller packaging/qlh-cpu.spec --noconfirm

# 4. Inno Setup 安装包编译
cd packaging
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
```

**独显版 (CUDA) 构建**（需另一独立 venv）：

```bash
# 发布构建在同级 qlh-release 仓库执行
cd ..\qlh-release
# 0. 创建并激活独显版 venv（仅首次）
python -m venv .venv-packaging-cuda
.venv-packaging-cuda\Scripts\activate

# 1. 安装依赖（仅首次，先 torch 后共享依赖，不会互相覆盖）
pip install torch                        # ★ CUDA 12.x（默认），不是 CPU 版
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2-4. 同集显版，但 spec 和 iss 分别用 qlh-cuda.spec / setup-cuda.iss
pyinstaller packaging/qlh-cuda.spec --noconfirm
cd packaging && "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup-cuda.iss
```

> ⚠️ **关键**：两个版本使用**不同的独立 venv**（`.venv-packaging/` vs `.venv-packaging-cuda/`）。
> 不能混用——集显版 venv 必须装 CPU-only torch，独显版 venv 必须装 CUDA torch。
> 装错会导致集显版体积从 180 MB 膨胀到 1.8 GB。
>
> **图像生成边界**：CPU/CUDA 安装包均不包含生图依赖、模型或工作区；生图只在 Koakumix 中安装和运行。主项目的图片输入与多模态理解不受影响。
>
> 安装后双击桌面快捷方式即可启动，无需配置 Python 环境。卸载时会询问是否同时删除 `models/` 目录，默认保留模型文件。
>
> 详细打包流程参见 [qlh-release README](../qlh-release/README.md)。

### Linux `.deb` 打包基线

Linux 构建脚本覆盖 Ubuntu 22.04+ / Debian 12+；下表的版本号和体积是历史示例，不应代替当前 release 清单或干净机验收结果：

| 版本 | 安装包 | 典型大小 | 适用场景 |
|------|--------|---------|---------|
| **CPU 版** | `qlh-edge-inference-cpu_<version>_amd64.deb` | ~200 MB（历史） | CPU / 集成显卡节点 |
| **CUDA 版** | `qlh-edge-inference-cuda_<version>_amd64.deb` | ~1.8 GB（历史） | NVIDIA GPU 节点 |

**构建**（需 Ubuntu/Debian 环境）：

```bash
cd ..\qlh-release\packaging\linux
bash build-deb.sh cpu     # 集显版
bash build-deb.sh cuda    # 独显版
```

**安装**：

```bash
sudo dpkg -i qlh-edge-inference-cpu_0.1.8.2_amd64.deb
# 安装后自动注册 systemd 服务、桌面入口和 /usr/local/bin/qlh-launcher
```

**使用**：

```bash
qlh-launcher --gui        # 独立图形启动器（普通界面 / TUI / 更新）
qlh-launcher app-ui       # 直接启动普通界面
qlh-launcher --headless   # 无头模式（仅 API，适合服务器）
sudo systemctl enable --now qlh-edge-inference  # 开机自启
```

> 前置依赖：`python3` (≥ 3.10)、`python3-venv`、`python3-tk`（图形 Launcher，推荐）、`tailscale`（分布式模式）。安装包内置独立 venv，不污染系统 Python。

### Android 客户端

> 前提：已安装 JDK 17 + Android SDK（API 34+），SDK 路径配置在 `../qlh-android/local.properties`
>
> 新克隆 `qlh-android` 后需先初始化其 llama.cpp submodule（Full 变体原生构建必需，Lite 不需要）：

```bash
cd ..\qlh-android
git submodule update --init --recursive
```

**编译**（无需 Android Studio）：

```bash
cd ..\qlh-android

# Debug APK（未压缩，开发用）
./gradlew.bat assembleDebug

# Release APK（R8 压缩 + 签名，分发用）
./gradlew.bat assembleRelease
```

产物：

| 产物 | 路径 | 典型大小 | 说明 |
|------|------|---------|------|
| Full Debug | `../qlh-android/app/build/outputs/apk/full/debug/app-full-debug.apk` | ~29 MB | 含 llama.cpp native 后端 |
| Full Release | `../qlh-android/app/build/outputs/apk/full/release/app-full-release.apk` | **~6.7 MB** | R8 + native strip |
| Lite Release | `../qlh-android/app/build/outputs/apk/lite/release/app-lite-release.apk` | **~1.5 MB** | 纯薄客户端，不含 native 库 |

**安装**：

```bash
adb install ../qlh-android/app/build/outputs/apk/full/release/app-full-release.apk
```

**使用**：

1. 启动 App → 底部导航选择「设置」
2. 全无模式：输入 PC 主节点 Tailscale IP 和端口 → 测试连接 → 开始对话
3. 全有模式：切换模式 → 选择包含 `.gguf` 的 SAF 外部目录 → 扫描并选中模型 → 离线推理

### 安装包分发服务器

在同一 Tailscale 网络内分发安装包，让其他设备浏览器直接下载：

```bash
cd packaging
python serve.py
# 默认端口 9090，浏览器访问 http://<本机Tailscale IP>:9090/
```

首页会列出：

- Windows PC 安装包 (.exe)
- Linux 安装包 (.deb)
- Android Full / Lite APK
- PC 模型压缩包 `models_pc.7z`
- Android 模型压缩包 `models_android.7z`（仅包含 GGUF 模型）

> 其他设备（包括 Android 手机）直接浏览器打开链接即可下载。

---

## 📊 历史量化基线（不构成当前发布或多机性能结论）

> 下表是早期固定环境的性能样例，只用于说明比较维度；真实模型采样质量、CUDA parity、双机吞吐与生产路由以[验收清单与资源限制登记](docs/验收清单与资源限制登记.md)和专项记录为准。

### CUDA 独显（PyTorch + bitsandbytes）

> 测试环境: NVIDIA RTX GPU + CUDA 12.6 + PyTorch 2.12.0 + Qwen-1.8B-Chat (24层)

| 配置 | GPU 显存 | 推理速度 | 备注 |
|------|---------|----------|------|
| FP16 | 3.47 GB | 53.2 tok/s | 基线对照组 |
| FP16 + compile | 3.47 GB | 55.1 tok/s | 算子融合 +3.6% |
| INT8 | 2.30 GB | 9.8 tok/s | 省显存但速度损失大 |
| **INT4** ⭐ | **1.75 GB** | **28.7 tok/s** | **推荐边缘设备：显存减半** |

### CPU / 集显（llama.cpp + GGUF）

> 测试环境: Intel i5-12400F / AMD R5 5600 + 16GB RAM + Windows 11

| 引擎 | 量化 | 内存 | 推理速度 | 备注 |
|------|------|------|----------|------|
| PyTorch CPU | FP16 | ~3.5 GB | ~3 tok/s | 无 CUDA 回退 |
| llama.cpp | Q4_K_M | ~1.2 GB | **~12 tok/s** | **推荐 CPU/集显** |

> llama.cpp 相比 PyTorch CPU：内存 **-65%**，速度 **+300%（3-5x）**

### Android 本地推理（理论预估，尚未作为真机验收结果）

| 芯片 | 等级 | Q4_K_M tok/s | 峰值 RAM |
|------|------|-------------|----------|
| 骁龙 8 Gen 3 | 旗舰 | 12-18 | 1.8 GB |
| 骁龙 8+ Gen 1 | 次旗舰 | 8-12 | 1.8 GB |
| 骁龙 865 | 中端 | 5-8 | 1.8 GB |

---

## 🧪 对照实验矩阵（设计与后续验收口径）

> 这不是已全部完成的实验结果。EX-N3 已完成既有历史记录的只读质量复核；真实模型多轮、CUDA、双机和生产路由采样仍在验收队列。

| 实验组 | 量化 | 算子融合 | KV缓存 | 编排策略 | 部署模式 |
|--------|------|----------|--------|----------|----------|
| 基线组 | FP16 | 无 | 传统KV | — | 单机 |
| 实验组1 | INT4 | 无 | 传统KV | — | 单机 |
| 实验组2 | INT4 | 融合 | 传统KV | — | 单机 |
| 实验组3 | INT4 | 融合 | 分页KV | — | 单机 |
| 实验组4 | INT4 | 融合 | 分页KV | 简单权重 | 分布式(3节点) |
| 实验组5 | INT4 | 融合 | 分页KV | 🧠 图算法 | 分布式(>5节点) |

---

## 📊 核心评判指标

- **显存占用**：量化、分页KV优化效果
- **推理时延 / Token生成速度**：算子融合、流水线延迟
- **网络带宽利用率**：图算法编排 vs 简单权重分配
- **CPU负载 / 网络延迟**：分布式通信开销
- **对话通顺度**：量化精度损失评估
- **长时间运行稳定性**：断线重连、心跳恢复、缓存清理

---

## 👥 团队分工

| 小组 | 职责 |
|------|------|
| 模型优化组 | 文献调研、模型量化、算子融合、KV缓存优化 |
| 分布式架构组 | 分布式架构设计、通信协议开发、多机调度逻辑 |
| 前端与文档组 | Web可视化平台、性能监控模块、文档与演示材料 |

**指导教师**：高博 副教授（北京交通大学软件学院）

---

## 📚 文档索引

### 设计文档

- [主线开发计划：分布式推理与边缘优化](docs/主线开发计划-分布式推理与边缘优化-2026-09-14.md) — 当前主线基线：llama.cpp/GGUF 模型分片、TUI、控制面和 Edge 无 torch 路径
- [支线开发计划：外置迁移与 Koakumix](docs/支线开发计划-外置迁移与Koakumix-2026-09-14.md) — 产品壳、侧车、发布和工具迁移边界
- [总体下一步计划](docs/总体下一步计划.md) — 历史总排期与当前计划索引
- [项目进展与下一步计划](docs/archive/项目进展与下一步计划.md) — **历史能力与证据快照**；当前主线/支线排期以两份 2026-09-15 基线计划为准
- [项目技术说明（新人入门）](docs/项目技术说明.md) — KV、算子融合、模型量化、分布式架构、并发调度与通信协议
- [文档状态与维护规则](docs/文档状态与清理清单.md) — 文档状态定义与后续维护规则
- [整体架构](docs/整体架构.md)
- [核心技术原理](docs/核心技术原理.md)
- [2-bit、3-bit 与 4-bit 量化调研与实施计划](docs/archive/2bit与4bit量化调研与实施计划.md) — 14B+ 低比特容量路线、Q2/Q3/IQ2 与 NF4/Q4 对照、GGUF/Android 验证、PyTorch sidecar 与 Go/No-Go 门槛
- [模块接口说明](docs/模块接口说明.md)
- [测试与评判标准](docs/测试与评判标准.md)
- [SD 1.5 引擎与分布式图像生成实施计划](docs/archive/SD%201.5引擎与分布式图像生成实施计划.md) — 历史验收记录；主项目实现已于 2026-09-14 裁撤，后续生图只在 Koakumix 演进
- [微服务架构改造计划](docs/archive/微服务架构改造计划.md) — 控制面/调度/推理三服务拆分、契约冻结与并行共存（阶段 3.2 完成；2.5/3.3 删除动作冻结至清理阶段）
- [一键模型部署与自治集群远期计划](../qlh-release/docs/一键模型部署与自治集群远期计划.md) — 模型注册、Sidecar、导入/下载、部署模拟与本机产品面已收口；下载治理采用 HF 直连 → 用户代理 → ModelScope 回退，真实大工件、CUDA、跨 PC 分发和生产路由仍待验收
- [测试通道运行说明](docs/测试通道运行说明.md) — 测试通道、标记（external/real_model）与运行方式
- [自动化优化实验与报告方案](docs/自动化优化实验与报告方案.md) — 固定提示词/seed/工件、串并行调度、统一 schema 与对照报告；EX-N3 只读生产质量门已复核既有记录 3/3 通过，真实模型采样、CUDA、双机和生产路由仍待验收
- [桌面赛博哥特与安卓原生界面复核](../qlh-shell/docs/桌面赛博哥特与安卓原生界面复核-2026-08-23.md) — PC 新前端与 pywebview/安装包交付链路、Android 原生极简风的边界、端间功能映射与 `CY-PKG-01` 计划

### 专项文档

- [小模型轻量推理 harness 工作台调研与方案](harness_workbench/docs/小模型轻量推理harness工作台调研与方案.md) — Koakumix 子项目：小模型定制化与唯一生图工作台；S1-S8 本机/离线开发门完成，真实运行时验收后置
- [联网搜索与轻量 Fetch 工具调用可行性调研与分期计划](docs/archive/联网搜索与轻量Fetch工具调用可行性调研与分期计划.md) — WEB-TOOL G0-G6/AUDIT 主节点 Tool Gateway 支线：能力探测、fail-closed 联网策略、adapter、TaskGraph Stage、显式持久化缓存与质量门；真实网络验收后置
- [DistilQwen2.5-DS3-0324 替代 R1 判题模型专项计划](docs/DistilQwen2.5-DS3-0324替代R1判题模型专项计划.md) — 快思考替代不可关闭 thinking 的 R1：v2 全口径 **2/4×3、8/11×3**，替代 R1 已批准候选；附多模型 0/4 判题口径问题专项分析
- [亚 1B 小模型专项实验计划](docs/亚1B小模型专项实验计划.md) — Qwen2.5-0.5B / Qwen3-0.6B / MiniCPM4-0.5B 用途（链路轻载体/thinking 开关标杆/新架构探针）与 M-SM-B1~B5 实验票
- [文档维护 Agent 工具子项目化与通用化专项计划](tools/docagent/docs/文档维护Agent工具子项目化与通用化专项计划.md) — 独立仓库 [qlh-docagent](https://github.com/SgfKrc/qlh-docagent)（主项目 submodule 引入）、规则数据化、规则变更机械扫描与演进门控（P1-P5）
- [reasonix-codex-bridge 完善方向](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge完善方向-2026-09-12.md) — 独立仓库 [reasonix-codex-bridge](https://github.com/SgfKrc/reasonix-codex-bridge)（主项目 submodule 引入）：Reasonix 子智能体接入 Codex 的 P0-P2 完善方向与验收门；W1/W2/W3 已落地默认关闭的受控写入、变更证据、显式回滚和读写 profile 分工，审计修复 `AUD-01`～`AUD-08`、G3 跨平台实跑与 R4 输出截断确定性已完成
- [reasonix-codex-bridge ACP 会话级恢复专项计划](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge-ACP会话级恢复专项计划-2026-09-13.md) — ACP 侧实测可用（`loadSession` / `session/{list,resume,close,delete}`），bridge 已完成 ACP-01～ACP-05，ACP-06 离线验收通过（强杀、orphan/resume、compact/rotate、并发取消与零泄漏）；默认仍 per-call，真实 provider 空会话跨进程 resume 返回 `unknown session`，恢复门待 provider 持久化语义补证
- [reasonix-codex-bridge Harness 工具扩展与能力补齐排期](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge-Harness工具扩展与能力补齐排期-2026-09-13.md) — 本机 Reasonix capability 调研与 EXEC/NET/LOOP/EVT/ACP/MESSAGE 票排期；`TOOL-RXB-EXEC-01`、`TOOL-RXB-NET-01`、`TOOL-RXB-NET-02` 与 `TOOL-RXB-LOOP-01` 已完成，搜索票在本机按 provider unavailable 门控，阶段编排保持主 agent 显式推进
- [reasonix-codex-bridge 工具面现状与能力归属](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge工具面现状-2026-09-13.md) — `reasonix doctor` 曾发现 profile 中 `git_log`/`git_diff` 为未知工具身份，现已由 `TOOL-RXB-TOOL-01` 收敛并验证归零；`web_fetch` 归 Reasonix，`web_search` 归 provider 并由 `providerSearch` fail-closed 门控；LOOP-01 阶段标记已接入 status/job/checkpoint
- [DeepSeek 缓存机制借鉴与 QLH 落地专项计划](docs/缓存机制专项计划-2026-09-13.md) — 登记 V4.1 磁盘上下文缓存与 SWA 单元匹配机制（三种持久化时机、hit/miss 25–50× 差价、KV 1/4 HBM 与 1/8 SSD），对照 `paged_kv_cache`/harness/bridge 现状，给出前缀稳定性与命中观测（零成本）、两级缓存与单元对齐（工程改造）、架构级压缩（仅跟踪）三档动作与 `CACHE-01`～`CACHE-06` 票
- [CACHE-05 非对称分工论证](docs/非对称分工论证-2026-09-14.md) — DeepSeek 输入/输出非对称仅作架构参考；QLH 以 draft-verify 与亚 1B 岗位化定义可验证控制变量、指标和停止规则
- [reasonix-codex-bridge 全面审计与多次实测报告](tools/reasonix-codex-bridge/docs/reasonix-codex-bridge全面审计与多次实测报告-2026-09-13.md) — 真实覆盖 MCP 控制面、inspect、plan、受控写入/回滚、checkpoint/resume、ACP-06、命名命令执行、Reasonix 原生 `web_fetch` 与 provider 搜索不可用门控；结论为可作为受限文件型低价替代，跨进程 ACP 恢复、自主多阶段编排和 provider 搜索接通验收仍待后续
- [答辩辅助工具细化与发散方案](../qlh-toolbox/docs/答辩辅助工具细化与发散方案.md) — P1-P4 细化与整体辅助工具发散；[模型文件 LZ4 压缩调研](docs/archive/模型文件LZ4压缩必要性调研与评估.md)（结论：不做本地转换）
- [抗弱网通信协议专项计划](docs/抗弱网通信协议专项计划.md) — 校园网 UDP 阻断、Tailscale/自建 DERP 现状、路径感知、应用层 WSS、Transport v2 与 UDP-over-WSS sidecar 分阶段计划
- [集群接入稳定性与本地RAG实施计划](docs/集群接入稳定性与本地RAG实施计划.md) — 手动入群一次性授权（CLUSTER-JOIN）、分布式角色/可用性审计、SSH 补丁传输、主节点本地 SQLite FTS5 + 向量 RAG、竞态/时序测试（T-RACE/G5.3）分期
- [前端、Android 与后端接口缺口审查](../qlh-shell/docs/前端安卓后端接口与功能缺口审查-2026-08-22.md) — 历史审查记录；产品壳已转入支线计划，Image Studio 已随 2026-09-14 生图裁撤移除
- [图算法智能编排](docs/图算法.md) — 最大带宽生成树 + DFS 路径搜索
- [分布式推理流水线实施计划](docs/分布式推理流水线实施计划.md) — 链式拓扑、LAYER_FORWARD 协议、KV Cache
- [混合分布式推理体系规划](docs/archive/混合分布式推理体系规划.md) — PyTorch 层间流水线、任务链、张量并行、exo 与 Mesh-LLM/GGUF stage 调研
- [三种分布式拆分细化实施方案](docs/三种分布式拆分细化实施方案.md) — PyTorch 层间待测试项、任务链和张量并行的协议、容错与实施阶段
- [Android 与 PC 功能差距清单](../qlh-shell/docs/安卓与PC功能差距清单.md) — 当前 Android Full/Lite 与 PC 的能力边界；presence、Full Worker/Stage、Gemma4 MTMD、更新/日志/诊断已完成本机开发门，真机/生产验收后置
- [Android 版本远期计划](docs/archive/Android版本远期计划.md) — Android 完整 Worker、任务链、GPU 平板与层间拆分的历史架构基线与远期边界
- [Android SAF 模型存储方案](docs/Android SAF模型存储方案.md) — SAF 外部目录、`/proc/self/fd` 加载、缓存副本 fallback
- Android llama.cpp 已随 `qlh-android` 迁移为 git submodule（`47e1de77`）；版本与维护事实源见 [`LLAMA_CPP_VERSION.md`](../qlh-android/app/src/main/cpp/LLAMA_CPP_VERSION.md)，迁移方案文档已废弃并移入 qlh-android 的 `_to_delete/`
- [任务链下一阶段实施计划](docs/任务链下一阶段实施计划.md) — dual_candidate DAG、journal、Provider registry、PC/Android Full Worker；开发门与短程双机证据已具备，`task_dispatch` 生产准入、长时/断电恢复仍后置
- [分布式推理仿真测试计划](docs/分布式推理仿真测试计划.md) — 无真实从节点时的仿真测试矩阵与运行方式
- [从节点部署配置指南](docs/从节点部署配置指南.md) — 从节点注册、模型目录与启动配置
- [数据库测试指南](docs/数据库测试指南.md) — 存储层测试现状：SQLite 契约、退场 fail-closed 用例与运行方式（PG 已退场）
- [离线资产一键整合包设计](../qlh-release/docs/离线资产一键整合包设计.md) — M1 已完成容量预检、清单、原子 ZIP、7z/分卷与恢复校验；真实全量出包和 Android SAF 导入后置
- [文档维护 Agent 工具设计](tools/docagent/docs/文档维护Agent工具设计.md) — M1-M3 已完成本机检索/语义质量门；工具只提供证据和建议，不自动改写文档
- [通用工具与子项目候选计划](docs/通用工具与子项目候选计划-2026-09-12.md) — 从主仓与 harness 筛选可跨项目复用的模块：A 档 5 项（进程归属/重置、演示证据链、签名更新、零依赖 TUI、多机同步与补丁分发）、B 档 7 项（harness 内部独立包，RAG 优先）、C 档 venv 环境治理入选；含判定标准、优先级与毕设选题映射

### 外部算力辅助（张量并行在异构 mesh 内不可行，改走集群外辅助）

- [张量并行外部辅助与混合拆分调研方案](docs/张量并行外部辅助与混合拆分调研方案.md) — 为什么 mesh 内 TP 必死的通信量化、三条外部辅助路线、与层间流水线的组合可行性、RQ/实验/里程碑
- [TP 孤岛接入指南](docs/TP孤岛接入指南.md) *(路线 A，PoC)* — 孤岛部署（vLLM/SGLang/llama.cpp rpc）、网关配置、验证与排障
- [外部推理服务 Provider 接入指南](docs/外部推理服务Provider接入指南.md) *(路线 B，PoC)* — 数据作用域门控、按请求路由、故障回退
- [投机解码外部辅助实施说明](docs/投机解码外部辅助实施说明.md) *(路线 C，实验)* — draft-verify 原理、分布等价性与已知偏差、接生产前的阻塞项

### 工程文档

- [TUI 使用指南](docs/TUI使用指南.md) — TUI 一键启动（自动带后端）、参数表、远程管理、故障排查
- [已知问题记录](docs/已知问题记录.md) — 缺陷/审计条目与修复状态（含 Full Worker/重启/IPv6 等实证登记）
- [待完成工作清单与推进顺序](docs/待完成工作清单与推进顺序-2026-08-23.md) — 当前可开发、条件性架构和外部验收队列
- [验收清单与资源限制登记](docs/验收清单与资源限制登记.md) — A 硬件限制类（显存/RAM，如 Qwen3-VL-8B 待内存条）/ B 网络限制类（B3 IPv6 实跑已验）验收入口
- [TUI 适配与聊天页实施计划](docs/TUI适配实施计划.md) — T1-T8 管理 TUI 网关适配与验收（Active）；T9.0-T9.5、T9.6 接线、T9.6-R2 Windows 开发机实装门和 UP-N6.4W 跨卷保留门已完成；外部干净机/Linux/真实模型会话、分布式真机与默认入口仍待（L4 Candidate）
- [TUI 指令集](docs/TUI指令集.md) — 27 条 `/` 命令全量参考（别名/参数/退出语义）
- TUI 技术栈与实现机制说明已并入 [TUI 适配与聊天页实施计划](docs/TUI适配实施计划.md) 和 [TUI 指令集](docs/TUI指令集.md)
- [双机补丁分发工具专项计划](../qlh-toolbox/docs/双机补丁分发工具专项计划.md) — 从节点签名补丁收发（`tools/patch_dispatch.py` / `patch_listener.py` + 根目录 bat；推送走本地 7897 代理、帧/分支校验、force-clean 保护节点身份）
- [打包说明](../qlh-release/README.md) — PyInstaller + Inno Setup 打包流程
- [独立安装包启动器与自动更新方案](../qlh-release/docs/安装包自动更新引导器方案.md) — 独立 Bootstrap、GUI/TUI、清单下载、Ed25519 验签/key rotation、UP-N3 原子版本与 UP-N4 A/B 自更新回滚；Launcher ZIP 发布链路已实测，Windows/Linux 干净机与 Android 更新仍待

---

## 📄 许可证

本项目为北京交通大学 2026 年大学生创新创业训练计划项目，以 [MIT License](LICENSE) 发布（Copyright (c) 2026 SgfKrc）。

---

© 2026 SgfKrc（QLH Project, Beijing Jiaotong University）· MIT License
