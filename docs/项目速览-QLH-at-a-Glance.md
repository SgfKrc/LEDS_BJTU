# QLH 速览 / 新人 · 评审快速入口

> 状态：**现行**
>
> 更新日期：2026-09-22

> **Language**: [English](项目速览-QLH-at-a-Glance.en.md) · [简体中文](项目速览-QLH-at-a-Glance.md)
>
> 本页是给**不熟悉仓库的人**的 2 分钟导览；详细能力、边界与证据见 [README](../README.md) 全文与专项文档。

---

## 这是什么？

QLH 是面向异构边缘设备的**轻量化分布式大模型推理系统**（北京交通大学 2026 大创项目）。它把"谁持有哪些层、用什么引擎、在哪里、容量多大"统一成一条抽象 `(layer_range, engine, location)`，据此在**装不下整模**的机器群里把模型**分层接力**跑起来。

**两条引擎档位**（按设备画像选，不是"有 GPU 就用 torch"）：

| 档位 | 引擎 | 适用 |
|---|---|---|
| **L 档** | llama.cpp / GGUF（含裁层 GGUF + `embd` 注入） | 轻量、边缘、无 torch 的节点 |
| **D 档** | PyTorch（层拆分、层间流水线、多节点层段） | 有 CUDA 的 PC；也是层切点搜索/实验的平台 |

**三种节点同构**，共享同一套合同与 fail-closed 校验：`local`（进程内）、`remote_rpc`（网络借算力，Android/PC 上的 `ggml-rpc-server`）、`cross_framework`（跨引擎层段接力）。可选的 Relay R 轨道（裁层 GGUF + `embd` 注入，同机/跨机）用于**容量合并与异构能力组合**。

**定位边界（重要）**：接力是**容量**手段，不是提速手段。同机实测里，跨框架接力最优（129.1 ms/token）比"llama.cpp + CUDA 全层上 GPU"（26.6 ms/token）**慢 4.85×**；有 CUDA 且装得下整模时，最佳实践是**不要接力**，让 CUDA 节点直接跑整模（或作 RPC/分片 worker）。默认路由始终优先"可直接整模跑"的 L / RPC 路径。

核心设计原则：**数据不出集群、断网可自治、可复现验收**。

## 已验证的能力（每条都带判据与日期）

| 能力 | 证据（判据 / 日期） |
|---|---|
| 双机真机分层推理（QW1.8B 0-21 / 21-24） | 3 次 `distributed_required` 全过，RTT 6-12 ms（2026-08-20） |
| 任务链重启/杀进程恢复 + Tailnet IPv6 | `wf_330d0aa1…`，重派 0（2026-08-21） |
| **D→L 跨框架接力正确性** | 主仓双引擎矩阵 **27/27 逐 token 一致**：qwen2.5-0.5B K=4/8/12/16/20、qwen3.5-2B K=8/12/16/20；负载 prefill 32/128/512、decode 32/64/256；batch 2/4；混合精度 fp16·f32·NF4 × Q4_K_M（2026-09-21） |
| **D→L 容量收益** | qwen2.5-0.5B **1.568×** / qwen3-5-2b **1.547×**；同一受控 3.0 GB CUDA 预算下整模被拒而 12 层上游通过（2026-09-21） |
| **L→L keep-head 通道** | `--path l2l_keep_head` / `d2l2l_keep_head` **32/32**，含「1 torch 上游 + 2 llama 下游」三段（2026-09-21） |
| **纯 L→L 多跳（三段全设备）** | y700 head8 → y700 mid8-16 → y700 cut-k16 tail，本机只发命令 ⇒ **32/32**（2026-09-22） |
| **纯 L→L 跨设备（三段）** | y700 head8 → **Surface** mid8-16 → y700 tail ⇒ **32/32**，两台真机协作、本机不参与计算（2026-09-22） |
| **Android 层段数值验证** | 真 ARM64（Snapdragon 8 Gen 3 / Android 15）与 x86_64 **逐 token 完全一致**；dotprod/i8mm kernel 已证实；`qlh-android` P0 交叉编译/JNI 完成（2026-09-21/22） |
| 切点求解闭环 | `scripts/relay_cut_plan.py` 从**实测**拟合段画像（固定开销 + 每层耗时）再求解；2 段闭环在 qwen2.5（r² 0.96/0.99）与 qwen3.5（0.79/0.96）上通过 |
| Windows 原生编译路径 | `triton-windows==3.8.0.post28` 实测可用（`torch.compile` 不再是死开关）；`PYTHONUTF8=1` 是前置条件 |
| 判题口径修复 | 多模型 0/4 → loose+512 可区分；DS3-0324-7B 替代 R1（v2 口径 2/4×3、格式率 8/11×3） |
| 子项目：小模型 harness 工作台（S1-S8） | 上下文预算 / STATE 记忆 / RAG / MCP，本机门 |
| 联网搜索 / 轻量 Fetch（WEB-TOOL G1-G6） | 本机开发门，`production_network_enabled=false` |

## 还不能宣称什么

- **跨机 RPC vs 接力的对照仍未测**；D→L 的**长时、跨机与多段故障验收**待补。
- **中段必须与调用方同局域网**：实测把中段放在**跨网络**节点（Surface 走 Tailscale **DERP 中继**、RTT 777 ms）时，每步一次往返 ⇒ 端到端从中位 ~140 ms 涨到 ~800 ms。跨网络尤其经中继的路径**不可用**。
- 生产路由准入（`task_dispatch` 关闭）、长时多轮/断电恢复、真实 443/WSS、IPv6-only 安装包、真实 7B/12B 三节点峰值——均在**后置验收队列**，不能凭本机门或模拟结果声称通过。
- Android 只到 `P0`（交叉编译/JNI + 层段数值验证）；设备运行、RPC worker、断线、热/电与安全证据**未完成**。
- 张量并行（TP）仅作集群外 PoC；投机解码为实验路径。
- **验收判据纪律**：接力一致性只认 **per-token argmax**（不用 cosine 代替）；分叉即如实标 FAIL，不做"可接受近似"。

## 首次启动需要做什么

### 0. 前置依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.10（推荐 3.12） | 主运行时与工具脚本 |
| Node.js + npm | Node ≥ 18 | 仅产品壳支线（`qlh-shell`） |
| JDK 17 + Android SDK (API 34+) | — | 仅构建 Android 时需要 |
| Tailscale | 最新 | 分布式模式必须；⚠️ **节点在同一局域网时最稳**（跨网络会走 DERP 中继，层段接力会因每步往返而不可用） |
| Git | — | clone（含 submodule） |
| NVIDIA 驱动 + CUDA（可选） | — | 仅 D 档 / PC 独显版需要 |

### 1. 克隆与一键配环境

```bash
git clone --recurse-submodules https://github.com/SgfKrc/LEDS_BJTU
cd LEDS_BJTU
# 当前 4 个子模块：android / packaging(qlh-release) / harness_workbench(Koakumix) / frontend_cybergothic(qlh-shell)
# 注：两个桥接器（reasonix-codex-bridge / dsh-codex-bridge）是**工作区独立目录**、不是子模块
python scripts/setup_envs.py --all            # 主线 Python 环境（默认不含 Node）
python scripts/setup_envs.py --all --with-node # 另行配置产品壳 Node 迁移源
python scripts/setup_envs.py --only test,tui  # 只配指定环境
python scripts/setup_envs.py --check          # 只校验不安装（无副作用）
```

**覆盖清单**：主环境 + `.venv-test` / `.venv-tui` / `.venv-qwen3-sidecar` 等；产品壳、Android、打包与 Node 环境属于支线迁移源，不是主线运行前置。Koakumix 的生图依赖只在其独立环境管理。

> ⚠️ **torch 等平台相关大件不自动安装**：脚本会过滤并打印各环境安装命令（如 `--torch-index-url https://download.pytorch.org/whl/cu126`），避免 CPU/CUDA 版本互相污染。按提示装完后再跑一次 `--check`。

### 2. 获取模型（**按设备画像分档**）

默认模型不是固定一个，而是按画像选（`src/model_config.py` 的 `DEFAULT_MODEL_BY_TIER`，消费点用 `config.get_active_model_paths()`）：

| 设备档 | 默认模型 |
|---|---|
| 移动端 / 边缘 / 超极本（共享显存 ≤ 2 GB） | `<1B` 档：`qwen3-0.6b` |
| PC（独显） | `~2B` 档：`qwen3-5-2b` |

> ⚠️ 这是**硬约束**不是偏好：2B 档实测 **VRAM 峰值 2.24 GB** ⇒ ≤2 GB 共享显存的超极本**放不下**。

```bash
# GGUF Q8_0（CPU/集显/Android 起点）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
```

其余模型见 [README](../README.md)「模型下载」；模型文件**不进 git**（`models/` 已 gitignore）。

### 3. 启动与验证

```bash
python src/api_server.py               # 后端 http://localhost:8000
python qlh.py                         # 主线 TUI（Textual 外壳，9 个功能屏 + 1 个调试兜底）
# 或终端版：./start_tui.sh（Windows: start_tui.bat，自动带后端）
```

```powershell
# 测试环境就绪（当前串行全量基线见 README）
.\.venv-test\Scripts\python.exe -m pytest tests/ -q --collect-only
# 跨机接力前先做健康检查（探活 + 心跳新鲜度；0 = 全健康 / 1 = 有异常）
python scripts/relay_health.py --check tail=tcp:127.0.0.1:50188 --json
```

分布式：各节点同一 Tailscale 账号 → TUI 的集群/节点命令 → 节点上线。

## 文档怎么读

- **总入口**：[README](../README.md)（中）· [README.en](README.en.md)（英）
- **接力（本仓最活跃的方向）**：[跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) —— **以它为索引**，旧报告中的矛盾数字按其有效性分级处理；配套[项目报告](跨框架层接力-项目报告.md)与[容量收益实测](跨框架层接力-容量收益实测-2026-09-21.md)
- **新人入门**：[项目技术说明](archive/distributed/项目技术说明.md) → [整体架构](整体架构.md) → [模块接口说明](模块接口说明.md)
- **计划**：[主线开发计划：分布式推理与边缘优化](主线开发计划-分布式推理与边缘优化-2026-09-14.md) · [P4.5 立项：主节点动态选举与配套分布式管理](主节点动态选举与分布式管理-P4.5立项-2026-09-21.md) · [总体下一步计划](archive/distributed/总体下一步计划.md)（历史总排期）
- **协议与节点**：[层段协议立项](层段协议立项-2026-09-17.md) · [层流水线节点类型与顶层透明性](层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md)
- **TUI**：[TUI 使用指南](TUI使用指南.md) · [TUI 指令集](TUI指令集.md)
- **测试与判据**：[测试与评判标准](测试与评判标准.md) · [测试通道运行说明](测试通道运行说明.md)
- **Android**：[Android 验证替代路径](../android/Android验证替代路径-2026-09-18.md)
- **文档自身**：[文档状态与清理清单](文档状态与清理清单.md)

## 工程文化

- **证据链优先**：每个"已完成"必须附测试/日志/真机证据（判据与日期缺一不可）。
- **判据纪律**：一致性只认 per-token argmax；分叉如实标 FAIL；区分 `mainrepo_end_to_end` / `raw_binding_probe` / `capacity_only` 三类记录，不混用。
- **机制化而非绿化**：测试通道分单元/契约/浏览器/竞态/仿真/真机；修复必配负例，防止"固定用例反复通过却漏掉新缺陷"。
- **先测再解释**：性能结论必须先量出瓶颈在哪一环（分段耗时 / RTT / 离散度），不要用"设备档次"讲故事。
- **用户主权**：模型工件、密钥、知识库归用户；开发组不代管不代持；离线可恢复。

## 许可证

MIT（Copyright (c) 2026 SgfKrc）——详见仓库根 [LICENSE](../LICENSE)。
