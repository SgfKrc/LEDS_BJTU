# QLH 速览 / QLH at a Glance（双语摘要 · 快速入口）

> 本页是给**不熟悉仓库的人**的 2 分钟导览；详细能力、边界与证据见 README 全文与专项文档。
> This page is a 2-minute tour for newcomers; the full feature list, boundaries and evidence live in the root README and the specialized plans.

---

## 这是什么？/ What is this?

**中文**：QLH 是面向异构边缘设备的**轻量化分布式大模型推理系统**（北京交通大学 2026 大创项目）。PC 主节点与从节点（PC/Surface/Android）通过 Tailscale 组网，按算力/内存/网络把模型分层协同推理；支持 PyTorch + llama.cpp 双引擎、INT4/INT8 量化、分页 KV 缓存、任务链编排、本地 RAG、SD 1.5 生图、Auth App 本地鉴权与多端客户端；核心设计原则是**数据不出集群、断网可自治、可复现验收**。

**English**: QLH is a lightweight distributed LLM inference system for heterogeneous edge devices (a 2026 BJTU innovation program project). Master and worker nodes (PC/Surface/Android) join a Tailscale mesh and split models across nodes by compute/memory/network; it features twin engines (PyTorch + llama.cpp), INT4/INT8 quantization, paged KV cache, task-graph orchestration, local RAG, SD 1.5 image generation, local Auth-App, and multi-client UIs. Design principles: **data stays in-cluster, offline autonomy, reproducible acceptance**.

---

## 已验证的能力 / Verified capabilities

| 中文 | English | 证据 Evidence |
|---|---|---|
| 双机真机分层推理（QW1.8B 0-21 / 21-24） | Dual-machine layer pipeline (verified 2026-08-20) | 3× `distributed_required` all passed, RTT 6-12ms |
| 任务链重启/杀进程恢复 + Tailnet IPv6 | Task-graph recovery + IPv6 (2026-08-21) | `wf_330d0aa1…`, reassignment=0 |
| 判题口径修复：多模型 0/4 → loose+512 可区分 | Judging-policy fix (P5) | Qwen3-4B 1/4 vs 1.8B 0/4 |
| DS3-0324-7B 替代 R1 判题模型（已批准候选） | DS3 replaces R1 as judging model | **2/4×3, format 8/11×3** (v2 policy) |
| 子项目：小模型 harness 工作台（S1-S8） | Sub-project: small-model harness workbench | context budget / STATE memory / RAG / MCP, local gates |
| 子项目：文档维护 Agent（独立仓库） | Sub-project: docagent (own repo, submodule) | rule-as-data + evolution gates |
| 联网搜索/轻量 Fetch（WEB-TOOL G1-G6 本机门） | Web search & Fetch tools (local gates) | `production_network_enabled=false` |

## 还不能宣称什么 / What is NOT claimed

- 生产路由准入（`task_dispatch` 关闭）、长时多轮/断电恢复、真实 443/WSS、IPv6-only 安装包、SD 分布式跨机、真实模型 7B/12B 三节点峰值——均在后置验收队列，不能凭本机门或模拟结果声称通过。
- 张量并行（TP）仅作集群外 PoC（route A）；投机解码为实验路径（route C）。

## 快速上手 / Quick start（5 分钟）

```bash
# 0. 克隆（含子模块 llama.cpp / docagent）
git clone --recurse-submodules https://github.com/SgfKrc/LEDS_BJTU
# 1. 一键环境（跳过 torch 大件，脚本会给出平台安装命令）
python scripts/setup_envs.py --all     # 或 setup_all_envs.sh/.bat
# 2. 下载默认模型（Qwen-1.8B-Chat，二选一或都装）
#    模型下载清单: README.md → 「克隆后资产获取清单」
# 3. 启动后端 + 产品前端
python src/api_server.py               # http://localhost:8000
cd frontend_cybergothic && npm run dev # http://localhost:5174
#    或终端版：./start_tui.sh 或 start_tui.bat（自动带后端）
# 4. 分布式：各节点装 Tailscale 同一账号 → 管理面板连接主节点
```

## 文档怎么读 / Reading map

- **主计划**：[总体下一步计划](总体下一步计划.md)（唯一排期入口）· [项目进展与下一步计划](项目进展与下一步计划.md)（证据快照）
- **新人入门**：[项目技术说明](项目技术说明.md) → [整体架构](整体架构.md) → [模块接口说明](模块接口说明.md)
- **子项目**：[harness 方案](小模型轻量推理harness工作台调研与方案.md) · [qlh-docagent](https://github.com/SgfKrc/qlh-docagent) · [联网工具调研](联网搜索与轻量Fetch工具调用可行性调研与分期计划.md)
- **实验与判题**：[DS3 替代 R1 专项](DistilQwen2.5-DS3-0324替代R1判题模型专项计划.md) · [亚1B 专项](亚1B小模型专项实验计划.md) · [测试与评判标准](测试与评判标准.md)

## 工程文化 / Engineering culture

- **证据链优先**：每个"已完成"必须附测试/日志/双人目视证据（EX-N3 只读质量门复核历史记录 3/3）。
- **机制化而非绿化**：测试通道分单元/契约/浏览器/竞态/仿真/真机；fix 必配负例，防止"固定用例反复通过却漏掉新缺陷"。
- **用户主权**：模型工件、密钥、知识库归用户；开发组不代管不代持；离线可恢复。

## 许可证 / License

MIT（Copyright (c) 2026 SgfKrc）——详见仓库根 [LICENSE](../LICENSE)。
