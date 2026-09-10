# 多子 agent 测试质量审计汇总

后续开发票与答辩演示排期见：[审计收口与答辩演示开发票计划](../../开发票计划-审计收口与答辩演示-2026-09-09.md)

日期：2026-09-10
范围：S7-MCP-01/S7-MCP-02 后续变更，重点覆盖四个新模型资产、模型下载与注册、Harness model_profiles、B1/DSW-D1 探测、三个前端和 MCP/Harness API。

## 审计范围校正

本轮开始时主任务曾误落到 `DOCAGENT-P5B` 的文档审计工具，最初产生的 docagent 专项结果不计入本报告。发现偏差后，主 agent 重新启动了四个独立子 agent，分别从后端资产、前端契约、测试质量、安全可靠性四个维度审计整个相关变更面；本报告只汇总这四个全局审计和主 agent 的代码 review。

## 子 agent 结果

| 维度 | 结论 | 主要发现 |
| --- | --- | --- |
| 后端/模型资产 | 发现高风险 | 预设目录大小写不一致；`model.manifest.json` 与本地资产扫描契约不一致；声明式哈希没有重新计算；模型 ID 与画像 ID 大小写/别名不一致；API 暴露绝对路径 |
| 前端/API 契约 | 发现中风险 | 主前端模型目录请求存在单点失败；CyberGothic 本地资产详情绑定不稳定；Harness SSE 错误会误标记为离线；运行模型和候选画像应保持可选性边界 |
| 测试质量 | 发现覆盖缺口 | 实际 load/switch 失败路径、B1 真实 runtime smoke、DOCAGENT CLI 磁盘变更、MCP 负向 UI 状态仍缺稳定自动门 |
| 安全/可靠性/打包 | 发现高风险 | 下载目标和显式 GGUF 路径缺少边界；本地目录复制可跟随符号链接；远程模型无 revision/SHA 固定；MiniCPM4 探测使用 remote code；发布后注册失败可能遗留目录 |

## 主 agent 代码 review 追加发现

画像注册表 `get()` 中 `or` 与后续过滤条件的优先级有缺陷：精确 `model_id` 命中时可能绕过 backend、artifact sha 和 revision 过滤。该问题不在子 agent 报告中，已由主 agent 修复并添加回归测试。

## 已完成修复

### P1：下载和资产完整性

- `src/model_download_jobs.py`
  - 四个新增预设使用与内置资产一致的稳定目录名，不再直接使用 Hugging Face repo 展示名。
  - 远程下载的 `target` 解析强制位于 `models` 根目录内，拒绝根目录本身、路径穿越和解析后越界路径；本地目录导入保留显式目标目录的既有兼容契约，并将其父目录作为本次导入根。
  - 本地 source 目录及其内容拒绝符号链接，避免复制阶段越界读取。
  - 远程 source 约束为 `provider/repository` 形状。
  - 显式 GGUF 选择器限定在 staging 目录内并校验 `.gguf` 后缀。
  - 注册失败时清理已经发布的目标目录，避免“目录已存在、任务无法重试”的孤儿状态。

- `src/local_model_assets.py`
  - 同时识别项目 manifest 和 `model.manifest.json` 兼容格式。
  - 兼容 manifest 支持 `size_bytes`，并对每个声明 SHA-256 重新计算；尺寸或哈希不匹配时 fail closed。
  - 未提供完整哈希的旧 manifest 标记为 `manifest_unverified`，不再伪称 `manifest_verified`。
  - safetensors 与 GGUF 合并资产只在组合中的所有 manifest 部分均完成校验时才标记为已验证。

- `src/api_server.py`
  - `/api/models` 与 `/api/models/registry` 对外只返回相对资产路径或文件名，不再泄露服务器绝对文件系统路径（包括 `expected_paths` 提示字段）；进程内加载仍使用原始路径。

### P1：Harness 和前端契约

- `harness_workbench/model_profiles/schema.py`、`registry.py`、`builtin.py`
  - 新模型画像加入稳定别名，兼容主项目实际 model ID。
  - 别名解析与精确 ID 使用同一组 backend、artifact sha、revision 过滤条件。
  - malformed aliases 现在会被拒绝，而不是静默转换为空数组。

- `frontend/src/components/ModelSelector.jsx`
  - 模型目录、运行引擎和 current model 改用独立 `Promise.allSettled` 结果，单个接口失败不会清空另一侧可用状态。

- `frontend_cybergothic/src/pages/ModelsPage.tsx`
  - 本地资产即使没有 registry 条目也能生成详情模型并正确绑定右侧详情区。

- `harness_workbench/ui_react/src/data.ts`、`App.tsx`
  - 保留后端错误的 status/code/retryable 信息。
  - 模型错误、权限错误和 SSE 业务错误不会再把健康连接错误标成 offline。
  - Harness 画像页展示新模型别名，同时继续区分 live adapter model 与 candidate profile；候选画像不会被误当作可运行模型。

- `scripts/model_tools/small_model_probe.py`
  - 模板探测 worker 改用最小环境白名单，过滤 token、password、secret、API key、credential、authorization 等变量，并固定离线标志。

## 回归验证

以下命令均在仓库 `.venv-test` 环境或各前端自身 Node 环境执行：

- 模型下载、资产扫描、画像、B1/DSW-D1：`30 passed, 1 skipped`
- API 模型、model config、Harness API：`74 passed`
- 主前端 Node 测试：`65 passed`
- 主前端 `npm run build`：通过
- CyberGothic `npm run build`：通过
- Harness UI `npm run build`：通过
- Python `compileall`：通过
- `git diff --check`：通过
- 最终受影响接口回归：`81 passed, 1 skipped`
- 全量 Python suite：`3374 passed, 20 skipped`（201.18s；并行 worker 争用共享 task-journal 的告警未导致测试失败）

## 仍保留的发布门

这些项目不能在没有上游可信材料或运行环境的情况下伪造为已修复：

1. `AUD-PIN-01` 已关闭：四个远程预设现在有官方仓库的完整 revision、允许/必需文件清单和逐权重 SHA-256，下载服务会固定版本并 fail-closed 校验。当前环境未执行真实模型下载，仍需在有发布源网络和足够磁盘的环境完成联网验收。
2. `AUD-SBX-01` 已关闭：MiniCPM4 的 `trust_remote_code=True` 探测现在运行在 OS 级低权限 worker 中；Linux/macOS 使用 sandbox backend，Windows 使用 restricted token + low-integrity token + Job Object，并对 metadata 使用一次性 snapshot。Windows 低完整性不提供内核级禁网，因此仍强制离线环境变量，网络隔离增强不作为已完成能力宣称。
3. `AUD-AUTH-01` 已关闭：模型 API 的认证/来源门、Bearer 透传和 control-svc loopback 默认绑定已完成；跨机部署仍必须显式配置受控 CIDR/Tailnet 和防火墙策略。
4. 尚缺真实 llama-cpp/MiniCPM4 runtime smoke 和实际 model switch/load 负向场景；MCP 负向 UI 自动门与 CyberGothic 全量 Playwright 回归均已关闭。

## 待完成工作统计

统计口径：按可以独立验收的行动计数；复合发现拆分为独立工作项。截至 `AUD-CY-E2E-01` 关闭后，共有 **2 项** 待完成工作，分为 **1 类**：

| 类别 | 数量 | 待完成项 |
| --- | ---: | --- |
| 真实运行验收 | 2 | 在真实依赖环境执行 llama-cpp/MiniCPM4 runtime smoke；补真实模型 load/switch 成功与失败场景断言 |
| 自动化回归与契约 | 0 | 无 |

当前已完成代码回归、三个前端构建、MCP 负向 UI 门、CyberGothic Playwright 全量回归和全量 Python suite；剩余 2 项仍不能以 fixture 定向测试或构建成功替代，等待真实模型环境后逐项关闭。

## 后续验收顺序

### AUD-AUTH-01 已关闭

模型 API 的认证/来源门已按后续开发票完成：gateway 对模型 registry/GGUF/download 使用 Web Bearer 和管理员角色；单体 api_server 与 inference-svc 默认只接受 loopback，跨机来源必须显式配置 `QLH_MODEL_API_TRUSTED_CIDRS`；control-svc 默认绑定 `127.0.0.1`。相关来源门、Bearer 透传和 HTTP 拒绝测试已通过。原清单中的该项因此从待完成统计中扣除，当前剩余 2 项按真实模型环境条件执行。

1. `AUD-RT-01`：在具备固定工件、发布源网络和真实 llama-cpp/MiniCPM4 runtime 的环境跑模板/thinking/架构兼容 smoke。
2. `AUD-SW-01`：在同一真实模型环境补 load/switch 成功、失败、回滚和量化不匹配断言。
3. `AUD-MCP-UI-01` 与 `AUD-CY-E2E-01` 已关闭：MCP 负向 UI 自动门五类失败态均通过，CyberGothic 全量 Playwright 为 `64 passed`；后续只剩上述两张真实模型票。

## `AUD-MCP-UI-01` 复核结论

- `harness_workbench/ui_react/scripts/mcp_negative_ui_gate.mjs` 通过 Vite + Playwright fixture 覆盖 permission denied、unknown tool、malformed/invalid arguments、backend business error 和 SSE error 五类失败态。
- UI 保留 status/code/retryable 信息；HTTP 失败和 MCP `isError` 不会进入成功结果区，SSE 业务错误不会把健康连接误标成 offline。
- 验证证据：Harness `npm run typecheck` 与 `npm run test:mcp-negative` 通过；相关 Python 回归 `14 passed, 1 skipped`；不依赖真实模型或外网。

## `AUD-CY-E2E-01` 复核结论

- `frontend_cybergothic` 使用系统 Edge + Vite fixture 执行完整 Playwright 矩阵，最终 `64 passed`。
- 回归覆盖 API 错误、角色/权限、下载/会话/生图、审计/集群/任务、响应式、键盘可达、工作台分屏和 SSE fixture；不依赖真实模型或外网。
- 回归期间修复移动端 pane 边界抖动与日志字体基线布局不稳定，均限定在 `frontend_cybergothic/src/styles/workbench.css`。

## `AUD-PIN-01` 复核结论

- 四个新模型的 pin 文件为 `docs/agent_tool/model-artifacts/remote-model-pins-2026-09-09.json`，包含仓库 revision、下载白名单、必需文件和权重文件 SHA-256。
- 下载链已把 pin 约束传递到 Hugging Face `snapshot_download`，并在发布前验证文件集合和逐文件哈希；pin 缺失、非法、缺文件、越界路径或哈希不匹配都会阻断 job。
- 专项测试 `tests/test_model_download_jobs.py`：`15 passed, 1 skipped`。真实模型下载和 runtime smoke 留给具备网络、磁盘和运行时依赖的后续验收票。

## `AUD-SBX-01` 复核结论

- `scripts/model_tools/sandbox_runner.py` 已成为模板 worker 的唯一启动路径；无 OS sandbox backend 时返回 `os_sandbox_unavailable`，不会退回普通 `subprocess`。
- Windows 实机验证通过：`windows-restricted-token-low-integrity` worker 成功启动并完成 MiniCPM4 tokenizer/template 探测；权重未加载，metadata snapshot 在 worker 退出后清理。
- `tests/test_small_model_probe.py` + `tests/test_model_probe_sandbox.py`：`9 passed`。Windows low-integrity 不能等同于内核级网络隔离，当前网络控制仍是 sandbox 外的 offline 环境变量，已明确记录为后续增强项。
