# Gemma-4-12B 多模态支持方案（实施计划）

> 状态：Active（G4.1 后端与官方模型实机门完成；G4.2 前端/TUI 图片交互完成；G4.3.1 原生绑定/ABI 预检完成；G4.3.2+ 工件、音频、分布式和资源门 Candidate）
>
> 更新日期：2026-08-11
> 适用范围：原版 Gemma 4 12B 经本机 Ollama `external_api` 提供文本与图像理解；G4.3.1 只完成原生 `llama-cpp-python` MTMD ABI 预检，音频和原生实际推理另列后续票
>
> 目标：先提供可验证、可回滚的原版 Gemma 4 12B 图像理解能力，再补产品交互；`myheretic:latest` 仍只作为用户私有资产，不进入项目基线。

---

## 1. 背景与目标

- 模型：**Google Gemma 4 12B Unified**（2026-06-03 发布，`gemma4` 架构，11.9B，256K 上下文，官方能力为 Text + Image + Audio 输入、Text 输出）。
- 本机显卡:RTX 4060 Laptop,**8GB 显存**。
- 本机现状：官方 Ollama `gemma4:12b` 已通过 `127.0.0.1:7897` 子进程代理完整拉取并验证；用户私有 `myheretic:latest` 保留但不参与验收。
- 模型基线：当前唯一已验收工件为 Ollama 官方 `gemma4:12b`；直接 GGUF 来源、revision 和 Python 绑定 ABI 尚未冻结，不作为 G4.1 依赖。
- 当前目标：QLH 已具备可用的 Gemma 4 图像理解产品路径；G4.3.1 已完成只读的原生绑定/ABI 预检，后续仅在冻结工件后验证原生 llama.cpp、音频、分布式和 8K/16K 资源门。

### 1.1 "多模态实验缺最后一种模型"的含义

对项目现状盘点(`docs/`、`src/`、`control/` 全量检索):

| 模态能力 | 现状 | 载体 |
|---|---|---|
| 文本推理(LLM) | ✅ 已有 8 个内置槽位 | `BUILTIN_MODELS`(`src/model_config.py:85-198`) |
| 图像生成(文生图) | ✅ 已有 SD 1.5 全系 | `src/diffusion/assets.py` |
| **图像理解(图生文)** | ❌ **缺失——多模态实验缺的最后一种** | 本方案补 gemma-4-12B |

项目内**没有**现成的"多模态实验"清单文档(仅 `docs/一键模型部署与自治集群远期计划.md:309,340` 提及多模态 projector 为远期依赖);本方案同时定义该实验的接入与验证基线。

---

## 2. 调研结论

### 2.1 模型规格(已核实)

| 项目 | 结论 |
|---|---|
| 模型名 | Google Gemma 4 12B Unified；Ollama 固定标签 `gemma4:12b` |
| 架构/参数量 | `gemma4`，11.9B，Ollama 模型 ID `4eb23ef187e2` |
| 模态 | 官方为 Text + Image + Audio → Text；G4.1 只开放 Text + Image，音频尚无 QLH 契约 |
| 上下文 | 262144；当前 8GB 开发门使用 2K/4K smoke，不把 256K 设计上限当本机可用承诺 |
| 量化/工件 | Q4_K_M；Ollama 清单显示 7.6GB，运行时 `ollama ps` 显示约 9.9GB 工作集 |
| 本机支持状态 | Ollama 0.32.8 已实测 completion/vision/audio/tools/thinking；QLH G4.1 已实测 OpenAI `image_url` |
| 原生支持状态 | 项目锁定 llama.cpp `47e1de77…`；本机 `llama-cpp-python 0.3.28` 的 MTMD 必需 ABI 已由 G4.3.1 实测可导入，但未提供经 SHA 校验的 Gemma 4 GGUF/mmproj 对，也未初始化 handler 或处理图片，保持 Candidate |
| 标签边界 | `gemma4:latest` 当前指向 E4B；12B 必须显式写 `gemma4:12b` |

### 2.2 本机部署现状（2026-08-11 已实测）

```text
ollama 0.32.8
gemma4:12b  4eb23ef187e2  7.6 GB
architecture=gemma4  parameters=11.9B  quantization=Q4_K_M
capabilities=completion/vision/audio/tools/thinking
projector=clip 52.38M
```

- 下载：只给 `ollama pull gemma4:12b` 子进程注入 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897`，耗时约 886.9 秒；未修改系统或项目全局代理。
- 文本：`think=false` 后返回精确 `GEMMA4_OK`；首次加载约 12.9 秒，热请求总计约 1.46 秒。
- 图像：Ollama 原生 API 能正确描述测试图标；QLH `ExternalChatClient` 经 `/v1/chat/completions`、`reasoning_effort=none` 返回合理描述，`finish_reason=stop`，93 prompt + 19 completion tokens。
- 资源：驻留时 `ollama ps` 为 `56% CPU / 44% GPU`；RTX 4060 Laptop 8GB 当时显存占用约 6248MiB。该证据证明可运行，不代表 256K 上下文可用。

> ⚠️ **`myheretic:latest` 是用户私人微调模型,不是原版**:
> 它是用户基于 gemma-4-12B 私人微调导入的版本,行为与官方原版**不一定一致,不具备通用能力**。
> 本机实测仅能证明「gemma4 架构 + Q4_K_M 在 8GB 显存可运行」,**不能**作为多模态实验的模型基线;
> 实验验证必须用原版(见 4.1 与方案 B)。

```
$ ollama list
NAME                 ID              SIZE      MODIFIED
myheretic:latest     e0a12e370be1    7.4 GB    2 hours ago
nomic-embed-text     0a109f422b47    274 MB    9 days ago

$ ollama show myheretic
  architecture        gemma4
  parameters          11.9B
  quantization        Q4_K_M
  Capabilities        completion / tools / thinking   ← 无 vision
```

- `myheretic:latest` 是用户私人微调版 gemma-4-12B Q4_K_M(自定义 tag),**非官方原版权重**;多模态实验以原版为基线。
- **缺 vision 的原因**:导入的 GGUF 未附带 `mmproj-*.gguf`(视觉投影器),Ollama 据此判定为非多模态。
- **结论**:现有 `myheretic` 不可作为多模态实验模型——**必须重新获取原版模型**(路线 A1 拉官方 tag,或路线 B 下载原版 GGUF)。
- Ollama 服务与 OpenAI 兼容端点均已实测；QLH 配置的 base URL 必须是 `http://127.0.0.1:11434`，由 `IslandEngine` 自行追加 `/v1/*`。

### 2.3 硬件可行性(RTX 4060 8GB)

| 约束 | 结论 |
|---|---|
| 显存/内存 | 实测由 Ollama 自动分配为 56% CPU / 44% GPU；不得宣称 8GB 可全 GPU 驻留 |
| 上下文 | 当前只验收 2K/4K；8K–16K 和长会话 OOM 必须在 G4.3 单独取数，256K 不进入本机承诺 |
| 时延 | 文本热请求约 1.46 秒；图像原生 smoke 总计约 24.4 秒，其中模型加载约 13.1 秒、图像 prompt 约 10.0 秒 |
| 结论 | 可作为单用户实验/验证模型；高并发、长上下文、音频与分布式均未验收 |

---

## 3. 项目接入点现状(改动基线)

| 层 | 现状 | 新增 gemma-4-12B 需要的改动 |
|---|---|---|
| 模型发现 | `external_api` 通过 `/v1/models` 发现 `gemma4:12b`，不伪装成本地 GGUF 注册项 | G4.1 Completed；状态继续标记 `engine=external_api` |
| 外部 Provider | `IslandEngine` 保持 OpenAI 兼容传输，支持结构化 `content` 和可选 `reasoning_effort` | G4.1 Completed；未添加 Ollama 私有客户端 |
| chat 契约 | `image_data_urls` 最多 4 张，只接受 PNG/JPEG/WebP data URL；单张 8MiB、总计 16MiB | G4.1 Completed；远程 URL、SVG、MIME 伪装和无显式授权均拒绝 |
| 安全/持久化 | 图片必须 `allow_external=true + prefer_external=true`；scope deny/端点失败均禁止丢图回退 | G4.1 Completed；历史和 SQLite 只保存文字问题/回复，不保存原图 |
| 前端/TUI | Web 当前内存图片选择/预览/删除/大小提示，TUI 本地路径队列；二者均仅在本次请求发送 data URL | G4.2 Completed；桌面/移动与真实 Edge 会话已验收 |
| 原生 llama_cpp | `gemma4-native-probe` 已验证 MTMD 必需 ABI，但没有任何已校验工件时不会初始化 `Llama` 或 handler | G4.3.1 Completed；G4.3.2 起先验工件/handler/打包再定 recipe |

---

## 4. 方案 A:Ollama 外部 Provider 接入(快速验证,改动最小)

**思路**：本机 Ollama 已运行且 OpenAI 兼容端点可用；QLH 继续复用 `external_api`，只扩展通用 OpenAI 结构化消息，不增加 Ollama 专属引擎。

> G4.1/G4.2 已完成官方模型下载、后端接线和产品交互。该路线目前是唯一实机通过的 Gemma 4 路径。

### 4.1 前置:获取带 vision 的**原版** gemma-4-12B

已完成命令（代理只作用于当前子进程）：

```powershell
$env:HTTP_PROXY='http://127.0.0.1:7897'
$env:HTTPS_PROXY='http://127.0.0.1:7897'
ollama pull gemma4:12b
ollama show gemma4:12b
```

> ~~A2(复用现有 myheretic 重建)~~ **已废弃**:myheretic 是私人微调版,即便补上 mmproj 重建,得到的仍是微调版的视觉能力,**不具备通用能力**,不满足实验基线,不再提供该路径。

### 4.2 QLH 配置(环境变量)

```bat
set QLH_EXTERNAL_ENABLED=true
set QLH_EXTERNAL_BASE_URL=http://127.0.0.1:11434
set QLH_EXTERNAL_MODEL=gemma4:12b        :: 原版(唯一选择)
set QLH_EXTERNAL_DATA_SCOPE=opt_in       :: 请求仍需 allow_external=true
set QLH_EXTERNAL_REASONING_EFFORT=none   :: 避免 thinking 占满短回复预算
set QLH_EXTERNAL_LABEL=gemma-4-12B(Ollama 本机)
```

对应实现:`src/config.py:211-229`;端点健康检查、故障回退、流式等均已支持(`docs/外部推理服务Provider接入指南.md`)。

### 4.3 图像消息管线（G4.1 Completed）

后端已完成：

1. `api_server.ChatRequest` 与 `inference_service.protocol.ChatRequest` 新增 `image_data_urls`，两侧同一校验口径。
2. `multimodal.py` 只接受内联 PNG/JPEG/WebP，校验 base64、文件签名、数量和大小；不接受远程 URL，避免由推理端代取任意地址。
3. `external_provider` 与 `IslandEngine` 保留 OpenAI `content[]/image_url`，不再把数组 `str()`；`reasoning_effort` 仅在显式配置时发送。
4. 带图请求必须显式授权外部路由；外部被禁、失败或 scope deny 时明确失败，不降级到忽略图片的本地文本推理。
5. 会话标题、追问、内存历史与 SQLite 仅记录文字问题和模型回复；原图 data URL 不持久化。

### 4.4 Web/TUI 图片交互（G4.2 Completed）

- Web 聊天页仅在当前内存队列中读取 PNG/JPEG/WebP，先核验 MIME、magic bytes、单张 8 MiB、总计 16 MiB 与最多 4 张；预览 URL 在移除、发送、切换会话或卸载时撤销，图片 data URL 不写入消息、localStorage、历史或 SQLite。
- 带图发送自动固定 `execution_mode=auto` 并携带 `allow_external=true + prefer_external=true`；不会误入当前不支持图片的 task graph，也不会由前端静默删除图片。外部模型可达而本地模型未加载时，聊天页以实际外部模型名作为可用状态。
- Textual TUI 增加 `/image <path>`、`/images`、`/image-clear`；本地路径读取复用后端格式/大小校验，空输入可发送默认图片描述提示，待发送 data URL 仅存活到本次 SSE 请求创建。
- 浏览器桩契约覆盖桌面/移动选择、预览、删除、请求字段和 localStorage 零 data URL；真实 Edge 会话经过 QLH `external_api` 调用 `gemma4:12b`，测试图返回“这是一张展示戏剧艺术象征——悲剧与喜剧面具的图标。”。

### 4.5 落地步骤

> 注:gemma4 的视觉 token 预算默认约 512(可调 70–1120),低分辨率即可,图片不必预处理成高清。

### 4.4 落地步骤

1. [x] 经 7897 子进程代理获取官方 `gemma4:12b` 并核对模型能力。
2. [x] Ollama 原生文本、图像与 OpenAI `image_url` smoke。
3. [x] QLH 结构化消息、reasoning 配置、数据作用域、禁止丢图回退和零图片持久化。
4. [x] G4.2 Web/TUI 产品交互、桌面/移动契约及真实 Gemma 浏览器会话。
5. [ ] G4.3 8K/16K 长会话资源门，以及原生/音频/分布式候选验证。

---

## 5. 方案 B：原生 llama_cpp 接入（G4.3 Candidate）

原生化仍有长期价值，但当前不能仅凭 vendored llama.cpp 中出现 `LLM_ARCH_GEMMA4` 就宣称 Python 运行时可用。必须同时冻结官方/可信工件 revision、视觉/音频 handler、`llama-cpp-python` 版本、Windows/Linux 打包 ABI 和与 Ollama 基线一致的输出。

### 5.1 文件放置(原版,需下载)

当前不冻结文件名和仓库，也不启动第二份 7GB 级下载。G4.3 先用 resolver 读取候选仓库的固定 revision、许可、文件关系和 SHA/LFS 元数据，再决定单 GGUF、projector/handler 配对或官方 QAT 工件方案。

### 5.2 代码改动点清单

1. [x] G4.3.1 新增 `gemma4-native-probe`：报告项目锁定 revision、当前 `llama-cpp-python` 版本和 MTMD 必需 ABI；绑定导入不能通过产品能力门。
2. [ ] 冻结相容的上游 llama.cpp revision 与 `llama-cpp-python` wheel/source build；当前仅记录二者版本事实，不声称二进制来源相同。
3. [ ] 对候选工件执行固定 revision resolve、许可确认、SHA 校验和隔离预检；缺少或摘要不匹配的 GGUF/mmproj 对 fail-closed。
4. [ ] 只在前项通过后扩展 `llama_engine.py` 的模型/聊天 handler，而不是向 `Llama()` 猜测传入未经当前绑定支持的 `mmproj` 参数。
5. [ ] 与本票相同测试图做 Ollama vs 原生语义对照，再补取消、卸载、8K/16K OOM、CPU/GPU offload 和打包门。
6. [ ] 只有上述门通过后才新增 `model_config`/curated recipe 并允许分布式调度选择。

### 5.3 G4.3.1 原生绑定/ABI 预检（Completed）

统一入口为 `python scripts/model_tools.py gemma4-native-probe --json`。它在独立子进程中只导入 `llama_cpp` 与 `llama_cpp.mtmd_cpp`，检查 `mtmd_init_from_file`、上下文参数、vision/audio 能力查询和释放函数；不加载模型、不访问网络、不输出用户绝对路径。当前开发机的结果为项目锁定 revision `47e1de77…`、`llama-cpp-python 0.3.28`、MTMD ABI 完整，但 `artifact_probe_requested=false`、`gate_passed=false`，这是预期的非能力结论。

后续 G4.3.2 只有在显式提供本地 `--model`、`--mmproj` 及两者 SHA-256 时才会运行。worker 先校验两个摘要，再以 CPU-only `n_gpu_layers=0` 创建 `Llama`，调用 `mtmd_init_from_file` 和 `mtmd_support_vision/audio`；成功状态仅为 `ready_for_image_smoke`，仍不等于图片语义、取消、长上下文、音频或打包已验收。

---

## 6. 方案对比与推荐

| 维度 | 方案 A(Ollama Provider) | 方案 B(原生 llama_cpp) |
|---|---|---|
| 模型获取 | 官方 `gemma4:12b` 已下载并验收 | 来源/revision/文件关系尚未冻结 |
| 当前状态 | **G4.1/G4.2 Completed** | **G4.3 Candidate** |
| 代码范围 | 通用结构化 OpenAI 消息 + 显式 reasoning 配置 | Python handler、ABI、工件、打包和调度均需验证 |
| 模型能力 | Text/Image 已由 QLH 实测；Audio 尚未开放 | 未实测，不作能力承诺 |
| 与分布式调度 | `external_api` 整请求，统计如实标注 | 通过 G4.3 后才可作为 `llama_cpp` 候选 |
| 回滚 | 关闭 `QLH_EXTERNAL_ENABLED` 即回滚 | 需版本与工件级回滚 |

**当前推荐：路线 A 已是产品基线。** 它已经使用官方原版工件完成 Web、TUI 与实机图像调用，不再只是临时 PoC。路线 B 不因“原生化”自动获得优先级，只有在安装体积、时延、分布式调度或离线交付上给出明确收益且通过 ABI/工件门后再实施。

### 6.1 分期票

| 子票 | 状态 | 范围 | 完成判据 |
|---|---|---|---|
| G4.1 后端与官方模型门 | Completed（2026-08-11） | 官方模型下载、文本/图像 smoke、结构化消息、reasoning、大小/格式/作用域边界、零图片持久化 | 定向 `160 passed`；QLH Provider 真实图片返回非空描述；全量（测试进程 `NO_PROXY=*`）`1908 passed / 26 skipped / 2 failed`，失败仅为当前 WSL 无法访问 `/g/...` 的 Linux env-register 路径 |
| G4.2 Web/TUI 图片交互 | Completed（2026-08-11） | 选择/预览/删除图片，发送状态，4 图/8MiB/16MiB 提示，外部授权与 TUI 本地路径队列 | Python 定向 `179 passed`；Web 桩契约 `1 passed`；真实 Edge → QLH → `gemma4:12b` `1 passed`；全量（测试进程 `NO_PROXY=*`）`1915 passed / 26 skipped` |
| G4.3.1 原生绑定/ABI 预检 | Completed（2026-08-11） | 独立只读 worker、MTMD 必需 ABI、项目锁定 revision/当前 binding 版本、SHA 强制与路径脱敏契约 | 定向 `56 passed / 2 skipped`；全量 `1934 passed / 13 skipped`；开发机 `llama-cpp-python 0.3.28` 报告 MTMD ABI 完整；无工件时明确 `gate_passed=false` |
| G4.3.2 原生工件与图片 smoke | Candidate | 冻结来源/revision/许可/两个 SHA、GGUF/mmproj 配对、隔离初始化与真实测试图 | `ready_for_image_smoke` 后与 Ollama 图像语义对照一致；未知配对/摘要/初始化均 fail-closed |
| G4.3.3 音频、资源、打包与分布式 | Candidate | 音频契约、8K/16K、取消/卸载/OOM、CPU/GPU offload、Windows/Linux 离线包、调度 | 所有门通过后才可新增 recipe 或让调度选中原生路径 |

---

## 7. 多模态实验补全验证清单

- [x] 模型基线为官方 `gemma4:12b`，Capabilities 含 vision/audio；未使用 `myheretic`
- [x] Ollama OpenAI 兼容端点接受 base64 `image_url`；默认 thinking 的空正文风险已有实测
- [x] QLH Provider 携带 `reasoning_effort=none` 后，带图请求返回非空合理描述
- [x] 远程 URL/SVG/MIME 伪装/超限/未授权/丢图回退均有拒绝测试
- [x] 图片不进入会话历史与 SQLite 持久化参数
- [x] G4.2 Web/TUI 图片会话与桌面/移动浏览器验收
- [x] G4.3.1 原生 `llama-cpp-python` MTMD ABI 隔离预检；无冻结工件时拒绝进入原生能力门
- [ ] 上下文限制生效:8K 上下文下连续多轮对话不 OOM
- [x] 2K/4K smoke 的加载、图像 prompt、资源分配基线已记录；8K/16K 留 G4.3

**实验矩阵补全后形态**(多模态实验 = 三模态闭环):

```
文本推理(qwen / deepseek)  +  图像生成(SD 1.5)  +  图像理解(gemma-4-12B)  ✅ 闭环
```

---

## 8. 风险与注意事项

1. **8GB 显存是硬约束**：当前实测 56% CPU / 44% GPU；8K/16K 未过门前不要提高默认上下文，更不能启用 256K。
2. **`gemma4:latest` 不是 12B**：必须显式使用 `gemma4:12b`。
3. **`myheretic` 是私人微调版，不是原版**：它不能作为项目多模态验收基线；G4.1 只使用官方 `gemma4:12b`，G4.3 的直接工件需重新冻结来源。
4. **Ollama keep_alive**:默认模型驻留 5 分钟后卸载,连续实验建议 `OLLAMA_KEEP_ALIVE=-1` 或调用时传 `keep_alive` 参数,避免反复加载。
5. **端口占用**:Ollama 占 11434;若 QLH 后端或其他服务也用此端口需调整(`OLLAMA_HOST`)。
6. **外部 Provider 数据门控**:`QLH_EXTERNAL_DATA_SCOPE` 默认 `opt_in`;本机 Ollama 场景可 `allow_all`,但若日后指向远端实例务必收紧。
7. **reasoning 默认值**：Ollama OpenAI 端点默认 thinking 时，小 `max_tokens` 可能只有 reasoning、正文为空；Gemma 4 本机配置固定 `QLH_EXTERNAL_REASONING_EFFORT=none`，其他 Provider 留空不发送。
8. **路线 B 未冻结**：不得把 vendored 源码符号当成 Python/打包支持，也不得在未固定 revision、许可和 SHA 前新增一键 recipe。

---

## 9. 参考

- Google Gemma 模型总览：https://ai.google.dev/gemma/docs
- Google Gemma 4 模型说明：https://ai.google.dev/gemma/docs/core
- Google Gemma 发布记录：https://ai.google.dev/gemma/docs/releases
- Ollama 官方库 `gemma4:12b`：https://ollama.com/library/gemma4
- 模型基线约定:`myheretic:latest` 为私人微调版,不具备通用能力,仅作架构可行性参考,不作为实验模型
- 项目 vendored llama.cpp 仅作为 G4.3 候选证据，不代表当前 Python 运行时已支持
- 外部 Provider 接入:`docs/外部推理服务Provider接入指南.md`
- 多模态远期规划:`docs/一键模型部署与自治集群远期计划.md`(:309,:340,:958-963)
