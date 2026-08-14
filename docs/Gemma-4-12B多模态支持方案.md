# Gemma-4-12B 多模态支持方案（实施计划）

> 状态：Active（G4.1 后端与官方模型实机门完成；G4.2 前端/TUI 图片交互完成；G4.3.1 原生绑定/ABI 预检完成；G4.3.2A 官方工件身份/资源准入完成；**G4.3.2B 原生初始化与图片语义完成（2026-08-13，后续开发缺口见 §5.6）**；EX-N3-GEMMA-S1 静态质量计数桥接完成；2.4 已登记 PyTorch 工件候选与下载治理 S0）
>
> 更新日期：2026-08-13
> 适用范围：原版 Gemma 4 12B 经本机 Ollama `external_api` 提供文本与图像理解；G4.3.1/2A 已完成原生候选的 ABI 与资产身份门，音频和原生实际推理另列后续票
>
> 目标：先提供可验证、可回滚的原版 Gemma 4 12B 图像理解能力，再补产品交互；`myheretic:latest` 仍只作为用户私有资产，不进入项目基线。

---

## 1. 背景与目标

- 模型：**Google Gemma 4 12B Unified**（2026-06-03 发布，`gemma4` 架构，11.9B，256K 上下文，官方能力为 Text + Image + Audio 输入、Text 输出）。
- 本机显卡:RTX 4060 Laptop,**8GB 显存**。
- 本机现状：官方 Ollama `gemma4:12b` 已通过 `127.0.0.1:7897` 子进程代理完整拉取并验证；用户私有 `myheretic:latest` 保留但不参与验收。
- 模型基线：当前唯一已验收工件为 Ollama 官方 `gemma4:12b`；直接 GGUF 来源、revision 和 Python 绑定 ABI 尚未冻结，不作为 G4.1 依赖。
- 当前目标：QLH 已具备可用的 Gemma 4 图像理解产品路径；G4.3.1 已完成只读的原生绑定/ABI 预检，G4.3.2A 已冻结本机官方 Ollama 工件身份。G4.3.2B 仅在可用 RAM 达到资源门后验证原生图片语义，音频、分布式和 8K/16K 继续后置。

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
| 原生支持状态 | 项目锁定 llama.cpp `47e1de77…`；**G4.3.2B 原生图片语义已通过（2026-08-13）**：以 47e1de77 源码重编 llama-cpp-python 0.3.28（独立 venv `.venv-gemma4-native`）+ 补全 MTMD 图像管线绑定（`scripts/model_tools/llama-cpp-python-mtmd.patch`，含 mtmd_tokenize/batch 系列/wrapper 结构修复），`scripts/gemma4_native_image_smoke.py` 实测固定测试图（sd-001：red apple on wooden table）输出语义正确描述（two red apples / weathered wooden surface）。**已知限制（2026-08-13 实测）**：原生路径无 `reasoning_effort=none` 等效控制，gemma4 思考段长度不可控（96–234+ tokens 波动，CPU 推理下可能占满短输出预算）；验证脚本已实现 `<channel|>`(token 101) 正文剥离（遇不到时输出警告）。**产品图像路径仍推荐 Ollama `external_api`（reasoning_effort=none，G4.1/4.2 已验收）**；原生绑定定位为 ABI/语义验证与未来增强。音频/8K-16K/offload 仍后置 |
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

### 2.4 PyTorch 工件候选登记与下载治理（2026-08-13）

> **定位**：以下为**未来 PyTorch 候选登记**，本机（8GB 显存）**绝不允许下载后直接准入**；本机当前唯一已验收工件仍为 Ollama 官方 `gemma4:12b`（Q4_K_M，G4.1）。

**工件身份**（HF API 实查，经 7897 代理）：

| 项 | 值 |
|---|---|
| 仓库 | `google/gemma-4-12B`（Hugging Face 官方） |
| gated | `False`（无需同意即可下载） |
| 许可证 | `apache-2.0` |
| repo SHA | `023679ed352de9bb66cc873c9009ce3482585c08` |
| model.safetensors | 单文件，**23,919,549,408 bytes（≈23.9 GB）**（HF `X-Linked-Size` 实测） |
| 架构 | `gemma4_unified`（Transformers 需支持该架构的版本） |
| 文件集 | 8 个：model.safetensors / config.json / generation_config.json / processor_config.json / tokenizer.json / tokenizer_config.json / README.md / .gitattributes |

**准入红线（本机 8GB 显存）**：fp16 12B 权重加载需 **≈24GB VRAM**，本机显存门不通过，禁止本机准入；PyTorch 候选仅面向**大显存/多卡/分布式/微调**场景。

**侧车定位（transformers 版本不兼容为预期设计）**：本机现有 transformers 4.47.1（主 venv 与 CUDA venv）均不识别 `gemma4_unified` 架构（实测 `AutoConfig.from_pretrained` 报"does not recognize this architecture"）。**不升级/污染既有 venv**——PyTorch 工件按 SD 1.5 侧车模式独立成 venv（如 `.venv-packaging-gemma4-pt`），独立安装支持 `gemma4_unified` 的新版 transformers，主解释器与 LLM 推理环境不受影响。

**容量预算**（本机实测：磁盘总 166GB / 可用 59GB）：

| 形态 | 磁盘占用 | VRAM 需求 | 准入条件 |
|---|---|---|---|
| 下载缓存（LFS 分片） | ~23.9 GB | — | 磁盘可用 ≥ 50GB（本机 59GB 临界，建议外部机） |
| 权重落盘（fp16 safetensors） | ~23.9 GB | — | 同上 |
| fp16 加载推理 | — | ~24 GB | 外部大显存机（≥ 24GB VRAM） |
| Q4 量化转换产物 | ~7.5 GB | — | 与 Ollama 工件同量级，仅对照用 |

**下载治理**：
- 网络：官网直连不通，一律走 `127.0.0.1:7897` 代理（与 gemma4 Ollama 拉取同一路径）
- 前置门：磁盘容量预算检查（下载 + 落盘 ≥ 48GB 空闲）+ 显存准入（≥ 24GB VRAM 或已规划量化目标）
- 校验：下载后对 model.safetensors 全量 SHA-256 校验（HF LFS 提供 xet/lfs hash；旁车先于文件落盘）
- 断点：huggingface_hub 分片/续传下载（有界超时与重试）

**分期计划**：

| 阶段 | 内容 | 准入条件 | 生命周期 |
|---|---|---|---|
| S0（本票） | 工件登记 + 下载治理 + 容量预算 | 无（只读调研） | `规划` |
| S1 | 外部大显存机下载 + 校验 + fp16 加载冒烟 + Transformers 官方加载路径验证 | ≥ 24GB VRAM 设备 / 磁盘 ≥ 50GB | `规划` |
| S2 | PyTorch 引擎接入：注册表条目、模型配置、CUDA 加载路径、分布管线候选 | S1 通过 + 侧车 venv 建好 | `规划` |
| S3 | 与 Ollama Q4 路径对照实验（同模型两种工件形态：质量/速度/内存对比） | S2 通过 | `规划` |

> 本登记不改变 G4.1/G4.2 已验收工件基线；在 S1 完成前，PyTorch 候选不进入任何实验、注册表或质量门。

---

## 3. 项目接入点现状(改动基线)

| 层 | 现状 | 新增 gemma-4-12B 需要的改动 |
|---|---|---|
| 模型发现 | `external_api` 通过 `/v1/models` 发现 `gemma4:12b`，不伪装成本地 GGUF 注册项 | G4.1 Completed；状态继续标记 `engine=external_api` |
| 外部 Provider | `IslandEngine` 保持 OpenAI 兼容传输，支持结构化 `content` 和可选 `reasoning_effort` | G4.1 Completed；未添加 Ollama 私有客户端 |
| chat 契约 | `image_data_urls` 最多 4 张，只接受 PNG/JPEG/WebP data URL；单张 8MiB、总计 16MiB | G4.1 Completed；远程 URL、SVG、MIME 伪装和无显式授权均拒绝 |
| 安全/持久化 | 图片必须 `allow_external=true + prefer_external=true`；scope deny/端点失败均禁止丢图回退 | G4.1 Completed；历史和 SQLite 只保存文字问题/回复，不保存原图 |
| 前端/TUI | Web 当前内存图片选择/预览/删除/大小提示，TUI 本地路径队列；二者均仅在本次请求发送 data URL | G4.2 Completed；桌面/移动与真实 Edge 会话已验收 |
| 原生 llama_cpp | `gemma4-native-assets` 已冻结官方 Ollama 双工件；`gemma4-native-ollama-probe` 先做完整 SHA 再执行资源门 | G4.3.1/2A Completed；当前 `resource_rejected`，G4.3.2B 仅在 RAM 条件满足后初始化 handler |

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

### 5.1 文件放置（原版，复用本地官方工件）

G4.3.2A 已冻结本机已有官方 Ollama `registry.ollama.ai/library/gemma4:12b`：manifest SHA-256 为 `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`，主 GGUF blob SHA-256 为 `1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606`（7,381,382,048 bytes），mmproj blob SHA-256 为 `675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842`（175,115,584 bytes），许可为 Apache-2.0。锁定文件 `scripts/model_tools/gemma4_native.lock.json` 同时约束 manifest、layer、`from=mmproj-gemma-4-12B-it-bf16.gguf` 和 GGUF 元数据配对；不复制、不移动，也不重新下载第二份 7GB 级模型。

### 5.2 代码改动点清单

1. [x] G4.3.1 新增 `gemma4-native-probe`：报告项目锁定 revision、当前 `llama-cpp-python` 版本和 MTMD 必需 ABI；绑定导入不能通过产品能力门。
2. [ ] 冻结相容的上游 llama.cpp revision 与 `llama-cpp-python` wheel/source build；当前仅记录二者版本事实，不声称二进制来源相同。
3. [x] G4.3.2A 对本机官方候选执行固定 manifest/许可/双工件 SHA 与元数据配对校验；缺少、摘要不匹配或配对不符的 GGUF/mmproj 均 fail-closed。
4. [ ] 只在前项通过后扩展 `llama_engine.py` 的模型/聊天 handler，而不是向 `Llama()` 猜测传入未经当前绑定支持的 `mmproj` 参数。
5. [ ] 与本票相同测试图做 Ollama vs 原生语义对照，再补取消、卸载、8K/16K OOM、CPU/GPU offload 和打包门。
6. [ ] 只有上述门通过后才新增 `model_config`/curated recipe 并允许分布式调度选择。

### 5.3 G4.3.1 原生绑定/ABI 预检（Completed）

统一入口为 `python scripts/model_tools.py gemma4-native-probe --json`。它在独立子进程中只导入 `llama_cpp` 与 `llama_cpp.mtmd_cpp`，检查 `mtmd_init_from_file`、上下文参数、vision/audio 能力查询和释放函数；不加载模型、不访问网络、不输出用户绝对路径。当前开发机的结果为项目锁定 revision `47e1de77…`、`llama-cpp-python 0.3.28`、MTMD ABI 完整，但 `artifact_probe_requested=false`、`gate_passed=false`，这是预期的非能力结论。

后续 G4.3.2 只有在显式提供本地 `--model`、`--mmproj` 及两者 SHA-256 时才会运行。worker 先校验两个摘要，再以 CPU-only `n_gpu_layers=0` 创建 `Llama`，调用 `mtmd_init_from_file` 和 `mtmd_support_vision/audio`；成功状态仅为 `ready_for_image_smoke`，仍不等于图片语义、取消、长上下文、音频或打包已验收。

### 5.4 G4.3.2A 官方工件身份与资源准入（Completed）

统一只读入口为 `python scripts/model_tools.py gemma4-native-assets --full-hash` 和 `python scripts/model_tools.py gemma4-native-ollama-probe --json`。前者完整哈希本机官方 manifest、主 GGUF 和 mmproj，并验证 `gemma4`/`mmproj`、Gemma 4 12B 基模型、262144 context、视觉 encoder 与 `gemma4uv` projector；报告不输出用户绝对路径。后者仅在上述资产门通过后才调用独立 worker。

原生 worker 在重新哈希或导入/初始化 `Llama` 前先检查文件存在、大小和可用物理 RAM。当前主 GGUF 加 projector 的保守门为约 8.74GiB（配对工件的 110% 加 1GiB OS headroom，且不低于 2GiB），两次实测可用 RAM 约 5-6GiB，因此返回 `resource_rejected / insufficient_ram`，不触发文件全量复哈希、MTMD 初始化、handler 创建或图片处理。这是资源保护门，不是原生图像能力失败结论。

G4.3.2B 已于 2026-08-13 完成（原生图片语义通过，见 §2.1「原生支持状态」）：CPU-only `n_ctx=128` 初始化 + 固定测试图（sd-001）语义对照通过，`<channel|>` 正文剥离生效，资源门（`required_free_ram_bytes`）fail-closed 实测有效（不足时返回 `resource_rejected / insufficient_ram`，不触发文件复哈希、MTMD 初始化或图片处理；两次实测可用 RAM 约 5-6GiB 均被正确拒绝）。不得以增加虚拟内存、隐式换页或降低身份校验来绕过此门。

**后续开发缺口（逐项验收标准见 §5.6）**：思考段控制、音频输入、8K/16K 上下文、GPU offload、产品接线、打包、卸载/取消——均未开始，按 §5.6 顺序推进。

### 5.6 G4.3.2B 后续开发缺口清单（2026-08-14 整理，供后续开发）

> G4.3.2B 本身已闭环（图像语义验收完成）。以下为**后续票**清单，按依赖顺序排列；每项列出缺口现状、验收标准与验证方式。产品路径在全部完成前仍以 Ollama `external_api` 为准（§6）。

| # | 缺口 | 现状（2026-08-13 实测） | 验收标准 | 验证方式 |
|---|---|---|---|---|
| 1 | **思考段控制**（`reasoning_effort=none` 等效） | **Completed（2026-08-14）**：屏蔽思考开始 token(100) 无效（模型改用文本形式思考）；**剥离式有效**——等思考段结束 token(101) 后收集正文，默认预算 512（`gemma4_native_image_smoke.py` 默认 `--max-tokens 512 --n-ctx 768`，思考段 96-234 tokens 波动被剥离，正文完整）。实测固定图（sd-001）3/3 次正文完整稳定、无思考段残留、语义与 Ollama `think=false` 一致 | 固定测试图多次运行正文完整（3/3 实测通过）；短预算 256 下思考段仍可能吃满（文档标注：原生路径正文预算需 ≥512，产品路径仍推荐 Ollama） |
| 2 | **产品接线**（原生路径入 chat 接口） | llama_engine 已有 MTMD 能力注册与 handler 骨架（`bf9cdc4`），但图像输入未接 `describe_image` 式 API | `llama_cpp` 引擎在 QLH chat 接口接受 `image_url`，走原生 MTMD 管线返回描述；资源门拒绝时 fail-closed | 在 api_server 对话接口传图 → 原生引擎返回；无 mmproj/内存不足时返回结构化拒绝 |
| 3 | **音频输入** | 官方能力含 Audio→Text；Ollama 实测 capability 含 audio；原生 MTMD `mtmd_support_audio` 能力查询可用，但音频编码管线未绑定/未实测 | 固定音频样本 → 文本输出；无音频工件时 fail-closed | 扩展现有 MTMD 绑定（参考图像管线补 audio 编码符号）；`gemma4_native_audio_smoke` 脚本 |
| 4 | **8K/16K 上下文** | 当前 CPU 门用 `n_ctx=128` smoke；文档不把 256K 当本机承诺 | 8K/16K 下长上下文对话不崩、KV 缓存内存预算正确 | `n_ctx=8192/16384` 加载 + 长文本生成 + 内存监控（复用 EX-N3 资源门工具） |
| 5 | **GPU offload** | 8GB 显存与 SD/Gemma 并驻留规则未定；未测 `n_gpu_layers` 真实 offload | 显存预算内 offload 部分层，与 SD 侧车不互斥；超预算拒绝 | `n_gpu_layers` 扫描 + `mem_get_info` 预算门 + 与 SD 并驻留实测 |
| 6 | **打包** | 原生绑定（重编 wheel + MTMD 补丁）未进 PyInstaller | 打包产物可加载 gemma4 GGUF + mmproj 完成图像描述 | 集显/独显 spec 加 `.venv-gemma4-native` 依赖与 mmproj 工件，冒烟打包 |
| 7 | **卸载/取消** | 进程生命周期（推理中取消、模型卸载、Ollama 驻留冲突）未验 | 取消不泄漏进程；卸载释放内存；与 Ollama 驻留互斥规则生效 | 取消/卸载矩阵测试 + RAM 观测 |

**优先级建议**：#1（思考段控制）是唯一影响"输出质量"的缺口，且 Ollama 已有参照实现，先做；#2（产品接线）价值最高（原生路径入产品）；#3-#7 视需求排期，音频/长上下文/offload 都依赖真实设备窗口。

### 5.5 EX-N3-GEMMA-S1 静态质量计数桥接（Completed，非模型验收）

`scripts/experiment_gemma_quality_unit.py` 与 `plan-quality-gemma-bridge-fixture-v1` 只验证实验质量链路能安全接收 Gemma 的受限计数证据。记录 schema v2 只保留锁定的 `gemma4:12b`、判题契约 ID/SHA-256、主题命中和关键要素覆盖的已评估/通过计数；prompt、图片、base64、路径、URL、模型输出、判题解释与 reasoning 均被拒绝。该票不导入或调用 Gemma、Ollama、CUDA、网络或图像文件。

fixture 以 `0.8/0.6` 穿过预注册的 `0.70/0.50` advisory 阈值，仅证明 schema、脱敏和汇总通路；它不证明图片语义、Gemma 正确率或阈值合理性。在实际 SD→Gemma 三轮取数和人工复核完成前，框架禁止将 `gemma_judge` 配置为 `quality.required=true`。这条边界与 G4.3.2B 的 RAM 门互不替代。

---

## 6. 方案对比与推荐

| 维度 | 方案 A(Ollama Provider) | 方案 B(原生 llama_cpp) |
|---|---|---|
| 模型获取 | 官方 `gemma4:12b` 已下载并验收 | 本机官方 Ollama 主 GGUF/mmproj 已以 manifest、许可、双 SHA 与元数据关系冻结；不另建下载 recipe |
| 当前状态 | **G4.1/G4.2 Completed** | **G4.3.2A Completed；G4.3.2B Completed（2026-08-13，图像语义已验）；后续缺口见 §5.6** |
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
| G4.3.2A 官方工件身份与资源准入 | Completed（2026-08-11） | 固定官方 manifest/许可、主 GGUF/mmproj 双 SHA 与元数据配对；资源门先于原生初始化 | `gemma4-native-assets --full-hash` 通过；两次预检可用 RAM 约 5-6GiB 小于约 8.74GiB 门，安全返回 `resource_rejected`，未加载模型；定向 `59 passed / 2 skipped`，当前全量 `1942 passed / 8 skipped` |
| EX-N3-GEMMA-S1 静态质量证据桥接 | Completed（2026-08-13） | 模型/判题契约 SHA 绑定、两项计数投影、脱敏拒绝、v2 schema 与报告汇总 | 7 项专项静态回归通过；不运行 Gemma/Ollama/图像，`0.8/0.6` fixture 不构成模型或阈值验收 |
| G4.3.2B 原生初始化与图片 smoke | Completed（2026-08-13） | CPU-only 初始化、固定测试图与 Ollama 图像语义对照 | `gemma4_native_image_smoke.py` 实测 sd-001（red apple on wooden table）→ 语义正确描述；`<channel|>` 剥离生效；资源门 fail-closed 实测有效；产品路径仍推荐 Ollama，后续缺口见 §5.6 |
| G4.3.3 音频、资源、打包与分布式 | Candidate | 音频契约、8K/16K、取消/卸载/OOM、CPU/GPU offload、Windows/Linux 离线包、调度 | 所有门通过后才可新增 recipe 或让调度选中原生路径（逐项验收见 §5.6） |

---

## 7. 多模态实验补全验证清单

- [x] 模型基线为官方 `gemma4:12b`，Capabilities 含 vision/audio；未使用 `myheretic`
- [x] Ollama OpenAI 兼容端点接受 base64 `image_url`；默认 thinking 的空正文风险已有实测
- [x] QLH Provider 携带 `reasoning_effort=none` 后，带图请求返回非空合理描述
- [x] 远程 URL/SVG/MIME 伪装/超限/未授权/丢图回退均有拒绝测试
- [x] 图片不进入会话历史与 SQLite 持久化参数
- [x] G4.2 Web/TUI 图片会话与桌面/移动浏览器验收
- [x] G4.3.1 原生 `llama-cpp-python` MTMD ABI 隔离预检；无冻结工件时拒绝进入原生能力门
- [x] G4.3.2A 官方 Ollama 主 GGUF/mmproj 双 SHA、许可、manifest 与元数据配对；资源不足时原生初始化前拒绝
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
8. **路线 B 仍未成为产品能力**：G4.3.2A 只冻结本机官方工件身份；不得把 vendored 源码符号、资产可读或 ABI 通过当成原生推理/打包支持。资源门不足时不可以虚拟内存、隐式换页或跳过 SHA 作为替代。

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
