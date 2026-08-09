# Gemma-4-12B 多模态支持方案(调研 + 接入设计)

> 目标:为 QLH 项目添加 **gemma-4-12B** 支持,补全多模态实验缺少的**图像理解(图生文)**能力。
> 本机 OLLAMA 部署的 `myheretic:latest` 为**用户私人微调版(非原版)**,不具备通用能力,不能作为实验基线;
> 本文以**原版模型**为准设计两条接入路线,**推荐路线 B(下载原版 GGUF 原生接入)**。

---

## 1. 背景与目标

- 模型:**Google Gemma 4 12B**(2026 年发布,`gemma4` 架构,11.9B dense 参数,256K 上下文,**Text + Image 多模态**,无音频输入)。
- 本机显卡:RTX 4060 Laptop,**8GB 显存**。
- 本机现状:Ollama 已导入 `myheretic:latest`(7.4GB,实测可运行),**但它是用户私人微调版(基于 gemma-4-12B 微调,非官方原版权重),不具备通用能力**,仅证明 gemma4 架构在本机 8GB 显存可运行;且缺视觉 projector,capabilities 只有 completion/tools/thinking,无 vision。
- 模型基线:**多模态实验必须使用原版 gemma-4-12B**(Ollama 官方 `gemma4:12b` 或 bartowski 原版 GGUF),私人微调版不满足通用能力要求。
- 目标:**让 QLH 能调用 gemma-4-12B 的多模态(图像理解)能力**,并把它登记为项目多模态实验模型。

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
| 模型名 | `gemma-4-12B`(Ollama 官方库 `gemma4:12b`;GGUF 由 **bartowski/gemma-4-12B-it-GGUF** 提供,ggml-org 官方仓库无 12B) |
| 架构/参数量 | `gemma4` 架构,11.9B,dense(非 MoE);词表 262K;Gemma 3 12B 的直接继任者 |
| 模态 | **Text + Image**(视觉 projector 独立文件,clip 架构 52.4M 参数 / 175MB BF16);音频仅 E2B/E4B 支持 |
| 上下文 | 256K(设计值;8GB 显存下必须大幅限制,见 2.3) |
| 量化 | Q4_K_M 权重 **7.4GB** + mmproj **175MB** ≈ 7.6GB |
| 支持状态 | **llama.cpp 完整支持**(本项目 vendored 子模块 `android/app/src/main/cpp/llama.cpp` 已含:`LLM_ARCH_GEMMA4`、visual/audio projector 类型 gemma4v/gemma4a、conversion 脚本、聊天模板) |
| Ollama 名称 | `gemma4:12b`(⚠️ `gemma4:latest` 是 9.6GB 的 E4B 档,不是 12B) |

### 2.2 本机部署现状(已实测)

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
- Ollama 服务在运行,OpenAI 兼容端点可用(已实测 `GET http://localhost:11434/v1/models` 返回模型列表)。

### 2.3 硬件可行性(RTX 4060 8GB)

| 约束 | 结论 |
|---|---|
| 显存 | 权重 7.4GB + projector + KV cache + 计算缓冲 > 8GB,**必然 CPU offload**(Ollama 默认自动分配,能塞多少层进 GPU 塞多少) |
| 上下文 | 256K 在 8GB 下不可行;**必须限制**,建议 8K–16K(`OLLAMA_CONTEXT_LENGTH` 或 `num_ctx`) |
| 速度 | 估算 8–15 tok/s(部分 offload 后),可交互但非流畅;纯 GPU 理论 15–25 tok/s(无实测) |
| 结论 | **可运行,定位为"实验/验证型"模型**,不适合高吞吐生产路径 |

---

## 3. 项目接入点现状(改动基线)

| 层 | 现状 | 新增 gemma-4-12B 需要的改动 |
|---|---|---|
| 模型注册 | `src/model_config.py:85` `BUILTIN_MODELS` 8 槽全文本 | 加 `ModelConfig(model_type="gguf", ...)`(或 DB 注册) |
| llama_cpp 引擎 | `src/llama_engine.py` 的 `Llama(model_path=...)` **无 mmproj 参数** | 加 mmproj 加载 + 图像预处理(仅路线 B 需要) |
| 外部 Provider(引擎) | `src/external_provider.py` 完整可用,`QLH_EXTERNAL_*` 配置(`src/config.py:211-229`),OpenAI 兼容消息构造 `_build_stage_messages`(:742) | **消息为纯文本,无 image 字段** → 需透传图像消息 |
| chat 契约 | `src/api_server.py` / `src/inference_service` 请求模型无 image 字段 | 需支持 `content` 数组 / `image_url`(OpenAI 风格) |
| control 同步 | `control/src/data/model-registry-store.ts:29-39` `BUILTIN_MODEL_IDS` 与 8 槽对齐 | 加模型 id(内置时) |
| Ollama 集成 | 项目自身**无** Ollama 调用代码(仅远期对标,`docs/一键模型部署与自治集群远期计划.md:958-963`) | 方案 A 通过通用 OpenAI 兼容端点接入,无需 Ollama 专属代码 |

---

## 4. 方案 A:Ollama 外部 Provider 接入(快速验证,改动最小)

**思路**:本机 Ollama 已运行且 OpenAI 兼容端点可用(`http://localhost:11434/v1`),QLH 的外部 Provider(`external_api` 引擎,即外部 Provider 接入指南中的路线 B——注意与本文方案 B 命名无关)正好对接——**零新引擎**,只需补齐"带 projector 的原版模型"与"图像消息透传"两处。

> 注意:本方案的"零模型文件下载"优势**已不成立**——现有 `myheretic` 是私人微调版,不能当原版用,**必须先拉取原版 `gemma4:12b`**。因此方案 A 定位为**快速验证通道**,最终落地以路线 B 为准(见第 6 节)。

### 4.1 前置:获取带 vision 的**原版** gemma-4-12B

唯一途径(原版,自带 175MB projector):

- **A1**:拉官方多模态 tag:
  ```bat
  ollama pull gemma4:12b
  ollama show gemma4:12b   :: Capabilities 应含 vision
  ```

> ~~A2(复用现有 myheretic 重建)~~ **已废弃**:myheretic 是私人微调版,即便补上 mmproj 重建,得到的仍是微调版的视觉能力,**不具备通用能力**,不满足实验基线,不再提供该路径。

### 4.2 QLH 配置(环境变量)

```bat
set QLH_EXTERNAL_ENABLED=true
set QLH_EXTERNAL_BASE_URL=http://localhost:11434/v1
set QLH_EXTERNAL_MODEL=gemma4:12b        :: 原版(唯一选择)
set QLH_EXTERNAL_DATA_SCOPE=allow_all    :: 本机 Ollama,数据不出本机;按需收紧为 opt_in
set QLH_EXTERNAL_LABEL=gemma-4-12B(Ollama 本机)
```

对应实现:`src/config.py:211-229`;端点健康检查、故障回退、流式等均已支持(`docs/外部推理服务Provider接入指南.md`)。

### 4.3 图像消息管线改造(多模态的关键工作)

QLH 全链路目前无图像消息。需三处小改:

1. **chat 请求契约**:允许消息 `content` 为 `str | list[{type:"text"|"image_url", ...}]`(OpenAI 风格),新增可选 `images: [base64]` 字段透传(api_server 与 inference_service 双侧)。
2. **external_provider 消息构造**(`src/external_provider.py:742 _build_stage_messages`):把上游图像字段转换为 OpenAI `image_url`(`data:image/...;base64,...`)并入 messages。
3. **前端/TUI**(可选):聊天输入附加图片按钮(纯 API 验证阶段可跳过)。

> 注:gemma4 的视觉 token 预算默认约 512(可调 70–1120),低分辨率即可,图片不必预处理成高清。

### 4.4 落地步骤

1. 按 4.1 获取带 vision 模型,`ollama run gemma4:12b "描述这张图片: <图片路径>"` 先人工验证。
2. 按 4.2 配置环境变量,重启后端。
3. 按 4.3 实现图像消息透传(预估 1–2 个文件,几十行)。
4. 按第 7 节验证清单跑通。

---

## 5. 方案 B:原生 llama_cpp 接入(推荐,原生化)

**思路**:不走 Ollama,从 HF 下载**原版** GGUF + mmproj 直接放进 `models/`,扩展 llama_cpp 引擎支持多模态。原版权重保证通用能力,完全内聚于项目、可纳入分布式调度,是本次多模态实验的**最终落地路线**。

### 5.1 文件放置(原版,需下载)

```
models/
  gemma-4-12b-it-Q4_K_M.gguf            (7.4GB, bartowski/gemma-4-12B-it-GGUF 原版)
  mmproj-gemma-4-12B-it-Q4_K_M.gguf     (175MB, 同仓库原版)
```

> 约 7.6GB,按项目红线在**家用宽带**下载;下载后可用现有 sha256 校验管线核对(原版仓库发布页有 checksum)。

### 5.2 代码改动点清单

1. **注册**:`src/model_config.py:85` `BUILTIN_MODELS` 加 `ModelConfig(model_type="gguf", gguf_path="models/gemma-4-12b-it-Q4_K_M.gguf", quant_types=["Q4_K_M"], is_experimental=True)`;同步 `control/src/data/model-registry-store.ts:29` `BUILTIN_MODEL_IDS`。
2. **引擎 mmproj**:`src/llama_engine.py` 加载处加 `mmproj=...` 参数(llama-cpp-python 支持),并新增图像→`content` 数组的调用封装;`src/inference_service/engine_host.py:3054` 的自动加载逻辑同步。
3. **GGUF 自动发现**(`src/llama_engine.py:455-503`)会优先挑 Q4_K_M,gemma 文件可直接被现有逻辑发现,但多模态模型必须显式配对 mmproj(按文件名前缀匹配 `mmproj-*`)。
4. **curated-catalog(可选)**:`control/src/data/curated-catalog.ts:20` 加 recipe(HF 仓库 + 固定 revision + `allow_patterns` 限定上述两文件)。
5. **图像预处理**:gemma4 视觉编码在 llama.cpp 侧完成(clip),引擎侧只需传入原始图像字节,工作量主要在消息契约(同 4.3 第 1 点)。
6. **硬件约束**:llama-cpp-python 侧需 `n_gpu_layers` 调低(offload 策略)、`n_ctx` 限制 8K–16K,与 2.3 一致。

---

## 6. 方案对比与推荐

| 维度 | 方案 A(Ollama Provider) | 方案 B(原生 llama_cpp) |
|---|---|---|
| 模型获取 | 需先拉原版 `gemma4:12b`(7.4GB+175MB) | HF 下载原版 7.6GB + 手动放置 |
| 代码改动 | 图像消息透传(1–2 文件) | 引擎 mmproj + 注册 + 发现 + catalog(4–6 处) |
| 落地时间 | 小时级(快速验证) | 1–2 天(正式落地) |
| 模型通用性 | 原版 `gemma4:12b`,✅ 通用 | 原版 bartowski GGUF,✅ 通用 |
| 与分布式调度 | 走 `external_api` 引擎,统计如实标注 | 走 `llama_cpp` 引擎,可纳入任务图 |
| 回滚 | 关环境变量即回滚 | 需改代码 |
| 适用阶段 | **快速验证多模态能力** | **正式补全多模态实验(推荐)** |

**推荐:走路线 B(下载原版模型,原生接入)**。理由:现有 `myheretic` 是私人微调版、不具备通用能力,方案 A 的"复用现有模型"优势已不存在;路线 B 用原版权重保证实验结论可信,且模型内聚于项目、可纳入分布式调度。路线 A 可作**快速验证通道**提前确认多模态管线(需先 `ollama pull gemma4:12b` 原版),验证结论写入 `docs/一键模型部署与自治集群远期计划.md` 的多模态条目后,再实施路线 B。

---

## 7. 多模态实验补全验证清单

- [ ] **模型基线为原版**:`ollama show gemma4:12b`(或 models/ 下 bartowski 原版 GGUF)的 Capabilities 含 **vision**;全程不使用私人微调版 `myheretic`
- [ ] `curl http://localhost:11434/v1/chat/completions` 以 `image_url`(base64)发送测试图,返回合理图文描述
- [ ] QLH 配置 `QLH_EXTERNAL_*` 后,`/api/models/available` 与前端健康检查显示 `external_api` 引擎可达
- [ ] QLH chat 接口带图调用 gemma-4-12B,输出与 Ollama 直调一致
- [ ] 图像理解用例:给定一张图(如 fixtures 中任一样例图),能回答图中物体/场景;中文提问可用
- [ ] 上下文限制生效:8K 上下文下连续多轮对话不 OOM
- [ ] 性能基线记录:首 token 延迟、tok/s(8GB 显存 offload 预期 8–15 tok/s),写入实验记录

**实验矩阵补全后形态**(多模态实验 = 三模态闭环):

```
文本推理(qwen / deepseek)  +  图像生成(SD 1.5)  +  图像理解(gemma-4-12B)  ✅ 闭环
```

---

## 8. 风险与注意事项

1. **8GB 显存是硬约束**:权重 7.4GB + projector 后几乎满载,ollama 默认行为可能 OOM,务必限制 `OLLAMA_CONTEXT_LENGTH`(建议 8192–16384);OOM 时降低 `num_ctx` 或调低 `OLLAMA_NUM_GPU` 让更多层跑 CPU。
2. **`gemma4:latest` ≠ 12B**:官方 latest tag 是 E4B(9.6GB),拉取时必须显式 `gemma4:12b`。
3. **`myheretic` 是私人微调版,不是原版**:它是用户私人微调模型,行为与官方原版**不一定一致、不具备通用能力**,不能作为多模态实验的验证基线;多模态实验必须用原版(`ollama pull gemma4:12b` 或 bartowski 原版 GGUF),不要基于 myheretic 重建。
4. **Ollama keep_alive**:默认模型驻留 5 分钟后卸载,连续实验建议 `OLLAMA_KEEP_ALIVE=-1` 或调用时传 `keep_alive` 参数,避免反复加载。
5. **端口占用**:Ollama 占 11434;若 QLH 后端或其他服务也用此端口需调整(`OLLAMA_HOST`)。
6. **外部 Provider 数据门控**:`QLH_EXTERNAL_DATA_SCOPE` 默认 `opt_in`;本机 Ollama 场景可 `allow_all`,但若日后指向远端实例务必收紧。
7. **路线 B 依赖 llama-cpp-python 版本**:确认其绑定版本支持 gemma4 架构与 mmproj(项目 vendored llama.cpp 已支持,Python 绑定需同步升级)。
8. **unsloth 仓库未经验证**:12B GGUF 以 **bartowski/gemma-4-12B-it-GGUF** 为准,不要引用未验证的镜像仓库。

---

## 9. 参考

- Ollama 官方库 `gemma4:12b`(原版):https://ollama.com/library/gemma4
- bartowski/gemma-4-12B-it-GGUF(原版,HF,Q4_K_M + mmproj,带 checksum)
- 模型基线约定:`myheretic:latest` 为私人微调版,不具备通用能力,仅作架构可行性参考,不作为实验模型
- 项目 vendored llama.cpp 多模态文档:`android/app/src/main/cpp/llama.cpp/docs/multimodal.md`、`docs/multimodal/gemma3.md`(gemma4 支持已并入)
- 外部 Provider 接入:`docs/外部推理服务Provider接入指南.md`
- 多模态远期规划:`docs/一键模型部署与自治集群远期计划.md`(:309,:340,:958-963)
