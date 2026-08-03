# SD 1.5 引擎与分布式图像生成实施计划

> 文档状态：部分实施（`L4 Candidate`；SD-N1 本地单机基线、8-bit U-Net Linear 量化和 QKV 融合已验证，仍由《总体下一步计划》L4-SD1.5 管理优先级）
>
> 调研日期：2026-07-30
> 更新日期：2026-07-30
>
> 适用范围：Windows CUDA PC 上的 Stable Diffusion 1.5 本地推理、可选 ControlNet 图生图，以及基于现有 TaskGraph/PC Full Worker 的跨 PC 图像批次分布式展示
>
> 总计划入口：[总体下一步计划](总体下一步计划.md)

本文同时记录计划与已验证边界。任何阶段只有完成对应自动化、真实模型、真实 GPU、跨设备和安装包验收后，才能把状态改为“已实现”。当前不得因为单机基线通过而宣称已支持分布式图像生成。

## 0. 2026-07-30 实施状态

| 范围 | 状态 | 已验证证据 |
|---|---|---|
| `SD-N0` 资产识别与固定下载 | 验证中 | 固定 `stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14`；仅下载 SD 1.5 FP16 推理所需文件，完整快照为 2,742,233,847 bytes，Inspector 识别为可加载 `sd15_pipeline`。完整 single-file checkpoint 的离线结构初始化和全部资产 fixture 仍待完成。 |
| `SD-N1` 本地单机文生图 | 基线与优化已验证，阶段未完成 | RTX 4060 Laptop GPU（8,188 MiB）/ 16 GB RAM、`torch 2.5.1+cu121`、`diffusers 0.35.2`、`transformers 4.47.1`：固定 seed `19950101` 起的 10 次连续 512x512/28 steps FP16 baseline 均实际生成成功，耗时 6.909–11.778 s（平均 8.844 s）；另以 8-bit U-Net Linear 量化 + QKV 融合组合完成 512x512/28 steps，耗时 10.030 s，峰值 reserved 显存 3,735,027,712 bytes。CUDA 打包 venv 内的 LLM `model_module` 导入也已通过。`diffusers 0.38.0` 会要求本项目兼容窗口外的 DINOv2 配置，故不作为打包组合。真实取消、资产注册/API、分布式 Worker 和真实 LLM 模型切换回归仍待完成。 |
| single-file checkpoint、LoRA、ControlNet | 未实施 | Inspector 可以拒绝把 ControlNet 当完整模型；实际加载和组合仍未接入。 |
| API/UI、资产登记、图片 blob、TaskGraph Stage、完整 PC Worker、跨 PC fan-out/fan-in | 未实施 | 不得显示为分布式，不得计入分布式任务统计。 |
| Android SD 推理 | 未实施 | Android 仍不承担完整 SD Worker 或层间拆分。 |

本轮运行脚本为 `scripts/smoke_sd15.py`，可复现实测；运行时只接受已下载的本地完整 Diffusers SD 1.5 目录，不会在推理路径访问 Hub。模型下载器和推理侧车仅安装进 CUDA 打包虚拟环境，尚未进入 PC/Android 发布产物。

### 0.1 单机优化能力边界

| 能力 | 状态 | 实现与实测边界 |
|---|---|---|
| U-Net 8-bit 量化 | 已验证 | `bitsandbytes 0.49.2` 只替换 U-Net 的 184 个 `torch.nn.Linear` 为 `Linear8bitLt`，卷积、CLIP、VAE 和 safety checker 仍为 FP16。512x512/28 steps 实机出图成功，耗时 9.629 s，峰值 reserved 显存为 3,409,969,152 bytes。量化 U-Net 为常驻 CUDA 路径，代码禁止与 CPU offload 混用。 |
| Attention QKV 算子融合 | 已验证 | 通过 Diffusers `fuse_qkv_projections(unet=True, vae=False)`。512x512/8 steps 实机出图成功；与 8-bit U-Net 组合后 512x512/28 steps 也成功，单次耗时 10.030 s，峰值 reserved 显存为 3,735,027,712 bytes。同一已加载 pipeline 以三个不同 seed 连续运行 28 steps，峰值 reserved 显存均保持为该值；耗时会受 GPU 背景负载波动，不能据此宣称融合提升吞吐。它与 attention slicing 不能同时启用，QKV profile 保留 VAE slicing。 |
| `torch.compile` / Inductor | 当前不可用，显式拒绝 | 当前 CUDA sidecar 没有可工作的 Triton，首次真实调用会失败。引擎在加载模型前检查并给出短错误，不会等到去噪首步才抛出长异常；不为了该可选优化把未验证的 Triton 依赖加入侧车。 |
| LLM `PagedKVCache` | 不适用，非缺陷 | SD 1.5 U-Net 的每一步输入 latent 都会变化，没有可跨去噪步复用的自回归 KV Cache。不得将文本引擎的 token KV 分页标为 SD 优化；SD 的受支持显存分块手段是 Diffusers attention slicing。 |

这些优化全部位于 `src/diffusion/sd15_engine.py` 的延迟导入侧车中。它们不导入或改写 LLM `ModelManager`、文本 Worker 契约、集显包、Android 依赖和既有全局解释器；SD 仍未注册为 Worker，因此不能显示为分布式推理或计入分布式任务统计。真正进入分布式前仍须完成 SD-N3 的图像 Stage/blob/Worker 协议和资源准入，避免与已加载的 LLM 争用同一块 GPU 显存。

---

## 1. 调研结论

### 1.1 推荐结论

1. 新增独立引擎标识 `diffusers_sd15`，使用 Hugging Face Diffusers 承载 SD 1.5，而不是把扩散模型塞进现有 LLM `ModelManager` 的 Transformer 层接口。
2. 同时支持本地 **Diffusers 多目录格式**和**完整 single-file `.ckpt`/`.safetensors` checkpoint**；模型可以位于电脑上任意可读目录，不要求复制进仓库或安装目录。
3. 加载前必须识别资产角色。文件扩展名只能说明张量容器格式，不能证明它是完整 SD checkpoint。
4. `control_v11e_sd15_ip2p_fp16.safetensors` 对应 SD 1.5 的 ControlNet Instruct-Pix2Pix 辅助网络，不包含完整的 text encoder、U-Net、VAE 和 scheduler，必须与一个完整 SD 1.5 基座或兼容 DreamBooth full fine-tune 及输入图像组合使用。
5. 第一期分布式采用**任务级/批次级数据并行**：多个完整 Worker 使用同一模型、固定 prompt 和不同 seed 并行生成，再由主节点聚合成图片网格。这是实际发生在多个节点上的分布式推理，目标是提高多图吞吐，不虚构为“单张图片延迟加速”。
6. 单张图内部的 U-Net/ControlNet/CFG/patch 拆分保留为后续研究。SD 1.5 每个去噪步都依赖上一步 latent，U-Net 与 ControlNet 在每一步紧耦合；在异构 PC 和 Tailscale/Wi-Fi 上高频同步大概率抵消计算收益。
7. 产品内同时提供：
   - 原版 SD 1.5 preset；
   - 固定的 90 年代日式动漫 DreamBooth preset；
   - 用户自选完整模型；
   - 可选 ControlNet IP2P 编辑 preset。
8. 演示模型不直接塞进主安装包。提供固定 revision 的一键下载或独立模型资产包，随包附模型卡、许可证、SHA-256 和固定参数；已经存在于其他目录的用户模型只登记路径和本机哈希。

### 1.2 为什么不能直接加载用户给出的文件出图

Hugging Face 元数据已经确认：

| 项目 | 结论 |
|---|---|
| 仓库 | `lllyasviel/control_v11e_sd15_ip2p` |
| pipeline tag | `image-to-image` |
| 模型角色 | ControlNet v1.1 Instruct-Pix2Pix condition model |
| 基座 | `runwayml/stable-diffusion-v1-5`，当前应映射到可用的 SD 1.5 本地工件/镜像 |
| 文件 | `diffusion_pytorch_model.fp16.safetensors` 等 ControlNet 权重与 `config.json` |
| 参数规模 | 约 3.61 亿参数，不是完整 SD 1.5 pipeline |
| 额外输入 | 一张输入图像和编辑指令 prompt |

所以用户本地类似 `control_v11e_sd15_ip2p_fp16.safetensors` 的文件应走：

```text
完整 SD 1.5 base / 兼容 full fine-tune
                  +
ControlNet IP2P component
                  +
输入图像 + 编辑 prompt
                  ↓
StableDiffusionControlNetPipeline
```

不能走：

```text
ControlNet safetensors -> StableDiffusionPipeline -> 文生图
```

### 1.3 分布式方案选择

| 方案 | 单图延迟 | 多图吞吐 | 网络同步 | 当前适配性 | 结论 |
|---|---:|---:|---:|---|---|
| 完整模型按 prompt/seed 扇出 | 不降低 | 明显可提升 | 每图仅输入/结果 | 与现有 Provider/DAG 高度一致 | **一期实施** |
| CLIP -> U-Net -> VAE 组件流水线 | 通常不降低 | 有限 | 每图传 embedding/latent | U-Net 仍占绝大多数时间 | 只作容量实验 |
| CFG 两分支并行 | 可能降低 | 中等 | 每个去噪步同步 | 需要低延迟稳定互联 | 暂缓 |
| U-Net block/layer pipeline | 风险高 | 风险高 | 每步多次激活传输 | 现有 LLM 层协议不可复用 | 暂缓 |
| patch/tensor/sequence parallel | 可降低 | 可提升 | 高频 collective | 更适合同机 NVLink/高速以太网与成熟 runtime | 独立调研 |
| Diffusers `device_map` | 可解决单机容量 | 非跨机吞吐方案 | 单机设备传输 | 官方能力，适合单机多 GPU | 可选单机能力 |

Diffusers 官方分布式推理指南把“每个进程持有完整 pipeline、拆分 prompts”作为基础数据并行方法，也提供单机 `device_map` 组件放置。DistriFusion/xDiT 等细粒度方案证明扩散模型内部并行可行，但依赖专门 runtime、GPU collective 和不同硬件假设；不应直接套进当前 HMAC TCP Worker 协议。

---

## 2. 目标与非目标

### 2.1 目标

- 在 CUDA PC 上加载并运行 SD 1.5 文生图。
- 允许用户选择项目外的本地 Diffusers 目录或完整 single-file checkpoint。
- 支持原版 SD 1.5 和一个固定 90 年代日式动漫 DreamBooth 演示 preset。
- 支持固定 seed、scheduler、尺寸、步数、CFG 和 negative prompt，结果可复测。
- 将图片请求纳入现有 Workflow/Stage/Attempt、reservation、lease、fencing、取消和统计体系。
- 至少两台物理 PC 完成多 seed fan-out/fan-in，UI 明确显示每张图片的实际执行节点。
- 在一期稳定后，支持本地 ControlNet IP2P 图像编辑。
- 在相同模型、参数和图片数量下，对比单节点串行与多节点并行 wall time，给出真实吞吐收益。

### 2.2 非目标

- 一期不训练或重新 DreamBooth 微调模型。
- 一期不实现 SDXL、FLUX、视频扩散或 Android 本地扩散推理。
- 一期不把 single image 拆成跨 Tailscale 的 U-Net 层/张量流水线。
- 不把 ControlNet 文件、LoRA 文件或 VAE 文件伪装成完整 SD checkpoint。
- 不通过 Worker Stage JSON 内嵌 base64 传输大图片。
- 不自动扫描全盘寻找模型；用户必须明确选择文件或目录。
- 不把主节点绝对模型路径发送给从节点；每个节点独立映射同一逻辑资产 ID。
- 不因 UI 选择“分布式”就统计为分布式，必须有至少一个远端已完成 attempt。

---

## 3. 模型与文件格式

### 3.1 需要支持的资产类型

| 类型 | 典型形态 | 加载入口 | 能否独立出图 |
|---|---|---|---|
| Diffusers 完整 pipeline | 含 `model_index.json`、`unet/`、`vae/`、`text_encoder/`、`tokenizer/`、`scheduler/` 的目录 | `StableDiffusionPipeline.from_pretrained(local_dir, local_files_only=True)` | 是 |
| 完整 single-file checkpoint | 通常为数 GB 的 `.ckpt` 或 `.safetensors` | `StableDiffusionPipeline.from_single_file(path, config=local_config)` | 是 |
| DreamBooth full fine-tune | 完整 Diffusers 目录或完整 checkpoint | 同上 | 是 |
| LoRA adapter | 通常几十到数百 MB 的 `.safetensors` | base pipeline + `load_lora_weights` | 否 |
| ControlNet component | `config.json` + component 权重，或可识别的 component single-file | `ControlNetModel.from_pretrained/from_single_file` | 否 |
| VAE component | `AutoencoderKL` 权重/目录 | base pipeline 替换 `vae` | 否 |
| textual inversion | embedding 文件 | base pipeline + `load_textual_inversion` | 否 |

### 3.2 资产探测器

新增只读 `DiffusionArtifactInspector`，不靠文件名拍脑袋判断：

1. 目录先读取 `model_index.json`，只允许 `_class_name` 在 SD 1.5 allow-list。
2. single-file 使用 Safetensors header/ckpt state dict key pattern 判断完整 checkpoint、ControlNet、LoRA 或未知组件。
3. `.safetensors` 只读 header，不在探测阶段把所有张量物化进 RAM/VRAM。
4. 对完整 checkpoint 验证 SD 1.x 关键结构和 cross-attention dimension，不只看扩展名。
5. 对 ControlNet 要求对应 `config.json` 或用户选择一个本地 Diffusers config 目录；离线模式不得在加载途中偷偷访问 Hub。
6. 返回结构化结果：

```json
{
  "artifact_kind": "sd15_pipeline|sd15_checkpoint|controlnet|lora|vae|unknown",
  "pipeline_family": "stable_diffusion_1",
  "precision": "fp16|fp32|mixed|unknown",
  "sha256": "...",
  "size_bytes": 0,
  "loadable": true,
  "missing_components": [],
  "warnings": []
}
```

7. `unknown` 必须 fail-closed，UI 展示“这是 Safetensors 容器，但无法确认它是完整 SD 1.5 模型”，而不是尝试加载后 OOM。

### 3.3 本地路径与持久化

- 新增本机配置 `diffusion_models.json`，开发环境放项目忽略文件，安装包放 `%LOCALAPPDATA%/QLH-Edge-Inference/`。
- 配置只保存用户明确选择的路径，不复制权重。
- API 返回逻辑 `artifact_id`、显示名、类型、大小和哈希；默认不向远端返回绝对路径。
- Worker hello 只上报 `artifact_set_id` 与 manifest SHA-256，不上报本机目录。
- 同一 artifact 在各节点路径可以不同：主节点 `D:\Models\sd15`，从节点 `E:\AI\sd15`，只要 manifest 一致即可。
- 文件移动、修改时间/大小变化或哈希不符后立即失效 ready 状态。
- 大文件哈希使用分块读取并缓存 `(resolved_path, size, mtime_ns, sha256)`；安全模式和发布验收仍执行完整 SHA-256。
- UNC/网络盘默认允许只读选择，但探测到吞吐过低时给出警告；推理过程中不自动把文件复制到系统盘。

### 3.4 资产集合身份

SD pipeline 不是单个 LLM 权重，因此把现有单一 `ModelIdentity` 扩展为 `ArtifactSetIdentity`：

```json
{
  "artifact_set_id": "sd15_90s_retrovers_v1",
  "engine": "diffusers_sd15",
  "pipeline_kind": "text_to_image",
  "base": {"logical_id": "...", "sha256": "...", "revision": "..."},
  "controlnet": null,
  "adapters": [],
  "scheduler": {"class": "PNDMScheduler", "config_sha256": "..."},
  "runtime": {"torch": "...", "diffusers": "...", "transformers": "..."},
  "manifest_sha256": "..."
}
```

以下任何一项不同都不能进入同一固定演示批次：base/full fine-tune、ControlNet、LoRA 权重与 scale、VAE、scheduler 配置、Diffusers 兼容版本、dtype 策略。

---

## 4. 演示模型与固定参数

### 4.1 原版 SD 1.5 preset

推荐逻辑 ID：`sd15_original_v1`。

上游候选：`stable-diffusion-v1-5/stable-diffusion-v1-5`，调研时 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`，CreativeML OpenRAIL-M。模型卡说明这是已弃用 RunwayML 仓库的镜像，因此实施时必须：

- 固定实际使用的 repo/revision 或本地完整 checkpoint SHA；
- 把上游模型卡与许可证随资产清单保留；
- 不在运行时跟随 `main` 自动更新；
- 优先支持用户已有的 `v1-5-pruned-emaonly.safetensors` 或完整本地 Diffusers snapshot。

### 4.2 90 年代日式动漫 DreamBooth preset

首选候选：`Aleksandra11/90style_anime_face_model`。

| 字段 | 调研结果 |
|---|---|
| 基座 | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| 训练方式 | 模型卡声明 DreamBooth full fine-tune |
| 用途 | 90 年代日式动漫风格人脸/肖像 |
| 触发词 | `retrovers` |
| pipeline | `StableDiffusionPipeline` |
| 文件形态 | 完整 Diffusers 目录，包含 text encoder、U-Net、VAE Safetensors |
| revision | 调研时 `aa8a082c6a12d66ed995cca1ccb491bb171b9713` |
| 许可 | Hugging Face 元数据为 `openrail` |
| 风险 | 下载量和社区验证较少，且模型自身 `requires_safety_checker=false` |

因此该候选只有完成以下门槛后才成为正式演示资产：

1. 固定 revision 可完整下载并离线重载。
2. 许可证原文与基础模型许可经人工复核，资产包附副本。
3. 10 个固定 seed 均能生成有效图像，无黑图、NaN 或明显模型损坏。
4. 由至少两人目视确认触发词确实产生目标风格，而非只相信仓库标签。
5. 应用加载独立安全检查组件或输出门控；不能因为 fine-tune 没带 safety checker 就默认公开未过滤结果。
6. 完整 snapshot 和每个文件 SHA-256 写入 manifest。

备选 `nitrosocke/classic-anim-diffusion` 是 OpenRAIL-M、DreamBooth、触发词 `classic disney style`，但它描述的是经典动画工作室风格，并非严格的“90 年代日式动漫”，只能在首选验收失败后作为“经典动画”备选，UI 名称不得写错。

### 4.3 固定展示 preset

固定主题使用原创的非特定角色，避免把模型能力展示绑定到真实人物或受保护角色：

```text
preset_id: sd15_retrovers_space_courier_v1
prompt: retrovers, portrait of an adult woman space courier on a rain-soaked neon train platform, 1990s Japanese cel animation, expressive eyes, hand-painted background, cinematic lighting, detailed face
negative_prompt: low quality, worst quality, blurry, deformed, malformed hands, extra fingers, text, watermark, logo, duplicate
width: 512
height: 512
num_inference_steps: 40
guidance_scale: 7.5
scheduler: PNDMScheduler（完整配置写入 manifest）
seeds: [19950101, 19950102, 19950103, 19950104]
num_images: 4
```

原版 SD 1.5 对照 preset 使用同一主题、negative prompt、尺寸、scheduler、步数、CFG 和 seeds，只去掉 `retrovers`，由此展示“引擎一致、模型资产可切换”，不是用不同参数人为制造风格差异。

固定参数的目的有三个：

- 所有 Worker 输入一致，只有 seed 按确定规则分配；
- 单节点串行与多节点并行可以公平比较 wall time；
- UI 演示可以一键复现，不受用户临时 prompt 质量影响。

固定 seed 不保证不同 GPU、驱动和算子后端逐像素一致。正确性检查以同环境复现、非黑图/非 NaN、尺寸/元数据一致和感知指标为主，不把跨硬件 PNG SHA 相同作为硬门。

### 4.4 模型“提供”方式

产品中“提供模型”定义为以下任一方式，不能只写一个失效链接：

1. **默认方式：一键下载固定 snapshot。** 下载器显示仓库、revision、许可证、预计空间，支持断点续传和最终 SHA 校验。
2. **离线方式：独立演示资产包。** 模型不进入主安装包，资产包包含 manifest、许可证、模型卡、哈希和导入说明。
3. **已有模型：本地目录映射。** 用户选择电脑其他位置的模型，由 Inspector 校验后注册。

不得自动重新分发一个来源或许可不清楚的社区 checkpoint；不得把 ControlNet 文件当作“已经提供完整 SD”。

---

## 5. 本地引擎设计

### 5.1 模块边界

建议新增：

```text
src/diffusion/
  artifacts.py       # 路径、类型、manifest、SHA 检查
  presets.py         # 固定展示 preset 与参数校验
  engine.py          # Diffusers SD 1.5 pipeline 生命周期
  provider.py        # Local/Remote image generation Provider adapter
  blobs.py           # 图片 blob 存储、摘要、配额和清理
  schemas.py         # image_generate/image_edit/image_grid schema
```

不建议直接扩写 `model_module.py`：LLM 的 tokenizer、逐 token decode、KV Cache、Transformer layer slicing 与扩散模型的 CLIP、scheduler、迭代去噪和 VAE 生命周期差异太大。上层统一在 Provider 和 API，不强迫底层类共享错误抽象。

### 5.2 依赖与环境

首轮兼容性 spike 使用项目现有 CUDA 虚拟环境，新增独立可选依赖清单，不先污染 CPU/集显包：

- `diffusers`：锁定为 `0.35.2`。实测 `0.38.0` 在导入阶段需要 `transformers 4.47.x` 不具备的 `Dinov2WithRegistersConfig`，不能进入本项目打包环境；候选模型导出版本为 `0.34.0.dev0`，只作兼容线索，不直接使用 dev 版本。
- `Pillow`：输入/输出图片。
- `safetensors`、`accelerate`、`transformers`、`torch`：复用 CUDA 环境，锁定已验证组合。
- `xformers`：一期不设为必需；先以 PyTorch SDPA 建立正确性基线，确认 wheel/CUDA 兼容和收益后再作为可选优化。
- 不在 PC 集显包声明 SD 可用，除非 CPU 512x512 基准达到单独定义的可接受门槛。

建议新增 `packaging/requirements-sd-cuda.txt` 或 CUDA extra，并在 PyInstaller spec 中按功能开关收集 Diffusers pipeline/config；模型权重始终外置。

### 5.3 加载策略

- CUDA 默认 `torch.float16`，进入推理前 `eval()` 与 inference mode。
- `from_pretrained(local_dir, local_files_only=True, use_safetensors=True)` 加载目录。
- `from_single_file(path, config=local_sd15_config, local_files_only=True)` 加载完整 checkpoint。
- ControlNet 使用 `ControlNetModel.from_pretrained/from_single_file`，再构造 `StableDiffusionControlNetPipeline`。
- 默认关闭上游网络回退。缺 config 时返回可执行的缺失项，不在用户点击生成后临时联网。
- 先尝试完整 GPU 常驻；显存不足时允许用户显式选择 model CPU offload/attention slicing。offload 模式纳入 capability 和基准标签，不能和常驻模式混算性能。
- 每个进程同一 pipeline 默认 `max_concurrency=1`；不要用两个线程并发调用同一 scheduler/pipeline 实例。
- 模型切换先阻止新 reservation，等待或取消活动 attempt，释放旧 pipeline、GC/empty cache，再加载新资产并重新上报 ready。

### 5.4 生成与取消

`DiffusionEngine.generate()` 接受强类型参数并返回图片 blob 与指标：

- prompt、negative prompt；
- width/height，必须是 8 的倍数并受 512 默认/最大值限制；
- steps、CFG、seed、batch size；
- pipeline kind 和可选 control image；
- attempt cancel event。

取消通过 Diffusers step-end callback 在去噪步边界检查。取消后：

- 不提交半成品图片为成功结果；
- 释放临时 latent、ControlNet residual 和 image buffers；
- Provider reservation 走既有幂等 release；
- 旧 attempt 即使稍后返回也由 fencing 拒绝。

指标至少包含：load time、queue wait、CLIP/denoise/VAE decode/encode 总耗时（可测时）、steps/sec、peak VRAM、seed、尺寸、scheduler、artifact manifest、provider/node、fallback reason。

---

## 6. API、Blob 与前端

### 6.1 API 形状

新增命名空间，不复用文本 `/api/chat`：

| API | 用途 |
|---|---|
| `GET /api/diffusion/capabilities` | 当前节点 CUDA、引擎、支持 pipeline、显存与 Worker 能力 |
| `POST /api/diffusion/artifacts/inspect` | 检查本机路径，返回类型与缺失组件 |
| `POST /api/diffusion/artifacts/register` | 注册逻辑资产，不上传权重 |
| `GET /api/diffusion/artifacts` | 列出本机可用资产/preset |
| `POST /api/diffusion/load` | 加载资产集合并建立 ready 状态 |
| `POST /api/diffusion/generate` | 本地单图或分布式批次任务 |
| `POST /api/diffusion/edit` | 二期 ControlNet IP2P |
| `GET /api/diffusion/jobs/{id}` | 查询 Workflow/Stage/Attempt 与图片结果 |
| `POST /api/diffusion/jobs/{id}/cancel` | 协作取消 |
| `GET /api/diffusion/blobs/{id}` | 读取有权限、未过期的图片 blob |

路径选择本身必须由本机桌面壳或受信本机 API 完成。远程浏览器不能提交任意服务器路径进行探测，避免把接口变成文件存在性/哈希探针。

### 6.2 为什么需要 blob 协议

Worker v2 目前是严格 JSON 信封，Stage 类型仅允许 `full_inference`/`aggregate`，单消息有大小上限。PNG/JPEG 经 base64 会增加约三分之一体积，ControlNet 输入与多图结果还会重复进 journal。因此新增图片 Stage 时不能直接把图像字节塞进 `root_input` 或 `StageResult.output`。

推荐实现受控 blob store：

- 图片先写入主节点 `STATE_DIR/diffusion/blobs/`；
- Stage 只传 `{blob_id, sha256, size, mime, width, height}`；
- Worker 使用 attempt 范围、短 TTL、HMAC 签名的一次性 URL 下载输入或上传结果；
- 流式分块计算 SHA 和大小，不信任 Content-Length；
- 只接受 PNG/JPEG/WebP allow-list，解码后再次检查像素数，防止解压炸弹；
- 默认单图 encoded 16 MiB、最大 2048x2048、单 Workflow 64 MiB，可配置但有硬上限；
- journal 只记录 blob descriptor，不写图片正文；
- terminal + TTL 到期后清理，正在被读取的 blob 使用引用计数/租约保护。

### 6.3 前端页面

新增“图像生成”工作区，而不是把十几个参数塞进聊天输入框：

- 模式 tabs：文生图 / 图像编辑（二期）。
- 模型选择：原版 SD 1.5 / 90s anime demo / 本地模型。
- 文件或目录选择、资产类型、哈希与缺失组件提示。
- 固定演示 preset 一键填入且参数可只读展开。
- 生成数量、尺寸、steps、CFG、seed 模式、negative prompt。
- 本地 / 分布式批次 segmented control；不可用原因直接显示。
- 2x2 固定比例结果网格，单图显示 seed、node、provider、耗时和实际模式。
- 进度来自 `step/total_steps` 或 Workflow 状态，不用假进度条。
- 取消、单图下载、整组 ZIP、复制参数与重新生成。
- 只有完成的远端 attempt 才显示“分布式：是”；一张图本地生成后不能冒充多节点。

---

## 7. 分布式任务图设计

### 7.1 固定工作流模板

模板 ID：`sd15_seed_fanout_v1`。

```text
validate_input
      |
      +--> generate_seed_19950101 -->+
      +--> generate_seed_19950102 -->+--> image_grid --> final result
      +--> generate_seed_19950103 -->+
      +--> generate_seed_19950104 -->+
```

- `validate_input` 在主节点校验 preset、manifest、blob 和所有候选 Provider。
- 每个 `image_generate` Stage 是独立可重试、完整模型推理，固定一个 seed。
- `image_grid` 在主节点读取已完成 blobs，生成缩略网格与结果清单，不重新编码原图。
- 默认至少 2 张成功才生成降级网格；少于阈值 Workflow 失败。
- 失败 seed 可以跨兼容 Provider 重派；已成功 seed 不重复生成。
- 聚合顺序永远按 seed/preset 序号，不按完成先后，保证 UI 稳定。

### 7.2 Stage schema

新增 allow-list Stage：

| Stage | 输入 | 输出 | 推荐 Provider |
|---|---|---|---|
| `image_generate` | prompt、negative、seed、尺寸、steps、CFG、scheduler、artifact set | image blob descriptor + metrics | 完整 SD CUDA Worker |
| `image_edit` | 上述参数 + input blob + ControlNet identity/scale | image blob descriptor + metrics | 完整 SD+ControlNet Worker |
| `image_grid` | 有序 image descriptors | grid blob + ordered result descriptors | 主节点轻量 Provider |

所有数值做上下限和有限值校验，prompt 长度、总像素、steps 和图片数在建图前冻结预算。Worker 不接受任意 Python pipeline 类名、任意 config URL 或任意本地路径。

### 7.3 Worker 协议升级

不修改 v2 现有字段含义，新增协议 v3 或协商 feature：

- stage types 增加 `image_generate`、`image_edit`、`image_grid`；
- engine 增加 `diffusers_sd15`；
- model identity 升级为 artifact-set manifest；
- capabilities 增加 `pipeline_kinds`、`dtypes`、`max_width/height/pixels`、`max_batch`、`supports_controlnet`、`supports_step_cancel`；
- Stage 只传 blob descriptors；下载/上传走独立受控 HTTP 数据面；
- v2 文本 Worker 与 v3 图像 Worker 可同时连接，协商不到 v3 时只保留原能力，不能因升级导致现有 Qwen/DeepSeek 任务失效。

### 7.4 Provider 选择与调度

硬约束先于评分：

1. 已认证、心跳健康、协议 v3/feature 通过。
2. `diffusers_sd15` 和目标 Stage 在能力 allow-list。
3. manifest SHA 完全一致，所需 ControlNet/adapter 均 ready。
4. 分辨率、dtype、显存与 blob 预算可满足。
5. 有可用 reservation，且节点不是正在执行不兼容 LLM 模型的单 GPU 资源。

满足硬约束后按以下指标评分：历史该 preset 的 steps/sec、当前队列、剩余显存、模型是否已热加载、RTT 和 blob 吞吐。对生成任务，模型已热加载通常比 CPU 核心数更重要。

任务图并行数取以下最小值：用户图片数、兼容 Provider 数、各 Provider 可用槽位总和、全局图片并行上限。一个单槽 Worker 不能同时拿两个 image Stage。

### 7.5 故障与降级

| 故障 | 行为 |
|---|---|
| Worker 缺模型/哈希不一致 | 建图前排除，不下发任务 |
| Worker OOM | attempt 失败；允许换兼容 Provider 重派一次，不自动改分辨率破坏固定 preset |
| Worker 断线/租约过期 | cancel + fencing；迟到图片 blob 不进入 winner，随后 TTL 清理 |
| 图片上传中断/哈希不符 | blob 无效，attempt 不可 completed |
| 一个 seed 失败 | 达到最少成功数时生成降级网格并标明缺失；否则失败 |
| 所有远端不可用 | 用户明确允许时可回退本地串行，并返回 `distributed=false` 与原因 |
| 用户取消 | 取消所有活动 attempt、终止后续 Stage、回收 reservation/blob |
| 主节点重启 | journal 恢复为保守终态，不声称恢复 U-Net 中间 latent；用户可重跑整个 seed Stage |

---

## 8. 分阶段实施

### SD-N0：依赖与资产识别 spike

**状态：** `In Progress`（已完成固定下载、目录 Inspector、ControlNet 防误识别与 CUDA 侧车共存实测；其余完成判据未满足）

**工作：**

- 在项目 CUDA 虚拟环境试装候选 Diffusers/Pillow 组合，不改变全局解释器。
- 对原版 full checkpoint、完整 Diffusers 目录、90s DreamBooth、ControlNet、LoRA 和随机 Safetensors 建立 Inspector fixture。
- 验证项目现有 `transformers`、PyTorch、bitsandbytes 与 Diffusers 版本共存。
- 冻结 offline config 方案和依赖版本。

**完成判据：** 六类资产识别正确；ControlNet 不被认成 full checkpoint；完整 checkpoint 离线加载到 CPU 完成结构初始化；无网络时错误可解释；输出兼容矩阵与依赖锁。

### SD-N1：本地原版 SD 1.5 引擎

**状态：** `In Progress`（已在与 LLM 相同 `transformers 4.47.1` 兼容窗口中完成本地加载/卸载、文生图、step 回调、基础取消单测和同一 pipeline 10 轮稳定性；尚未完成注册/API、真实取消与真实 LLM 模型切换回归）。

**依赖：** SD-N0。

**工作：** `DiffusionEngine`、本地资产注册、加载/卸载、文生图、step 取消、指标、API 单元测试。

**完成判据：** RTX 4060 Laptop 或目标 CUDA 设备上用原版 SD 1.5 固定 preset 连续生成 10 轮；无 NaN/黑图/显存持续增长；取消在下一去噪步收敛；切换回 LLM 后显存和模型状态正确。

### SD-N2：90s DreamBooth 演示资产与前端

**依赖：** SD-N1、模型许可/样例人工验收。

**工作：** 固定 `retrovers` preset、原版对照 preset、一键下载/离线资产包、图像生成 UI、结果参数卡和下载。

**完成判据：** 固定 revision 和 SHA 可重建；10 seed 质量门通过；原版和 fine-tune 均可选择；无模型时 UI 不假装可用；安装包不内置大权重但能导入外置模型。

### SD-N3：图像 blob 与 Worker v3

**依赖：** SD-N1、TC-N2.4 状态机/Worker 故障边界保持稳定。

**工作：** blob store、受控传输、图像 Stage schema、artifact manifest、v3 能力协商、RemoteDiffusionProvider、取消/lease/fencing。

**完成判据：** 本机双进程能完成单 `image_generate` Stage；恶意尺寸、错误 MIME、哈希不符、断传、重复上传、迟到结果和路径探测全部 fail-closed；v2 文本 Worker 全量回归不变。

### SD-N4：固定多 seed 分布式展示

**依赖：** SD-N2、SD-N3、至少两台物理 CUDA PC。

**工作：** 固定四 seed DAG、Provider 调度、网格聚合、实际参与节点 UI、单节点串行基线、故障注入。

**完成判据：**

- 同一 artifact manifest 在至少两台 PC ready。
- 四张图片至少由两个物理节点完成，其中至少一个远端节点。
- UI、API、journal 和任务统计对每张图的 node/provider/attempt 一致。
- 与同设备集合中的单节点串行四图相比，热模型 wall time 有正收益；目标加速比不硬编码，在验收报告记录实际值和瓶颈。
- 拔网、Worker OOM、重连、主节点重启、单 seed 失败和用户取消均无假成功、重复图或 blob 泄漏。
- 页面明确写“分布式批次/多图吞吐”，不写“单图已加速”。

### SD-N5：ControlNet IP2P

**依赖：** SD-N2、本地 blob 安全路径；分布式支持可在 SD-N4 后接入。

**工作：** 加载本地 `control_v11e_sd15_ip2p_fp16.safetensors` 及本地 config；输入图上传/选择；固定编辑 prompt；ControlNet scale；结果对照。

**完成判据：** 原图、prompt、base、ControlNet、seed 和参数均写入 manifest/结果元数据；缺 base/config/input image 时在执行前拒绝；本地 10 轮无泄漏；若进入分布式，输入只上传一次并由 blob 引用复用。

### SD-N6：细粒度分布式可行性研究

**依赖：** SD-N4 有真实网络、U-Net step profile 和批次收益基线。

按以下顺序做微基准，不直接承诺产品化：

1. 单机多 GPU Diffusers `device_map` 容量分配。
2. CFG conditional/unconditional 两分支同步成本。
3. CLIP/U-Net/VAE 组件拆分的传输与负载占比。
4. DistriFusion、xDiT 或其他成熟 sidecar 对 SD 1.5 的实际支持矩阵。
5. 同机高速互联与 Tailscale/Wi-Fi 分别测量，不混为一组。

只有单图 P50 延迟至少改善 20%、质量回归通过、30 分钟稳定且断线能在完整 image attempt 边界恢复，才另立生产计划；否则标记 `Frozen`，保留任务级分布式。

---

## 9. 测试与验收矩阵

### 9.1 自动化

| 领域 | 必测项 |
|---|---|
| Inspector | 完整目录/checkpoint、ControlNet、LoRA、VAE、损坏/未知 Safetensors、超大 header |
| 路径 | 外置盘、空格/中文、文件移动、权限拒绝、UNC、路径越界、远程 API 路径探测 |
| 参数 | seed 边界、NaN/Inf、尺寸倍数、steps/CFG 上限、prompt 长度、总像素预算 |
| Engine | load/unload、切换、取消、OOM、offload、重复调用、异常后可恢复 |
| Blob | MIME、magic bytes、尺寸炸弹、SHA、分块中断、TTL、并发读写、journal 不含正文 |
| TaskGraph | fan-out/fan-in、稳定排序、部分成功阈值、重派、winner、fencing、恢复 |
| 协议 | v2 兼容、v3 协商、未知 engine/stage 拒绝、manifest 不一致 |
| API/UI | 本地/分布式真实元数据、进度、取消、下载、回退原因、未加载状态 |

单元测试不得要求下载 5 GB 模型：使用 fake pipeline、微型随机 Diffusers components 或固定小图片覆盖控制逻辑。真实质量和性能由条件集成测试负责，缺模型时明确 skip reason。

### 9.2 真实模型

| 组合 | 必测模式 |
|---|---|
| 原版 SD 1.5 full checkpoint | 本地 single-file、离线 config、连续 10 轮 |
| 原版 SD 1.5 Diffusers 目录 | 本地目录、冷/热加载 |
| 90s DreamBooth | 固定 preset、10 seed、许可证/manifest |
| 用户外置完整模型 | 项目外绝对路径、重启后恢复 |
| ControlNet IP2P | 原图编辑、错误 base、错误 config、输入复用 |
| 两物理 Worker | 四 seed 扇出、掉线、OOM、重连、取消 |

### 9.3 性能记录

每组必须记录：

- CPU/GPU/显存/驱动/CUDA/PyTorch/Diffusers；
- 模型 manifest SHA 和加载策略；
- 分辨率、steps、scheduler、CFG、seed；
- 冷启动、热启动、单图和四图 wall time；
- 每节点 queue/load/denoise/decode/upload 时间；
- 峰值显存、系统内存、图片字节数、Tailscale RTT/吞吐；
- 单节点串行 vs 多节点 fan-out 的吞吐和效率；
- 失败、重试、回退和实际参与节点。

---

## 10. 安全、许可与发布

- 模型选择接口只对本机受信调用开放；不能让远程用户读取任意服务器文件。
- Safetensors 优先于 pickle ckpt；如支持 `.ckpt`，默认提示来源风险并在隔离进程/受信资产策略下加载。
- 模型与输出遵守 OpenRAIL 使用限制，下载/导入时展示并保存许可确认。
- 社区模型卡声称的许可不自动解决训练数据、角色或风格相关权利风险；公开展示使用原创 prompt，不模仿在世艺术家，不使用真实人物和受保护角色。
- safety checker 或等价输出门控必须成为公开服务的发布门；演示模型缺 checker 时从固定 SD 1.5 安全组件补齐并纳入 artifact manifest。
- 图片包含 prompt、seed、模型、节点等元数据时，不写本机绝对路径、用户账户名或集群密钥。
- 模型权重不写入 Git、不默认写入安装目录、不通过普通 Stage payload 同步。
- 发布清单包含依赖锁、支持的 pipeline、外置资产说明、许可证、固定 revision、SHA 和已知限制。

---

## 11. 风险与止损

| 风险 | 处理/止损 |
|---|---|
| Diffusers 与现有 Transformers/PyTorch 冲突 | 使用独立可选依赖锁；无法共存则拆 SD sidecar 进程，不降级 LLM 环境 |
| 4060 Laptop 显存不足 | 先 512x512 FP16；offload 作为显式慢路径；仍 OOM 则不提高分辨率 |
| 社区 90s 模型质量/许可不足 | 候选不通过即冻结资产；保留原版支持和允许用户自选，不伪造验收 |
| Worker 都要完整复制模型 | 接受任务级并行的容量成本；以本地路径映射/资产包解决，不在请求期同步数 GB |
| 多节点没有单图加速 | 产品文案只承诺多图吞吐；单图内部并行需 SD-N6 独立过门 |
| 图片传输吃掉收益 | 使用 blob、一次传输和缩略图；基准若网络占比过高，限制远端图片数或只返回压缩预览 |
| 不同硬件像素不一致 | 固定输入并记录环境；质量用结构/感知门，不要求跨 GPU 文件哈希一致 |
| ControlNet 被误识别为 full model | Inspector 硬分类、组件 schema 和回归 fixture 三层阻止 |
| 安全检查缺失 | 公开入口 fail-closed；仅本地开发模式也明确显示未启用，不静默绕过 |

触发以下任一条件时，细粒度分布式路线转 `Frozen`，但本地 SD 与批次 fan-out 不受影响：

- 实测通信时间超过单图计算节省；
- 需要破坏现有文本 Worker v2 正确性或在 adapter 里另建 winner 状态机；
- 目标 runtime 不支持 SD 1.5/ControlNet 或只能依赖不可分发的私有组件；
- 单图延迟改善不足 20% 或质量回归不可接受；
- 普通 Tailscale/Wi-Fi 下无法连续稳定运行 30 分钟。

---

## 12. 已冻结决策与待决策项

### 12.1 已冻结

- 引擎使用 Diffusers，标识 `diffusers_sd15`。
- 模型外置，支持目录和完整 single-file。
- 加载前区分 full checkpoint、DreamBooth full fine-tune、ControlNet、LoRA、VAE。
- `control_v11e_sd15_ip2p_fp16.safetensors` 是 ControlNet 组件。
- 一期分布式是固定多 seed 的完整 Worker fan-out/fan-in。
- 原版 SD 1.5 与 90s DreamBooth 均支持。
- 图片走 blob descriptor，不进 Stage JSON/journal 正文。
- 远端 Worker 按 artifact manifest 匹配，不按路径匹配。
- 分布式展示以真实节点统计和热模型 wall time 为准。

### 12.2 实施前待决策

- Diffusers/Pillow/Transformers/PyTorch 最终版本锁。
- 90s 候选的人工质量和许可证复核结果；不通过时采用哪个新候选。
- 演示资产采用一键下载、独立压缩包或两者并存。
- blob encoded/decoded 硬上限和 TTL 的最终值。
- Worker 协议直接升 v3，还是 v2 feature negotiation + 独立 blob API；不得静默扩展 v2 schema。
- safety checker 资产来源、revision 与输出门控策略。
- CPU/集显是否只展示“不支持”，还是保留极慢的显式实验模式。

---

## 13. 调研来源

- [Diffusers：Distributed inference](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference)
- [Diffusers：Single files](https://huggingface.co/docs/diffusers/main/en/api/loaders/single_file)
- [Diffusers：Pipeline loading and device placement](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading)
- [SD 1.5 当前 Hugging Face 镜像](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet v1.1 IP2P 模型卡](https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p)
- [90style Anime Face DreamBooth 候选](https://huggingface.co/Aleksandra11/90style_anime_face_model)
- [Classic Animation Diffusion 备选](https://huggingface.co/nitrosocke/classic-anim-diffusion)
- [DistriFusion 论文](https://arxiv.org/abs/2402.19481)
- [xDiT 项目](https://github.com/xdit-project/xDiT)

外部仓库、默认分支、依赖和许可证会变化。真正实施时必须重新查询并把 revision、许可证副本、文件清单与 SHA 固定到本地 manifest，本文记录的 2026-07-30 调研快照不能代替发布审计。
