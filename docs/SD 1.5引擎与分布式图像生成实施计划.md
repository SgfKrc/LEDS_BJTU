# SD 1.5 引擎与分布式图像生成实施计划

> 文档状态：部分实施（`L4 Candidate`；SD-N1 与 SD-N5.2 已完成；SD-N2 本地图像工作区、固定资产下载/导入和 DreamBooth 十种子自动门已实测；img2img 与 IP-Adapter `reference` 的后端、前端和双模型完整自动质量/显存门均已通过，双人目视审核于 2026-08-06 完成（Siegfried Kkm./浅草爱音，5 份报告均 passed）；SD-N5.3 InstructPix2Pix 指令编辑已完成实现，十指令自动门、Edge 链路与双人目视均已完成；SD-N3.3 已完成，SD-N3.4 已接通主节点 Provider、单 Stage TaskGraph API 和独立进程真实 TCP+HTTP fake executor 闭环，但真实 Diffusers/GPU Worker、物理 PC 和多 seed 分布式验收仍未完成；继续由《总体下一步计划》L4-SD1.5 管理优先级）
>
> 调研日期：2026-07-30
> 更新日期：2026-08-07
>
> 适用范围：Windows CUDA PC 上的 Stable Diffusion 1.5 本地文生图，以及后续图生图、局部重绘、指令式编辑和基于现有 TaskGraph/PC Full Worker 的跨 PC 图像批次分布式展示
>
> 总计划入口：[总体下一步计划](总体下一步计划.md)

本文同时记录计划与已验证边界。任何阶段只有完成对应自动化、真实模型、真实 GPU、跨设备和安装包验收后，才能把状态改为“已实现”。当前不得因为单机基线通过而宣称已支持分布式图像生成。

## 0. 2026-08-07 实施状态

| 范围 | 状态 | 已验证证据 |
|---|---|---|
| `SD-N0` 资产识别与固定下载 | 验证中 | 固定 `stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14`；仅下载 SD 1.5 FP16 推理所需文件，完整快照为 2,742,233,847 bytes，Inspector 识别为可加载 `sd15_pipeline`。完整 single-file checkpoint 的离线结构初始化和全部资产 fixture 仍待完成。 |
| `SD-N1` 本地单机文生图 | `Completed`（2026-08-05） | RTX 4060 Laptop GPU（8,188 MiB）/ 16 GB RAM、`torch 2.5.1+cu121`、`diffusers 0.35.2`、`transformers 4.47.1`：固定 seed `19950101` 起的 10 次连续 512x512/28 steps FP16 baseline 均实际生成成功，耗时 6.909–11.778 s（平均 8.844 s）；另以 8-bit U-Net Linear 量化 + QKV 融合组合完成 512x512/28 steps，耗时 10.030 s，峰值 reserved 显存 3,735,027,712 bytes。新增本地资产登记、加载/卸载、异步 job、step 取消、结果查询/删除和有界内存 PNG blob，并接通单体 FastAPI、inference-svc 与 Nest 网关；自动化覆盖 LLM/SD 双向生命周期互斥、加载失败及取消/编码竞态。真实 inference-svc 路由验收完成 512x512/4 steps PNG（542,085 bytes），取消在第 1/50 步后 0.244 秒收敛且无 blob；SD 卸载后 Qwen 1.8B GGUF 在 5.471 秒内重载并完成最短对话。SD 卸载后仅余约 20 MiB CUDA 上下文 reserved，不残留模型显存。`diffusers 0.38.0` 会要求本项目兼容窗口外的 DINOv2 配置，故不作为打包组合。前端、发布包和分布式 Worker 属于后续阶段。 |
| `SD-N2` 本地图像工作区 | `In Progress`（2026-08-05） | Web 主节点已接通本机 Diffusers 检查/登记、profile 加载、LLM/SD 显式互斥切换、原版/90s preset、异步 step 进度、取消和 PNG 生命周期。DreamBooth `aa8a082c...` 及逐权重 SHA 已冻结，4,874,690,864-byte 固定下载集合已实际下载、完整校验并生成 16 文件 manifest；独立脚本下载后服务端也能在目录刷新时自动发现并注册。90s 组合资产固定原版 SD 1.5 safety checker，组件缺失时 fail-closed。第二轮固定十种子自动门 10/10 通过：10 个唯一 PNG、0 个 safety flag、最小熵 7.855、最小文件 446,750 bytes，总耗时 111.712 s。**双人目视与许可复核于 2026-08-06 完成（Siegfried Kkm./浅草爱音，十种子报告 status=passed）**。正式离线发布包仍未完成，因此 SD-N2 不得标记完成。 |
| single-file checkpoint、LoRA、ControlNet | 未实施 | Inspector 可以拒绝把 ControlNet 当完整模型；实际加载和组合仍未接入。 |
| 图生图 img2img、参考图一致性、局部重绘、指令式编辑 | img2img 与 IP-Adapter `In Progress`（自动门和双人目视已完成）；inpaint `Completed`（2026-08-07）；instruction `In Progress`（自动/Edge 门和双人目视均已完成） | **img2img（2026-08-06）**：引擎 `SD15Engine.edit()` 复用已加载组件（`StableDiffusionImg2ImgPipeline.from_pipe`/`components`）、服务层 `submit_edit`/`validate_edit`（blob 租约、源图/mask 维度校验、strength 0.05–1.0）、单体 FastAPI 与 inference-svc 路由、Nest 网关 multipart 转发与 Web 工作区均已落地。已修复生成结果续编的 `purpose`/`owner_scope` 契约，并完成真实 inference-svc `文生图 → output Blob 图生图 → 引用删除 → 取消 → 卸载` 快速门：512×512/4 steps 文生图 6.739 s，strength 0.55 图生图 7.428 s，源 Blob 被结果引用时删除正确返回 409，取消在第 1/20 step 后 0.054 s 收敛。真实 Edge 150 浏览器门完成“上传源图 → img2img → 继续编辑 → 再次 img2img → 卸载”，两轮引擎耗时 5.998/4.842 s、像素采样范围 245/239。源图 SHA-256 固定为 `a6fd131b...e93264`；原版 28 steps 与 90s 40 steps 各完成 10 seed × 0.25/0.55/0.85 共 30 张，均 30/30 自动通过且 30 张唯一，耗时 291.993/361.291 s，连续 allocated/reserved span 均为 0，卸载后 allocated 增量 8,519,680 bytes。两份报告的自动门均通过，目视签字结果见本行后文。**IP-Adapter（2026-08-06）**：冻结并下载 `h94/IP-Adapter@018e4027...` 的 2,573,016,776-byte 离线资产，稳定 artifact SHA-256 为 `671c7452...e94c4`；Inspector、服务/API、Web `reference` 模式和真实 GPU 生命周期均已通过。修复 adapter 后注册的 2.5 GB image encoder 未进入 Accelerate CPU-offload hook 的性能缺陷，并在 pipeline 级关闭 tqdm，4-step 回归为 4.537 s、无 `[Errno 22]`。固定人物参考图 SHA-256 为 `3271580f...fe38f`；原版 28 steps 完成 36/36 有效图，90s 40 steps 完成 35 张有效图并正确保留 1 张 safety 黑图证据，三档 scale 有效数为 11/12/12；两轮峰值 reserved 均为 2,789,212,160 bytes，显存门通过，自动报告随后已完成双人目视签字。**img2img（2 份）与 IP-Adapter（2 份）报告于 2026-08-06 双人目视签字完成（Siegfried Kkm./浅草爱音，均 status=passed），目视门已满足；IP-Adapter 只承诺人物主要要素保持，不承诺精确身份锁定。** **inpaint（2026-08-07）**：固定 `stable-diffusion-v1-5/stable-diffusion-inpainting@8a4288a...` 的 2,742,261,613-byte FP16 下载集合，专用 9-channel U-Net 识别为 `sd15_inpaint_pipeline`，稳定 artifact SHA-256 为 `ddd6d69a...1c605`。引擎按需加载并复用专用 pipeline，服务/API/网关接通 source + mask 双租约，Web 提供鼠标/触控画笔、橡皮、撤销/重做、反转、缩放/平移和黑白 PNG 导出；缺专用资产或非 balanced CUDA profile 时排队前拒绝。固定源图 SHA `a6fd131b...e93264` 的 10 seed × 20 steps 完整门 10/10 通过：全黑 mask MAE 4.16–4.98，局部白 mask 外侧 MAE 4.59–5.03、内外差 20.16–30.01，全白 mask MAE 82.04–85.57；0 safety flag、10 张唯一，连续 allocated span 为 0，峰值 reserved 3,649,044,480 bytes，卸载后 allocated/reserved 为 8,519,680/20,971,520 bytes。Edge 150 真实画布上传与 inpaint 任务通过，4-step 引擎/浏览器链路耗时 4.683/7.015 s。** **instruction（2026-08-07）**：默认冻结完整 InstructPix2Pix `31519b5c...`（稳定 SHA `a6626f7f...c872ab`）；API/网关/Web 与独立 guidance 参数已贯通，10 条固定指令自动门 10/10、0 safety、显存门和 Edge 链路通过，报告已完成双人目视并签字为 `passed`。** |
| API/UI、资产登记、图片 blob、TaskGraph Stage、完整 PC Worker、跨 PC fan-out/fan-in | 本地 API/UI、Blob 数据面、v3 adapter、隔离 Provider、单 Stage 实验接口和独立进程传输闭环已实现；多图生产链未接通 | SQLite WAL 内容寻址对象、受控 HTTP 分块数据面、attempt lease、HMAC 传输授权、v3 image schema、Worker/协调器 adapter、`RemoteDiffusionProvider`、校验式 Blob transfer client 以及默认关闭的 `/api/diffusion/distributed/generate` 单 `image_generate` TaskGraph seam 已实现。接口要求 TaskGraph + SD v3 双开、主节点、健康 Worker 和精确 artifact manifest；结果由 coordinator CAS owner 校验读取，取消复用 `/api/workflows/{workflow_id}/cancel`。本机独立 Worker 子进程已通过真实 TCP hello/offer/result 和真实 FastAPI HTTP 分块回传固定 PNG；该证据使用确定性 fake SD executor，不等价于真实 Diffusers/GPU、两台物理设备、固定多 seed fan-out/fan-in 或 UI 分布式统计验收，前端仍不能显示为已发布分布式能力。 |
| Android SD 推理 | 未实施 | Android 仍不承担完整 SD Worker 或层间拆分。 |

单引擎基准由 `scripts/smoke_sd15.py` 复现；真实 HTTP 文生图/续编/取消生命周期由 `scripts/validate_sd15_api_lifecycle.py` 复现；固定资产下载由 `scripts/download_sd15.py --asset-id ... --accept-license` 复现；文生图十种子门由 `scripts/quality_gate_sd15.py` 复现，图生图三 strength/多 seed 门由 `scripts/quality_gate_sd15_img2img.py` 复现，IP-Adapter 三 scale/多 seed 门由 `scripts/quality_gate_sd15_ip_adapter.py` 复现，inpaint 黑/局部白/全白十轮语义门由 `scripts/quality_gate_sd15_inpaint.py` 复现，InstructPix2Pix 十条固定指令门由 `scripts/quality_gate_sd15_instruction.py` 复现，真实浏览器门由 `frontend npm run test:e2e:sd15` 复现。图生图、IP-Adapter 和 inpaint 完整门都必须显式传入固定源图的 `--source-sha256`；快速/改参运行只会得到 `partial_pass`。IP-Adapter 完整门每档固定生成 12 个 seed，至少 10 个未被 safety 拦截的有效结果才计为通过；被拦截黑图保留在报告中，不计入质量和唯一性。inpaint 完整门固定执行 3 次全黑、4 次局部白和 3 次全白 mask，以像素 MAE 和显存跨度同时判定。指令编辑自动门只证明输出有效、结构相关、唯一且显存稳定，指令语义是否正确仍需两名独立审核者签字。推理路径只接受已下载的本地完整 Diffusers 目录，不会在加载或生成途中访问 Hub。下载器和推理侧车只使用 CUDA 可选虚拟环境，不安装或升级全局解释器依赖，尚未进入 PC/Android 发布产物。

### 0.1 单机优化能力边界

| 能力 | 状态 | 实现与实测边界 |
|---|---|---|
| U-Net 8-bit 量化 | 已验证 | `bitsandbytes 0.49.2` 只替换 U-Net 的 184 个 `torch.nn.Linear` 为 `Linear8bitLt`，卷积、CLIP、VAE 和 safety checker 仍为 FP16。512x512/28 steps 实机出图成功，耗时 9.629 s，峰值 reserved 显存为 3,409,969,152 bytes。量化 U-Net 为常驻 CUDA 路径，代码禁止与 CPU offload 混用。 |
| Attention QKV 算子融合 | 已验证 | 通过 Diffusers `fuse_qkv_projections(unet=True, vae=False)`。512x512/8 steps 实机出图成功；与 8-bit U-Net 组合后 512x512/28 steps 也成功，单次耗时 10.030 s，峰值 reserved 显存为 3,735,027,712 bytes。同一已加载 pipeline 以三个不同 seed 连续运行 28 steps，峰值 reserved 显存均保持为该值；耗时会受 GPU 背景负载波动，不能据此宣称融合提升吞吐。它与 attention slicing 不能同时启用，QKV profile 保留 VAE slicing。 |
| `torch.compile` / Inductor | 当前不可用，显式拒绝 | 当前 CUDA sidecar 没有可工作的 Triton，首次真实调用会失败。引擎在加载模型前检查并给出短错误，不会等到去噪首步才抛出长异常；不为了该可选优化把未验证的 Triton 依赖加入侧车。 |
| LLM `PagedKVCache` | 不适用，非缺陷 | SD 1.5 U-Net 的每一步输入 latent 都会变化，没有可跨去噪步复用的自回归 KV Cache。不得将文本引擎的 token KV 分页标为 SD 优化；SD 的受支持显存分块手段是 Diffusers attention slicing。 |

这些优化全部位于 `src/diffusion/sd15_engine.py` 的延迟导入侧车中，本地生命周期由 `src/diffusion/service.py` 托管。它们不导入或改写 LLM `ModelManager`、文本 Worker 契约、集显包、Android 依赖和既有全局解释器；SD 与 LLM 当前通过同一模型生命周期锁显式互斥，SD 仍未注册为 Worker，因此不能显示为分布式推理或计入分布式任务统计。真正进入分布式前仍须完成 SD-N3 的图像 Stage/blob/Worker 协议和资源准入。

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
- 在不改变现有文生图请求契约的前提下，依次提供图生图、局部重绘和指令式编辑；三种模式复用同一异步 job、取消、blob 与安全边界。
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
- SD-N5 不实现 Photoshop 类图层、PSD 工程、滤镜栈、自由变形、自动抠图或跨多次编辑的无限历史；第一阶段只保存当前源图、mask、参数和结果引用。
- 不把普通图生图包装成“理解自然语言指令”；无法确认专用 InstructPix2Pix/ControlNet 资产和参数时，指令式编辑必须显示不可用或降级原因。

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
| revision | 已冻结 `aa8a082c6a12d66ed995cca1ccb491bb171b9713` |
| 许可 | Hugging Face 元数据为 `openrail` |
| 推理体积 | DreamBooth 核心文件 4,266,667,701 bytes；组合原版 safety checker 后固定下载集合 4,874,690,864 bytes |
| 风险 | 下载量仅 8、社区验证很少，且模型自身 `requires_safety_checker=false` |

因此该候选只有完成以下门槛后才成为正式演示资产：

1. **已实测：** 固定 revision 断点下载、直接连接失败后可回退 `127.0.0.1:7897`；4,874,690,864-byte 下载集合已落到 `models/sd15-90s-retrovers-v1`，完整哈希校验通过，推理全程离线加载。
2. **已完成（2026-08-06）：** 许可证原文与基础模型许可复核（Siegfried Kkm./浅草爱音 确认 `openrail` 许可与 CreativeML OpenRAIL-M 基础模型条款允许演示分发并附副本）；正式离线资产包仍须附许可副本和模型卡。
3. **已实测：** `scripts/quality_gate_sd15.py` 固定 10 seed，自动拒绝黑图、低熵图、损坏图和重复输出；当前固定集合 10/10 自动通过，报告位于 `build/sd15-quality/sd15_90s_retrovers_v1/quality-report.json`（status=passed，2026-08-06 双人目视签字）。
4. **已完成（2026-08-06）：** 至少两名不同审核者（Siegfried Kkm./浅草爱音）确认触发词产生目标风格，并把 `NAME=pass` 写入质量报告（status=passed）。
5. **已实现：** QLH 把固定原版 SD 1.5 safety checker 权重组合进 90s 资产，`safety_checker_required=true` 时 pipeline 实际缺组件会拒绝加载。
6. **已实现：** 大权重按 Hugging Face LFS SHA-256 验证；下载/导入成功后生成 `.qlh-sd-asset.json` 全文件清单。该清单证明本地完整性，不等同发布者数字签名。

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
scheduler: DPMSolverMultistepScheduler
seeds: [19950101, ..., 19950108, 19950110, 19950111]
num_images: 10（UI 单图使用首个 seed；质量门使用全部 seed）
```

首次真实十种子运行中 `19950109` 被组合 safety checker 标记并替换为黑图，自动门因此正确失败；固定集合改用已单独验证未触发安全门的 `19950111`，不通过关闭 safety checker 绕过该结果。

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

当前 Web 图像页已经提供前两类入口的执行面：目录下载不进入 Git 或安装包，下载期间以已落盘文件计算真实进度；离线导入对固定目录做逐文件大小和权重 SHA 校验，不能用任意“同名目录”冒充已冻结资产。CLI 与 API 共用 `src/diffusion/assets.py` 的同一份目录事实源：

```powershell
.\.venv-packaging-cuda\Scripts\python.exe scripts\download_sd15.py --asset-id sd15_90s_retrovers_v1 --accept-license
.\.venv-packaging-cuda\Scripts\python.exe scripts\quality_gate_sd15.py --asset-id sd15_90s_retrovers_v1
```

质量报告自动门通过后仍是 `pending_manual_review`。至少两人审图后用两个 `--reviewer NAME=pass` 重跑或登记审核，才允许成为正式演示资产。模型卡链接和许可标识会在下载前显示；正式离线压缩包附带许可原文的发布流程仍未完成。

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
- 对编辑请求使用不可变 input/mask blob descriptor，而不是本机路径或 JSON Base64；
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
| `POST /api/diffusion/blobs` | 二期以 `multipart/form-data` 上传源图或 mask，返回不可变 blob descriptor |
| `POST /api/diffusion/edit` | 二期统一提交 `img2img`、`inpaint`、`instruction` 异步编辑任务 |
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

- 模式 segmented control：文生图 / 图生图 / 参考图 / 局部重绘 / 指令编辑；五种模式已接通本地 executor，后续未实施模式仍不得提前显示为可用。
- 模型选择：原版 SD 1.5 / 90s anime demo / 本地模型。
- 文件或目录选择、资产类型、哈希与缺失组件提示。
- 固定演示 preset 一键填入且参数可只读展开。
- 生成数量、尺寸、steps、CFG、seed 模式、negative prompt。
- 本地 / 分布式批次 segmented control；不可用原因直接显示。
- 2x2 固定比例结果网格，单图显示 seed、node、provider、耗时和实际模式。
- 进度来自 `step/total_steps` 或 Workflow 状态，不用假进度条。
- 取消、单图下载、整组 ZIP、复制参数与重新生成。
- 只有完成的远端 attempt 才显示“分布式：是”；一张图本地生成后不能冒充多节点。

### 6.4 本地图像编辑扩展计划

#### 6.4.1 目标、顺序与共同边界

三个方向共用一套输入资产、作业状态、取消和结果协议，但按风险从低到高实施：

| 模式 | 用户输入 | 首选 Diffusers pipeline | 第一阶段定位 |
|---|---|---|---|
| `img2img` | 源图 + prompt + `strength` | `StableDiffusionImg2ImgPipeline` | 先落地；验证输入 blob 和组件复用的最小闭环 |
| `inpaint` | 源图 + mask + prompt + `strength` | `StableDiffusionInpaintPipeline` | 第二步；优先使用 SD 1.5 专用 9-channel inpainting U-Net |
| `instruction` | 源图 + 自然语言编辑指令 | 专用 InstructPix2Pix pipeline，或兼容 SD 1.5 base 的 ControlNet IP2P 组合 | 第三步；以资产能力声明选择 pipeline，不承诺任意指令都能像素级准确执行 |

共同目标：

- 保持现有 `POST /api/diffusion/generate` 文生图调用、preset 和响应结构向后兼容。
- 输入图、mask 和输出图全部通过 blob ID 引用；API、journal、任务图和 Worker 信封不携带图片正文或本机绝对路径。
- 三种模式都使用现有单活动作业、step 边界取消和模型生命周期锁；不得绕开 LLM/SD 显式互斥。
- 一个完成结果可被显式“作为新源图继续编辑”，形成有界版本链；不自动永久保留整条历史。
- 本地能力先验收，再接 SD-N3/N4 的跨进程 blob 和完整 Worker。没有远端完成 attempt 时始终报告 `distributed=false`。

共同非目标：第一阶段不做训练、LoRA 在线合并、多人协同画布、实时逐笔扩散、视频编辑、语义分割服务或单张图跨节点 U-Net 拆分。

#### 6.4.2 API 与 blob 契约

新增输入上传端点，使用受控 multipart，不把 Base64 放进 JSON：

```http
POST /api/diffusion/blobs
Content-Type: multipart/form-data

purpose=input_image|mask
file=<PNG/JPEG/WebP bytes>
```

成功返回现有 descriptor 的兼容超集：

```json
{
  "blob_id": "img_...",
  "purpose": "input_image",
  "content_type": "image/png",
  "size_bytes": 123456,
  "sha256": "...",
  "width": 512,
  "height": 512,
  "created_at": 0,
  "expires_at": 0
}
```

上传时必须按 magic bytes 解码，不信任文件名/MIME/EXIF；应用 EXIF orientation 后重新编码为规范 PNG，剥离路径、账户名和非必要元数据。mask 规范化为与源图同尺寸的 8-bit 灰度 PNG，固定 **白色表示重绘、黑色表示保留**，拒绝空尺寸、超像素、动画、多页和解压炸弹。

统一编辑入口：

```json
POST /api/diffusion/edit
{
  "mode": "img2img|inpaint|instruction",
  "preset_id": "sd15_original_v1",
  "source_blob_id": "img_source",
  "mask_blob_id": null,
  "prompt": "...",
  "negative_prompt": "...",
  "seed": 19950101,
  "width": 512,
  "height": 512,
  "steps": 28,
  "guidance_scale": 7.5,
  "strength": 0.65,
  "instruction": null,
  "edit_adapter_id": null,
  "conditioning_scale": null,
  "image_guidance_scale": null
}
```

校验规则：

- `source_blob_id` 三种模式必填；`mask_blob_id` 仅 `inpaint` 必填，其他模式携带时拒绝而不是忽略。
- `strength` 仅在 pipeline 声明支持时允许，建议 UI 范围 `0.05–1.0`；服务端使用冻结硬上限，不接受 NaN/Inf。较低值偏向保留原图，较高值允许更大改动，UI 不承诺线性效果。
- `instruction` 仅 `instruction` 模式必填；普通 `prompt` 与指令的映射由固定 preset 定义，Worker 不接受任意 Python 类名或远程 config URL。
- 请求中的 width/height 是规范化工作尺寸。默认保持源图宽高比并缩放到受支持的 8 倍数；任何裁剪、填充或拉伸都必须在提交前由前端明确展示，并写入 job 参数。
- `edit_adapter_id` 只能引用本机已登记且 manifest 匹配的 InstructPix2Pix/ControlNet 资产。base、adapter、VAE、scheduler、dtype 和 runtime 共同组成 artifact-set identity。
- 响应继续返回 `202 + job snapshot`；`GET /jobs/{id}` 增加 `mode`、去敏后的输入 blob descriptors、编辑参数和实际 pipeline/adapter identity，不返回图片正文。

为兼容现有调用，输入 blob 与结果 blob 使用同一存储接口和 descriptor，但增加 `purpose`、`owner_scope`、`parent_blob_ids`、`lease_count` 元数据。`GET`/`DELETE` 行为保持一致；删除仍被活动 job 租用的输入时返回冲突或标记延迟删除，不得让运行中的 pipeline 读到半删除数据。

#### 6.4.3 引擎与 pipeline 设计

`SD15Engine` 不变成包含大量模式分支的单函数。新增强类型编辑请求，并由 pipeline factory 按 artifact-set capability 构造执行器：

```text
DiffusionService
  -> TextToImageExecutor       # 现有行为
  -> Img2ImgExecutor           # base components + source image
  -> InpaintExecutor           # dedicated inpaint bundle + source + mask
  -> InstructionEditExecutor  # InstructPix2Pix 或 base + ControlNet IP2P
```

- 图生图优先从已加载 SD 1.5 pipeline 的 `components` 构造 `StableDiffusionImg2ImgPipeline`，避免同一 base 权重重复常驻。组件复用必须有单元测试证明 scheduler/config 不被跨 job 污染。
- 局部重绘正式 preset 使用 SD 1.5 兼容的专用 inpainting checkpoint。普通 4-channel base 的兼容/旧式重绘只可作为标记清楚的实验路径，未过质量门不得自动回退。
- 指令式编辑首先评估专用 InstructPix2Pix full pipeline；已有 `control_v11e_sd15_ip2p_fp16.safetensors` 走“SD 1.5 base + ControlNet IP2P + source image”组合。两者输入字段和 guidance 语义不同，必须由 asset capability 显式区分，不能用一个模糊的 `scale` 猜测。
- pipeline bundle 同时只保留一个活动执行组合。模式切换可复用共享组件，但新增 ControlNet/inpaint U-Net 前先做显存预算；预计超限则在排队前拒绝或要求用户切换已验证 offload profile，不在生成中途偷偷降级。
- 所有 pipeline 都保持 `local_files_only`、侧车延迟导入、Diffusers 进度输出隔离、`torch.inference_mode()` 和 step callback 取消；不得安装或升级全局 Python 解释器。
- source/mask 在进入 pipeline 前完成 RGB/L 规范化并冻结；pipeline 不得原地修改 blob store 中的对象。每次 job 使用独立 generator 和 scheduler 状态，保证相同工件、输入哈希、参数和 seed 可复测。

#### 6.4.4 前端交互

在现有 `DiffusionPanel` 中增加模式 segmented control：文生图 / 图生图 / 局部重绘 / 指令编辑。模式切换只改变相关控件，不卸载当前兼容模型；不支持的模式显示具体缺失资产或显存原因。

- 图生图：支持文件选择、拖放和剪贴板粘贴；显示规范化后的尺寸/裁剪预览；使用 slider 设置 `strength`，保留 seed、steps、CFG 和 negative prompt。
- 局部重绘：在稳定尺寸画布上提供画笔/橡皮、画笔大小、撤销、重做、清空和反转 mask；遮罩叠层颜色只用于显示，提交时转换为黑白 mask。画布必须支持缩放/平移，鼠标和触控坐标都映射到源图像素。
- 指令编辑：以一条主指令为核心，按所选 pipeline 显示 `conditioning_scale` 或 `image_guidance_scale`，不同时暴露无效参数；提供原图/结果并排或拖动对比。
- 通用：提交后锁定本次参数快照，显示真实 step；取消沿用当前 job API。结果提供下载、删除、“作为新源图继续编辑”和复制参数；替换结果时撤销旧 `ObjectURL`，服务端 blob 删除失败不阻断 UI 回收本地 URL。
- 刷新能力或切换模型时保留尚未提交的本地表单，但若 blob 已过期必须提示重新上传，不保留失效 ID 后继续提交。

#### 6.4.5 安全、资源隔离与生命周期

- 输入图可能包含隐私信息。默认只保存在本机受控 blob store，日志和 journal 仅记录 descriptor/hash；不写 EXIF、原始文件名、浏览器路径或图片正文。
- 输入/结果 blob 使用 owner/session scope 和不可猜测 ID；ID 不是授权凭据。远程访问仍须经过现有集群认证，后续跨节点下载使用 attempt 范围短 TTL 签名。
- job 提交时为 source/mask 增加租约；`completed`、`failed`、`cancelled` 后在 finally 中释放。取消后不发布半成品结果，已编码但尚未 commit 的 PNG 立即回收。
- 默认输入与 mask TTL 至少覆盖最长 job 超时和排队窗口；活动租约期间不做容量淘汰。无租约 blob 按 TTL/LRU 清理，版本链只保存父 ID/hash，不阻止父 blob 正常过期。
- 一个侧车仍只并发一个扩散 job。图片预处理可以在受限 CPU executor 中进行，但不能占用文本任务的全局线程池或通过并发预处理绕过图片像素/内存预算。
- 输出继续经过 safety checker 或等价门控。公开服务中安全组件缺失、输入解码异常、mask 不匹配、ControlNet manifest 不一致或显存预算失败都 fail-closed，并返回稳定错误码而非底层 traceback。

#### 6.4.6 分布式接入边界

本地三模式完成前不扩 Worker 协议。进入分布式后，`image_edit` Stage 只携带不可变 descriptor；主节点上传一次 source/mask，各 attempt 按短租约读取，结果仍按 winner/fencing 提交。调度硬约束增加 `pipeline_kinds`、所需 adapter manifest、最大输入像素和预估峰值显存。

同一源图多 seed/多指令可以任务级 fan-out；单张编辑的去噪循环仍不跨节点拆分。远端全部不可用时，只有请求显式允许才回退本地，并在 job 中记录 `distributed=false`、`fallback_reason` 和实际 provider。

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
| `image_edit` | 上述参数 + mode + source/mask blob + edit artifact identity + 模式专用参数 | image blob descriptor + metrics | 声明对应 pipeline capability 的完整 SD Worker |
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

**状态：** `Completed`（2026-08-05；在与 LLM 相同 `transformers 4.47.1` 兼容窗口中完成本地加载/卸载、文生图、step 回调、同一 pipeline 10 轮稳定性、资产登记、异步 job、临时 PNG blob 和三层 HTTP API；自动化覆盖加载失败、并发拒绝、取消/编码竞态与 LLM/SD 双向互斥，真实 inference-svc 路由完成取消及 `SD → Qwen GGUF` 模型切换回归）。

**依赖：** SD-N0。

**工作：** `DiffusionEngine`、本地资产注册、加载/卸载、文生图、step 取消、指标、API 单元测试和真实 GPU API 生命周期验收均已完成。后续新增 UI、DreamBooth 资产或跨节点 Worker 时不得回写改变本阶段已冻结的本地 job/blob 契约。

**完成判据：** RTX 4060 Laptop 或目标 CUDA 设备上用原版 SD 1.5 固定 preset 连续生成 10 轮；无 NaN/黑图/显存持续增长；取消在下一去噪步收敛；切换回 LLM 后显存和模型状态正确。

### SD-N2：90s DreamBooth 演示资产与前端

**状态：** `In Progress`（2026-08-07；本地 Web 工作区、固定 revision/SHA 目录、一键下载、外置目录校验导入、组合 safety checker 和 10 seed 自动门均已接通并完成真实 DreamBooth 实测；双人目视已完成，许可复核和正式离线压缩包尚未完成）。

**依赖：** SD-N1、模型许可/样例人工验收。

**工作：** 固定 `retrovers`/原版 preset、图像生成 UI、结果参数、step 进度、取消、PNG 回收、固定资产获取和外置导入已实现。4.87 GB 组合资产与十种子自动门已实测通过，双人目视已经签字；下一步完成许可证复核和正式离线资产包。发布资产门未完成前不把 SD-N2 标记完成，但不再阻塞已经开始的 SD-N3 协议与数据面工程。

**完成判据：** 固定 revision 和 SHA 可重建；10 seed 质量门通过；原版和 fine-tune 均可选择；无模型时 UI 不假装可用；安装包不内置大权重但能导入外置模型。

### SD-N3：图像 blob 与 Worker v3

**状态：** `In Progress`（2026-08-07；`SD-N3.0/N3.1/N3.2` 已完成，`SD-N3.3` 已完成默认关闭的 TCP 控制面和本机 Worker executor 桥接；尚未形成可执行的分布式图像链路）。

**依赖：** SD-N1、TC-N2.4 状态机/Worker 故障边界保持稳定。

**已完成 SD-N3.0：** 新增独立于本地临时结果 store 的持久化图像 Blob 基础设施：SQLite WAL 元数据、内容寻址不可变对象、owner/父引用、容量和 TTL 回收、顺序分块上传、SHA/MIME/像素校验、attempt-scoped lease、短期 HMAC 传输授权及稳定 artifact manifest。对象文件删除采用事务内持久 GC 队列、提交后回收，避免数据库回滚后元数据指向已删除文件。Task Worker 协议新增显式 v3 图像 schema，限定专用 `pc_diffusion_worker`、`diffusers_sd15`、`image_generate`/`image_edit`/`image_grid`、不可变 Blob descriptor、模式专用参数和结果指标。现有适配器继续固定 v2，`preferred_version=2`，状态端点明确报告 v3 adapter/data plane 未启用。

**已完成 SD-N3.1（数据面与 wire schema）：** 新增不进入 OpenAPI 的内部 `/internal/v1/diffusion/data-plane` 路由。`STATE_DIR/diffusion/distributed_blobs` 只在 32+ byte 集群密钥可用时初始化；否则状态端点明确返回 disabled，绝不退化成无认证传输。下载 URL 绑定 attempt/blob/lease/grant，上传 URL 绑定 attempt/upload session/grant；两者均使用短期 HMAC Bearer grant。上传仅接受有界 octet-stream 分块，支持相同 offset/相同字节的幂等重放和重复 commit 返回同一 descriptor；下载支持有界 range chunk，不暴露本地路径。v3 `stage_offer` 与 `stage_result` 均新增并纳入摘要的 `transfer_plan`：输入由协调器所在节点授权 Worker 拉取，输出由 Worker 所在节点授权协调器拉取，避免在图像生成前伪造未知 SHA/大小的上传会话。

**已完成 SD-N3.1（控制面 library）：** 新增独立 `DiffusionCoordinatorControlPlane` 与 `DiffusionWorkerAdapter`，只接受 v3 image Worker 的 `hello` 协商和 capability snapshot；Worker 仅允许一个活跃 Stage，覆盖 accepted/terminal offer 重放、busy 拒绝、基于本地 monotonic deadline 的 lease-expiry fencing、取消的 first-terminal fencing、重复取消幂等以及执行器异常的无敏感细节错误回执。单进程 fake executor 已验证 `image_generate` 的 hello -> accept -> result 闭环，v2 文本 Worker 回归保持通过。它刻意不注册 TaskGraph Provider、不暴露运行时连接状态，也不在 adapter 内执行 HTTP Blob 拉取/上传或签发 transfer plan；因此 `adapter_connected=false`、生产 task dispatch 禁用和 UI 本地统计保持不变。

**已完成 SD-N3.2（隔离 Provider 与 Blob 回传）：** 新增仅支持 v3 image Worker 的 `RemoteDiffusionProvider`，显式要求 Worker 已广告相同 artifact manifest、外部显式打开 dispatch，且强制以 result ingestor 消费 `stage_result.transfer_plan`；grant 仅在内存中用于下载，绝不进入 `StageResult.metadata` 或 TaskGraph journal。新增标准库 `DiffusionBlobTransferClient`，逐段校验 `Content-Range`、MIME、响应 SHA、总长度和总 SHA 后才写入本地内容寻址 store；Worker 可用 `publish_output()` 生成本节点 output descriptor 与 attempt-scoped download grant。fake transport 已验证 Worker 发布 PNG -> Provider 收到 result -> 内部 HTTP 分段拉取 -> 协调器 store 生成新本地 descriptor，篡改 range 在写入前失败。Worker adapter 同时增加 provider identity、lease renewal replay 与过期 fencing。该闭环仍是同进程测试及 TestClient HTTP 路由，不是 TCP 双进程/双设备验收。

**已完成 SD-N3.3（受控 TCP 与本机 Worker 桥接）：** Scheduler 继续复用已有 `TASK_WORKER` TCP 帧，并按协议版本将 v2 文本 Worker 与 v3 图像 Worker 严格分流。`QLH_DIFFUSION_WORKER_EXPERIMENTAL_ENABLED=false` 时拒绝所有 v3 协商；开启后主节点仅接纳已注册的 PC client 的 `hello`，仍拒绝 Stage 响应且不创建图像 Provider。客户端只有由 API 生命周期显式安装 `DiffusionWorkerAdapter` 后才发送 v3 hello，并只把 `hello_ack`、`stage_offer`、`lease_renew`、`stage_cancel` 交给它；TCP 断开会立刻取消 active stage、清空协商状态，重连才重新 hello。`DiffusionWorkerRuntime` 将已经加载且具有 SHA 的本地 SD 基础工件映射为 artifact manifest；一期只广告/执行 `image_generate`，以本地 `DiffusionService` 的可取消 job 运行，成功 PNG 复制到持久数据面并签发 attempt-scoped 下载 grant，取消时绝不发布输出。此 runtime 还要求配置可被协调器访问的 `QLH_DIFFUSION_WORKER_DATA_PLANE_BASE_URL`，不从 `0.0.0.0` 或猜测的地址推导。Scheduler 不导入或加载 Diffusers，侧车仍与 LLM 生命周期互斥。自动化覆盖默认关闭、注册节点门、同通道 hello、客户端 offer/cancel、断连清理、artifact 不匹配和取消不发布；尚无主节点 Provider、TaskGraph 连接或真实 TCP 双进程证据。

**SD-N3.4 进行中：** 主节点现已能在 data plane 与 result ingestor 均由 API 显式注入后，把已协商 Worker 绑定为 `RemoteDiffusionProvider`；`stage_accept/result/error/cancelled` 由 Scheduler 交给该 Provider，断连会唤醒 pending attempt。`DiffusionCoordinatorRuntime` 使用标准库 HTTP 客户端逐段验证 Worker 输出，完整通过后才写入 coordinator CAS，返回新本地 descriptor；短期 grant 仅存在于 Provider 调用栈，不进入 `StageResult.metadata`。Provider 现在可在 TaskGraph 中按稳定 ID 安全替换，API 已接入默认关闭的单 Stage `image_generate_v1` 实验接口，结果通过 `workflow_id + blob_id` 的持久 owner scope 读取，取消复用 TaskGraph workflow 取消端点；TCP 心跳也会刷新图像控制面的健康时间。除 Scheduler 级 fake 闭环与 API workflow/取消/工件歧义/CAS owner 回归外，本机独立 Worker 子进程也已通过真实 TCP hello -> offer -> result 与真实 FastAPI HTTP 分块下载，协调器最终读取自己 CAS 中的新 PNG。该子进程使用确定性 fake SD executor，只证明传输和状态机边界。**尚未完成项：** 真实 Diffusers/GPU Worker、迟到结果与 journal 恢复、两台物理设备和固定多 seed fan-out/fan-in 仍待验证，因此 SD-N3.4 不能标记完成。下一步把相同双进程门替换为真实 Diffusers smoke，再进行两台物理 PC 验收。`image_grid` 在发往 Worker 前由 Provider 将已完成依赖物化为有序 Blob descriptor，协议本身不内联 TaskGraph dependency output；`image_edit` 保持关闭，待输入 Blob 拉取与各模式 asset 能力单独完成。

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

### SD-N5：本地图像编辑四方向分期

**总依赖：** SD-N2、本地输入 blob 安全路径；分布式支持可在 SD-N3/N4 后接入。SD-N5 各阶段只能逐项推进，不因图生图完成就宣称已支持局部重绘或指令式编辑。

#### SD-N5.0：输入 blob 与编辑公共契约

**实施进度（2026-08-06）：`Completed`。** 单体 API 与 inference-svc 的 multipart 输入 Blob、PNG/JPEG/WebP 解码、EXIF 方向归一化、RGB/灰度 mask 规范化、字节/像素上限、owner/TTL/lease/父引用、重复删除冲突、统一编辑请求模型、稳定错误码和前端客户端契约均已完成。相同 owner 下的输入 Blob 与生成结果 `output` Blob 都可作为后续编辑源；生成和编辑结果继承调用面的 owner scope。活动租约或父引用存在时删除 fail-closed，子结果删除后父引用释放。`img2img`、`reference`、`inpaint` 与 `instruction` 均已进入真实 executor；每种模式按 artifact capability 使用独立参数，不能互相冒充。

**工作：** 增加 multipart 输入上传、图片/mask 规范化、owner/lease/TTL、`POST /api/diffusion/edit` 强类型 schema、编辑 job snapshot 和稳定错误码；保持现有 `/generate` 不变。

**完成判据：** PNG/JPEG/WebP、EXIF orientation、错误 MIME、解压炸弹、像素上限、重复删除、活动租约删除、取消/编码竞态和服务关闭全部自动化通过；job/journal/log 不含图片正文和原始路径。

#### SD-N5.1：图生图

**实施进度（2026-08-06）：`In Progress`，自动门与双人目视完成。** 引擎、服务、单体/inference API、网关和 Web 工作区已接通；结果“继续编辑”的 Blob purpose/owner 契约及回归已修复。真实 inference-svc 已完成原版 SD 1.5 的连续编辑、引用回收、取消和卸载快速门；Edge 150 已完成上传、两轮结果续编和卸载的点击链路。独立质量脚本现在以源图 SHA、preset 参数、完整 seed/strength 矩阵和显存上限共同判定 full gate。原版与 90s 各 30 张的自动完整性、唯一性、安全标记和连续显存门均通过，两份报告的自动门初始状态为 `pending_manual_review`，双人目视签字已于 2026-08-06 完成（Siegfried Kkm./浅草爱音，两份报告均 passed）。

固定源图 SHA-256 为 `a6fd131b5008b77f3c39f01d4a073529cf8225a8a204997d4f01de9217e93264`。复现完整门时分别使用 `--asset-id sd15_original_v1` 与 `--asset-id sd15_90s_retrovers_v1`，并同时传入 `--source-image logs/sd15/sd15_original_v1_seed19950101.png --source-sha256 <上述完整值>`；审核者无需重复推理，可直接登记已有报告：`python scripts/quality_gate_sd15_img2img.py --review-report build/sd15-img2img-quality/full-original/quality-report.json --reviewer Alice=pass --reviewer Bob=pass`。两个不同姓名均通过且无人失败后，报告状态才会变为 `passed`。

**工作：** 接入 `StableDiffusionImg2ImgPipeline`，复用已加载 base components；加入 source image、`strength`、规范化尺寸和结果“继续编辑”入口。

**完成判据：** 原版 SD 1.5 与 90s full fine-tune 各完成固定输入 × 10 seed；低/中/高三个 strength 档均能生成有效 PNG，输入哈希和所有参数进入结果元数据；取消在下一 step 收敛；连续模式切换无显存持续增长，文生图回归不变。

#### SD-N5.1A：IP-Adapter 参考图一致性

**实施进度（2026-08-06）：`In Progress`，完整自动门与双人目视完成。** 首期采用 SD 1.5 兼容 IP-Adapter，把“参考人物/外观”与普通 img2img 的 source latent、ControlNet 的姿势/结构约束、InstructPix2Pix 的自然语言编辑分开。通用 VLM 只保留为后续可选提示词助手，不能替代图像条件或宣称精确身份保持。本地 Safetensors 目录识别、`reference` 请求、按需加载/卸载、适配器 scale、结果元数据、Web 入口、冻结资产、真实 GPU 和双 base 自动质量/显存门均已完成；两份完整报告于 2026-08-06 双人目视签字完成（Siegfried Kkm./浅草爱音，均 status=passed）。

**工作：**

- 冻结一个 SD 1.5 兼容 IP-Adapter revision、逐文件 SHA、许可证和本地离线目录；适配器权重与 CLIP image encoder 必须同时就绪，禁止运行期访问 Hub。
- `reference` 模式使用 `source_blob_id` 作为参考图、`edit_adapter_id` 绑定本地适配器、`ip_adapter_scale` 控制图像条件强度；它走完整文生图去噪步，不复用 img2img 的 `strength` 语义。
- 适配器按需装载到当前 base pipeline；切回 txt2img/img2img 时显式卸载，避免图像条件泄漏到下一任务。base、adapter、image encoder、dtype、runtime 和 scale 全部进入结果身份/元数据。
- 8 GB 首期只开放完成实测的 FP16 profile。8-bit U-Net、QKV fusion、多适配器叠加和 FaceID 在各自兼容/显存门通过前 fail-closed；FaceID 若引入额外人脸编码器，继续放在独立 CUDA 侧车依赖中。
- 第二阶段可组合 ControlNet OpenPose/Depth/Canny，分别约束姿势、几何和轮廓；不得把 IP-Adapter 单独描述为像素级身份锁定。

**已验证证据：** `h94/IP-Adapter@018e402774aeeddd60609b4ecdb7e298259dc729` 以 Apache-2.0 许可冻结，下载集合 2,573,016,776 bytes，包含 `models/ip-adapter_sd15.safetensors` 与完整 ViT-H image encoder；逐文件校验生成稳定 artifact SHA-256 `671c7452e97ce26faaf3e25dbdb11e2d78e3560d1d6fe7b6fbf7a59c3ebe94c4`。真实 RTX 4060 Laptop GPU 上修复了 adapter 动态注册后 image encoder 未进入 model CPU-offload hook 的缺陷；4-step `reference` 回归耗时 4.537 s，峰值 reserved 2,789,212,160 bytes，卸载后 allocated 增量 8,519,680 bytes。固定参考图 `3271580f...fe38f` 下，原版 SD1.5 28 steps 的 36 张均有效，完整门耗时 260.533 s；90s base 40 steps 生成 36 张，其中 35 张有效、1 张被 safety checker 正确替换为黑图，各 scale 有效数 11/12/12，完整门耗时 343.724 s。两轮连续 allocated span 为 0、reserved span 为 2,097,152 bytes，自动门均通过。目视抽查显示三档 scale 均保留金发、眼睛、脸型与蓝色外套等主要要素，`0.8` 更接近参考外观，但发型和表情仍会随 seed 改变。生命周期复核还发现 Diffusers 0.35.2 的 `from_pipe()` 默认会把共享组件转为 FP32；现已显式继承 base dtype，并在动态组件移除前拆除旧 offload hooks。修复后真实 `reference → txt2img → img2img → reference` 耗时依次为 6.985/2.338/2.177/2.991 s，adapter 与 attention slicing 状态按模式正确翻转，峰值 reserved 3,034,578,944 bytes，最终卸载后 allocated/reserved 为 8,519,680/20,971,520 bytes。

**完成判据：** Inspector 能区分完整 IP-Adapter 目录、单独 adapter 权重、缺 image encoder 和伪造 Safetensors；无网络时加载与生成成功且不会回退下载；固定人物参考图 × 原版/90s base × 至少 10 seed × 三档 scale 通过自动完整性、安全、唯一性和显存门，并由两名独立审核者评价人物主要要素保持；切换 `reference → txt2img → img2img → reference` 无条件泄漏、显存持续增长或参数串线；缺 adapter/encoder、错误 base family、QKV/量化未验组合在排队前稳定拒绝。

#### SD-N5.2：局部重绘

**实施进度（2026-08-07）：`Completed`。** 固定并下载 `stable-diffusion-v1-5/stable-diffusion-inpainting@8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` 的 16 文件 FP16 集合，逐权重 SHA 校验后得到稳定 artifact SHA-256 `ddd6d69af8e9324f38074e8624452f49c907cce380add8de2caf97ca0661c605`。Inspector 将其 9-channel U-Net 单独识别为 `sd15_inpaint_pipeline`，普通文生图加载路径明确拒绝；引擎在 balanced CPU-offload profile 下按需加载并复用专用 pipeline，量化/QKV/双常驻未验组合在排队前拒绝。source/mask 双租约、父引用、结果哈希与 `white=redraw, black=preserve` 元数据已贯通单体 API、inference-svc、Nest 网关和 Web 工作区。前端支持鼠标/触控画笔、橡皮、大小、撤销/重做、清空、反转、缩放/平移和黑白 PNG 上传。

**已验证证据：** 固定源图 SHA-256 `a6fd131b5008b77f3c39f01d4a073529cf8225a8a204997d4f01de9217e93264` 的 10 seed × 20 steps 完整门通过，mask 序列为 3 次全黑、4 次局部白、3 次全白。全黑整图 MAE 4.16–4.98；局部白外侧 MAE 4.59–5.03、内侧相对外侧增加 20.16–30.01；全白整图 MAE 82.04–85.57。10 张结果均唯一、0 safety flag；连续 allocated span 为 0，峰值 reserved 3,649,044,480 bytes，卸载后 allocated/reserved 为 8,519,680/20,971,520 bytes。真实 Edge 150 完成“选择 inpaint → 上传源图 → 画布绘制 → mask multipart 上传 → 4-step inpaint → PNG 展示”，引擎耗时 4.683 s、浏览器链路 7.015 s。报告位于 `build/sd15-inpaint-quality/full-original/quality-report.json` 与 `build/sd15-browser-e2e/inpaint-browser-report.json`（构建证据目录不进入 Git）。

**工作：** 登记并加载固定 revision/SHA 的 SD 1.5 inpainting 资产；接入 source + mask；前端完成触控/鼠标遮罩画布和黑白 mask 导出。

**完成判据：** mask 尺寸/方向/黑白语义稳定；全黑 mask 基本保留、局部白 mask 只开放目标区域、全白 mask 按重绘处理；至少 10 轮无资源泄漏；专用 inpaint 资产缺失时执行前拒绝，不静默改走普通 img2img。

#### SD-N5.3：指令式编辑

**实施进度（2026-08-07）：`In Progress`，实现、完整自动门、Edge 链路与双人目视均已完成。** 对固定 revision 的完整 InstructPix2Pix 与 ControlNet IP2P 做了同源图三指令真实对照。完整 InstructPix2Pix 在局部颜色、季节和水彩风格三类指令下更稳定地保留窗框、床和视角；ControlNet IP2P 在季节/风格指令下发生较大场景重构，且首轮峰值 reserved 约 3.39 GiB，高于完整 pipeline 的约 2.86 GiB。因此默认路线冻结为 `timbrooks/instruct-pix2pix@31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`，ControlNet IP2P 保留为未暴露的实验候选。

固定 InstructPix2Pix 下载集合为 2,742,242,939 bytes，稳定 artifact SHA-256 为 `a6626f7fedd356273f726b1707293266f11f6548a57730785ccbffe8efc872ab`。Inspector 使用 `sd15_instruction_pipeline` 独立分类；引擎按需加载并复用专用 pipeline，切换 inpaint 时释放另一条专用 pipeline，量化/QKV/非 offload CUDA 组合在排队前拒绝。服务/API/网关/Web 已贯通 source blob、instruction、`image_guidance_scale`、artifact identity、取消和结果元数据；ControlNet 的 `conditioning_scale` 不能传给默认 pipeline。

固定源图 SHA-256 `a6fd131b...e93264` 的 10 条指令 × 20 steps 完整自动门 10/10 通过：10 张结果均唯一、0 safety flag、MAE 27.38–102.13、连续 allocated span 为 0，峰值 reserved 3,682,598,912 bytes，卸载后 allocated/reserved 为 8,519,680/20,971,520 bytes，总耗时 94.02 s。Edge 150 完成“选择指令编辑 → 上传源图 → 选择专用 pipeline → 提交独立 instruction/guidance → PNG 展示”，最新 4-step 引擎/浏览器链路耗时 5.81/7.64 s。报告位于 `build/sd15-instruction-quality/full-original/quality-report.json` 与 `build/sd15-browser-e2e/instruction-browser-report.json`；自动报告经两名独立审核者签字后状态为 `passed`（Siegfried Kkm./浅草爱音，2026-08-07 双人目视签字；构建证据目录不进入 Git，本段为签字事实源）。

2026-08-07 收尾回归又完成一次真实 `instruction -> txt2img -> unload` 切换：基础模型加载 8.166 s，4-step 指令编辑 5.832 s，切回 4-step 文生图 9.752 s；文生图开始前专用 instruction pipeline 已释放，峰值 reserved 3,682,598,912 bytes，卸载后 allocated/reserved 8,519,680/20,971,520 bytes。Web 能力判断改为以后端实际 `engine_config` 为准，加载期间锁定 profile 下拉框，避免界面显示 balanced 却由后端 resident/QKV 配置执行。

**工作：** 对专用 InstructPix2Pix 与 `control_v11e_sd15_ip2p_fp16.safetensors` 两条兼容路线做真实 GPU/质量/显存对照；冻结一条默认 preset，另一条保留为可选能力。UI 只展示所选 pipeline 有效的 guidance 参数。

**完成判据：** base、adapter/full pipeline、输入哈希、指令、seed 和 guidance 全部进入 artifact manifest/结果元数据；缺 base/config/input/adapter 时执行前拒绝；固定 10 条编辑指令由自动完整性门 + 双人目视门通过；不支持的指令有明确能力说明，不用普通 img2img 冒充成功理解。

#### SD-N5.4：编辑任务分布式接入

**依赖：** SD-N3、SD-N4，以及对应本地阶段完成。

**工作：** 把已验收模式加入 `image_edit` Stage 和 Worker capability；输入 blob 只上传一次，attempt 通过 descriptor/短租约复用；纳入 reservation、lease、winner、fencing、取消和实际节点统计。

**完成判据：** 至少两个物理 PC 对同一输入完成多 seed 或多指令 fan-out；断传、迟到结果、节点掉线和用户取消无假成功、重复 blob 或租约泄漏；单张编辑不宣称跨机加速。

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
| Blob | 输入/结果 purpose、MIME、magic bytes、EXIF orientation、尺寸炸弹、像素上限、SHA、分块中断、owner、活动 lease、TTL、并发读写、journal 不含正文 |
| 图生图 | source 必填、strength 边界、尺寸规范化、组件复用、相同输入/seed 可复测、结果继续编辑 |
| IP-Adapter | 完整目录/缺 image encoder 识别、adapter id/scale 边界、离线加载、按需卸载、模式切换不串条件、未验 profile 拒绝 |
| 局部重绘 | mask 必填、尺寸一致、黑白语义、空/全黑/全白 mask、触控坐标映射、专用模型缺失拒绝 |
| 指令编辑 | pipeline capability、adapter manifest、guidance 字段互斥、缺组件拒绝、禁止静默退化为普通 img2img |
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
| SD 1.5 img2img | 原版/90s 各固定输入 × 10 seed、三个 strength 档、连续切换 |
| SD 1.5 IP-Adapter | 固定人物参考图、原版/90s base、三档 scale、10 seed、模式切换、8 GB 显存与双人目视一致性门 |
| SD 1.5 inpainting | 专用 inpaint 工件、三类 mask、边缘一致性、错误尺寸/方向 |
| InstructPix2Pix / ControlNet IP2P | 固定指令集、错误 base/config/adapter、输入复用、自动门 + 双人目视 |
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
- 用户上传的源图和 mask 视为敏感本地数据；剥离 EXIF/文件名，日志和 journal 只保存 blob descriptor/hash，默认不发送到未被本次请求授权的远端 Worker。
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
| 图生图 strength 被误解为精确相似度 | UI 解释为去噪强度并提供预览；质量门覆盖低/中/高档，不承诺线性对应 |
| mask 方向、尺寸或颜色语义错误 | 上传时规范化并固定白改黑留；前后端共用 fixture，任何隐式 resize/crop 都写入参数 |
| 普通图生图冒充指令理解 | pipeline capability 硬约束；无专用 InstructPix2Pix/ControlNet 资产时拒绝或显式降级，不改变模式标签 |
| 输入图片隐私或 blob 泄漏 | owner scope、短 TTL、活动 lease、finally 释放、日志去敏和跨节点请求级授权 |
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
- 文生图继续使用 `/api/diffusion/generate`；三类编辑统一使用 `/api/diffusion/edit`，输入通过 multipart blob 上传，不在 JSON 中内嵌 Base64。
- mask 固定白色表示重绘、黑色表示保留；局部重绘正式路径优先专用 SD 1.5 inpainting 工件。
- 指令式编辑必须由专用 InstructPix2Pix 或明确的 base + ControlNet IP2P artifact-set 提供，不用普通 img2img 冒充。
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
- 指令式编辑默认采用固定 revision 的完整 InstructPix2Pix；base + ControlNet IP2P 仅保留为实验候选，重新进入产品入口前需独立质量/显存门。
- 输入 blob 的 owner/session 映射、默认 TTL 和版本链最大保留深度。

---

## 13. 调研来源

- [Diffusers：Distributed inference](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference)
- [Diffusers：Single files](https://huggingface.co/docs/diffusers/main/en/api/loaders/single_file)
- [Diffusers：Pipeline loading and device placement](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading)
- [Diffusers：Image-to-image](https://huggingface.co/docs/diffusers/main/en/using-diffusers/img2img)
- [Diffusers：Inpainting](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint)
- [Diffusers：InstructPix2Pix](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix)
- [SD 1.5 当前 Hugging Face 镜像](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet v1.1 IP2P 模型卡](https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p)
- [90style Anime Face DreamBooth 候选](https://huggingface.co/Aleksandra11/90style_anime_face_model)
- [Classic Animation Diffusion 备选](https://huggingface.co/nitrosocke/classic-anim-diffusion)
- [DistriFusion 论文](https://arxiv.org/abs/2402.19481)
- [xDiT 项目](https://github.com/xdit-project/xDiT)

外部仓库、默认分支、依赖和许可证会变化。真正实施时必须重新查询并把 revision、许可证副本、文件清单与 SHA 固定到本地 manifest，本文记录的 2026-07-30 调研快照不能代替发布审计。
