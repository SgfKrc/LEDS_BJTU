# QLH — Lightweight Distributed Edge LLM Inference

> **Language**: [English](README.en.md) · [简体中文](../README.md)

**A multi-engine, evolvable distributed LLM inference system for heterogeneous edge devices.**

Model quantization · Operator fusion · Paged KV cache · Graph-algorithm orchestration · Multi-terminal collaborative inference · Visual monitoring · External-compute assistance

**v0.1.8.2** (updated 2026-08-12)

> 📌 Scheduling & lifecycle: **[Overall Next-Step Plan](../docs/总体下一步计划.md)** (Chinese); capability snapshot: **[Progress & Next Steps](../docs/项目进展与下一步计划.md)** (Chinese).
> This README describes **implemented** capabilities; items marked *PoC* are disabled by default and are not production capabilities — see the dedicated plans for boundaries.
> Scope: capability overview, quick start and documentation index; the authoritative capability boundary lives in the specialized plans, source code and tests. Specialized documentation is currently in Chinese.

---

## 📋 Project Introduction

QLH targets heterogeneous edge devices — Windows/Linux desktops, workstations, servers, laptops, Android phones and tablets. The PC PyTorch pipeline can split compatible models across nodes by Transformer layers; the PC/Android llama.cpp path runs full local GGUF inference. The system keeps INT4/INT8 quantization, operator fusion, paged KV cache, graph orchestration and graceful degradation, and plans full inference workers, task chains, standalone tensor parallelism and GGUF stages.

Coverage: **Windows PC + Linux PC + Android**. A device type is not sufficient for scheduling — eligibility depends on engine, model format, model fingerprint, available memory, accelerator and network topology.

### Software Editions (four tiers)

| Tier | Edition | Target devices | Core capabilities | Excludes / not recommended |
|------|----------|----------------|-------------------|----------------------------|
| 1 | **PC iGPU** | Windows / Linux PCs without NVIDIA dGPU | llama.cpp + GGUF CPU/iGPU inference, cluster join, remote request forwarding | PyTorch layer pipeline, heavy-model experiments, CUDA-only features |
| 2 | **PC dGPU** | Windows / Linux NVIDIA GPU master/experiment PCs | PyTorch + CUDA + bitsandbytes with CPU fallback, **SD 1.5 image sidecar** (txt2img/img2img/reference/inpaint/instruction), future multi-model & heavy-model experiments | Android minimal strategy |
| 3 | **Android standard** | Android phones/tablets | Full-on mode local GGUF inference, thin mode forwarding to PC, SAF model directory, full settings | Transformer layer splitting, heavy-model experiments |
| 4 | **Android lite** | Lightweight phone entry | Minimal chat, smallest APK/cache/model footprint, single recommended small/INT4 model | Full model directory |

### Core Features

| Feature | Description |
|------|------|
| 🧠 **Intelligent orchestration** | PyTorch layer pipeline assigns contiguous layer segments by compute/memory/network; large topologies use max-bandwidth spanning tree + DFS → [distributed resource scheduling](分布式资源调度系统.md) |
| 🔗 **PyTorch layer pipeline** | Compatible Safetensors models split by contiguous layers; hidden states passed node-to-node with KV cache incremental decoding |
| 🔄 **Dual engine** | PyTorch + bitsandbytes (CUDA) / llama.cpp + GGUF (CPU/iGPU), automatic switching |
| 📋 **MLFQ queue** | Three-level feedback queue: short-interaction priority + aging anti-starvation + FIFO compatibility → [scheduling doc](分布式资源调度系统.md) |
| 🗄️ **Local facts source** | Sessions/settings/model registry on the master-node SQLite (remote PostgreSQL retired); offline-safe |
| 🌐 **Tailscale networking** | Cross-subnet device interconnect, guided first-join |
| 📦 **Installers** | PC iGPU (~180 MB) / PC dGPU (~1.7 GB) / Linux .deb (~200 MB) / Android APK |
| 🎛️ **Admin panel** | Node register/deregister, layer overrides, role transfer, spare master, TCP status |
| 🖥️ **TUI** | Terminal admin menu, zero-dependency stdlib; `bjtu chat` built-in Textual chat page |
| 🎨 **SD 1.5 image gen** *(dGPU)* | Local workspace: txt2img, img2img, IP-Adapter, inpaint, InstructPix2Pix; offline asset packages ready (5 assets, 15 GB, offline-reproducible) |
| 📱 **Android client** | Standard: full-on (local GGUF) / thin (forward to PC cluster); Material 3 UI |
| 🏝️ **TP island** *(PoC)* | Out-of-cluster homogeneous GPU tensor-parallel subcluster (vLLM/SGLang/llama.cpp rpc) as one logical node → [guide](TP孤岛接入指南.md) |
| ☁️ **External provider** *(PoC)* | Route whole requests to OpenAI-compatible endpoints outside the cluster; **data scope defaults to deny** → [guide](外部推理服务Provider接入指南.md) |
| 🎯 **Speculative decoding** *(experiment)* | Local small draft + external verify; disabled by default, not wired into production decoding → [notes](投机解码外部辅助实施说明.md) |

### Project Design Philosophy

- **No third-party service dependency**: apart from GitHub official (source hosting and releases), QLH depends on no third-party service — update sources can be self-hosted signed stations; mirrors (e.g. Gitee) are bandwidth/availability fallbacks only; credentials and authorization data live only on the user's master node.
- **All assets belong to the user**: model artifacts, tensor-parallel external assets (vLLM/SGLang/llama.cpp rpc, etc.), API keys and credentials are **not provided by the development team** — you bring your own; the project only provides import, verification, registration and usage channels.
- **Data stays in-cluster**: external inference/image assistance defaults to `deny` scope; offline asset packages and signed manifests keep distribution auditable and rebuildable.

**Use cases**: smart terminals · IoT · edge computing · education & research

---

## 🌐 Tailscale Networking (Important)

The distributed inference mode relies on **Tailscale** to interconnect devices across subnets. All nodes participating in inference (PC, Android) are advised to install Tailscale first and join the same network.

### Installing Tailscale

**PC** (Windows / macOS / Linux):

> 🔗 https://tailscale.com/download

After installation, logging in with the same account automatically forms the network.

**Android**:

> 🔗 Search "Tailscale" on Google Play to install, or sideload it from APK Mirror

**Verifying the network**:

Open the Tailscale admin console at https://login.tailscale.com/admin/machines and confirm all nodes are online and have been assigned a `100.x.x.x` address.

### Why Tailscale?

- Campus/home networks usually do not assign public IPs, so devices cannot reach each other directly
- Tailscale builds a virtual LAN on top of WireGuard, giving each device a fixed `100.x.x.x` address
- The Windows packaged launcher automatically checks whether Tailscale is installed and logged in

> The campus network is currently observed to block UDP in real-world tests, so Tailscale cannot establish direct connections or relay via the self-hosted DERP over HTTPS/TCP 443. The self-hosted DERP has resolved basic reachability, but path observation, backup relays, the WSS data plane connecting directly to the primary node, and chunked resumption are still being planned/implemented; see the [Anti-Weak-Network Communication Protocol Plan](抗弱网通信协议专项计划.md) for diagnostic boundaries and the phased plan.

---

## 🏗️ Project Architecture

```
Project root
├── docs/                          # Project documentation
│   ├── 项目技术说明.md              # Newcomer entry: KV, fusion, quantization, distributed, scheduling & protocols
│   ├── 整体架构.md                 # Project overview, device scope, current execution paths
│   ├── 核心技术原理.md              # Multi-engine, quantization, KV cache & distributed-mode boundaries
│   ├── 模块接口说明.md              # Current module responsibilities (source code is authoritative for interfaces)
│   ├── 测试与评判标准.md            # Acceptance criteria for standalone and distributed execution
│   ├── 文档状态与清理清单.md         # Document status definitions & maintenance rules
│   ├── 图算法.md                   # Topology-path algorithms for the PyTorch layer pipeline
│   ├── 分布式资源调度系统.md          # MLFQ three-level feedback queue + graph-algorithm layer orchestration
│   ├── 分布式推理流水线实施计划.md    # Chain topology, LAYER_FORWARD protocol, KV cache plan
│   ├── 混合分布式推理体系规划.md      # Inter-layer, task-chain, tensor-parallel & GGUF stage multi-provider system
│   ├── 三种分布式拆分细化实施方案.md  # Inter-layer pending tests, task chain & tensor-parallel implementation
│   ├── Android版本远期计划.md       # Android plan evaluation
│   ├── Android SAF模型存储方案.md   # Android SAF external model directory plan
│   ├── 总体下一步计划.md             # ★ Sole master plan: L0-L5, lifecycle, dependencies, release gates
│   ├── 项目进展与下一步计划.md       # ★ Capability & evidence snapshot
│   ├── 张量并行外部辅助与混合拆分调研方案.md  # ★ Quantitative argument that in-mesh TP is infeasible + three external routes
│   ├── TP孤岛接入指南.md            # ★ Route A: island = single logical high-compute node (PoC)
│   ├── 外部推理服务Provider接入指南.md # ★ Route B: whole-request external routing + data-scope gating (PoC)
│   └── 投机解码外部辅助实施说明.md   # ★ Route C: draft-verify (experimental, disabled by default)
├── src/                           # Python source (PC side)
│   ├── config.py                  # Global configuration (network/model/KV/layering/run-mode/graph thresholds)
│   ├── model_module.py            # Model loading, quantization, operator fusion, layer split, forward inference
│   ├── llama_engine.py            # llama.cpp engine wrapper (CPU/iGPU GGUF inference)
│   ├── island_engine.py           # ★ TP island engine (OpenAI-compatible endpoint → single logical node, Route A)
│   ├── external_provider.py       # ★ External inference provider + data-scope gating (Route B)
│   ├── speculative.py             # ★ draft-verify speculative decoding (experimental, disabled, Route C)
│   ├── tui_admin.py               # ★ Cross-platform TUI admin menu (pure stdlib, zero dependencies)
│   ├── tui_chat.py                # ★ T9 chat page (Textual + httpx; bundled in installers, optional in source)
│   ├── tui_sse.py / tui_shared.py # T9 SSE incremental parser & shared layer
│   ├── paged_kv_cache.py          # Lightweight paged KV cache (memory page management, dynamic allocation)
│   ├── tcp_comm.py                # TCP master/worker communication (long-lived conns, heartbeats, framing, tensor serialization)
│   ├── scheduler.py               # Task scheduling (node management, layer assignment, pipeline control, request queue)
│   ├── graph_orchestrator.py      # ★ Graph-algorithm orchestration (max-bandwidth spanning tree + DFS)
│   ├── device_profiler.py         # Device profiling (CPU/GPU/RAM/network)
│   ├── api_server.py              # FastAPI server (REST API + WebSocket)
│   ├── local_store.py             # Primary-node SQLite local storage (one-time legacy JSON import)
│   ├── model_downloader.py        # Model download guidance (HuggingFace/ModelScope/Baidu Netdisk)
│   ├── model_host.py              # Model lifecycle host (manager-held, LLM/SD mutex)
│   ├── email_notifier.py          # SMTP alerts + IMAP voting (recipient configurable via node_config)
│   ├── scheduler_svc_http.py      # scheduler-svc HTTP shell (contract passthrough)
│   ├── diffusion/                 # ★ SD 1.5 sidecar (engine/assets/service, separate CUDA venv)
│   ├── inference_service/         # ★ inference-svc (engine_host/protocol/routes)
│   └── node_config.py             # Local node configuration (cluster secret/profile, not source-controlled)
├── control/                       # ★ control-svc (NestJS: 9 control-plane domains + SQLite local facts source)
├── gateway/                       # ★ api-gateway (NestJS + Fastify, 96+ endpoint passthrough)
├── schemas/                       # ★ MODEL-FLEET frozen contracts (artifact/pull-job/deployment/profile JSON Schema)
├── fixtures/                      # Test & walkthrough fixtures (SD SSE event streams, model-gate samples)
├── android/                       # Android client (Kotlin + Jetpack Compose)
│   ├── app/
│   │   ├── build.gradle.kts       # Gradle build script (release signing config)
│   │   └── src/main/java/com/qlh/inference/
│   │       ├── data/              # Room database + DataStore settings persistence
│   │       ├── network/           # OkHttp API client + ChatRepository
│   │       ├── service/           # InferenceService foreground service + ModelManager + LocalInferenceEngine
│   │       └── ui/                # ChatScreen / SettingsScreen / SessionListScreen
│   ├── keystore.properties        # Release signing config (git-ignored, generate locally)
│   ├── qlh-release.jks            # Release signing keystore (git-ignored)
│   └── gradlew / gradlew.bat      # Gradle Wrapper (no Android Studio required)
├── .venv-packaging/               # iGPU packaging venv (CPU torch + PyInstaller)
├── .venv-packaging-cuda/          # dGPU packaging venv (CUDA torch + PyInstaller)
├── packaging/                     # Packaging config + distribution server (no build artifacts)
│   ├── launcher.py                # Main-app launch payload (Tailscale → model check → engine selection → start)
│   ├── qlh_launcher.py            # ★ Standalone Bootstrap (GUI/TUI/update, no inference deps)
│   ├── updater.py                 # Update CLI
│   ├── update_core.py             # Manifest, version, download & SHA-256 core
│   ├── qlh-launcher.spec          # Standalone Launcher PyInstaller spec
│   ├── setup-launcher.iss         # Standalone Launcher Setup
│   ├── serve.py                   # ★ Minimal HTTP file distribution server (PC + Android + Linux installers)
│   ├── qlh-cpu.spec               # PyInstaller spec (iGPU edition)
│   ├── qlh-cuda.spec              # PyInstaller spec (dGPU edition, CUDA + CPU fallback)
│   ├── qlh-tui-chat.spec          # Textual chat companion console (bundled)
│   ├── setup.iss                  # Inno Setup script, iGPU edition
│   ├── setup-cuda.iss             # Inno Setup script, dGPU edition
│   ├── requirements-cpu.txt       # CPU-only dependency list
│   ├── linux/                     # Linux .deb packaging
│   │   ├── build-deb.sh           # deb build script
│   │   ├── launcher.py            # Linux cross-platform launcher
│   │   ├── control-cpu / control-cuda  # dpkg metadata
│   │   ├── postinst / prerm / postrm   # install/uninstall scripts
│   │   ├── qlh-edge-inference.service  # systemd unit
│   │   └── qlh-edge-inference.desktop  # desktop entry
│   ├── dist/                      # ★ Final installer output (git-ignored)
│   └── README.md                  # Packaging docs
├── frontend/                      # React frontend (Vite + FastAPI backend proxy)
│   └── src/
│       ├── App.jsx                # Main layout & settings state
│       ├── api/client.js          # API client wrapper
│       └── components/            # ChatPanel / AdminPanel / DevicePanel / SettingsModal etc.
├── tests/                         # Unit tests (2026-08-12 full run: 1994 passed / 8 skipped)
├── scripts/                       # Utility scripts
│   ├── quantize_model.py          # Model preparation & quantization verification
│   ├── benchmark_all.py           # Full quantization-tier benchmark
│   ├── benchmark_compile.py       # torch.compile fusion test
│   └── convert_to_gguf.py         # Safetensors → GGUF conversion
├── models/                        # Model files (download yourself)
│   ├── qwen-1_8b-chat/            # PC: Safetensors format
│   └── qwen-1_8b-chat-Q4_K_M.gguf # PC: GGUF format (llama.cpp engine)
├── logs/                          # Runtime logs
├── requirements.txt               # Python dependency list
└── README.md                      # This file
```

### PyTorch Layer Pipeline Example (device count is not fixed)

```
User input → Master → TCP → Worker 1 (Client) → TCP → Worker 2 (Client) → Result return
             Embed + L0-3        L4-14             L15-23 + LM Head
             The dGPU master participates in the first segment's compute; it no longer only coordinates
```

### Android's Two Current Modes

```
┌──────────────────────────────┬──────────────────────────────┐
│ Local mode (UI: Full mode)   │ Remote mode (UI: Thin mode)   │
│                              │                              │
│  Android local llama.cpp     │  Android chat UI             │
│  GGUF Q4_K_M (~1.16 GB)      │  HTTP → PC master node       │
│  Offline-capable             │  PC cluster distributed inference │
└──────────────────────────────┴──────────────────────────────┘
```

### Software Layering

| Layer | Function | Technology |
|------|------|------|
| Application | Visual interaction & node management & performance monitoring | React + TUI (standard library) + Jetpack Compose (Android) |
| Scheduling | Task scheduling, instruction dispatch, state management, request queue | Python threading + graph algorithms |
| Communication | Long-lived TCP connections, packet de-framing, heartbeats, tensor serialization | Python socket + struct |
| Inference | Multi-engine: model loading, quantization, fusion, KV cache | PyTorch (CUDA) / llama.cpp (CPU / Android) / island *(PoC)* |
| External assistance *(PoC)* | Whole-request external routing and data-scope gating, speculative-decoding verification | OpenAI-compatible HTTP (vLLM / SGLang, etc.) |
| Storage | Conversation persistence, node registration, configuration management | Primary-node SQLite (shared by Python/Node) + Room (Android) |
| Foundation | Runtime environment | Python / CUDA / bitsandbytes / llama.cpp |

---

## 📦 Environment Dependencies

### Core Frameworks

| Dependency | Version Requirement | Notes |
|------|----------|------|
| Python | ≥ 3.10 | Dev environment 3.12.10; source verified to parse on 3.10 / 3.11 / 3.12 |
| PyTorch | ≥ 2.2.0 | CUDA build for discrete GPUs; CPU build for integrated graphics |
| **transformers** | **≥ 4.45, < 5.0** | ⚠️ Must stay on 4.x! 5.x removed `load_in_4bit`/`load_in_8bit` |
| accelerate | ≥ 1.0.0 | Model loading acceleration (bitsandbytes dependency) |

### Model Quantization

| Dependency | Version Requirement | Notes |
|------|----------|------|
| bitsandbytes | ≥ 0.45.0 | INT4/INT8 quantization (required for discrete GPUs, optional for integrated graphics) |

### CPU / Integrated-GPU Inference Engine

| Dependency | Version Requirement | Notes |
|------|----------|------|
| llama-cpp-python | ≥ 0.3.0 | CPU-optimized GGUF inference, 3-5x faster than PyTorch on CPU |

### SD 1.5 Image Sidecar (Optional for Discrete-GPU Build)

| Dependency | Version Requirement | Notes |
|------|----------|------|
| diffusers | 0.35.2 (pinned) | Image workspace pipeline; 0.38+ requires DINOv2 config, outside the compatibility window |
| transformers | 4.47.1 (pinned) | Same library as the LLM side but in a separate CUDA venv (`packaging/requirements-sd15.txt`) |

> The SD sidecar lives in a separate CUDA venv (on the `.venv-packaging-cuda` side) and never imports into or upgrades the global interpreter; `torch.compile`/Inductor are explicitly rejected because Triton is unavailable.

### Web Visualization

| Dependency | Version Requirement | Notes |
|------|----------|------|
| fastapi | ≥ 0.110.0 | API backend framework |
| uvicorn[standard] | ≥ 0.29.0 | ASGI server |
| pywebview | ≥ 5.0 | Native window for packaged builds (replaces the browser) |
| python-multipart | ≥ 0.0.12 | File upload support |

### Database

> Remote PostgreSQL has been retired (M1.3, 2026-08-10): production runtime no longer connects to or packages the PG driver; data is carried by the primary node's SQLite. psycopg2 is installed on demand only when needed for historical migration audits.

### Networking (Required for Distributed Mode)

| Dependency | Version Requirement | Notes |
|------|----------|------|
| **Tailscale** | Latest | Cross-subnet virtual networking; must be installed on every distributed node |

> 🔗 Download: https://tailscale.com/download

### Tools

| Dependency | Version Requirement | Notes |
|------|----------|------|
| tqdm | ≥ 4.65.0 | Progress bars |
| psutil | ≥ 5.9.0 | System resource monitoring |

### Frontend

| Dependency | Version Requirement | Notes |
|------|----------|------|
| Node.js | ≥ 18 | Frontend build |
| npm | — | Package manager |

### Android Client

| Dependency | Version Requirement | Notes |
|------|----------|------|
| Android SDK | API 34+ | Compile target |
| Gradle | 8.11+ | Wrapper bundled; no separate installation needed |
| Kotlin | 2.1.0 | Downloaded automatically via Gradle |
| Java | JDK 17 | Required for compilation |

> The Android client does **not** need Android Studio; a JDK plus the Android SDK command-line tools is enough to build via `gradlew.bat`.

### One-Click Installation

```bash
# Python dependencies (primary node is self-contained on SQLite; no PostgreSQL needed)
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

---

## 🤖 Model Download

> **Default source**: the current control-svc has the Hugging Face official source built in and enabled, and also registers HF mirror and ModelScope endpoint descriptions (the latter two are disabled by default, pending the corresponding adapters / real-network acceptance); source priority, enable/disable and `credential_ref` are supported. Windows tokens are protected by the current user's DPAPI; the model proxy follows `QLH_HTTP_PROXY > user-persisted config > direct connection` and can be set or cleared via the local `/models/network/proxy` API without modifying the system proxy. Gated repositories require registered credentials and explicit license acceptance first; plaintext never enters SQLite/jobs/manifests/responses. See [Special Plan](一键模型部署与自治集群远期计划.md) §4.2/§7.1 for the mechanism.

The project's default example model is **Qwen-1.8B-Chat**, and the model registry provides additional Qwen/DeepSeek experimental slots. The following covers only the two formats of the default model and does not imply the system supports only that model:

| Format | Engine | Size | Use Cases |
|------|------|------|---------|
| **Safetensors** | PyTorch (CUDA) | ~3.5 GB | Discrete-GPU inference, distributed pipeline |
| **GGUF Q4_K_M** | llama.cpp (CPU / Android) | ~1.16 GB | Integrated-GPU/CPU, single-machine inference, Android local inference |

### Safetensors Format (PyTorch / Distributed)

**Option 1: ModelScope (recommended, faster in China)**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
```

**Option 2: Hugging Face**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen-1.8B-Chat --local-dir models/qwen-1_8b-chat
```

**Option 3: Baidu Netdisk**

> 🔗 https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3 | Extraction code: vtp3

### GGUF Format (llama.cpp / PC CPU Engine)

```bash
# Download the recommended Q4_K_M (~1.16 GB)
huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/
```

| Quantization | Size | Notes |
|------|------|------|
| Q3_K_M | ~0.94 GB | Experimental tier; for 14B+ capacity validation or small-model pipeline smoke, not recommended for daily small-model use |
| **Q4_K_M** ⭐ | **~1.16 GB** | **Recommended — best speed/quality balance** |
| Q5_K_M | ~1.31 GB | Higher quality |
| Q8_0 | ~1.82 GB | Near-lossless |

### SD 1.5 Image Models (Optional, Discrete-GPU Build)

The image workspace uses local Diffusers assets pinned at fixed revisions (fully offline inference, no Hub access):

| Asset | Pinned Source | Size | Purpose |
|------|----------|------|------|
| **Original SD 1.5** | `stable-diffusion-v1-5@451f4fe1…` | ~2.74 GB snapshot | Text-to-image / image-to-image baseline (CreativeML OpenRAIL-M) |
| **90s DreamBooth** | `aa8a082c…` (with original safety checker) | ~4.87 GB pinned set | 1990s retro anime preset (openrail, dual-reviewer visual pass) |
| **IP-Adapter reference** | `h94/IP-Adapter@018e4027…` (stable SHA `671c7452…`) | ~2.57 GB | Reference-image consistency (keeps key character elements, not exact identity lock) |
| **SD 1.5 Inpainting** | `stable-diffusion-inpainting@8a4288a7…` (stable SHA `ddd6d69a…`) | ~2.74 GB | 9-channel U-Net local repainting (white mask repaints, black preserves) |
| **InstructPix2Pix** | `timbrooks/instruct-pix2pix@31519b5c…` (stable SHA `a6626f7f…`) | ~2.74 GB | Natural-language instruction editing (MIT; auto gates, Edge pipeline and dual-reviewer visual pass all cleared; offline asset packages released, 5 assets 15 GB, offline-importable) |

Download and verification:

```bash
# One-click download (pinned revision + per-file SHA verification + manifest)
python scripts/download_sd15.py --asset-id sd15_90s_retrovers_v1 --accept-license
# Ten-seed automatic quality gate (black/low-entropy/corrupt/duplicate rejection + dual-reviewer registration)
python scripts/quality_gate_sd15.py --asset-id sd15_90s_retrovers_v1
# img2img / IP-Adapter full matrix gates (source SHA + strength/scale matrix + VRAM gate)
python scripts/quality_gate_sd15_img2img.py --review-report build/sd15-img2img-quality/full-90s/quality-report.json --reviewer 审核者=pass
python scripts/quality_gate_sd15_ip_adapter.py --review-report build/sd15-ip-adapter-quality/sd15_90s_retrovers_v1-v2/quality-report.json --reviewer 审核者=pass
# InstructPix2Pix ten fixed-instruction gate; two independent reviewers required after the auto gate
python scripts/quality_gate_sd15_instruction.py
```

You can also download/import directly in the Web image workspace (asset directory refresh auto-discovers new assets). License and gated status are shown before download; official offline asset packages (5 assets, 15 GB, with license copies + model cards) are released and offline-importable.

### GGUF Format (Android Local Inference)

In Android local mode (called "Full Mode" in the existing UI), models must be placed in a **user-selected external directory** (SAF `ACTION_OPEN_DOCUMENT_TREE`), **not in app internal storage**, so that models are kept by default when the APK is uninstalled.

**Android model storage locations**:

| Recommended Location | Notes |
|----------|------|
| `Download/QLH/models/` | The device's built-in download directory; uninstalling the APK does not delete it |
| User-chosen external SD card directory | Any directory authorized via SAF |

**How to obtain**:

1. **PC distribution**: start the distribution server on the PC, download from the Android browser, then move the model into the SAF model directory

   ```bash
   cd packaging
   python serve.py
   ```

2. **Direct download**: access Hugging Face from the Android browser or transfer files via USB

3. **Later**: in-app support for downloading directly from the PC primary node into the SAF directory

**Workflow**:

```text
Open app → Settings → switch to "Full mode" → Model management → pick directory
  → pick a directory containing .gguf → scan → select model → done
```

> For the detailed plan, see [Android SAF Model Storage Plan](Android SAF模型存储方案.md)

---

## 🚀 Quick Start

### Development Mode (PC)

```bash
# Terminal 1: start the Python backend (run from the project root)
python src/api_server.py

# Terminal 2: start the frontend dev server (optional; the backend serves built frontend assets)
cd frontend && npm run dev
```

Once the backend is ready:

- **Backend direct access**: `http://localhost:8000` (serves the frontend after `npm run build`)
- **Frontend dev server**: `http://localhost:5173` (Vite HMR, proxied to 8000)

### Standalone Mode (PC)

Edit `src/config.py`: set `RUN_MODE = "single"`, then:

```bash
python src/api_server.py
```

### Distributed Mode (PC)

> ⚠️ Prerequisite: all participating nodes have Tailscale installed and are logged in with the same account.

**Master node**:

```bash
python src/api_server.py
# Enable "distributed inference" in the admin panel and configure the Tailscale network
```

**Worker node**:

```bash
python src/api_server.py
# Enter the master node's Tailscale IP in the admin panel and click "connect to master"
```

> The system automatically completes: node registration → device profile reporting → layer assignment computation → layered configuration push.

### TUI Admin Menu (terminal, cross-platform)

For environments without a browser (SSH, servers, Raspberry Pi, etc.), a terminal admin menu is available, covering the same functions as the Web admin panel (system overview / node management / distributed & layered inference / request queue / device profile / logs / settings). It is implemented purely with the Python standard library and supports Windows 10+ / Linux / macOS.

**One-click launch** (auto-starts the backend + waits until ready + enters the TUI; the backend keeps running after the TUI exits):

```bash
bjtu                                        # Global command: works from any terminal (install below)
./start_tui.sh                              # Linux / macOS (no install needed)
start_tui.bat                               # Windows (double-click or command line)
```

**Install the global `bjtu` command** (recommended): the packaged Windows build offers PATH registration in the install wizard (silent flag `/ENVREG=0|1`); the Linux `.deb` always installs `/usr/local/bin/bjtu`, and you can additionally register the `/opt` PATH via `QLH_ENVREG=1` or `qlh-env-register enable`. For a source checkout, add the project root in the GUI environment-variable editor on Windows (avoid `setx` rewriting an overly long PATH); on Linux/macOS you can use `sudo ln -s <project root>/bjtu.sh /usr/local/bin/bjtu`.

**Manual / advanced usage** (run `python src/api_server.py` first if the backend is not running):

```bash
python src/tui_admin.py --host 100.x.x.x    # Manage a remote Tailscale master directly
python src/tui_admin.py --plain             # Fall back to a plain numbered menu on old terminals/pipes
python src/tui_admin.py --host 100.x.x.x --log-token xxx   # Remote mode with log token
bjtu --help                                 # Full command set and startup args (does not start the backend)
```

**TUI command set** (in any screen, type a `/` command and press Enter; ESC cancels; also works in `--plain` mode): common operations such as model / quantization / engine switching, GPU selection, distributed toggle, queue control, logs, settings, and graceful exit are available without entering the menu:

```bash
/help                     # Command help (inside the TUI)
/status  /models  /model  # Status & model info
/switch <model ID> [--quant precision] [--engine engine]   # Switch model (auto-rollback on failure)
/quant  <int4|int8|fp16|gguf>                    # Quantization switch (reloads current model)
/engine <auto|llama_cpp|pytorch|island>          # Engine switch (reloads current model)
/gpu <index>  /device auto                        # GPU selection / device auto-config
/dist on|off  /queue pause|resume|clear          # Distributed toggle / queue control
/logs  /host <host> [port]  /interval <seconds>   # Logs / settings
/quit                     # Exit the TUI (backend keeps running)
/shutdown                 # Graceful exit: backend cleans up resources, then the TUI exits
```

See the **[TUI User Guide](TUI使用指南.md)** for the full parameter table, the `QLH_BACKEND_PORT` override, troubleshooting, and the automated walkthrough; the **complete reference of the 27 `/` commands (aliases / parameters / options / exit semantics / menu mapping) is in [TUI Command Set](TUI指令集.md)**; gateway contract and tests are in [TUI Adaptation Implementation Plan](TUI适配实施计划.md) (T1–T8 current · Active; T9.0–T9.5 completed, terminal walkthrough 54/54; T9.6-R2 Windows dev-machine implementation gate and the UP-N6.4W cross-volume retention gate passed; external clean machine / Linux / real-model sessions and the default entry point are still pending).

### External Compute Assistance (three routes, all disabled by default)

Tensor parallelism is not feasible inside this project's heterogeneous Tailscale mesh (48 all-reduces per token; at a 20 ms RTT, synchronization alone costs ≥960 ms/token — see the quantization argument in the [research proposal](张量并行外部辅助与混合拆分调研方案.md), §1). TP therefore stays on fast interconnects **outside** the cluster, leveraged through three routes:

| Route | Form | Switch | Status |
|------|------|------|------|
| **A · TP island** | An out-of-cluster homogeneous GPU sub-cluster running TP, presenting itself to the cluster as a **single logical high-compute node** and handling whole-request inference (not participating in layer splitting) | `QLH_ISLAND_ENABLED=1` + `QLH_ISLAND_BASE_URL` | Phase 1 PoC, verified |
| **B · External inference service** | Whole requests routed by policy to an OpenAI-compatible endpoint outside the cluster; **nothing leaves the cluster by default** | `QLH_EXTERNAL_ENABLED=1` + `QLH_EXTERNAL_DATA_SCOPE` | Phase 1 PoC, verified |
| **C · Speculative decoding** | A local small model drafts γ tokens and the external large model verifies them in one pass; only token ids cross the slow network | `QLH_SPEC_ENABLED=1` (experimental endpoint returns 404 while disabled by default) | Phase 0–1 exploration, **not wired into the production decoding loop** |

```bash
# Route A: island side (multi-GPU machine / homogeneous LAN GPU group)
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
# Gateway side (run QLH, then connect to the master as usual)
set QLH_ISLAND_ENABLED=1 && set QLH_ISLAND_BASE_URL=http://10.0.0.2:8000
set QLH_ISLAND_GPU_COUNT=2 && set QLH_ISLAND_VRAM_GB=48 && set QLH_ISLAND_TP_SIZE=2
python src/api_server.py

# Route B: default opt_in — only requests explicitly marked allow_external may leave the cluster
set QLH_EXTERNAL_ENABLED=1 && set QLH_EXTERNAL_BASE_URL=https://gpu-box.example.com:8000
set QLH_EXTERNAL_DATA_SCOPE=opt_in
curl -X POST localhost:8000/api/chat -H "Content-Type: application/json" \
     -d "{\"message\":\"...\",\"allow_external\":true,\"prefer_external\":true}"
```

> ⚠️ **Data boundary**: Routes B / C send user content (including speculative-decoding draft tokens) out of the cluster. The scope levels `deny` / `opt_in` (default) / `allow_all` are a security boundary, not a performance switch; an invalid value fails closed to `deny`. Confirm compliance requirements before enabling.

### Packaged Build (Windows Installer)

Two versions are provided — pick whichever fits:

| Version | Installer | Typical size | Use case |
|------|--------|---------|---------|
| **iGPU (CPU) edition** | `QLH-Edge-Inference-Setup-vX.X.X.exe` | ~180 MB | CPU / integrated-graphics nodes (worker nodes) |
| **dGPU (CUDA) edition** | `QLH-Edge-Inference-Setup-vX.X.X-CUDA.exe` | ~1.7 GB | NVIDIA GPU nodes (master node); falls back to CPU automatically when no GPU is present |

**iGPU (CPU) build**:

```bash
# 0. Create and activate the iGPU venv (first time only)
python -m venv .venv-packaging
.venv-packaging\Scripts\activate

# 1. Install dependencies (first time only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2. Build the frontend
cd frontend && npm install && npx vite build && cd ..

# 3. PyInstaller packaging (★ run from the project root)
pyinstaller packaging/qlh-cpu.spec --noconfirm

# 4. Inno Setup installer compilation
cd packaging
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
```

**dGPU (CUDA) build** (requires a separate venv):

```bash
# 0. Create and activate the dGPU venv (first time only)
python -m venv .venv-packaging-cuda
.venv-packaging-cuda\Scripts\activate

# 1. Install dependencies (first time only; torch first, then shared deps — they don't overwrite each other)
pip install torch                        # ★ CUDA 12.x (default), NOT the CPU build
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2-4. Same as the iGPU edition, but use qlh-cuda.spec / setup-cuda.iss
pyinstaller packaging/qlh-cuda.spec --noconfirm
cd packaging && "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup-cuda.iss
```

> ⚠️ **Important**: the two versions use **different, separate venvs** (`.venv-packaging/` vs `.venv-packaging-cuda/`).
> Never mix them — the iGPU venv must install CPU-only torch, and the dGPU venv must install CUDA torch.
> Installing the wrong one will bloat the iGPU build from 180 MB to 1.8 GB.
>
> **SD 1.5 image sidecar**: the dGPU edition additionally installs `pip install -r packaging/requirements-sd15.txt` (pins diffusers 0.35.2 / transformers 4.47.1; the standalone sidecar does not pollute the LLM inference environment); image model assets are not bundled into the installer; official offline asset packages (with license copies + model cards) are released and imported offline.
>
> After installation, double-click the desktop shortcut to launch — no Python environment configuration needed. On uninstall you will be asked whether to also delete the `models/` directory; model files are kept by default.
>
> See [packaging/README.md](../packaging/README.md) for the detailed packaging workflow.

### Linux Package (.deb)

A `.deb` package corresponding to the Windows iGPU edition is provided, for Ubuntu 22.04+ / Debian 12+:

| Version | Package | Typical size | Use case |
|------|--------|---------|---------|
| **CPU edition** | `qlh-edge-inference-cpu_0.1.8.2_amd64.deb` | ~200 MB | CPU / integrated-graphics nodes |
| **CUDA edition** | `qlh-edge-inference-cuda_0.1.8.2_amd64.deb` | ~1.8 GB | NVIDIA GPU nodes |

**Build** (requires an Ubuntu/Debian environment):

```bash
cd packaging/linux
bash build-deb.sh cpu     # iGPU edition
bash build-deb.sh cuda    # dGPU edition
```

**Install**:

```bash
sudo dpkg -i qlh-edge-inference-cpu_0.1.8.2_amd64.deb
# Registers the systemd service, desktop entry and /usr/local/bin/qlh-launcher automatically
```

**Usage**:

```bash
qlh-launcher --gui        # Standalone graphical launcher (regular UI / TUI / update)
qlh-launcher app-ui       # Launch the regular UI directly
qlh-launcher --headless   # Headless mode (API only, good for servers)
sudo systemctl enable --now qlh-edge-inference  # Enable at boot
```

> Prerequisites: `python3` (≥ 3.10), `python3-venv`, `python3-tk` (graphical Launcher, recommended), `tailscale` (distributed mode). The package bundles its own venv and does not pollute the system Python.

### Android Client

> Prerequisite: JDK 17 + Android SDK (API 34+) installed, with the SDK path configured in `android/local.properties`
>
> After cloning the repository fresh, initialize the llama.cpp submodule first (required for the Full variant's native build; not needed for Lite):

```bash
git submodule update --init --recursive
```

**Build** (no Android Studio required):

```bash
cd android

# Debug APK (uncompressed, for development)
./gradlew.bat assembleDebug

# Release APK (R8 shrinking + signing, for distribution)
./gradlew.bat assembleRelease
```

Artifacts:

| Artifact | Path | Typical size | Notes |
|------|------|---------|------|
| Full Debug | `android/app/build/outputs/apk/full/debug/app-full-debug.apk` | ~29 MB | Includes the llama.cpp native backend |
| Full Release | `android/app/build/outputs/apk/full/release/app-full-release.apk` | **~6.7 MB** | R8 + native strip |
| Lite Release | `android/app/build/outputs/apk/lite/release/app-lite-release.apk` | **~1.5 MB** | Pure thin client, no native libraries |

**Install**:

```bash
adb install android/app/build/outputs/apk/full/release/app-full-release.apk
```

**Usage**:

1. Launch the app → select "Settings" in the bottom navigation
2. Full-Remote Mode: enter the PC master node's Tailscale IP and port → test the connection → start chatting
3. Full-Local Mode: switch modes → pick a SAF external directory containing `.gguf` files → scan and select a model → run inference offline

### Distribution Server

Distribute installers within the same Tailscale network so other devices can download them directly from a browser:

```bash
cd packaging
python serve.py
# Default port 9090; browse to http://<local Tailscale IP>:9090/
```

The homepage lists:

- Windows PC installer (.exe)
- Linux installer (.deb)
- Android Full / Lite APK
- PC model archive `models_pc.7z`
- Android model archive `models_android.7z` (GGUF models only)

> Other devices (including Android phones) can download directly by opening the link in a browser.

---

## 📊 Quantization Results

### CUDA dGPU (PyTorch + bitsandbytes)

> Test environment: NVIDIA RTX GPU + CUDA 12.6 + PyTorch 2.12.0 + Qwen-1.8B-Chat (24 layers)

| Config | GPU VRAM | Inference Speed | Notes |
|--------|----------|-----------------|-------|
| FP16 | 3.47 GB | 53.2 tok/s | Baseline control group |
| FP16 + compile | 3.47 GB | 55.1 tok/s | Operator fusion +3.6% |
| INT8 | 2.30 GB | 9.8 tok/s | Saves VRAM but large speed loss |
| **INT4** ⭐ | **1.75 GB** | **28.7 tok/s** | **Recommended for edge devices: VRAM halved** |

### CPU / iGPU (llama.cpp + GGUF)

> Test environment: Intel i5-12400F / AMD R5 5600 + 16GB RAM + Windows 11

| Engine | Quantization | Memory | Inference Speed | Notes |
|--------|--------------|--------|-----------------|-------|
| PyTorch CPU | FP16 | ~3.5 GB | ~3 tok/s | No CUDA fallback |
| llama.cpp | Q4_K_M | ~1.2 GB | **~12 tok/s** | **Recommended for CPU/iGPU** |

> llama.cpp vs PyTorch CPU: memory **-65%**, speed **+300% (3–5x)**

### Android Local Inference (estimates)

| Chip | Tier | Q4_K_M tok/s | Peak RAM |
|------|------|--------------|----------|
| Snapdragon 8 Gen 3 | Flagship | 12-18 | 1.8 GB |
| Snapdragon 8+ Gen 1 | Upper-mid | 8-12 | 1.8 GB |
| Snapdragon 865 | Mid-range | 5-8 | 1.8 GB |

---

## 🧪 Comparative Experiments

| Experiment | Quantization | Operator Fusion | KV Cache | Scheduling Strategy | Deployment Mode |
|------------|--------------|-----------------|----------|---------------------|-----------------|
| Baseline | FP16 | None | Traditional KV | — | Single node |
| Experiment 1 | INT4 | None | Traditional KV | — | Single node |
| Experiment 2 | INT4 | Fused | Traditional KV | — | Single node |
| Experiment 3 | INT4 | Fused | Paged KV | — | Single node |
| Experiment 4 | INT4 | Fused | Paged KV | Simple weighting | Distributed (3 nodes) |
| Experiment 5 | INT4 | Fused | Paged KV | 🧠 Graph algorithm | Distributed (>5 nodes) |

---

## 📊 Core Metrics

- **VRAM usage**: quantization and paged-KV optimization effect
- **Inference latency / token generation speed**: operator fusion, pipeline latency
- **Network bandwidth utilization**: graph-algorithm scheduling vs simple weight allocation
- **CPU load / network latency**: distributed communication overhead
- **Conversation fluency**: evaluation of quantization accuracy loss
- **Long-run stability**: reconnection, heartbeat recovery, cache cleanup

---

## 👥 Team

| Team | Responsibilities |
|------|------------------|
| Model Optimization | Literature review, model quantization, operator fusion, KV-cache optimization |
| Distributed Architecture | Distributed architecture design, communication protocol development, multi-node scheduling logic |
| Frontend & Documentation | Web visualization platform, performance monitoring module, documentation and demo materials |

**Advisor**: Gao Bo, Associate Professor (School of Software Engineering, Beijing Jiaotong University)

---

## 📚 Documentation Index

Specialized plans are currently in Chinese; start from the **[Overall Next-Step Plan](../docs/总体下一步计划.md)** and the **[Progress & Next Steps](../docs/项目进展与下一步计划.md)** snapshot. A full index of design docs, specialized plans and engineering docs: [文档索引](../README.md#-文档索引).

> **Translation status**: all sections are translated; the Chinese README remains the source of truth for ongoing changes.

## 📄 License

This project is a 2026 Beijing Jiaotong University Student Innovation and Entrepreneurship Training Program project.

---

© 2026 Beijing Jiaotong University · Project team
