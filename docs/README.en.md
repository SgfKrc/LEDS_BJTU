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

Distributed inference depends on **Tailscale** for cross-subnet device interconnect; install Tailscale on every participating node (PC, Android) and join the same tailnet. *Detailed installation steps are in the Chinese README ([安装 Tailscale](../README.md#tailscale-组网重要)) — Translation pending.*

---

## 🏗️ Project Architecture

*Architecture overview, layer-pipeline example, Android modes and software layering are documented in Chinese: [项目架构](../README.md#-项目架构). Translation pending.*

---

## 📦 Environment Dependencies

Python / CUDA / bitsandbytes / llama.cpp / Node.js (frontend, gateway, control) / Android SDK. *Detailed version table in Chinese: [环境依赖](../README.md#-环境依赖). Translation pending.*

---

## 🤖 Model Download

Models are user-managed assets; the project provides download guidance and offline import. *Formats (Safetensors / GGUF / SD 1.5 / Android) documented in Chinese: [模型下载](../README.md#-模型下载). Translation pending.*

---

## 🚀 Quick Start

### Development Mode (PC)

```bash
# Backend (single node)
python -m venv .venv-packaging
.venv-packaging\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

*Full instructions (standalone, distributed, TUI, packaging) in Chinese: [快速开始](../README.md#-快速开始). Translation pending.*

---

## 📊 Quantization Results

*Benchmark tables (CUDA dGPU / CPU iGPU / Android estimates) in Chinese: [量化效果](../README.md#-量化效果). Translation pending.*

---

## 🧪 Comparative Experiments

*Experiment groups documented in Chinese: [对照实验组](../README.md#-对照实验组). Translation pending.*

## 📊 Core Metrics

*Core acceptance metrics (latency, throughput, memory, quality gates) documented in Chinese: [核心评判指标](../README.md#-核心评判指标). Translation pending.*

---

## 👥 Team

*Team roles documented in Chinese: [团队分工](../README.md#-团队分工). Translation pending.*

---

## 📚 Documentation Index

Specialized plans are currently in Chinese; start from the **[Overall Next-Step Plan](../docs/总体下一步计划.md)** and the **[Progress & Next Steps](../docs/项目进展与下一步计划.md)** snapshot. A full index of design docs, specialized plans and engineering docs: [文档索引](../README.md#-文档索引).

> **Translation status**: core sections (intro, editions, features, philosophy, quick start) are translated; detailed operational chapters remain in Chinese with `Translation pending` markers. The Chinese README is the source of truth.

## 📄 License

This project is a 2026 Beijing Jiaotong University Student Innovation and Entrepreneurship Training Program project.

---

© 2026 Beijing Jiaotong University · Project team
