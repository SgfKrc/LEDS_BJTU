# QLH at a Glance (Newcomers · Reviewers Quick Entry)

> 状态：**现行**
>
> 更新日期：2026-09-16

> **Language**: [English](项目速览-QLH-at-a-Glance.en.md) · [简体中文](项目速览-QLH-at-a-Glance.md)
>
> A 2-minute tour for newcomers; the full feature list, boundaries and evidence live in the [root README](../README.md) and specialized plans.

---

## What is this?

QLH is a lightweight distributed LLM inference system for heterogeneous edge devices (a 2026 Beijing Jiaotong University student innovation program project). Edge nodes use <=1B llama.cpp/GGUF models locally by default and can join larger model sharding as RPC workers; the production path is built from single-node validation through same-host simulation and real PC/Android workers. The main repository also carries a PC-only Relay R compatibility track: cut GGUF plus `embd` injection reached 16/16 same-process L→L behavioral matches, while D→L, cross-process/network and long-sequence behavior remain unverified. PyTorch is retained only as a PC reference and Relay source; task graphs are temporary full-model fallback. The main repository keeps INT4/INT8 quantization, paged KV cache, TUI and reproducible experiments. Product Web, Android UI, knowledge base, release tooling and image generation are side-line work; image generation belongs exclusively to the standalone Koakumix harness. Design principles: **data stays in-cluster, offline autonomy, reproducible acceptance**.

## Verified capabilities

| Capability | Evidence |
|---|---|
| Dual-machine layer pipeline (QW1.8B layers 0-21 / 21-24) | 2026-08-20: 3× `distributed_required` all passed, RTT 6-12ms |
| Task-graph recovery + Tailnet IPv6 | 2026-08-21: `wf_330d0aa1…`, reassignment=0 |
| Judging-policy fix (multi-model 0/4 → loose+512 discriminative) | P5: Qwen3-4B 1/4 vs 1.8B 0/4 |
| DS3-0324-7B replaces R1 as judging model (approved candidate) | v2 policy **2/4×3, format 8/11×3** |
| PC-only Relay R: cut GGUF + `embd` L→L handoff | 2026-09-16: 16/16 same-process behavioral matches; `embd` is not bitwise deterministic, D→L/cross-boundary work pending |
| Sub-project: small-model harness workbench (S1-S8) | context budget / STATE memory / RAG / MCP, local gates |
| Sub-project: docagent (own repo, brought in as submodule) | rule-as-data + evolution gates |
| Sub-project: Reasonix ↔ Codex bridge (own repo, brought in as submodule) | read-only subagent for Codex; controlled writes W1/W2/W3 landed (off by default) |
| Web search & lightweight Fetch (WEB-TOOL G1-G6) | local gates, `production_network_enabled=false` |

## What is NOT claimed

- Production routing admission (`task_dispatch` off), long-running/multi-turn & power-loss recovery, real 443/WSS, IPv6-only installers, real 7B/12B three-node peak memory — all remain in the deferred acceptance queue; local gates or simulations must not be presented as passes.
- Relay currently has only same-process PC L→L behavioral evidence; no bitwise, PyTorch→llama.cpp, cross-process/network, long-sequence or sampling production claim is allowed.
- Tensor parallelism (TP) exists only as an out-of-cluster PoC (route A); speculative decoding is an experimental path (route C).

## First-time setup

### 0. Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 (3.12 recommended) | Main runtime & tool scripts |
| Node.js + npm | Node ≥ 18 | Product frontend / gateway / control (development only) |
| JDK 17 + Android SDK (API 34+) | — | Android builds only |
| Tailscale | latest | Required for distributed mode (campus networks may block UDP → relay) |
| Git | — | clone (with submodules) |
| NVIDIA driver + CUDA (optional) | — | PC dGPU only |

### 1. Clone & one-shot environment setup

```bash
git clone --recurse-submodules https://github.com/SgfKrc/LEDS_BJTU   # includes llama.cpp (third-party) plus the in-house docagent and reasonix-codex-bridge submodules
cd LEDS_BJTU
python scripts/setup_envs.py --all            # mainline Python envs (no Node by default)
python scripts/setup_envs.py --all --with-node # opt in to the product-shell Node source
python scripts/setup_envs.py --only test,tui  # pick specific envs
python scripts/setup_envs.py --skip frontend  # skip the frozen legacy frontend
python scripts/setup_envs.py --check          # verify only, no install (no side effects)
```

**Coverage**: main env + `.venv-test` / `.venv-tui` / `.venv-gemma4-native` / `.venv-gemma4-pipeline` / `.venv-qwen3-sidecar`; product shell, Android, packaging and Node environments are side-line migration sources, not mainline prerequisites. Koakumix owns its image-generation dependencies in its own environment.

> ⚠️ **torch and other platform-specific heavy packages are NOT auto-installed**: the script filters them out and prints the platform install commands (e.g. `--torch-index-url https://download.pytorch.org/whl/cu126`) to avoid CPU/CUDA cross-contamination. Install them manually, then re-run `--check`.

### 2. Get models (default: Qwen-1.8B-Chat)

```bash
# Safetensors (PyTorch / distributed, ~3.5 GB) — ModelScope preferred in China
pip install modelscope && python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
# GGUF Q4_K_M (CPU/iGPU/Android, ~1.16 GB)
huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/
```

Other models (Qwen3-4B, Gemma 4, etc.) see the root README "Post-clone asset checklist / Model download"; model files are **never committed** (`models/` is gitignored). Koakumix image assets are outside the QLH main-project asset list.

### 3. Run & verify

```bash
python src/api_server.py               # backend http://localhost:8000
python -m src.tui_admin --plain --host http://127.0.0.1:8000 # mainline TUI
# or the terminal UI: ./start_tui.sh (Windows: start_tui.bat, auto-starts the backend)
python -c "import src.api_server"                                   # backend importable
.venv-test\Scripts\python.exe -m pytest tests/ -q --collect-only    # test env ready
```

Distributed mode: all nodes sign in with the same Tailscale account → "Connect to master" in the admin panel → node online.

## Reading map

- **Mainline plan**: [Distributed Inference & Edge Optimization](主线开发计划-分布式推理与边缘优化-2026-09-14.md) · **Side-line plan**: [Externalization & Koakumix](支线开发计划-外置迁移与Koakumix-2026-09-14.md) · Historical schedule: [总体下一步计划](总体下一步计划.md)
- **Newcomer**: [项目技术说明](项目技术说明.md) → [整体架构](整体架构.md) → [模块接口说明](模块接口说明.md)
- **Sub-projects**: [harness plan](../harness_workbench/docs/小模型轻量推理harness工作台调研与方案.md) · [qlh-docagent](https://github.com/SgfKrc/qlh-docagent) · [reasonix-codex-bridge](https://github.com/SgfKrc/reasonix-codex-bridge) · [web-tool research](archive/联网搜索与轻量Fetch工具调用可行性调研与分期计划.md)
- **Experiments & judging**: [DS3 replaces R1](DistilQwen2.5-DS3-0324替代R1判题模型专项计划.md) · [sub-1B plan](亚1B小模型专项实验计划.md) · [tests & criteria](测试与评判标准.md)

*Note: most specialized documents are in Chinese (see the root README index).*

## Engineering culture

- **Evidence first**: every "Completed" must come with tests/logs/dual-review evidence (EX-N3 read-only production gate re-checked historical records 3/3).
- **Mechanization, not green-washing**: test channels separate unit/contract/browser/race/simulation/real-machine; fixes come with negative cases to avoid pesticide effects.
- **User sovereignty**: model artifacts, keys, and knowledge bases belong to the user; the dev group never holds, and offline recovery is guaranteed.

## License

MIT (Copyright (c) 2026 SgfKrc) — see [LICENSE](../LICENSE).
