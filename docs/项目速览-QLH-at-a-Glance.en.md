# QLH at a Glance (Newcomers · Reviewers Quick Entry)

> **Language**: [English](项目速览-QLH-at-a-Glance.en.md) · [简体中文](项目速览-QLH-at-a-Glance.md)
>
> A 2-minute tour for newcomers; the full feature list, boundaries and evidence live in the [root README](../README.md) and specialized plans.

---

## What is this?

QLH is a lightweight distributed LLM inference system for heterogeneous edge devices (a 2026 Beijing Jiaotong University student innovation program project). Master and worker nodes (PC / Surface / Android) join a Tailscale mesh and split models across nodes by compute/memory/network capacity. It features twin engines (PyTorch + llama.cpp), INT4/INT8 quantization, paged KV cache, task-graph orchestration, local RAG, SD 1.5 image generation, a local Auth-App control plane, and multi-client UIs. Design principles: **data stays in-cluster, offline autonomy, reproducible acceptance**.

## Verified capabilities

| Capability | Evidence |
|---|---|
| Dual-machine layer pipeline (QW1.8B layers 0-21 / 21-24) | 2026-08-20: 3× `distributed_required` all passed, RTT 6-12ms |
| Task-graph recovery + Tailnet IPv6 | 2026-08-21: `wf_330d0aa1…`, reassignment=0 |
| Judging-policy fix (multi-model 0/4 → loose+512 discriminative) | P5: Qwen3-4B 1/4 vs 1.8B 0/4 |
| DS3-0324-7B replaces R1 as judging model (approved candidate) | v2 policy **2/4×3, format 8/11×3** |
| Sub-project: small-model harness workbench (S1-S8) | context budget / STATE memory / RAG / MCP, local gates |
| Sub-project: docagent (own repo, brought in as submodule) | rule-as-data + evolution gates |
| Web search & lightweight Fetch (WEB-TOOL G1-G6) | local gates, `production_network_enabled=false` |

## What is NOT claimed

- Production routing admission (`task_dispatch` off), long-running/multi-turn & power-loss recovery, real 443/WSS, IPv6-only installers, distributed SD cross-machine, real 7B/12B three-node peak memory — all remain in the deferred acceptance queue; local gates or simulations must not be presented as passes.
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
| NVIDIA driver + CUDA (optional) | — | PC dGPU / SD sidecar only |

### 1. Clone & one-shot environment setup

```bash
git clone --recurse-submodules https://github.com/SgfKrc/LEDS_BJTU   # includes llama.cpp / docagent submodules
cd LEDS_BJTU
python scripts/setup_envs.py --all            # everything: 8 Python envs + Node projects
python scripts/setup_envs.py --all --no-node  # Python envs only
python scripts/setup_envs.py --only test,tui  # pick specific envs
python scripts/setup_envs.py --skip frontend  # skip the frozen legacy frontend
python scripts/setup_envs.py --check          # verify only, no install (no side effects)
```

**Coverage**: main env + `.venv-test` / `.venv-tui` / `.venv-gemma4-native` / `.venv-gemma4-pipeline` / `.venv-qwen3-sidecar` / `.venv-packaging` / `.venv-packaging-cuda` (incl. SD sidecar); Node: `frontend_cybergothic` (the only product frontend) / `gateway` / `control` (legacy `frontend` is frozen; installed by default, use `--skip frontend`).

> ⚠️ **torch and other platform-specific heavy packages are NOT auto-installed**: the script filters them out and prints the platform install commands (e.g. `--torch-index-url https://download.pytorch.org/whl/cu126`) to avoid CPU/CUDA cross-contamination. Install them manually, then re-run `--check`.

### 2. Get models (default: Qwen-1.8B-Chat)

```bash
# Safetensors (PyTorch / distributed, ~3.5 GB) — ModelScope preferred in China
pip install modelscope && python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
# GGUF Q4_K_M (CPU/iGPU/Android, ~1.16 GB)
huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/
```

Other models (Qwen3-4B, Gemma 4, SD 1.5 five-asset pack, etc.) see the root README "Post-clone asset checklist / Model download"; model files are **never committed** (`models/` is gitignored).

### 3. Run & verify

```bash
python src/api_server.py               # backend http://localhost:8000
cd frontend_cybergothic && npm run dev # product frontend http://localhost:5174
# or the terminal UI: ./start_tui.sh (Windows: start_tui.bat, auto-starts the backend)
python -c "import src.api_server"                                   # backend importable
.venv-test\Scripts\python.exe -m pytest tests/ -q --collect-only    # test env ready
```

Distributed mode: all nodes sign in with the same Tailscale account → "Connect to master" in the admin panel → node online.

## Reading map

- **Master plan**: [总体下一步计划](总体下一步计划.md) · Evidence snapshot: [项目进展与下一步计划](项目进展与下一步计划.md)
- **Newcomer**: [项目技术说明](项目技术说明.md) → [整体架构](整体架构.md) → [模块接口说明](模块接口说明.md)
- **Sub-projects**: [harness plan](小模型轻量推理harness工作台调研与方案.md) · [qlh-docagent](https://github.com/SgfKrc/qlh-docagent) · [web-tool research](联网搜索与轻量Fetch工具调用可行性调研与分期计划.md)
- **Experiments & judging**: [DS3 replaces R1](DistilQwen2.5-DS3-0324替代R1判题模型专项计划.md) · [sub-1B plan](亚1B小模型专项实验计划.md) · [tests & criteria](测试与评判标准.md)

*Note: most specialized documents are in Chinese (see the root README index).*

## Engineering culture

- **Evidence first**: every "Completed" must come with tests/logs/dual-review evidence (EX-N3 read-only production gate re-checked historical records 3/3).
- **Mechanization, not green-washing**: test channels separate unit/contract/browser/race/simulation/real-machine; fixes come with negative cases to avoid pesticide effects.
- **User sovereignty**: model artifacts, keys, and knowledge bases belong to the user; the dev group never holds, and offline recovery is guaranteed.

## License

MIT (Copyright (c) 2026 SgfKrc) — see [LICENSE](../LICENSE).
