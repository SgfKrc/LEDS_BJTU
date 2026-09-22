# QLH at a Glance (Newcomers · Reviewers Quick Entry)

> Status: **current**
>
> Updated: 2026-09-22

> **Language**: [English](项目速览-QLH-at-a-Glance.en.md) · [简体中文](项目速览-QLH-at-a-Glance.md)
>
> A 2-minute tour for people who are new to the repository; full capabilities, boundaries and evidence live in the [root README](../README.md) and the specialized documents.

---

## What is this?

QLH is a **lightweight distributed LLM inference system for heterogeneous edge devices** (a 2026 Beijing Jiaotong University student innovation project). It unifies "who holds which layers, with which engine, where, and at what capacity" into a single abstraction, `(layer_range, engine, location)`, and uses it to run models **layer-by-layer across machines** that cannot hold the whole model.

**Two engine tiers** (chosen by device profile — *not* "if there is a GPU, use torch"):

| Tier | Engine | Suited to |
|---|---|---|
| **L tier** | llama.cpp / GGUF (including cut GGUF + `embd` injection) | Lightweight, edge, torch-less nodes |
| **D tier** | PyTorch (layer splitting, inter-layer pipeline, multi-node segments) | CUDA PCs; also the platform for cut-point search / experiments |

**Three node kinds are isomorphic** and share one contract plus fail-closed validation: `local` (in-process), `remote_rpc` (borrowed compute over the network — `ggml-rpc-server` on Android/PC), and `cross_framework` (cross-engine layer handoff). The optional Relay R track (cut GGUF + `embd` injection, same-host or cross-host) exists for **capacity merging and heterogeneous capability composition**.

**Positioning boundary (important)**: relay is a **capacity** mechanism, not a speed-up. On the same host, the best cross-framework relay (129.1 ms/token) is **4.85× slower** than "llama.cpp + CUDA with all layers on GPU" (26.6 ms/token); when CUDA is available and the whole model fits, the best practice is **not to relay** — just run the full model on the CUDA node (or use it as an RPC/shard worker). Default routing always prefers the L / RPC paths that can run the model outright.

Design principles: **data stays in-cluster, offline autonomy, reproducible acceptance**.

## Verified capabilities (each with criterion and date)

| Capability | Evidence (criterion / date) |
|---|---|
| Dual-machine real-device layer pipeline (QW1.8B layers 0-21 / 21-24) | 3× `distributed_required` all passed, RTT 6-12 ms (2026-08-20) |
| Task-graph restart / kill-and-recover + Tailnet IPv6 | `wf_330d0aa1…`, reassignment=0 (2026-08-21) |
| **D→L cross-framework relay correctness** | Main-repo dual-engine matrix **27/27 per-token identical**: qwen2.5-0.5B K=4/8/12/16/20, qwen3.5-2B K=8/12/16/20; loads prefill 32/128/512, decode 32/64/256; batch 2/4; mixed precision fp16·f32·NF4 × Q4_K_M (2026-09-21) |
| **D→L capacity gain** | qwen2.5-0.5B **1.568×** / qwen3-5-2b **1.547×**; under the same controlled 3.0 GB CUDA budget the whole model is rejected while a 12-layer upstream passes (2026-09-21) |
| **L→L keep-head channel** | `--path l2l_keep_head` / `d2l2l_keep_head` **32/32**, including the three-stage "1 torch upstream + 2 llama downstream" case (2026-09-21) |
| **Pure L→L multi-hop (three stages, all on device)** | y700 head8 → y700 mid8-16 → y700 cut-k16 tail, host only sends commands ⇒ **32/32** (2026-09-22) |
| **Pure L→L cross-device (three stages)** | y700 head8 → **Surface** mid8-16 → y700 tail ⇒ **32/32**, two real machines cooperating with the host doing no compute (2026-09-22) |
| **Android layer-segment numerical validation** | Real ARM64 (Snapdragon 8 Gen 3 / Android 15) is **per-token identical** to x86_64; dotprod/i8mm kernels confirmed; `qlh-android` P0 cross-compile/JNI done (2026-09-21/22) |
| Cut-point solver closed loop | `scripts/relay_cut_plan.py` fits segment profiles (fixed overhead + per-layer cost) from **measurements**, then solves; two-segment loop passes on qwen2.5 (r² 0.96/0.99) and qwen3.5 (0.79/0.96) |
| Windows native compile path | `triton-windows==3.8.0.post28` verified working (`torch.compile` is no longer a dead switch); `PYTHONUTF8=1` is a prerequisite |
| Judging-policy fix | multi-model 0/4 → loose+512 becomes discriminative; DS3-0324-7B replaces R1 (v2 policy 2/4×3, format 8/11×3) |
| Sub-project: small-model harness workbench (S1-S8) | context budget / STATE memory / RAG / MCP, local gates |
| Web search / lightweight Fetch (WEB-TOOL G1-G6) | local dev gates, `production_network_enabled=false` |

## What is NOT claimed

- **Cross-host RPC vs relay has not been compared yet**; D→L **long-run, cross-host and multi-segment failure acceptance** are pending.
- **A middle stage must share a LAN with its caller**: with the middle stage on a **cross-network** node (Surface, going through Tailscale **DERP relay**, RTT 777 ms) every step costs a round trip ⇒ end-to-end went from ~140 ms median to ~800 ms. Cross-network paths, especially relayed ones, are **unusable**.
- Production routing admission (`task_dispatch` off), long-running/multi-turn and power-loss recovery, real 443/WSS, IPv6-only installers, and real 7B/12B three-node peak memory all remain in the **deferred acceptance queue**; local gates or simulations must not be presented as passes.
- Android is at `P0` only (cross-compile/JNI + layer-segment numerical validation); on-device running, RPC worker, disconnection, thermal/power and security evidence are **not done**.
- Tensor parallelism (TP) exists only as an out-of-cluster PoC; speculative decoding is an experimental path.
- **Acceptance discipline**: relay consistency is judged by **per-token argmax** only (never cosine); a divergence is reported as FAIL, without "acceptable approximation".

## First-time setup

### 0. Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 (3.12 recommended) | Main runtime & tool scripts |
| Node.js + npm | Node ≥ 18 | Product shell (`qlh-shell`) side-line only |
| JDK 17 + Android SDK (API 34+) | — | Android builds only |
| Tailscale | latest | Required for distributed mode; ⚠️ **most stable when nodes share a LAN** (cross-network traffic goes through a DERP relay, and layer relay then becomes unusable because every step costs a round trip) |
| Git | — | clone (with submodules) |
| NVIDIA driver + CUDA (optional) | — | D tier / PC dGPU only |

### 1. Clone & one-shot environment setup

```bash
git clone --recurse-submodules https://github.com/SgfKrc/LEDS_BJTU
cd LEDS_BJTU
# Current 4 submodules: android / packaging(qlh-release) / harness_workbench(Koakumix) / frontend_cybergothic(qlh-shell)
# Note: the two bridges (reasonix-codex-bridge / dsh-codex-bridge) are **workspace-local directories**, not submodules.
python scripts/setup_envs.py --all            # mainline Python envs (no Node by default)
python scripts/setup_envs.py --all --with-node # opt in to the product-shell Node source
python scripts/setup_envs.py --only test,tui  # pick specific envs
python scripts/setup_envs.py --check          # verify only, no install (no side effects)
```

**Coverage**: main env + `.venv-test` / `.venv-tui` / `.venv-qwen3-sidecar` and friends; product shell, Android, packaging and Node environments are side-line migration sources, not mainline prerequisites. Koakumix owns its image-generation dependencies in its own environment.

> ⚠️ **torch and other platform-specific heavy packages are NOT auto-installed**: the script filters them out and prints the platform install commands (e.g. `--torch-index-url https://download.pytorch.org/whl/cu126`) to avoid CPU/CUDA cross-contamination. Install them, then re-run `--check`.

### 2. Get models (**by device profile**)

The default model is not a single fixed one — it is selected from the device profile (`DEFAULT_MODEL_BY_TIER` in `src/model_config.py`; consumers use `config.get_active_model_paths()`):

| Device tier | Default model |
|---|---|
| Mobile / edge / ultrabook (≤ 2 GB shared VRAM) | `<1B` tier: `qwen3-0.6b` |
| PC (discrete GPU) | `~2B` tier: `qwen3-5-2b` |

> ⚠️ This is a **hard constraint**, not a preference: the 2B tier peaks at **2.24 GB VRAM**, so an ultrabook with ≤ 2 GB shared VRAM simply **cannot hold it**.

```bash
# GGUF Q8_0 (starting point for CPU / iGPU / Android)
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
```

Other models: see the root README "Model download"; model files are **never committed** (`models/` is gitignored).

### 3. Run & verify

```bash
python src/api_server.py               # backend http://localhost:8000
python qlh.py                         # mainline TUI (Textual shell, 9 function screens + 1 debug fallback)
# or the terminal UI: ./start_tui.sh (Windows: start_tui.bat, auto-starts the backend)
```

```powershell
# test env ready (current serial full-run baseline: see README)
.\.venv-test\Scripts\python.exe -m pytest tests/ -q --collect-only
# before a cross-host relay, run the health check (TCP probe + heartbeat freshness; 0 = healthy / 1 = problem)
python scripts/relay_health.py --check tail=tcp:127.0.0.1:50188 --json
```

Distributed mode: all nodes sign in with the same Tailscale account → TUI cluster/node commands → node online.

## Reading map

- **Entry point**: [README](../README.md) (Chinese) · [README.en](README.en.md) (English)
- **Relay (the most active track)**: [Current effective baseline & optimization plan](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) — **use it as the index**; conflicting numbers in older reports are triaged by validity. Companion docs: [project report](跨框架层接力-项目报告.md), [capacity-gain measurements](跨框架层接力-容量收益实测-2026-09-21.md)
- **Newcomer**: [项目技术说明](archive/distributed/项目技术说明.md) → [整体架构](整体架构.md) → [模块接口说明](模块接口说明.md)
- **Plans**: [Distributed Inference & Edge Optimization](主线开发计划-分布式推理与边缘优化-2026-09-14.md) · [P4.5: dynamic master election & distributed management](主节点动态选举与分布式管理-P4.5立项-2026-09-21.md) · [总体下一步计划](archive/distributed/总体下一步计划.md) (historical schedule)
- **Protocol & nodes**: [layer-segment protocol proposal](层段协议立项-2026-09-17.md) · [node types & top-level transparency](层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md)
- **TUI**: [TUI usage guide](TUI使用指南.md) · [TUI command set](TUI指令集.md)
- **Tests & criteria**: [tests & judging criteria](测试与评判标准.md) · [test channel runs](测试通道运行说明.md)
- **Android**: [Android alternative verification path](../android/Android验证替代路径-2026-09-18.md)
- **Docs themselves**: [document status & cleanup list](文档状态与清理清单.md)

*Note: most specialized documents are in Chinese.*

## Engineering culture

- **Evidence first**: every "Completed" must come with tests/logs/real-device evidence — criterion *and* date, both required.
- **Judging discipline**: consistency is judged by per-token argmax only; divergence is reported as FAIL; record kinds `mainrepo_end_to_end` / `raw_binding_probe` / `capacity_only` are kept separate and never mixed.
- **Mechanization, not green-washing**: test channels separate unit/contract/browser/race/simulation/real-machine; fixes ship with negative cases to avoid the pesticide effect.
- **Measure, then explain**: performance claims must first locate the bottleneck (per-segment timing / RTT / dispersion) — never explain a gap with "device class" alone.
- **User sovereignty**: model artifacts, keys, and knowledge bases belong to the user; the dev group never holds them; offline recovery is guaranteed.

## License

MIT (Copyright (c) 2026 SgfKrc) — see [LICENSE](../LICENSE).
