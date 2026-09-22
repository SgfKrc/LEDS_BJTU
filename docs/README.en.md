# QLH

QLH is a distributed inference core for heterogeneous edge devices. The mainline is the lightweight GGUF/llama.cpp engine; the repository also owns a PyTorch layered-distribution engine plus a **layer pipeline** (including cross-framework layer relay), and the user-facing entry point is a cross-platform TUI.

> Status: the main-repository baseline is being reorganized (2026-09-21, baseline `610f4b3`)
>
> This README describes only the current boundary of the main repository and its reproducible entry points. Experiment logs, historical implementations and external sub-projects are **not** equivalent to production capability.
>
> 中文: [../README.md](../README.md) - this file: docs/README.en.md

## What the Main Repository Does

- Runs or coordinates GGUF inference on Windows/Linux PCs, devices without CUDA, and Android nodes.
- Lets models too large for a single machine be carried jointly by multiple nodes under a verified layer-segment contract; each node should hold only the model part actually assigned to it.
- Lets an Edge node complete local inference with <=1B models by default, while retaining the ability to serve as an RPC worker for larger models.
- Manages three node kinds (**local**, **remote RPC**, **cross-framework**) under one layer-segment contract, and selects the engine by device profile and capability.
- Controls distributed admission and failure recovery through device profiling, capacity planning, model identity, layer-segment contracts, leases and epoch fencing.
- Uses a Textual TUI for chat, model assets, nodes, distributed layout, queue, device, logs and settings; only backend operations without a dedicated interaction fall back to the debug page. Read-only single commands use a standard-library thin layer.

The engine is **dual-track**, and both tracks live in the main repository - this is not a "primary path plus legacy comparison" arrangement:

| Track | Engine | Capabilities | Dependency |
| --- | --- | --- | --- |
| **L** | llama.cpp / GGUF | Single-machine inference, RPC partial residency, TUI chat, the optimization trio | No torch (Edge default) |
| **D** | PyTorch / Safetensors | Layer splitting and tensor placement, **inter-layer pipeline**, multi-node layer-segment hosting, the upstream side of cross-framework relay, and the control experiments against llama.cpp | torch |

**The D track is not part of the Edge default dependency set**, but layer splitting, the layer pipeline and cross-framework relay are in practice implemented by the PyTorch stack (`model_module.py`, `tcp_comm.py`, `qwen3_pipeline_*`), so it is not a "comparison-only path". On the same card llama.cpp is faster for a single sequence (about 4.3x), so the default production path remains the L track.

## Architecture Overview

QLH is **two layers in one process**: a control plane aimed at people, and an engine layer aimed at
machines and protocols. There is exactly one boundary between them — the layer-range contract
`(layer_range, engine, location)`.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Control plane (aimed at people)                                            │
│ Textual TUI · read-only single commands · HTTP API (/api/cluster/*)        │
│ device profile · capacity plan · layer-range contract · lease / epoch      │
│ fencing · admission                                                        │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ layer-range contract (layer_range, engine, location)
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ Engine layer (aimed at machines)                                           │
│ Tier L: llama.cpp / GGUF            Tier D: PyTorch / Safetensors          │
│ ├ single-machine inference          ├ layer splitting and tensor placement │
│ ├ ggml RPC worker (borrowed GPU)    ├ inter-layer pipeline (qwen3_pipeline)│
│ └ layer-segment forward (shim)      └ cross-framework upstream (model_mod) │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ segment channel: Relay TCP (HIDDEN / HIDDEN_SEQ / TOKEN)
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ Nodes and transport (cross-host, heterogeneous)                            │
│ local loopback · SSH tunnel · Surface (x86_64, Windows) · y700 (ARM64)     │
│ hidden compression: f32 / f16 / bf16 / int8_block128 · weak-net budgets    │
└────────────────────────────────────────────────────────────────────────────┘
```

One four-stage chain that has actually been measured end-to-end (records and method: the relay
baseline document linked from the [Documentation Index](#documentation-index)):

```
prompt → torch(0..7) →hidden→ Surface(8..15) →hidden→ y700(16..19) →hidden→ llama(20..23+head) → token
           local CUDA          x86_64 Windows        ARM64 Android           local llama.cpp
```

The acceptance criterion is **per-token argmax equality with the same-precision monolithic model**
(cosine is not a substitute); any divergence is reported as FAIL, never as an "acceptable
approximation".

## Is This System Software or User Software?

**Layered answer**: QLH ships as **a system-software core plus a user-software shell**.

- **Control plane ≈ user software**: TUI, model assets, node/layout/queue/log/settings pages, HTTP API.
  Its users are **people**; the failure mode is degraded experience (retry, switch model, switch
  layout), and its interfaces may evolve.
- **Engine layer ≈ system software**: layer-range contracts, the layer pipeline, cross-framework
  relay, heterogeneous node hosting, hidden-state compression and transport. Its users are **other
  software** (the control plane, higher-level orchestration, peer nodes), and its failure mode is
  **silently wrong numerics** — hence strong contracts and fail-closed criteria.

| Criterion | Control plane | Engine layer |
| --- | --- | --- |
| Primary users | People (end users / operators) | Other software (TUI, API orchestration, peer nodes) |
| Failure consequence | Degraded experience, retryable | Wrong numerics, possibly **silent** |
| Interface stability | May evolve (pages/commands can change) | Strong contract (layer contract, protocol version, record schema) |
| Replaceable alone | Yes (new frontend, same engine) | No (a new engine means new numeric semantics → re-run per-token comparison) |
| Analogy | Application / admin panel | Kernel + runtime + distributed subsystem |

Two engineering consequences:

1. **The change site decides the verification strength**: control-plane changes need UI/contract tests;
   engine-layer changes require per-token comparison + record-schema validation + matrix runs.
2. **"Borrowed compute" is a node type, not a fallback**: `remote_rpc` / `cross_framework` are
   isomorphic to `local`, so the engine layer is "one subsystem, many node kinds" rather than
   "main path + degraded path".

## Layer Pipeline and Cross-Framework Layer Relay

### Unified Node Abstraction

The layer pipeline unifies "which layers who holds, with which engine, where, and how large the capacity is" into one abstraction: `(layer_range, engine, location)`. The three node kinds are **isomorphic** and share the same contract and validation:

| `kind` | Meaning | Engine | Transport |
| --- | --- | --- | --- |
| `local` | Layer segment inside the local process | llama.cpp / pytorch | in-process |
| `remote_rpc` | **Borrowed compute** (a `ggml-rpc-server` on Android/PC) | llama.cpp | network (ggml RPC) |
| `cross_framework` | Layer-segment relay across engines (torch upstream + llama.cpp downstream) | both | in-process or stdio |

"Borrowing compute" is not a fallback scheme but **one node kind** inside the layer pipeline; all three kinds share one abstraction, so there is no "primary path vs. fallback" opposition.

Implementation and endpoints:

- `src/pipeline_node_contract.py`: the `PipelineNode` contract, mapping from existing artifacts, and fail-closed layout validation;
- `src/pipeline_capacity.py` capacity solving, `src/pipeline_assignment_manifest.py` assignment manifests;
- `src/pipeline_reshard.py`: capacity re-solve + artifact readiness gate + atomic epoch commit;
- `GET /api/cluster/layers`, `GET /api/cluster/pipeline-capacity`, `GET /api/cluster/pipeline-reshard`.

### Cross-Framework Layer Relay (D to L)

The upstream PyTorch layer segment computes up to layer N and hands the hidden states to the downstream layer-cut llama.cpp model to finish. The injection point is llama.cpp's **standard** `llama_batch.embd` field, so the PyPI build of `llama-cpp-python` suffices - **no fork and no recompilation**.

**Why we do it:**

1. **It is the twin mechanism of the layer pipeline** - both share "layer splitting + hidden passing". The layer pipeline can **completely avoid loading the whole model** (llama.cpp RPC needs the leader to hold the full GGUF), so this track decides whether "a model too large for one device can be carried by several devices", not single-sequence latency;
2. **It is the only interface that lets heterogeneous devices join one pipeline** - nodes with different engine capabilities (`local` / `remote_rpc` / `cross_framework`) can only coexist in one layer pipeline through it;
3. **It is the precondition for customizability** - cut-point assignment, mixed precision, operator substitution and batch overlap all build on "hidden states are transferable between layers";
4. **There is no direct academic precedent** - Petals is same-framework, KTransformers is operator-level, distributed-llama is TP; on this path we also filed a defect upstream and independently verified the fix (issue #28963).

**Current validity note (2026-09-21)**: the dual-engine sample is Qwen2.5-0.5B, 12+12 layers, gen=32 -
upstream `model_module.forward_layers` plus downstream `llama_engine.forward_layers_from_hidden`,
**47.501 ms/step**, token-identical to the pure llama.cpp control.

**Full matrix**: two models x cut points / loads (prefill 32/128/512, decode 32/64/256) / batch (2/4) /
mixed precision (upstream fp16-f32-NF4 x downstream Q4_K_M) - **27 runs, all token-identical**;
Qwen3.5 hybrid **K=8/12/16/20 all 32/32**.

⚠️ Before quoting a tier, verify what actually took effect: `quant_type="int4"` **silently falls back to
fp16** in the main-repository layer pipeline, and upstream compile is disabled by the size gate below 1.5B params.

**D-to-L capacity value**: two-segment cut points give **1.568x / 1.547x** for Qwen2.5-0.5B / qwen3-5-2b;
under the same controlled 3.0 GB CUDA budget the whole model is rejected while a 12-layer upstream passes.
That budget is a reproducible experiment constraint, not a physical OOM. D-to-L is positioned as capacity
merging and heterogeneous capability composition - **not** as a CUDA single-machine speed-up substitute.

**Evidence and boundaries**:

| Item | Current basis |
| --- | --- |
| Correctness | Main-repo dual-engine D-to-L matrix **27/27 token-identical** (Qwen2.5-0.5B K=4/8/12/16/20, Qwen3.5-2B K=8/12/16/20; loads prefill 32/128/512, decode 32/64/256; batch 2/4; mixed precision fp16-f32-NF4 x Q4_K_M); per-token criteria are recorded separately from speed and capacity |
| Mixed precision | A full-precision PyTorch upstream with a quantized GGUF downstream is a deliberate "incomplete quantization" strategy; it must be compared against a same-precision downstream full model |
| L-to-L upstream channel | The pip-bound `llama_get_embeddings_ith` returns `output_norm(H)` (measured cos 0.999998), so it **cannot** serve as a layer-relay upstream. The patched **keep-head channel is live**: `--path l2l_keep_head` / `d2l2l_keep_head` measured **32/32** (including a three-segment "1 torch upstream + 2 llama downstream" chain); the old `l2l_llama` stays as the fail-loud counter-example |
| Cut-point solver | `scripts/relay_cut_plan.py` + `src/relay_cut_objective.py`: fits segment profiles (fixed cost + per-layer cost) from **measured** records, then solves for the cut with `capacity_feasible` / `latency_estimate` / `risk_penalty` outputs; n-segment capable, with the Qwen3.5 4-layer-multiple hard constraint. The 2-segment loop passes on both Qwen2.5 (r2 0.96/0.99) and Qwen3.5 (0.79/0.96) |
| Windows operators | Native Windows `triton-windows==3.8.0.post28` is measured working; `PYTHONUTF8=1` is a prerequisite for the compile path; WSL2/fla is a parallel path, not the only option |
| Production positioning | Correctness evidence satisfies Relay contract admission; speed only shifts the default routing preference. Long-run, remote-artifact auto-distribution and multi-segment failure acceptance remain open |
| Current documents | Use the [current effective baseline and optimization plan](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) as the index; conflicting numbers in older reports are triaged by validity |
| Key quantitative results | **Compute is 99.3%** of the time on both sides (communication + sync + batching only 0.34%); the single largest win was removing the upstream's **idling through 20 layers** (**10.7x**); the layer pipeline loads only `embed_tokens + L0-3`, **1.47 GB vs 4.55 GB** (3.1x); falsified: the process boundary (only 8%), reusing `llama_batch` (0.09%), naive upstream layer truncation (numerically broken), treating `--override-tensor` as a speed-up (it is a capacity knob) |

**Where the bottleneck is**: the earlier "IPC is the main cost" judgement does not hold - that was an illusion masked while both sides were slow. **The leverage is in the compute on both sides (cut point, kernel, batching), not in the transport layer.**
See [Same-Process Dual-Backend Relay Implementation and Performance](archive/relay/同进程双后端接力实现与性能-2026-09-16.md) sections 12-14.

### Cut-Point Sweep Results (P0, measured 2026-09-18)

A full sweep over the upstream layer count N (N=0 means **no relay** - llama.cpp runs the whole model; the downstream is the corresponding f16 layer-cut GGUF, CPU / 8 threads).
**The conclusion depends on whether the upstream runs on CPU or GPU** - both were measured.

**Upstream on GPU (the real deployment direction).** Adding `--upstream-device cuda` to relay (with f16 + `--upstream-partial` loading only the first N layers, roughly N/24 x 4.3 GB of VRAM):

| Upstream layers N | CPU upstream total ms/step | **GPU upstream total ms/step** | Gain |
| ---: | ---: | ---: | ---: |
| 8 | 223.8 | **169.0** | 1.32x |
| 12 | 224.6 | **155.0** | 1.45x |
| 16 | 221.9 | **146.4** | 1.52x |
| **20** | 249.7 | **129.1** | **1.93x** |

- **Optimal N=20 = 129.1 ms/step**, **1.45x faster** than the 186.8 ms/step of **no relay** - relay has a clear benefit in this configuration;
- Upstream **GPU about 2.5-4.3 ms/layer**, downstream **CPU llama.cpp about 5.8-8.6 ms/layer** => **push as many layers as possible to the GPU upstream**;
- The 64-token sequence at every cut point remains **token-identical** (including the GPU upstream).

**Upstream on CPU: the opposite holds** (an early basis, kept only to delimit applicability):

- **No relay is fastest** (186.8 ms/step); relay costs about **+15%**;
- **Total time is nearly insensitive to the cut point** (N in {4,8,12,16} spans 214-225 ms/step, about +/-2.6%), **there is no intermediate valley**;
- An upstream layer (8.8-10.7 ms) is **not** cheaper than a downstream layer (7.7-8.6 ms) => in this configuration "moving layers to the torch GPU" gives no speed advantage.

=> So "no relay is fastest / cut points give no benefit" **holds only for a CPU upstream and must not be extrapolated**. In real deployments the downstream is usually a
**CUDA-less edge device**, which supports "put layers on the GPU upstream": **P1 (adding a GPU to the downstream) has narrow applicability; what is worth doing is "GPU-izing the upstream" and P2 overlap**.

**Two general constraints**:

- **Correctness does not vary with the cut point**: the greedy sequence is token-identical at every cut point;
- **The cut point must be a multiple of `full_attention_interval`** (Qwen3.5 = 4), otherwise the layer types of the layer-cut GGUF are misaligned and it fails to load (measured at N=2).

⚠️ **Methodology**: isolated measurements detached from the end-to-end chain are not trustworthy - an early sweep under-measured the upstream per-step cost by about 5.6x because the **upstream was actually running on CPU while being compared against GPU data** (same script, same basis, same 4 layers: cpu 34.9 ms / cuda 8.3 ms; the difference is explained by the device, not by KV shape or idle down-clocking). Cut-point conclusions must use the end-to-end basis.

Report: `local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-p0-corrected-2026-09-18.json`;
ticket: [acceptance list D29](验收清单与资源限制登记.md).

### Fair Comparison Against "All-llama + CUDA" + P2 Overlap (measured 2026-09-18)

**Q: When the cluster has a CUDA node, is all-llama.cpp worse than relay? A: No.**

| Configuration | ms/token | Relative |
| --- | ---: | ---: |
| **llama.cpp + CUDA** (build-cuda, `-ngl 24` all layers on GPU, f16, t=8) | **26.6** | 1.0x |
| Best cross-framework relay (torch GPU upstream N=20 + llama.cpp CPU downstream) | 129.1 | 4.85x slower |

The `-ngl` curve is monotonic (ms/token): `0->91.1`, `4->71.0`, `8->60.5`, `12->52.9`, `16->44.2`, `20->34.7`, `24->26.6`. The reference must be the **same build** - even at `-ngl 0` the build-cuda llama-bench is about 2x faster than build-cpu (91.1 vs 194.2 ms/token).

**P2 overlap** (software pipelining: the upstream GPU torch and the downstream CPU llama.cpp advance interleaved, each holding a lock; torch's CUDA calls and ctypes' llama.cpp calls both release the GIL, so they can genuinely run in parallel):

| Mode | ms/token | Throughput |
| --- | ---: | ---: |
| serial | 101.11 | 154.5 tok/s |
| **threaded (overlapped)** | **78.31** | **199.5 tok/s** |

A **1.291x** speed-up, and both sequences are **identical** to the single-sequence baseline; the theoretical ceiling is about 1.79x (taking the larger of upstream 72.3 / downstream 56.8 when fully overlapped), and the measurement reaches about 72% of it.

**Conclusion and positioning**: the 1.45x from P0 above is only the local gain of "CPU llama.cpp -> GPU **torch**"; the better move is "CPU llama.cpp -> GPU **llama.cpp**" (`-ngl`). So **cross-framework relay is not a faster inference path** - it is the mechanism for "**layers that can only run under torch**" (hybrid/custom operators) and for "**capacity merging / layer pipeline** (too large for a single machine)", plus an experiment platform. **When the cluster has a CUDA node, the best practice is to use it as a llama.cpp CUDA worker (RPC/sharding), not as a relay upstream**; P2 overlap only recovers part of the loss inside the "relay is unavoidable" scenario (78.3 ms/token is still about 3x slower than 26.6).

Report: `local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-p2-2026-09-18.json`. All of the above is **same-machine** data; the **cross-machine (GPU node + CUDA-less edge node) RPC vs. relay comparison is still unmeasured**.

### torch.compile and the "Layer Loop" Switches (`USE_COMPILE` / `USE_MONOLITHIC_FORWARD`)

Getting `torch.compile` gains out of segmented forward takes two switches (both in `src/config.py`, overridable by environment variables):

| Switch | Default | Effect |
| --- | --- | --- |
| `USE_COMPILE` | `True` | Enables compilation. When compilation is unavailable it **warns and falls back to eager** without blocking startup (this is the path taken on Windows without `triton-windows`; **installing it is enough** - native Windows compilation has been verified working since 2026-09-19, so it is no longer a 'dead switch'). |
| `USE_MONOLITHIC_FORWARD` | `False` | Additionally compiles the **"layer loop"** (`_LayerLoop`) used by `forward_layers()`; off by default. |

**Why only the layer loop is compiled, not the whole model**: `Qwen2Model.forward()`'s return value passes through `self.norm` (full-model semantics), while a distributed segmented forward must return the **pre-norm** raw hidden when `has_lm_head=False`. So only the loop is wrapped; the pre/post steps stay in `forward_layers()`, which is what keeps the semantics identical to the per-layer version.

**Measured gains** (`USE_MONOLITHIC_FORWARD=True`; see [Figure 3](figures/cross-frame-relay/fig3-compile-gains.png)):

| Scenario | Per-layer | Compiled layer loop | Speedup | Per-token argmax |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B, 12 layers (non-hybrid, prefill 64, repeats=5) | 10.269 ms/step | **6.133 ms/step** | **1.674x** | identical |
| Qwen3.5-2B, 24 layers (hybrid, prefill 32, repeats=3) | 41.058 ms/step | **32.349 ms/step** | **1.269x** | identical |

Hybrid models (Qwen3.5's 18 `linear_attention` + 6 `full_attention` layers) need **a different mask per layer type**; `_LayerLoop` supports this (using a "tuple + per-layer mask index", which also keeps `torch.compile` guards simple). NOTE: the two rows use different setups (model/layers/prefill), so the **speedups are not directly comparable**; hybrid's `linear_attention` (GatedDeltaNet) offers fewer fusion opportunities than pure attention+MLP, so a lower gain is expected.

**Three boundaries you must know**:

1. **compile and eager are not bit-identical**: the hidden difference is exactly **1 ULP of f16** (`0.015625 = 2^-6`); after ruling out every other candidate, the only remaining source is the attention implementation path (`fuse_attention` fusing bmm+softmax back into aten SDPA). **But it must not be described as "compile is worse"** - upstream reports compile has **better** rtol against a **float64** baseline; the correct wording is "**not bit-identical to eager**".
2. **Scenarios with a "per-token identical" acceptance criterion must not enable compile** (e.g. the cross-framework relay admission criterion).
3. **Windows needs two things**: `PYTHONUTF8=1` (otherwise torch/inductor decodes internally as GBK, fails, and **silently falls back to eager**) and [`triton-windows`](../requirements-compile.txt) (optional acceleration, declared in `requirements-compile.txt`; **verified working** - PyPI has no official Windows wheel, so use the community build `triton-windows-3.8.0.post28`). Missing either is non-fatal - you just do not get the gain.

Current **serial full-suite** baseline: `2991 passed / 13 skipped / 0 failed` (`-n 0`; recorded after `610f4b3`; xdist concurrency occasionally flakes - judge by the serial run).

Reports: `local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-a4-layer-loop-2026-09-18.json`, `...-b14-hybrid-layer-loop-2026-09-18.json`, `...-compile-numerics-2026-09-18.json`.

### Top-Level Transparency

The TUI and API top layer only needs to know the **aggregate resources** (GPU/CPU/memory) and "whether it is distributed"; it does not need to know who is local and who is remote. Engine choice is decided by **resources + capability + goal**, not by "torch whenever there is a GPU". Optional policies (privacy, bandwidth) are not implemented yet and are left to a later policy ticket. See [Layer Pipeline Node Kinds and Top-Level Transparency](archive/relay/层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md).

## Current Status

| Capability | Current position |
| --- | --- |
| Textual TUI | Wired into the unified `qlh` entry point; chat, 9 feature screens and 1 debug fallback screen share one process and can start the local backend on demand; write operations such as model download/search/preflight/registration, cluster config, node management, log filtering/stats/export, device config and user settings go through a confirmation gate; the old hand-drawn ANSI TUI is archived |
| Edge <=1B single machine | Model profiling, the GGUF/llama.cpp path and edge preflight exist; torch is not loaded by default |
| Same-machine two-process RPC | A llama host + `ggml-rpc-server` simulation and contract tests exist; not equivalent to cross-machine production admission |
| PC RPC | Device scoring, automatic layer planning, lease/disconnect fallback and asset-sync contracts exist; real large-model throughput is not claimed, while capacity evidence is tracked separately |
| Layer-segment contract and auto-reshard | The contract, fail-closed layout validation, capacity re-solve and atomic epoch commit development gate are done; real PC/Android fault injection, long-run and performance acceptance remain |
| Cross-framework layer relay (D to L) | Correctness evidence is admitted separately from routing speed. Dual-engine sample: Qwen2.5-0.5B, 12+12 layers, **47.501 ms/step, 32 steps identical**, plus the **2026-09-21 full matrix of 27/27 token-identical runs** (Qwen3.5-2B K=8/12/16/20 all 32/32, load tiers, batch 2/4, mixed-precision tiers). Capacity evidence is 1.568x / 1.547x under a controlled budget; long-run, cross-machine, multi-segment and remote-asset acceptance remain |
| PyTorch D track | The actual implementer of layer splitting / inter-layer pipeline / multi-node layer-segment hosting, doubling as the control experiment; not part of the Edge default dependency |
| Relay R | L to L, D to L, the f32/sampling matrix and SSH cross-machine evidence have completed correctness verification; no performance advantage, off by default, does not replace RPC |
| Android | `qlh-android` P0 cross-compilation/JNI is done; P1's on-device run, RPC worker, disconnect, thermal/power and security evidence is not |
| Runtime | **Main runtime `transformers` 5.17.0** (`huggingface_hub` 1.32 / `tokenizers` 0.23); all PyTorch sidecars (`.venv-qwen3-sidecar` / `.venv-gemma4-pipeline`) unified to 5.17.0; `.venv-test` carries `triton-windows`. Native Windows `torch.compile` **works** |
| Model assets | Model files are not in Git; the repository asset list registers Qwen2.5-0.5B, Qwen3-0.6B, MiniCPM4-0.5B, DistilQwen2.5-DS3-0324-7B and others |

Experiments that do not state real device, cross-machine or production acceptance may only be used as development evidence or PoC.

## Main Repository Boundary

| Kept in the main repository | Externalized or not flowing back |
| --- | --- |
| llama.cpp/GGUF engine adaptation, RPC/layer-segment contracts, scheduling and failure recovery | Android UI/JNI project: `qlh-android` |
| FastAPI control plane, model/node/capability contracts | Product shell and frontend: `qlh-shell` |
| Cross-platform Textual TUI, read-only command thin layer, Edge entry point and quality gates | Release/installer: `qlh-release` |
| Single-machine, same-machine two-process and PC/Android mainline experiment interfaces | Toolbox: `qlh-toolbox` |
| **PyTorch D-track engine (layer splitting, layer pipeline) and cross-framework relay implementation** | Image generation, web product UI, mail, operations workbench |
| Model registration, download verification, device profiling and distributed observability | Koakumix harness custom experiments, image generation and sidecar capability |

The main project keeps no image-generation runtime or assets; image generation belongs solely to Koakumix. Multimodality is not a fixed mainline dependency - the model fleet picks a text or vision model per device capability.

## Directory Layout

| Path | Content |
| --- | --- |
| `src/` | QLH main code: control plane, engines, layer-segment/layer-pipeline contracts, TUI (grouped below) |
| `tests/` | pytest suite (TUI, RPC/layer-segment, scheduling, contracts, doc gates) |
| `scripts/` | Verification, experiment, environment and documentation tools (`edge_preflight.py`, `android_validation.py`, `llama_rpc_*.py`, `doc_maintenance_audit.py`, ...) |
| `docs/` | Current documents; historical and migrated content lives in `docs/archive/` |
| `schemas/` | Cross-process/cross-repo JSON Schema contracts (artifact-manifest, cluster-profile, experiment-record, ...) |
| `fixtures/` | Test fixtures (including offline chat event replay, used by `qlh chat --fixture`) |
| `local_docs/` | Local experiment and acceptance raw records; not a public source interface |
| `runtime/` | Runtime logs and the llama.cpp runtime directory |
| `qlh.py` / `qlh_edge.py` | Interactive TUI/CLI entry point and Edge entry point |
| `qlh.bat` / `qlh.sh` / `bjtu.*` / `koakuma.*` | Launchers; `bjtu` and `koakuma` are compatibility aliases, the unified entry point is still `qlh` |
| `start_tui.*` / `start_backend.bat` / `setup_all_envs.*` | One-click start and multi-environment install scripts |
| `requirements*.txt` / `pytest.ini` / `pyrightconfig.json` / `reasonix.toml` | Dependency lists and tool configuration |
| `models/`, `chat_history/`, `dist/`, `build/`, `test-results/`, `logs/`, `_to_delete/` | Local artifacts or archive areas, not in Git (`logs/`, `_to_delete/` are gitignored) |

**Submodules (directly tied to main-repo functionality — 4)** — declared in `.gitmodules`, fetched by `git submodule update --init`:

| Submodule path | Remote |
| --- | --- |
| `android/` | `qlh-android` |
| `frontend_cybergothic/` | `qlh-shell` |
| `packaging/` | `qlh-release` |
| `harness_workbench/` | `Koakumix` |

**Related repositories (dev/experiment tooling — no longer submodules as of 2026-09-20)** — clone them
**separately** into the workspace; they are gitignored and **intentionally kept out of the main repository**:

| Local directory | Remote | Purpose |
| --- | --- | --- |
| `tools/docagent/` | `qlh-docagent` | Documentation maintenance scanner |
| `tools/toolbox/` | `qlh-toolbox` | Tool collection |
| `tools/reasonix-codex-bridge/` | `reasonix-codex-bridge` | Reasonix ↔ Codex controlled bridge (MCP + ACP) |
| `tools/dsh-codex-bridge/` | `dsh-codex-bridge` | Twin project (DSH side, vendors the bridge runtime) |
| `packages/spawnledger/` | `spawnledger` | Process-ownership ledger |

> **Note** — `tools/`, `packages/`, `logs/`, `docs/agent_tool/` and the one-off experiment scripts under
> `scripts/` have been **dropped from the repository** (kept locally only). Scripts that are **imported by
> `src/` or by tests**, plus onboarding/pipeline tools (`setup_envs.py`, `cut_layers.py`,
> `run_test_channels.py`, …), remain tracked.

`src/` grouped by responsibility (for navigation; per-module interfaces are in [Module Interfaces](模块接口说明.md)):

| Group | Representative modules |
| --- | --- |
| Control plane and entry points | `api_server.py` (FastAPI control plane), `api_errors.py`, `config.py`, `bootstrap.py`, `model_api_access.py`, `review.py`, `local_store.py` |
| Scheduling and cluster | `scheduler.py`, `scheduler_svc_http.py`, `cluster_join.py`, `cluster_transport.py`, `edge_cluster.py`, `node_config.py`, `node_runtime.py`, `transport_runtime.py`, `transport_port.py`, `network_address.py`, `network_path.py`, `proxy_config.py`, `wss_loopback.py` |
| Engines | `llama_engine.py` (L track), `model_module.py` (D track layer splitting), `island_engine.py`, `koakuma_engine.py`, `tcp_comm.py` (torch tensor transport), `inference_client.py`, `inference_svc_main.py`, `inference_service/`, `paged_kv_cache.py`, `external_provider.py`, `speculative*.py` |
| RPC and layer relay | `llama_rpc_contract.py`, `llama_rpc_device.py`, `llama_rpc_planner.py`, `relay_contract.py`, `relay_planner.py`, `relay_transport.py` |
| Layer-segment / layer-pipeline contracts | `pipeline_node_contract.py`, `pipeline_capacity.py`, `pipeline_assignment_manifest.py`, `pipeline_model_descriptor.py`, `pipeline_reshard.py`, `cache_unit_layout.py` |
| Cross-framework / multimodal pipelines | `qwen3_pipeline_*.py`, `qwen3_multimodal_*.py`, `gemma4_pipeline_*.py`, `multimodal.py` |
| Models and assets | `model_config.py`, `model_host.py`, `model_sync.py`, `model_downloader.py`, `model_download_jobs.py`, `model_search.py`, `model_registry_validation.py`, `model_runtime_contracts.py`, `local_model_assets.py` |
| Task graph and workflows | `task_graph*.py`, `task_journal.py`, `task_provider.py`, `task_worker_*.py`, `graph_orchestrator.py` |
| TUI and interaction | `tui_textual.py`, `tui_api.py`, `tui_shared.py`, `tui_sse.py`, `tui_backend.py`, `tui_commands.py` |
| Devices and soak | `device_profiler.py`, `provider_soak.py`; RAG is externalized to `harness_workbench`/Koakumix |

## Quick Start

### 1. Get the Code and Submodules

```bash
git clone https://github.com/SgfKrc/qlh.git
cd qlh
git submodule update --init --recursive
```

Submodule remotes are in `.gitmodules`. Common sibling repositories:

- `https://github.com/SgfKrc/qlh-android.git`
- `https://github.com/SgfKrc/qlh-shell.git`
- `https://github.com/SgfKrc/qlh-release.git`
- `https://github.com/SgfKrc/qlh-toolbox.git`
- `https://github.com/SgfKrc/qlh-docagent.git`
- `https://github.com/SgfKrc/Koakumix.git`

> **Bridges (not submodules since 2026-09-20)** — clone separately into `tools/` if you need them;
> they are gitignored and intentionally kept out of the main repository:
> - `https://github.com/SgfKrc/reasonix-codex-bridge.git`
> - `https://github.com/SgfKrc/dsh-codex-bridge.git`

### 2. Choose a Runtime Environment

The main environment includes torch and suits the D-track PyTorch layer pipeline, cross-framework relay and the full API:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The main requirements include the pinned CPU/GGUF fallback
`llama-cpp-python==0.3.35` alongside the PyTorch D track. This is the default CPU
wheel contract; CUDA llama.cpp builds are optional separately built artifacts and
are not implied by the main lock.

The Edge environment installs only GGUF/llama.cpp and control-plane dependencies, without torch, Transformers or bitsandbytes:

```powershell
python -m venv .venv-edge
.\.venv-edge\Scripts\python.exe -m pip install -r requirements-edge.txt
.\.venv-edge\Scripts\python.exe scripts/edge_preflight.py --python .venv-edge\Scripts\python.exe --json
python scripts/llama_dependency_contract.py --json
```

The managed Gemma 4 MTMD profile is separate: `.venv-gemma4-native` uses the
frozen `llama-cpp-python==0.3.28` binding and must not reuse the ordinary CPU
wheel. Its ABI marker and lock are checked independently.

On Linux/macOS replace `Scripts\python.exe` with `bin/python`. The interactive TUI needs `Textual`; read-only commands, the protocol layer and CI checks do not. `uvicorn/FastAPI` is only needed when the local backend is started automatically - a remote TUI will not start a backend on the remote host.

If you only need the interactive TUI, you can install it on top of a complete environment:

```powershell
python -m pip install -r requirements-tui.txt
```

### 3. Start the TUI

```bash
python qlh.py chat
python qlh.py chat --route distributed_preferred --thinking
python qlh.py chat --fixture fixtures/chat.json
python qlh.py status
python qlh.py models
```

When the local backend is not running, `qlh chat` starts it in a daemon thread of the current process and shows the liveness stage on the Textual splash screen. Backend logs are not flushed into the TUI; they go to the existing log file and the logs screen. Single commands such as `status` and `models` do not start the backend automatically; `--fixture` is the offline chat event replay path.

Write operations are initiated from the shell: the models screen uses `L` to load / `U` to unload, the queue screen uses `P` pause-resume / `S` policy / `C` clear queued, and the chat screen supports `/model`, `/queue`, `/new`, `/resume`, `/rename`, `/sessions`, `/delete-session`, `/reset`; destructive and long-running operations first show a confirmation box. Model control endpoints are allowed by default on loopback; to control a main node remotely the main node must configure `QLH_MODEL_API_TRUSTED_CIDRS`.

On Windows you can use `qlh.bat` directly; on Linux/macOS use `qlh.sh`. `bjtu`/`koakuma` are compatibility launchers, and the unified repository entry point is still `qlh`.

The TUI's 9 feature screens are the main interaction and acceptance boundary: the models screen covers local assets/presets/download jobs, search, preflight, registration, load and unload; the distributed/nodes screen covers toggles, capacity, max nodes, invite, connect, join-request code/authorization consumption and deregistration; the logs screen covers filtering, statistics, export and clearing; the device screen covers auto-configuration and GPU selection; the settings screen reads and writes user settings. The final "debug" screen reads routes dynamically from `/openapi.json` of the running backend and serves only as a JSON fallback for operations that have no dedicated interaction yet; it does not count as product feature coverage. The current main backend OpenAPI snapshot is 152 operations; the actual number is whatever the target backend returns. Streaming chat and file upload are still handled by the chat page specifically.

## Models and Distribution

Model artifacts, download caches and large GGUF files do not enter Git. The model list and capability profiles are managed by the repository API/TUI, and a model must pass format, digest, architecture, template, thinking, device budget and provenance validation before it can enter the usable list.

Current recommended validation order:

1. Load a single <=1B GGUF locally and verify the template, thinking toggle and streaming output.
2. Start a host and `ggml-rpc-server` on the same machine and verify partial residency, capacity merging, output comparison and worker disconnect.
3. On a PC node, complete real RPC, asset sync, lease and failure-recovery acceptance.
4. When cross-framework or inter-layer pipeline is involved, first re-check the numerical consistency and performance boundary of the D to L relay locally.
5. Then move on to ARM64/Android worker acceptance.

Do not claim "the model is sharded" based on full-model replication, whole-request parallelism through the task graph, or old two-machine PyTorch results. Small-model bypass on node failure is a scheduling policy, not a separate Lite product.

## Android Validation

The Android project lives in the external submodule `android/`. When the development machine has no real Android device, use layered evidence:

```powershell
# JVM protocol/state-machine/capability contract tests
python scripts/android_validation.py

# Also build the fullDebug APK
python scripts/android_validation.py --assemble

# With an emulator or adb device connected, install and launch the control plane
python scripts/android_validation.py --assemble --install --launch --serial emulator-5554

# Emit machine-readable evidence
python scripts/android_validation.py --assemble --json
```

An x86_64 emulator can validate the APK, UI, permissions, networking and lifecycle, but cannot prove the `arm64-v8a` JNI RPC worker; an ARM64 AVD/QEMU can only add ARM compatibility and cannot replace real-phone thermal/power, background-reclaim, weak-network and long-run testing. The full Android P1 criteria and the AVD/QEMU/remote-adb plan are in [Android Validation Alternative Paths](../android/Android验证替代路径-2026-09-18.md). The old hand-drawn ANSI TUI has been moved to `_to_delete/`; do not treat it as the current interaction implementation or test entry point.

## Testing

Use an environment separate from the main one for tests:

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pytest -q
```

Targeted checks for high-risk mainlines:

```powershell
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_tui_textual.py tests/test_tui_write_ops.py tests/test_tui_shared.py tests/test_tui_sse.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_llama_rpc_planner.py tests/test_llama_rpc_device.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_pipeline_node_contract.py tests/test_pipeline_reshard.py tests/test_pipeline_capacity.py
```

Real hardware, cross-machine networking, Android ARM64, performance and long-run soak must additionally preserve the raw commands, environment, model digests, topology, output and failure boundaries; a green test run alone does not replace that evidence.

**Documentation checks** (purely static, standard library only): `python scripts/run_doc_checks.py` runs the relative-link check and the README bilingual-structure check. The very same suite runs twice — in CI ([`.github/workflows/checks.yml`](../.github/workflows/checks.yml)) and in a local pre-push hook ([`.githooks/`](../.githooks/README.md), enabled with `git config core.hooksPath .githooks`) — i.e. **defined once, reused in both places**. The immediate reason for adding it: the repository had no automated checks at all, and archived documents easily leave behind "reference not updated" dead links — a single pass turned up 20 of them.

## Documentation Index

- [Current D-to-L Baseline and Optimization Plan (2026-09-21)](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)
- [Test Quality Audit (2026-09-21): Parallel Flakiness and Race-Coverage Gaps](archive/misc/测试质量审计-2026-09-21.md)
- [P4.5 Proposal: Dynamic Master Election and Distributed Management](主节点动态选举与分布式管理-P4.5立项-2026-09-21.md)
- [Mainline Development Plan: Distributed Inference and Edge Optimization](主线开发计划-分布式推理与边缘优化-2026-09-14.md)
- [Overall Architecture](整体架构.md)
- [Layer-Segment Protocol Proposal (2026-09-17)](archive/relay/层段协议立项-2026-09-17.md)
- [Layer Pipeline Node Kinds and Top-Level Transparency](archive/relay/层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md)
- [Same-Process Dual-Backend Relay Implementation and Performance](archive/relay/同进程双后端接力实现与性能-2026-09-16.md)
- [Engine Single-Sequence and Concurrency Comparison](archive/relay/引擎单序列与并发性能对比-2026-09-16.md)
- [Distributed Inference Parallelism and Cross-Framework Route Survey](分布式推理并行与跨框架路线调研汇总-2026-09-15.md)
- [TUI User Guide](TUI使用指南.md)
- [TUI Feature Screens and Debug Fallback](TUI使用指南.md#调试兜底非功能验收)
- [TUI Command Set](TUI指令集.md)
- [Edge Device Simulation Environment Plan](archive/edge/边缘设备模拟环境计划-2026-09-15.md)
- [Android Validation Alternative Paths](../android/Android验证替代路径-2026-09-18.md)
- [Baseline Rewrite Plan](archive/runtime/基线重写方案-2026-09-16.md)
- [Module Interfaces](模块接口说明.md)
- [Testing and Evaluation Criteria](测试与评判标准.md)
- [Document Status and Cleanup List](文档状态与清理清单.md)

Historical plans and migrated capabilities live in `docs/archive/`; local experiment artifacts and acceptance raw records live in `local_docs/` and are not a public source interface.

## License

The main repository uses the [MIT License](../LICENSE). Each submodule and the upstream `llama.cpp` keep their own license and version lock, unchanged by being referenced here.
