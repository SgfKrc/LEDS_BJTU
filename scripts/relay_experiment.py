#!/usr/bin/env python
"""relay_experiment.py — P1 统一（只读）跨框架接力实验驱动。

取代此前散落的两个脚本：
  * `build/cross-framework-layer-poc/relay_mainrepo_both.py`（主仓下游端到端）
  * `build/cross-framework-layer-poc/relay_mainrepo_upstream.py`（裸 llama_cpp 下游探针）

四条链路（`--path`），`kind` 由引擎接口**自动判定**并强制写入 relay experiment record
（`schemas/relay-experiment-record.schema.json`）：

| `--path` | `kind` | 上游 | 下游 |
|---|---|---|---|
| `d2l_mainrepo` | `mainrepo_end_to_end` | `model_module.load_layer_range` + `forward_layers` | `llama_engine.forward_layers_from_hidden` |
| `d2l_raw_binding` | `raw_binding_probe` | 同上 | 裸 `llama_cpp`（`llama_decode`） |
| `l2l_llama` | `raw_binding_probe` | `llama_engine.forward_layers_to_hidden`（head GGUF） | `llama_engine.forward_layers_from_hidden` |
| `capacity_scan` | `capacity_only` | `model_module.load_layer_range`（只加载） | `llama_cpp.llama_model_load_from_file`（只加载） |

判据：`criterion = per_token_argmax`（引用 `src/relay_contract.RELAY_ACCEPTANCE`）—— 每条序列的
贪心 token 必须与**同精度整模**（裸 llama.cpp）逐 token 一致。记录在写盘前必须过 schema 校验
（fail-closed），因此不同 kind 的数字无法被混标成端到端基线。

本驱动**只读仓库**：不修改任何源码/文档，只把记录写到 `--json-out` 指定的路径。

用法::

    # 端到端（主仓下游），Qwen2.5-0.5B / K=12
    python scripts/relay_experiment.py --path d2l_mainrepo \
        --model-dir models/qwen2.5-0.5b-instruct --layers 12 \
        --cut-model build/cross-framework-layer-poc/out/qwen25-05b-f16-cut-k12.gguf \
        --whole-model build/cross-framework-layer-poc/out/qwen25-05b-f16.gguf \
        --prefill 32 --gen 32 --batch 1 --json-out build/relay-records/x.json

    # 预检：不加载模型，只产出（并校验）记录骨架
    python scripts/relay_experiment.py --path capacity_scan --dry-run --json-out -
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.relay_contract import (  # noqa: E402
    RELAY_ACCEPTANCE,
    RelayHiddenSpec,
    RelayXFrameEvidence,
)
from src.relay_experiment_record import (  # noqa: E402
    IFACE_KEEP_HEAD_UPSTREAM,
    IFACE_LLAMA_ENGINE_DOWNSTREAM,
    IFACE_LLAMA_MODEL_LOADER,
    IFACE_LLAMA_UPSTREAM,
    IFACE_MODEL_MODULE_LOADER,
    IFACE_MODEL_MODULE_UPSTREAM,
    IFACE_RAW_LLAMA_DOWNSTREAM,
    PATH_CAPACITY,
    PATH_D2L2L_KEEP_HEAD,
    PATH_D2L_MAINREPO,
    PATH_D2L_RAW,
    PATH_L2L,
    PATH_L2L_KEEP_HEAD,
    build_record,
    write_record,
)

#: path -> (上游接口, 下游接口, 中间段接口)。kind 由 build_record()/classify_path() 反推。
#: ⚠️ 中间段接口必须显式登记：否则三段链路会与两段 D→L 撞键（那就是"混表"）。
_IFACES_BY_PATH: dict[str, tuple[str, str, str]] = {
    PATH_D2L_MAINREPO: (IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, ""),
    PATH_D2L_RAW: (IFACE_MODEL_MODULE_UPSTREAM, IFACE_RAW_LLAMA_DOWNSTREAM, ""),
    PATH_L2L: (IFACE_LLAMA_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, ""),
    PATH_L2L_KEEP_HEAD: (IFACE_KEEP_HEAD_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, ""),
    PATH_D2L2L_KEEP_HEAD: (IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM,
                           IFACE_KEEP_HEAD_UPSTREAM),
    PATH_CAPACITY: (IFACE_MODEL_MODULE_LOADER, IFACE_LLAMA_MODEL_LOADER, ""),
}

#: 用 keep-head 通道当上游/中间段的链路（需要 `--keep-head-shim`）。
KEEP_HEAD_PATHS = (PATH_L2L_KEEP_HEAD, PATH_D2L2L_KEEP_HEAD)

_QUANT_TYPE_BY_UPSTREAM = {
    "fp16": None,      # 沿用主仓 profile / ckpt 的 fp16
    "f32": None,       # 加载后 post-load cast
    "int8": "int8",
    "int4": "int4",    # ⚠️ 主仓层流水线实测回退 fp16（见记录里的 dtype_effective）
    "nf4": None,       # 加载后显式替换 nn.Linear -> Linear4bit
}


# --------------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def _rss_gb() -> float | None:
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 1e9, 4)


def _cuda_gb() -> float | None:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1e9, 4)
    except Exception:  # noqa: BLE001
        return None


def _param_bytes(param: Any) -> int:
    """参数字节数；bitsandbytes `Params4bit` 按 4-bit 打包估算。"""
    if type(param).__name__ == "Params4bit":
        return int(param.numel() // 2)
    return int(param.numel() * param.element_size())


def _replace_linear_nf4(model: Any, device: Any) -> int:
    """把 `nn.Linear` 递归替换为 bitsandbytes `Linear4bit`（NF4）。返回替换个数。

    为什么必须显式替换：`model_module.load_layer_range()` 按 key 手工物化权重，
    **不走** `BitsAndBytesConfig` ⇒ 传 `quant_type="int4"` 实测回退 fp16。
    """
    import bitsandbytes as bnb  # noqa: PLC0415
    import torch  # noqa: PLC0415

    replaced = 0

    def _walk(parent: Any) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
                new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                        bias=child.bias is not None,
                                        compute_dtype=torch.float16, quant_type="nf4",
                                        compress_statistics=True)
                new.weight = bnb.nn.Params4bit(child.weight.detach().clone(),
                                               requires_grad=False, quant_type="nf4",
                                               compress_statistics=True)
                if child.bias is not None:
                    new.bias = torch.nn.Parameter(child.bias.detach().clone())
                setattr(parent, name, new.to(device))
                replaced += 1
            else:
                _walk(child)

    _walk(model)
    return replaced


def _n_pos_per_embd(gguf_path: Path) -> int:
    """M-RoPE 的 pos 段数（`<arch>.rope.dimension_sections` 长度）；1D RoPE 为 1。"""
    try:
        import gguf  # noqa: PLC0415

        reader = gguf.GGUFReader(str(gguf_path))
        for key, field in reader.fields.items():
            if key.endswith(".rope.dimension_sections"):
                return max(1, len(field.data))
    except Exception:  # noqa: BLE001
        pass
    return 1


def _build_prompt_tokens(tok: Any, prompt_path: Path, prefill: int) -> list[int]:
    text = prompt_path.read_text(encoding="utf-8").strip()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    if prefill <= 0:
        return ids
    if len(ids) >= prefill:
        return ids[:prefill]
    out: list[int] = []
    while len(out) < prefill:
        out.extend(ids)
    return out[:prefill]


def _performance_verdict(relay_ms: float | None, baseline_ms: float | None) -> str:
    if not relay_ms or not baseline_ms:
        return "unknown"
    return "advantageous" if relay_ms < baseline_ms else "not_advantageous"


def _check_l2l_upstream_channel(path: str, allow_normed_upstream: bool,
                                keep_head_shim: str | None = None) -> None:
    """L→L 上游通道守卫（fail-loud）。

    实测证据（`scripts/relay_diag_head_norm.py`，qwen2.5-0.5B）：
    pip 绑定的 `llama_get_embeddings_ith`（即 `llama_engine.forward_layers_to_hidden()` 的底层）
    返回的是 **`output_norm(H)`** —— 与 `RMSNorm(H) * model.norm.weight` 的
    `rel_err=0.0018 / cos=0.999998`。而层接力上游需要的是**未过 final norm** 的层输出，
    因此该通道直接当上游会多一次归一化，必然与整模对拍分叉（实测 `first_mismatch=2`）。

    正确做法（P2 路线 C，已验证 32/32）：
      用**补丁版 keep-head 通道**（`--path l2l_keep_head` / `d2l2l_keep_head`）——
      `llama_set_embeddings_nextn` / `llama_get_embeddings_nextn_ith` 给出**末层输出
      （`output_norm` 之前）**，配 head 裁层工件即得前 K 层输出。
    """
    if path in KEEP_HEAD_PATHS:
        if not keep_head_shim:
            raise SystemExit(
                f"FAIL: --path {path} 需要 --keep-head-shim 指向 qlh_keep_head.dll"
                "（用 scripts/model_tools/build_keep_head_shim.ps1 生成）")
        return
    if path == PATH_L2L and not allow_normed_upstream:
        raise SystemExit(
            "FAIL: l2l_llama 的上游通道语义不成立 —— pip 绑定的 embeddings 通道返回 "
            "output_norm(H)（实测 vs RMSNorm(H)*w：rel_err=0.0018 / cos=0.999998），"
            "比层接力上游所需的 hidden 多一次归一化。\n"
            "  正确做法：(a) 用 PyTorch 上游：--path d2l_mainrepo；或 (b) 用补丁版 keep-head "
            "通道：--path l2l_keep_head（+ --keep-head-shim / --upstream-model=head 工件），"
            "三段链路用 --path d2l2l_keep_head。\n"
            "  若你明确只要「含 output_norm 的对照」，显式加 --allow-normed-upstream 自行承担口径。")


# --------------------------------------------------------------------------- args
def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P1 统一跨框架接力实验驱动（只读）")
    ap.add_argument("--path", required=True, choices=sorted(_IFACES_BY_PATH))
    ap.add_argument("--model-dir", default=None, help="上游 PyTorch 模型目录（d2l_* 必需）")
    ap.add_argument("--upstream-model", default=None,
                    help="上游 GGUF（l2l_llama / l2l_keep_head 必需：前者是 head 工件、"
                         "后者是前 K 层的裁层工件）")
    ap.add_argument("--keep-head-shim", default=None,
                    help="补丁版 keep-head shim（qlh_keep_head.dll）；l2l_keep_head 与 "
                         "d2l2l_keep_head 必需。生成：scripts/model_tools/build_keep_head_shim.ps1")
    ap.add_argument("--mid-model", default=None,
                    help="d2l2l_keep_head 的中间段工件（保留 blk.K1..K2-1 的裁层 GGUF）")
    ap.add_argument("--mid-layers", type=int, default=None,
                    help="d2l2l_keep_head 的第二个切点 K2（中间段覆盖 blk.K1..K2-1）")
    ap.add_argument("--layers", type=int, default=12, help="切点 K：上游层数")
    ap.add_argument("--cut-model", required=True, help="下游裁层 GGUF（保留后 N-K 层）")
    ap.add_argument("--whole-model", required=True, help="对照整模 GGUF（同精度）")
    ap.add_argument("--prompt", default=str(ROOT / "build" / "cross-framework-layer-poc" / "out"
                                            / "prompt-france.txt"))
    ap.add_argument("--prefill", type=int, default=32)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--upstream-quant", default="fp16",
                    choices=["fp16", "int8", "int4", "f32", "nf4"])
    ap.add_argument("--downstream-dtype", default=None, help="记录用：下游工件量化（f16/q4_k_m…）")
    ap.add_argument("--downstream-quant", default=None,
                    help="容量专项用：下游工件量化标签（默认取 --downstream-dtype）")
    ap.add_argument("--allow-normed-upstream", action="store_true",
                    help="⚠️ 允许 l2l_llama 用 pip 绑定的 embeddings 通道作上游（该通道 = "
                         "output_norm(H)，比层接力所需的 hidden 多一次归一化 ⇒ 结果必然分叉）")
    ap.add_argument("--json-out", default=None, help="记录输出路径；`-` 表示打到 stdout")
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载任何模型，只产出并校验记录骨架（链路/字段预检）")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- run
def _load_upstream(args: argparse.Namespace) -> dict[str, Any]:
    """上游 PyTorch 层段（d2l_* / capacity_scan 用）。"""
    import torch  # noqa: PLC0415

    import config as _cfg  # noqa: PLC0415
    import model_module  # noqa: PLC0415

    _cfg.USE_COMPILE = True
    model_module.USE_COMPILE = True
    _cfg.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False

    started = time.perf_counter()
    mgr = model_module.ModelManager()
    mgr.load_layer_range(0, args.layers, has_embedding=True, has_lm_head=False,
                         model_path=args.model_dir,
                         quant_type=_QUANT_TYPE_BY_UPSTREAM.get(args.upstream_quant))
    load_s = round(time.perf_counter() - started, 2)

    device = mgr.get_device()
    if args.upstream_quant == "f32":
        mgr.model.to(torch.float32)          # 主仓没有「加载时 f32」档 ⇒ post-load cast
    nf4_replaced = 0
    if args.upstream_quant == "nf4":
        nf4_replaced = _replace_linear_nf4(mgr.model, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if nf4_replaced == 0:
            raise SystemExit("FAIL: nf4 档未替换任何 Linear（上游仍是 fp16？）")

    dtype = next(mgr.model.parameters()).dtype
    return {
        "mgr": mgr,
        "load_s": load_s,
        "device": str(device),
        "dtype": str(dtype),
        "nf4_replaced": nf4_replaced,
        "param_bytes": int(sum(_param_bytes(p) for p in mgr.model.parameters())),
        "load_mode": (getattr(mgr, "_layer_load_metrics", {}) or {}).get("mode"),
        "compiled": getattr(mgr, "_compiled_transformer", None) is not None,
        "n_embd": int(mgr.model.config.hidden_size),
    }


def _load_keep_head_segment(args: argparse.Namespace, model_path: str, *,
                            role: str) -> dict[str, Any]:
    """用补丁版 keep-head 通道加载一个段（`role` 仅用于记录）。"""
    from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

    extra = [d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "").split(os.pathsep) if d]
    started = time.perf_counter()
    upstream = KeepHeadUpstream(args.keep_head_shim, model_path, mode="nextn",
                                n_ctx=args.n_ctx, n_threads=args.threads,
                                extra_dll_dirs=extra)
    return {
        "keep_head": upstream,
        "load_s": round(time.perf_counter() - started, 2),
        "device": "cpu",
        "dtype": "float32",
        "nf4_replaced": 0,
        "param_bytes": Path(model_path).stat().st_size,
        "load_mode": f"keep_head_shim(nextn, {role})",
        "compiled": False,
        "n_embd": upstream.n_embd,
        "n_layer": upstream.n_layer,
    }


def _baseline_tokens(args: argparse.Namespace, prompt: list[int]) -> tuple[list[int], dict[str, Any]]:
    """对照：纯 llama.cpp 整模贪心（裸绑定，同精度）。"""
    import numpy as np  # noqa: PLC0415
    import llama_cpp.llama_cpp as M  # noqa: PLC0415

    whole = Path(args.whole_model)
    model_params = M.llama_model_default_params()
    model = M.llama_model_load_from_file(str(whole).encode("utf-8"), model_params)
    if not model:
        raise SystemExit(f"FAIL: 加载整模 {whole} 失败")
    ctx_params = M.llama_context_default_params()
    ctx_params.n_ctx = args.n_ctx
    ctx_params.n_batch = 512
    ctx_params.n_ubatch = 512
    ctx_params.n_threads = args.threads
    ctx = M.llama_init_from_model(model, ctx_params)
    if not ctx:
        raise SystemExit(f"FAIL: 初始化 {whole} 的 context 失败")

    model_bytes = int(M.llama_model_size(model))
    n_vocab = int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(model)))
    batch = M.llama_batch_init(512, 0, 1)
    tokens: list[int] = []
    step_ms: list[float] = []
    pos = 0
    try:
        for step in range(args.gen):
            toks = list(prompt) if step == 0 else [tokens[-1]]
            for i, tid in enumerate(toks):
                batch.token[i] = tid
                batch.pos[i] = pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1 if i == len(toks) - 1 else 0
            batch.n_tokens = len(toks)
            started = time.perf_counter()
            if M.llama_decode(ctx, batch) != 0:
                raise SystemExit("FAIL: 整模 decode 失败")
            logits = M.llama_get_logits_ith(ctx, len(toks) - 1)
            tokens.append(int(np.ctypeslib.as_array(logits, shape=(n_vocab,)).argmax()))
            step_ms.append((time.perf_counter() - started) * 1000)
            pos += len(toks)
    finally:
        M.llama_batch_free(batch)
        M.llama_free(ctx)
        M.llama_model_free(model)

    return tokens, {
        "model_bytes": model_bytes,
        "artifact_sha256": _sha256(whole),
        "decode_ms": _stat(step_ms[1:]),
    }


def _load_downstream_engine(args: argparse.Namespace):
    """下游走主仓 `llama_engine`（mainrepo_end_to_end / l2l_llama）。"""
    import llama_cpp.llama_cpp as M  # noqa: PLC0415

    from llama_engine import LlamaCppEngine  # noqa: PLC0415

    engine = LlamaCppEngine()
    started = time.perf_counter()
    engine.load_model(model_path=str(args.cut_model), n_ctx=args.n_ctx,
                      n_threads=args.threads, n_seq_max=max(1, args.batch))
    if not engine.is_loaded:
        raise SystemExit("FAIL: LlamaCppEngine 未加载成功")
    native = engine._model._model.model
    return {
        "engine": engine,
        "load_s": round(time.perf_counter() - started, 2),
        "model_bytes": int(M.llama_model_size(native)),
        "layers": int(M.llama_model_n_layer(native)),
        "n_embd": int(M.llama_model_n_embd_inp(native)),
    }


def _run_relay(args: argparse.Namespace, prompt: list[int], upstream: dict[str, Any],
               downstream: dict[str, Any]) -> dict[str, Any]:
    """接力生成（d2l_* 与 l2l_llama 共用；下游模式由 --path 决定）。"""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import llama_cpp.llama_cpp as M  # noqa: PLC0415

    batch_n = max(1, args.batch)
    upstream_tokens: list[list[int]] = [[] for _ in range(batch_n)]
    up_prefill_ms = dn_prefill_ms = None
    up_decode: list[float] = []
    dn_decode: list[float] = []
    mid_decode: list[float] = []
    failure: str | None = None
    pos = 0

    if args.path in (PATH_L2L, PATH_L2L_KEEP_HEAD):
        keep_head_up = upstream.get("keep_head")
        llama_up = upstream.get("upstream_engine")

        def _up_hidden(tokens, pos, want_all):
            """上游取 hidden：keep-head 通道（补丁版，末层输出）或 llama_engine embeddings。"""
            if keep_head_up is not None:
                return np.asarray(keep_head_up.forward_tokens_to_hidden(tokens, n_past=pos),
                                  dtype=np.float32)
            return np.asarray(
                llama_up.forward_layers_to_hidden(tokens, n_past=pos, all_positions=want_all),
                dtype=np.float32)

        ids_t = [[int(t) for t in prompt]] * batch_n
        upstream_pos = 0
        while len(upstream_tokens[0]) < args.gen:
            is_prefill = not upstream_tokens[0]
            started = time.perf_counter()
            if is_prefill and batch_n == 1:
                hidden = _up_hidden(ids_t[0], 0, True)
            else:
                hidden = np.stack([_up_hidden(ids_t[b], upstream_pos, False)
                                   for b in range(batch_n)], axis=0)
            up_ms = (time.perf_counter() - started) * 1000
            up_prefill_ms = up_ms if is_prefill else up_prefill_ms
            if not is_prefill:
                up_decode.append(up_ms)
            # ⚠️ 末位通道返回 1D（长度 n_embd）、整段通道返回 2D ⇒ 先规范成 2D，
            #    并按**行数**（token 数）取 n_tok —— 曾误用 shape[1]（特征维）导致索引越界。
            if hidden.ndim == 1:
                hidden = hidden[None, :]
            n_tok = int(hidden.shape[0])
            hidden = np.ascontiguousarray(hidden.reshape(-1, hidden.shape[-1]), dtype=np.float32)
            logits, dn_ms = downstream["forward"](hidden, is_prefill, pos)
            if logits is None:
                failure = f"下游未返回 logits（step {len(upstream_tokens[0])}）"
                break
            dn_prefill_ms = dn_ms if is_prefill else dn_prefill_ms
            if not is_prefill:
                dn_decode.append(dn_ms)
            for b in range(batch_n):
                upstream_tokens[b].append(int(np.asarray(
                    logits[b * n_tok + n_tok - 1]).argmax()))
            ids_t = [[upstream_tokens[b][-1]] for b in range(batch_n)]
            upstream_pos += n_tok
            pos += n_tok
    else:
        mgr = upstream["mgr"]
        device = upstream["device"]
        forward = downstream["forward"]
        middle = upstream.get("keep_head_middle")     # 三段链路的中段（keep-head）
        mid_pos = 0
        for _ in range(max(0, args.warmup)):
            with torch.no_grad():
                out = mgr.forward_layers(
                    input_ids=torch.tensor([prompt], dtype=torch.long, device=device),
                    past_key_values=None, use_cache=True)
            del out
        ids_t = torch.tensor([list(prompt)] * batch_n, dtype=torch.long, device=device)
        past = None
        for step in range(args.gen):
            is_prefill = step == 0
            started = time.perf_counter()
            with torch.no_grad():
                out = mgr.forward_layers(input_ids=ids_t, past_key_values=past, use_cache=True)
            # ★ hybrid（Qwen3.5）必须回传 cache 对象：tuple 只带 KV，recurrent state 会丢。
            past = out.get("cache") or out.get("past_key_values", past)
            up_ms = (time.perf_counter() - started) * 1000
            up_prefill_ms = up_ms if is_prefill else up_prefill_ms
            if not is_prefill:
                up_decode.append(up_ms)

            hidden_t = out["hidden_states"]
            n_tok = int(hidden_t.shape[1])
            hidden = np.ascontiguousarray(
                hidden_t.reshape(batch_n * n_tok, -1).to(torch.float32).cpu().numpy(),
                dtype=np.float32)
            if middle is not None:
                # 三段：把上游 hidden 交给 keep-head 中段，吃 hidden 吐 hidden
                started = time.perf_counter()
                hidden = middle.forward_hidden_to_hidden(hidden, n_past=mid_pos)
                mid_ms = (time.perf_counter() - started) * 1000
                if not is_prefill:
                    mid_decode.append(mid_ms)
                mid_pos += n_tok
            logits, dn_ms = forward(hidden, is_prefill, pos)
            if logits is None:
                failure = f"下游未返回 logits（step {step}）"
                break
            dn_prefill_ms = dn_ms if is_prefill else dn_prefill_ms
            if not is_prefill:
                dn_decode.append(dn_ms)
            for b in range(batch_n):
                upstream_tokens[b].append(int(np.asarray(logits[b * n_tok + n_tok - 1]).argmax()))
            ids_t = torch.tensor([[upstream_tokens[b][-1]] for b in range(batch_n)],
                                 dtype=torch.long, device=device)
            pos += n_tok

    return {
        "tokens": upstream_tokens,
        "failure": failure,
        "upstream_prefill_ms": up_prefill_ms,
        "downstream_prefill_ms": dn_prefill_ms,
        "upstream_decode_ms": _stat(up_decode),
        "downstream_decode_ms": _stat(dn_decode),
        "middle_decode_ms": _stat(mid_decode),
    }


def _make_downstream_forwarder(args: argparse.Namespace, downstream: dict[str, Any]):
    """返回 `forward(hidden, is_prefill, pos) -> (logits, elapsed_ms)`，按 path 选择下游实现。"""
    import numpy as np  # noqa: PLC0415
    import llama_cpp.llama_cpp as M  # noqa: PLC0415

    if args.path == PATH_D2L_RAW:
        n_pe = _n_pos_per_embd(Path(args.cut_model))
        native_ctx = downstream["raw_ctx"]
        n_vocab = downstream["n_vocab"]

        def forward(hidden, is_prefill, pos):
            n_tokens = int(hidden.shape[0])
            batch = M.llama_batch_init(n_tokens * n_pe, downstream["n_embd"], 1)
            try:
                for i in range(n_tokens):
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = 1
                for section in range(n_pe):
                    for i in range(n_tokens):
                        batch.pos[section * n_tokens + i] = pos + i
                batch.n_tokens = n_tokens
                ctypes.memmove(batch.embd, hidden.ctypes.data, hidden.nbytes)
                started = time.perf_counter()
                rc = M.llama_decode(native_ctx, batch)
                # ★ 契约：各分支一律返回 [n_tokens, n_vocab] 的 ndarray（主仓分支同形）。
                #   曾经在这里直接返回 ctypes 指针 ⇒ 调用方按数组取行得到 0 维对象 ⇒ 全 0 输出。
                rows = ([] if rc != 0 else
                        [np.ctypeslib.as_array(M.llama_get_logits_ith(native_ctx, i),
                                               shape=(n_vocab,)).copy()
                         for i in range(n_tokens)])
                elapsed = (time.perf_counter() - started) * 1000
                if rc != 0 or not rows:
                    return None, elapsed
                return np.stack(rows, axis=0), elapsed
            finally:
                M.llama_batch_free(batch)

        return forward

    engine = downstream["engine"]

    def forward(hidden, is_prefill, pos):
        n_tokens = int(hidden.shape[0])
        n_seq = max(1, args.batch)
        per_seq = n_tokens // n_seq
        seq_ids = [b for b in range(n_seq) for _ in range(per_seq)]
        positions = [pos + i for _ in range(n_seq) for i in range(per_seq)]
        started = time.perf_counter()
        logits = engine.forward_layers_from_hidden(hidden, seq_ids=seq_ids, positions=positions,
                                                   all_logits=True)
        return logits, (time.perf_counter() - started) * 1000

    return forward


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import llama_cpp.llama_cpp as M  # noqa: PLC0415

    from transformers import AutoTokenizer  # noqa: PLC0415

    cut_path = Path(args.cut_model)
    whole_path = Path(args.whole_model)
    prompt_path = Path(args.prompt)
    upstream_iface, downstream_iface, middle_iface = _IFACES_BY_PATH[args.path]
    _check_l2l_upstream_channel(args.path, args.allow_normed_upstream, args.keep_head_shim)
    if args.path in (PATH_L2L, PATH_L2L_KEEP_HEAD) and args.batch != 1:
        raise SystemExit("FAIL: L→L（含 keep_head）目前只支持 --batch 1")
    if args.path == PATH_D2L2L_KEEP_HEAD and (not args.mid_model or args.mid_layers is None):
        raise SystemExit("FAIL: --path d2l2l_keep_head 需要 --mid-model 与 --mid-layers")
    if not args.model_dir:
        raise SystemExit(
            "FAIL: 需要 --model-dir（HF 模型目录）—— 它同时是 tokenizer 来源；"
            "l2l_llama 的 --upstream-model 是 GGUF，不能当 tokenizer")

    models: dict[str, Any] = {}
    layer_layout: dict[str, Any] = {"upstream_layers": args.layers}
    device_profile: dict[str, Any] = {"threads": args.threads}
    metrics: dict[str, Any] = {}
    verdict: dict[str, Any] = {"passed": False, "criterion": RELAY_ACCEPTANCE,
                               "failure": None, "tokens_match": None}
    n_embd = None

    M.llama_backend_init()

    # ---- 容量专项：只加载、不生成 ----
    if args.path == PATH_CAPACITY:
        upstream = _load_upstream(args)
        n_embd = upstream["n_embd"]
        model_params = M.llama_model_default_params()
        native = M.llama_model_load_from_file(str(cut_path).encode("utf-8"), model_params)
        if not native:
            raise SystemExit(f"FAIL: 加载 {cut_path} 失败")
        downstream_bytes = int(M.llama_model_size(native))
        downstream_layers = int(M.llama_model_n_layer(native))
        M.llama_model_free(native)
        whole_native = M.llama_model_load_from_file(str(whole_path).encode("utf-8"), model_params)
        whole_bytes = int(M.llama_model_size(whole_native)) if whole_native else None
        if whole_native:
            M.llama_model_free(whole_native)
        max_segment = max(upstream["param_bytes"], downstream_bytes)
        models = {
            "upstream": {"id": Path(args.model_dir or "").name or None, "path": args.model_dir,
                         "quant_requested": args.upstream_quant, "dtype_effective": upstream["dtype"],
                         "param_bytes": upstream["param_bytes"], "load_mode": upstream["load_mode"],
                         "nf4_replaced_linears": upstream["nf4_replaced"]},
            "downstream": {"path": str(cut_path), "quant": args.downstream_quant or args.downstream_dtype,
                           "artifact_sha256": _sha256(cut_path), "model_bytes": downstream_bytes,
                           "layers": downstream_layers, "n_embd": n_embd},
            "whole": {"artifact_sha256": _sha256(whole_path), "model_bytes": whole_bytes,
                      "quant": args.downstream_quant or args.downstream_dtype},
        }
        layer_layout.update({"downstream_layers": downstream_layers,
                             "trim_layers": args.layers,
                             "kept_block_count": downstream_layers})
        metrics = {
            "capacity_gain_x": (round(whole_bytes / max_segment, 4)
                                if whole_bytes and max_segment else None),
            "resident_weight_bytes": {"whole": whole_bytes, "upstream": upstream["param_bytes"],
                                      "downstream": downstream_bytes, "max_segment": max_segment},
            "peak_vram_gb": _cuda_gb(),
            "rss_gb": _rss_gb(),
        }
        verdict.update({"passed": bool(whole_bytes and max_segment),
                        "failure": None if whole_bytes else "整模工件无法加载"})
        device_profile.update({"gpu": _gpu_name(), "vram_gb": _vram_gb(),
                               "upstream_device": upstream["device"],
                               "upstream_dtype_effective": upstream["dtype"],
                               "upstream_compiled": upstream["compiled"]})
        return build_record(
            upstream_iface=upstream_iface, downstream_iface=downstream_iface,
            middle_iface=middle_iface, path=args.path,
            models=models, layer_layout=layer_layout, handoff=_handoff(n_embd),
            load={"prompt": str(prompt_path), "prefill_tokens": args.prefill,
                  "gen_tokens": args.gen, "batch": args.batch},
            verdict=verdict, metrics=metrics, device_profile=device_profile,
            evidence=RelayXFrameEvidence(correctness_verified=False,
                                         performance_verdict="unknown").to_dict(),
            artifacts={"log_path": None})

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    prompt = _build_prompt_tokens(tokenizer, prompt_path, args.prefill)
    baseline_tokens, baseline = _baseline_tokens(args, prompt)
    models["whole"] = {"artifact_sha256": baseline["artifact_sha256"],
                       "model_bytes": baseline["model_bytes"],
                       "quant": args.downstream_quant or args.downstream_dtype}

    if args.path == PATH_L2L_KEEP_HEAD:
        upstream = _load_keep_head_segment(args, args.upstream_model, role="upstream")
    elif args.path == PATH_L2L:
        from llama_engine import LlamaCppEngine  # noqa: PLC0415

        up_engine = LlamaCppEngine()
        up_engine.load_model(model_path=str(args.upstream_model), n_ctx=args.n_ctx,
                             n_threads=args.threads, n_seq_max=max(1, args.batch))
        if not up_engine.is_loaded:
            raise SystemExit("FAIL: 上游 head 模型未加载成功")
        head_native = up_engine._model._model.model
        upstream = {
            "load_s": None, "device": "cpu", "dtype": "float32", "nf4_replaced": 0,
            "param_bytes": int(M.llama_model_size(head_native)),
            "load_mode": "llama_engine(裁层 head GGUF)", "compiled": False,
            "n_embd": int(M.llama_model_n_embd(head_native)),
            "upstream_engine": up_engine,
        }
    else:
        upstream = _load_upstream(args)

    if args.path == PATH_D2L2L_KEEP_HEAD:
        # 三段：torch 上游 → keep-head 中段 → llama 末段
        upstream["keep_head_middle"] = _load_keep_head_segment(
            args, args.mid_model, role="middle")["keep_head"]
        layer_layout["middle_layers"] = int(args.mid_layers) - int(args.layers)

    downstream: dict[str, Any] = {}
    if args.path == PATH_D2L_RAW:
        native = M.llama_model_load_from_file(str(cut_path).encode("utf-8"),
                                              M.llama_model_default_params())
        if not native:
            raise SystemExit(f"FAIL: 加载 {cut_path} 失败")
        ctx_params = M.llama_context_default_params()
        ctx_params.n_ctx = args.n_ctx
        ctx_params.n_batch = 512
        ctx_params.n_ubatch = 512
        ctx_params.n_threads = args.threads
        raw_ctx = M.llama_init_from_model(native, ctx_params)
        if not raw_ctx:
            raise SystemExit("FAIL: 下游 context 初始化失败")
        downstream = {"raw_ctx": raw_ctx, "native": native,
                      "n_embd": int(M.llama_model_n_embd_inp(native)),
                      "n_vocab": int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(native))),
                      "model_bytes": int(M.llama_model_size(native)),
                      "layers": int(M.llama_model_n_layer(native)),
                      "load_s": None}
    else:
        downstream = _load_downstream_engine(args)

    n_embd = downstream["n_embd"]
    if upstream["n_embd"] != n_embd:
        raise SystemExit(f"FAIL: hidden 宽度不匹配 上游 {upstream['n_embd']} vs 下游 {n_embd}")

    downstream["forward"] = _make_downstream_forwarder(args, downstream)
    relay = _run_relay(args, prompt, upstream, downstream)

    tokens_match = (relay["failure"] is None
                    and all(seq == baseline_tokens for seq in relay["tokens"]))
    matched = sum(1 for seq in relay["tokens"] if seq == baseline_tokens)
    first_mismatch = next(
        (i for i, (got, want) in enumerate(zip(relay["tokens"][0], baseline_tokens)) if got != want),
        None)
    print(f"[relay] baseline[:8]={baseline_tokens[:8]} relay[:8]={relay['tokens'][0][:8]} "
          f"first_mismatch={first_mismatch}", flush=True)
    relay_total = None
    if relay["upstream_decode_ms"]["mean"] is not None and relay["downstream_decode_ms"]["mean"] is not None:
        relay_total = relay["upstream_decode_ms"]["mean"] + relay["downstream_decode_ms"]["mean"]

    models.update({
        "upstream": {"id": Path(args.model_dir or args.upstream_model or "").name or None,
                     "path": args.model_dir or args.upstream_model,
                     "quant_requested": args.upstream_quant, "dtype_effective": upstream["dtype"],
                     "param_bytes": upstream["param_bytes"], "load_mode": upstream["load_mode"],
                     "nf4_replaced_linears": upstream["nf4_replaced"]},
        "downstream": {"path": str(cut_path), "quant": args.downstream_dtype,
                       "artifact_sha256": _sha256(cut_path), "model_bytes": downstream["model_bytes"],
                       "layers": downstream["layers"], "n_embd": n_embd},
    })
    layer_layout.update({"downstream_layers": downstream["layers"], "trim_layers": args.layers,
                         "kept_block_count": downstream["layers"]})
    max_segment = max(upstream["param_bytes"], downstream["model_bytes"])
    metrics = {
        "upstream_prefill_ms": relay["upstream_prefill_ms"],
        "downstream_prefill_ms": relay["downstream_prefill_ms"],
        "upstream_decode_ms": relay["upstream_decode_ms"],
        "downstream_decode_ms": relay["downstream_decode_ms"],
        "middle_decode_ms": (relay.get("middle_decode_ms")
                             if args.path == PATH_D2L2L_KEEP_HEAD else None),
        "baseline_ms_per_step": baseline["decode_ms"],
        "capacity_gain_x": (round(baseline["model_bytes"] / max_segment, 4)
                            if max_segment else None),
        "resident_weight_bytes": {"whole": baseline["model_bytes"],
                                  "upstream": upstream["param_bytes"],
                                  "downstream": downstream["model_bytes"],
                                  "max_segment": max_segment},
        "peak_vram_gb": _cuda_gb(),
        "rss_gb": _rss_gb(),
    }
    verdict.update({
        "passed": bool(tokens_match),
        "tokens_match": bool(tokens_match),
        "matched_runs": matched,
        "total_runs": len(relay["tokens"]),
        "first_mismatch_index": first_mismatch,
        "failure": relay["failure"],
    })
    device_profile.update({"gpu": _gpu_name(), "vram_gb": _vram_gb(),
                           "upstream_device": upstream["device"],
                           "upstream_dtype_effective": upstream["dtype"],
                           "upstream_compiled": upstream["compiled"]})
    evidence = RelayXFrameEvidence(
        correctness_verified=bool(tokens_match),
        correctness_cases=matched,
        max_tested_prefill=len(prompt),
        prompt_distribution_verified=False,
        long_sequence_verified=False,
        weak_network_verified=False,
        protocol_consistency_verified=False,
        performance_verdict=_performance_verdict(
            relay_total, (baseline["decode_ms"] or {}).get("mean")),
    ).to_dict()

    return build_record(
        upstream_iface=upstream_iface, downstream_iface=downstream_iface,
        middle_iface=middle_iface, path=args.path,
        models=models, layer_layout=layer_layout, handoff=_handoff(n_embd, args.path),
        load={"prompt": str(prompt_path), "prefill_tokens": len(prompt),
              "gen_tokens": args.gen, "batch": args.batch},
        verdict=verdict, metrics=metrics, device_profile=device_profile, evidence=evidence,
        artifacts={"log_path": None})


def _handoff(n_embd: int | None, path: str | None = None) -> dict[str, Any]:
    """线上 hidden 规范：本驱动的接力一律以 f32 传递。"""
    spec = RelayHiddenSpec(n_embd=int(n_embd or 0), dtype="float32")
    result = spec.to_dict()
    if path == PATH_L2L:
        # ★ 如实记录：pip 绑定的 embeddings 通道返回的是 output_norm(H)（见
        #   _check_l2l_upstream_channel 的证据），因此该 path 只允许在显式开关下运行。
        result["upstream_applies_output_norm"] = True
        result["notes"] = ("上游经 llama.cpp embeddings 通道 = output_norm(H)；"
                          "只在 --allow-normed-upstream 下允许，结论不得与 d2l_mainrepo 混表")
    return result


def _gpu_name() -> str | None:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _vram_gb() -> float | None:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def _dry_run_record(args: argparse.Namespace) -> dict[str, Any]:
    """不加载模型，只产出记录骨架（仍过 schema 校验）。"""
    upstream_iface, downstream_iface, middle_iface = _IFACES_BY_PATH[args.path]
    return build_record(
        upstream_iface=upstream_iface, downstream_iface=downstream_iface,
        middle_iface=middle_iface, path=args.path,
        models={
            "upstream": {"id": Path(args.model_dir or args.upstream_model or "").name or None,
                         "path": args.model_dir or args.upstream_model,
                         "quant_requested": args.upstream_quant, "dtype_effective": None,
                         "param_bytes": None, "load_mode": None, "nf4_replaced_linears": None},
            "downstream": {"path": args.cut_model, "quant": args.downstream_dtype,
                           "artifact_sha256": None, "model_bytes": None,
                           "layers": None, "n_embd": None},
            "whole": {"artifact_sha256": None, "model_bytes": None,
                      "quant": args.downstream_quant or args.downstream_dtype},
        },
        layer_layout={"upstream_layers": args.layers, "trim_layers": args.layers},
        handoff={"dtype": "float32", "n_embd": None, "bytes_per_token": None, "supported": None},
        load={"prompt": args.prompt, "prefill_tokens": args.prefill,
              "gen_tokens": args.gen, "batch": args.batch},
        verdict={"passed": False, "tokens_match": None, "criterion": RELAY_ACCEPTANCE,
                 "failure": "dry_run"},
        metrics={},
        device_profile={"threads": args.threads},
        evidence=None,
        artifacts={"log_path": None},
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    record = _dry_run_record(args) if args.dry_run else _run(args)

    if args.json_out == "-":
        print(json.dumps(record, ensure_ascii=False, indent=1))
    elif args.json_out:
        path = write_record(record, args.json_out)
        print(f"[record] {path}")
    print(f"[verdict] kind={record['kind']} path={record['path']} "
          f"passed={record['verdict']['passed']} "
          f"tokens_match={record['verdict'].get('tokens_match')} "
          f"failure={record['verdict'].get('failure')}")
    if args.dry_run:
        # 预检成功 ≠ 实验通过：dry-run 只要记录合法就算成功（不把 passed=False 当失败）。
        return 0
    return 0 if record["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
