#!/usr/bin/env python
"""relay_l2l_net_probe.py — 纯 llama（L→L）跨机接力的**可执行证据**（P4.5 退化路径）。

场景：边缘集群里**不一定有 PC**，也可能一台能跑 torch 的节点都没有。此时层接力必须退化为
**全部由 llama.cpp 承载**（L→L）：上游用 head 工件吐 hidden，中间段吐 hidden，末段吐 token。

本探针用**主仓组件**串起这条路径（不手写前向）：

- 上游：`src/llama_keep_head.py` 的 `KeepHeadUpstream`（补丁版 keep-head 通道，head 工件）；
- 中间段 / 末段：`src/relay_transport.py` 的 `RelayTcpClient.request_hidden` / `.request_token`
  （远端由 `scripts/relay_mid_service.py --role middle|tail` 提供，端口经 SSH 隧道映射到 loopback）；
- 对照：`llama_cpp` 直接跑**同精度整模**，逐 token argmax 对比。

三种拓扑：

    A. head(本机) → tail(远端)                        两段
    B. head(本机) → middle(远端) → tail(远端)          三段（中段吐 hidden、末段吐 token）
    C. head(本机) → tail(本机)                        对照：全本地（验证拓扑本身正确）

判据：**per-token argmax 与同精度整模一致**（不得用 cosine 代替）；不一致即如实标 FAIL。

用法：

    python scripts/relay_l2l_net_probe.py \
        --head-model build/cross-framework-layer-poc/out/qwen25-05b-f16-head8.gguf \
        --tail-endpoint 127.0.0.1:50183 \
        --whole-model build/cross-framework-layer-poc/out/qwen25-05b-f16.gguf \
        --prompt build/relay-records/prompts/natural-short.txt \
        --prefill 32 --gen 32 --json-out build/relay-records/l2l-net-2seg.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _build_prompt_tokens(tokenizer, prompt_path: Path, prefill: int) -> list[int]:
    text = prompt_path.read_text(encoding="utf-8").strip()
    ids = tokenizer.encode(text)
    if len(ids) < prefill:
        reps = (prefill // max(1, len(ids))) + 1
        ids = (ids * reps)[:prefill]
    return [int(i) for i in ids[:prefill]]


def _whole_tokens(whole: Path, prompt: list[int], gen: int, threads: int) -> list[int]:
    """同精度整模逐 token argmax（对照）。"""
    import llama_cpp.llama_cpp as M
    import numpy as np

    params = M.llama_context_default_params()
    # ★ 2026-09-24：余量从 8 提到 256 —— 旧值紧贴 `len(prompt) + gen`，一旦 llama.cpp 的
    #   ctx 分配/取整策略变化就会在长 decode 中途失败（同一类坑见
    #   `KeepHeadUpstream.ctx_per_seq` 的说明）。
    params.n_ctx = len(prompt) + gen + 256
    params.n_batch = max(512, len(prompt))
    params.n_ubatch = params.n_batch
    params.n_threads = int(threads)
    model = M.llama_model_load_from_file(str(whole).encode("utf-8"),
                                         M.llama_model_default_params())
    if not model:
        raise SystemExit(f"FAIL: 整模加载失败：{whole}")
    ctx = M.llama_init_from_model(model, params)
    if not ctx:
        raise SystemExit("FAIL: 整模 context 初始化失败")
    n_vocab = int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(model)))
    batch = M.llama_batch_init(params.n_batch, 0, 1)
    tokens: list[int] = []
    pos = 0
    try:
        for step in range(gen):
            toks = list(prompt) if step == 0 else [tokens[-1]]
            for i, tid in enumerate(toks):
                batch.token[i] = int(tid)
                batch.pos[i] = pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1 if i == len(toks) - 1 else 0
            batch.n_tokens = len(toks)
            if M.llama_decode(ctx, batch) != 0:
                raise SystemExit("FAIL: 整模 decode 失败")
            logits = M.llama_get_logits_ith(ctx, len(toks) - 1)
            tokens.append(int(np.ctypeslib.as_array(logits, shape=(n_vocab,)).argmax()))
            pos += len(toks)
    finally:
        M.llama_batch_free(batch)
        M.llama_free(ctx)
        M.llama_model_free(model)
    return tokens


def _stat(values: list[float]) -> dict[str, object]:
    """均值/标准差/样本数（分段计时的口径统一在这里）。"""
    import statistics

    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 3),
            "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
            "min": round(min(values), 3), "max": round(max(values), 3)}


def _segment_builds(args) -> dict[str, object]:
    """逐段收集**构建标识**（§10.2 待办）：本地段直接收集，远端段读 ready 文件，否则显式 unknown。

    为什么必须落进记录：`engines` 过去是固定字符串 ⇒ 无法回答「这一段实际用哪套 llama.cpp 构建」，
    而那正是同一 head 段下 **pip 末段 1/32 与 shim 末段 32/32** 的分界（见文档 §10.7）。
    远端段没给 ready 文件时宁可写 `remote_unknown`，也不省略字段 —— 记录要能区分「未知」与「没写」。
    """
    from relay_segment_info import (  # noqa: PLC0415
        collect_local_build,
        load_ready_build,
        unknown_remote_build,
    )

    digest = bool(getattr(args, "digest_artifacts", False))
    segments: dict[str, object] = {}

    if args.head_endpoint:
        ready = getattr(args, "head_ready_file", None)
        segments["head"] = {
            "runner": "relay_mid_service --role head（远端）",
            "endpoint": args.head_endpoint,
            "build": (load_ready_build(ready) if ready
                      else unknown_remote_build(args.head_endpoint)),
        }
    else:
        segments["head"] = {
            "runner": "llama_keep_head.KeepHeadUpstream",
            "mode": "nextn",
            "channel": "keep_head_layer_out",
            "build": collect_local_build(shim=args.shim, model=args.head_model,
                                         digest_artifacts=digest),
        }

    if getattr(args, "middle_endpoint", None):
        ready = getattr(args, "middle_ready_file", None)
        segments["middle"] = {
            "runner": "relay_mid_service --role middle（远端）",
            "endpoint": args.middle_endpoint,
            "build": (load_ready_build(ready) if ready
                      else unknown_remote_build(args.middle_endpoint)),
        }

    ready = getattr(args, "tail_ready_file", None)
    segments["tail"] = {
        "runner": "relay_mid_service --role tail（远端；shim 或 pip 绑定由该服务决定）",
        "endpoint": args.tail_endpoint,
        "build": (load_ready_build(ready) if ready
                  else unknown_remote_build(args.tail_endpoint)),
    }
    return segments


def _parse_mid_layers(raw: str | None) -> tuple[int, int] | None:
    """★ #30：解析 `--mid-layers` 的 `K1-K2`（与工件文件名同义，如 `8-16`）。非法 ⇒ `None`。"""
    if not raw:
        return None
    text = str(raw).strip().replace("_", "-")
    left, _, right = text.partition("-")
    if not left.isdigit() or not right.isdigit():
        return None
    start, end = int(left), int(right)
    return (start, end) if 0 <= start < end else None


def _manifest_layers(path) -> tuple[int, int] | None:
    """★ #30：从段工件的 manifest 读 `source_layer_range`（区间 `[start,end)`）。

    manifest 由 `scripts/cut_layers.py`（生成时自带）或 `scripts/relay_artifact_manifest.py`
    （对早期工件**事后补录**）产出 ⇒ 工件**从此可自证**自己覆盖哪几层。
    读不到 / 格式非法一律返回 `None`（调用方据此回落到显式参数或 `unverified`，**不猜**）。
    """
    if not path:
        return None
    try:
        payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        span = payload.get("source_layer_range")
        start, end = int(span[0]), int(span[1])
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None
    return (start, end) if 0 <= start < end else None


def _discover_manifest(path) -> str | None:
    """★ #30：`<artifact>.gguf` ⇒ 试 `<artifact>.gguf.manifest.json`（`relay_artifact_manifest.py` 的默认落点）。

    找不到就返回 `None`（**不报错** —— 缺 manifest 是常态，只意味着退回到"人填层范围"）。
    """
    if not path:
        return None
    candidate = Path(str(path) + ".manifest.json")
    return str(candidate) if candidate.is_file() else None


def _respect_manifests(args) -> dict[str, str]:
    """★ #30：用 manifest 补齐**未显式给出**的层覆盖参数（显式参数永远优先）。

    优先级：显式 `--*-layers` > 显式 `--*-manifest` > 自动发现（`<head-model>.manifest.json`）。
    只补空位，**绝不覆盖**已经由人显式给出的值。返回实际用到的 manifest 路径（写进证据）。
    """
    used: dict[str, str] = {}

    # head：[0, K) —— 显式 `--head-manifest` 优先；否则自动试 `<head-model>.manifest.json`
    head_manifest = getattr(args, "head_manifest", None)
    if not head_manifest and getattr(args, "head_model", None):
        head_manifest = _discover_manifest(args.head_model)
    if head_manifest and not getattr(args, "head_layers", None):
        span = _manifest_layers(head_manifest)
        if span:
            args.head_layers = str(span[1])
            used["head"] = str(head_manifest)

    # middle：[K1, K2)
    if getattr(args, "mid_manifest", None) and not getattr(args, "mid_layers", None):
        span = _manifest_layers(args.mid_manifest)
        if span:
            args.mid_layers = f"{span[0]}-{span[1]}"
            used["middle"] = str(args.mid_manifest)

    # tail：[K2, N)
    if getattr(args, "tail_manifest", None) and getattr(args, "tail_start", None) is None:
        span = _manifest_layers(args.tail_manifest)
        if span:
            args.tail_start = span[0]
            used["tail"] = str(args.tail_manifest)

    return used


def _layer_coverage(args) -> dict[str, object]:
    """★ #30：校验各段层覆盖**恰好铺满 `[0,total)` 且不重叠**，并给出可落进证据的结论。

    为什么需要它：L→L 路径此前**不校验**「head 工件覆盖 + 中段来源起点 == 末段起点」，也没有任何东西
    要求各段恰好铺满 ⇒ **缺层 / 重复层静默通过**，只在 `per-token argmax` 上表现为不一致
    （与"代码算错"同型）。2026-09-26 已因"端点实际载的工件与记录不符"误判过一次。

    返回值（写进证据的 `layer_coverage` 字段）：
    - `status="verified"`：四元组齐全且**恰好铺满**，附 `segments` 明细；
    - `status="invalid"`：齐全但**不衔接 / 重叠 / 越界**，`detail` 说明原因；
    - `status="unverified"`：参数不全或格式非法 ⇒ **不拦**（默认只 WARN），但证据里明确标出。
    """
    head_layers = getattr(args, "head_layers", None)
    mid_raw = getattr(args, "mid_layers", None)
    tail_start = getattr(args, "tail_start", None)
    total = getattr(args, "total_layers", None)
    spec = {
        "head_layers": head_layers, "mid_layers": mid_raw,
        "tail_start": tail_start, "total_layers": total,
    }

    if not head_layers or tail_start is None or not total:
        return {"status": "unverified", "spec": spec,
                "detail": "缺少 --head-layers / --tail-start / --total-layers"}

    try:
        head_end = int(head_layers)
        total = int(total)
        tail_start = int(tail_start)
    except (TypeError, ValueError):
        return {"status": "unverified", "spec": spec, "detail": "层数参数非整数"}

    mid = _parse_mid_layers(mid_raw)
    if mid_raw and mid is None:
        return {"status": "unverified", "spec": spec, "detail": f"--mid-layers 非法: {mid_raw!r}"}

    if getattr(args, "middle_endpoint", None):
        # 三段：head [0,K) + middle [K1,K2) + tail [K2,N)
        if mid is None:
            return {"status": "unverified", "spec": spec,
                    "detail": "三段拓扑需要 --mid-layers（如 `8-16`）"}
        mid_start, mid_end = mid
        if head_end != mid_start:
            detail = f"head 终点 {head_end} != middle 起点 {mid_start}（缺口或重叠）"
        elif mid_end != tail_start:
            detail = f"middle 终点 {mid_end} != tail 起点 {tail_start}（缺口或重叠）"
        elif not (0 < head_end < total) or not (mid_end < total):
            detail = f"切点越界：head_end={head_end} mid_end={mid_end} total={total}"
        else:
            detail = ""
        return {
            "status": "verified" if not detail else "invalid",
            "detail": detail,
            "spec": spec,
            "segments": {"head": [0, head_end], "middle": [mid_start, mid_end],
                         "tail": [tail_start, total]},
        }

    # 两段：head [0,K) + tail [K,N)
    if head_end != tail_start:
        detail = f"head 终点 {head_end} != tail 起点 {tail_start}（缺口或重叠）"
    elif not (0 < head_end < total):
        detail = f"切点越界：head_end={head_end} total={total}"
    else:
        detail = ""
    return {
        "status": "verified" if not detail else "invalid",
        "detail": detail,
        "spec": spec,
        "segments": {"head": [0, head_end], "tail": [tail_start, total]},
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="纯 llama（L→L）跨机接力的可执行证据探针")
    ap.add_argument("--head-model", default=None,
                    help="本地上游 head 工件；与 --head-endpoint 二选一")
    ap.add_argument("--head-endpoint", default=None,
                    help="★ P4.5：**远端**上游段 host:port（吃 token 吐 hidden）—— "
                         "给它就完全不用本机跑模型")
    ap.add_argument("--n-embd", type=int, default=896,
                    help="远端 head 模式下本机不知道隐藏宽度，需显式给（默认 896）")
    ap.add_argument("--tail-endpoint", required=True, help="末段（tail）host:port（loopback 隧道）")
    ap.add_argument("--middle-endpoint", default=None,
                    help="可选：中间段（middle）host:port；给了就是三段")
    ap.add_argument("--whole-model", required=True, help="同精度整模（对照）")
    ap.add_argument("--model-dir", default=None, help="HF 目录（tokenizer 来源）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefill", type=int, default=32)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shim", default="build/keephead/build-cpu/bin/qlh_keep_head.dll")
    ap.add_argument("--hidden-quant", default="none",
                    choices=("none", "f16", "int8_block128", "int4_block128"),
                    help="★ A5：上行（本机 → 远端段）的 hidden 压缩档。判据仍是 **per-token argmax**；"
                         "远端段必须能处理同一档位（旧对端看到非零 flags 会 fail-closed 拒）")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tail-ready-file", default=None,
                    help="★ 末段服务端的 ready 文件（内含构建标识）⇒ 记录里逐段写 runner/构建")
    ap.add_argument("--middle-ready-file", default=None, help="★ 中段服务端的 ready 文件")
    ap.add_argument("--head-ready-file", default=None, help="★ 远端 head 段的 ready 文件")
    ap.add_argument("--digest-artifacts", action="store_true",
                    help="★ 对段工件也算 sha256（GB 级文件会明显变慢；默认只记大小/名字）")
    # ★ #30（2026-09-26）：**层覆盖校验** —— 工件名自带的层范围**不可自证**
    #   （`head*/mid*/tail*` 这批无 manifest；裁层生成器只覆盖 `block_count`、不记录层号重命名），
    #   而 L→L 路径此前**完全不做**覆盖校验 ⇒ 缺层 / 重复层会静默通过、只表现为数值不一致
    #   （与"代码算错"同型，实测已误判过一次）。这里要求显式给出每段层区间，校验
    #   **恰好铺满 `[0, total)` 且不重叠**；参数不全时不拦、只标 `unverified`（加 `--strict-coverage` 才失败）。
    ap.add_argument("--head-manifest", default=None,
                    help="★ #30：head 段工件的 manifest（据 `source_layer_range` 自动填 `--head-layers`）；"
                         "缺省时自动试 `<--head-model>.manifest.json`")
    ap.add_argument("--mid-manifest", default=None,
                    help="★ #30：middle 段工件的 manifest（自动填 `--mid-layers`）")
    ap.add_argument("--tail-manifest", default=None,
                    help="★ #30：tail 段工件的 manifest（自动填 `--tail-start`）")
    ap.add_argument("--head-layers", default=None,
                    help="★ #30：上游 head 段覆盖层数 K（区间 `[0,K)`）；不给则记 unverified")
    ap.add_argument("--mid-layers", default=None,
                    help="★ #30：中段覆盖区间 `K1-K2`（与文件名同义，如 `8-16`）")
    ap.add_argument("--tail-start", type=int, default=None,
                    help="★ #30：末段起点 K2（区间 `[K2,total)`）")
    ap.add_argument("--total-layers", type=int, default=None,
                    help="★ #30：整模层数 N（应与 --whole-model 一致）")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="★ #30：层覆盖参数不全或校验不通过时**直接失败**（默认只 WARN + 记 unverified）")
    args = ap.parse_args()

    # ★ #30：先用 manifest 补齐**未显式给出**的层覆盖参数（显式参数永远优先），再做校验。
    used_manifests = _respect_manifests(args)
    if used_manifests:
        print(f"[coverage] 从 manifest 自动读入层范围：{used_manifests}", file=sys.stderr)

    coverage = _layer_coverage(args)
    # ★ #30：证据里留痕 —— 这些层范围是**从哪个 manifest 读来的**（便于事后自证）。
    coverage["manifests"] = used_manifests
    if coverage["status"] == "invalid":
        detail = f"层覆盖校验不通过：{coverage['detail']}"
        if args.strict_coverage:
            raise SystemExit(f"FAIL: {detail}")
        print(f"[warn] {detail}", file=sys.stderr)
    elif coverage["status"] == "unverified":
        detail = ("未提供完整层覆盖参数"
                  "（--head-layers / --mid-layers / --tail-start / --total-layers）"
                  "⇒ 证据记 unverified，**无法排除载错工件**")
        if args.strict_coverage:
            raise SystemExit(f"FAIL: {detail}")
        print(f"[warn] {detail}", file=sys.stderr)

    # ★ P4.5 健康检查：**先探活** —— 放在最前面，避免为一次注定失败的运行白跑整模对照；
    # 也把"远端段已退出"与"模型算错"分开（见 `_preflight` docstring）。
    if args.head_endpoint:
        _preflight(args.head_endpoint, role="上游(head)")
    if args.middle_endpoint:
        _preflight(args.middle_endpoint, role="中间(middle)")
    _preflight(args.tail_endpoint, role="末段(tail)")

    import numpy as np

    from llama_keep_head import KeepHeadUpstream
    from relay_transport import RelayTcpClient

    if args.model_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    else:
        raise SystemExit("FAIL: 需要 --model-dir（HF tokenizer）—— 避免手写 tokenize")

    prompt = _build_prompt_tokens(tokenizer, Path(args.prompt), int(args.prefill))
    started = time.perf_counter()
    baseline = _whole_tokens(Path(args.whole_model), prompt, int(args.gen), int(args.threads))
    baseline_s = round(time.perf_counter() - started, 2)

    # ★ P4.5：上游可以是**本机 head 段**（shim）或**远端 head 段**（TOKENS → HIDDEN）。
    #   后者让本机全程不跑任何模型 —— 无 PC 集群的完整形态。
    upstream = None
    head_client = None
    if args.head_endpoint:
        n_embd = int(args.n_embd)
        head_client = RelayTcpClient(*_split(args.head_endpoint), n_embd=n_embd, timeout=300.0)
    else:
        if not args.head_model:
            raise SystemExit("FAIL: 需要 --head-model（本地上游）或 --head-endpoint（远端上游）")
        upstream = KeepHeadUpstream(str(Path(args.shim).resolve()), args.head_model,
                                    mode="nextn", n_ctx=len(prompt) + int(args.gen) + 256,
                                    n_threads=int(args.threads),
                                    n_batch=max(512, len(prompt)), n_seq_max=1)
        n_embd = int(upstream.n_embd)
    tail = RelayTcpClient(*_split(args.tail_endpoint), n_embd=n_embd, timeout=180.0)
    middle = (RelayTcpClient(*_split(args.middle_endpoint), n_embd=n_embd, timeout=180.0)
              if args.middle_endpoint else None)

    used_middle = 0
    tokens: list[int] = []
    pos = 0
    # ★ P4.5 ①：**分段计时** —— 正确性之外还要有性能数字（各段各步耗时 + 端到端）
    seg_ms: dict[str, list[float]] = {"head": [], "middle": [], "tail": []}
    e2e_ms: list[float] = []
    started = time.perf_counter()
    try:
        for step in range(int(args.gen)):
            t_step = time.perf_counter()
            toks = prompt if step == 0 else [tokens[-1]]
            if head_client is not None:
                # ★ P4.5：上游在远端（吃 token 吐 hidden）—— 本机不跑任何模型
                payload = head_client.request_hidden_from_tokens(toks)
                n_tok = len(toks)
            else:
                hidden = upstream.forward_tokens_to_hidden(toks, n_past=pos)
                payload = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
                n_tok = int(hidden.shape[0])
            t_head = time.perf_counter()
            if middle is not None:
                # ★ A5：上行按档位压缩（远端解回 f32；旧对端会 fail-closed 拒非零 flags）。
                payload = middle.request_hidden(payload, n_tokens=n_tok,
                                                quant=args.hidden_quant)
                used_middle += 1
            t_mid = time.perf_counter()
            token = tail.request_token(payload, n_tokens=n_tok, quant=args.hidden_quant)
            t_tail = time.perf_counter()
            seg_ms["head"].append((t_head - t_step) * 1000.0)
            if middle is not None:
                seg_ms["middle"].append((t_mid - t_head) * 1000.0)
            seg_ms["tail"].append((t_tail - t_mid) * 1000.0)
            e2e_ms.append((t_tail - t_step) * 1000.0)
            tokens.append(int(token))
            pos += n_tok
    finally:
        # ⚠️ 关闭失败**不得掩盖原始异常**（实测踩到：close 抛 ConnectionAbortedError，
        #    把循环里真正的错误盖掉了，只能看到"关闭失败"这个假象）。
        for client in (tail, middle, head_client):
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
        if upstream is not None:
            try:
                upstream.close()
            except Exception:  # noqa: BLE001
                pass
    relay_s = round(time.perf_counter() - started, 2)

    matched = sum(1 for a, b in zip(tokens, baseline) if a == b)
    first_bad = next((i for i, (a, b) in enumerate(zip(tokens, baseline)) if a != b), None)
    passed = tokens == baseline
    report = {
        "schema_version": "qlh.relay_l2l_net_probe.v1",
        "path_kind": ("l2l_net_3seg" if middle is not None else "l2l_net_2seg"),
        "ifaces": {
            "upstream": "llama_keep_head.KeepHeadUpstream.forward_tokens_to_hidden",
            "middle": ("relay_transport.RelayTcpClient.request_hidden" if middle else None),
            "downstream": "relay_transport.RelayTcpClient.request_token",
        },
        "engines": "纯 llama.cpp（无 torch / 无 D 档组件参与推理）",
        # ★ 2026-09-23（§10.2）：**逐段**写出实际 runner 与构建标识（shim/libllama 摘要、
        #   pip llama_cpp 版本、段工件大小/摘要）。远端段没给 ready 文件时显式记 remote_unknown。
        "segment_engines": _segment_builds(args),
        # ★ #30（2026-09-26）：**层覆盖**是否已校验（`verified` / `invalid` / `unverified`）。
        #   此前证据里既没有层号、也不校验各段衔接 ⇒ 载错工件**无法自证**（见 `docs/已知问题记录.md` #30）。
        "layer_coverage": coverage,
        "endpoints": {"tail": args.tail_endpoint, "middle": args.middle_endpoint},
        "load": {"prompt": args.prompt, "prefill_tokens": len(prompt), "gen_tokens": int(args.gen),
                 "n_embd": n_embd},
        # ★ 2026-09-23（A5）：上行 hidden 压缩档（`none` = f32 原样）。判据仍是 per-token argmax。
        "hidden_quant": args.hidden_quant,
        "tokens_relay": tokens,
        "tokens_baseline": baseline,
        "verdict": {"criterion": "per_token_argmax", "passed": passed,
                    "tokens_match": passed, "matched": matched, "total": len(baseline),
                    "first_mismatch_index": first_bad},
        "timing": {
            "baseline_s": baseline_s, "relay_s": relay_s, "middle_hops": used_middle,
            # ★ P4.5 ①：分段耗时与端到端（`head` 段在远端时含其往返；每步一个样本）
            "segments_ms": {name: _stat(values) for name, values in seg_ms.items() if values},
            "end_to_end_ms": _stat(e2e_ms),
            "per_step_e2e_ms": [round(value, 3) for value in e2e_ms],
        },
        "host_total_layers_note": "本机只跑 head 段（llama.cpp/CPU）；中段/末段在远端设备上",
    }
    print(json.dumps(report, ensure_ascii=False))
    icon = "PASS" if passed else "FAIL"
    print(f"[verdict] {icon} {report['path_kind']} matched={matched}/{len(baseline)} "
          f"first_mismatch={first_bad}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"[record] {args.json_out}")
    return 0 if passed else 1


def _split(endpoint: str) -> tuple[str, int]:
    host, _, port = str(endpoint).rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"FAIL: endpoint 需要 host:port，实得 {endpoint!r}")
    return host, int(port)


def _preflight(endpoint: str, *, role: str, timeout: float = 5.0) -> None:
    """★ P4.5 健康检查（客户端侧）：收发之前先做一次 **TCP 探活**。

    为什么必须单独做这一步：远端段若已退出（典型情形是它跑在 ssh 会话里、被一次网络抖动
    **静默带走**），**隧道端口仍在本机监听** ⇒ 第一次收发才会失败，且错误形如连接被 reset，
    **与"模型算错"几乎无法区分**（实测踩到，见文档 §8.4）。探活让这种失败在"开始推理之前"
    就以明确诊断终止，并指向正确的排查方向。
    """
    import socket  # noqa: PLC0415

    host, port = _split(endpoint)
    try:
        with socket.create_connection((host, port), timeout=float(timeout)):
            return
    except OSError as exc:
        raise SystemExit(
            f"FAIL: {role} 段不可达（{endpoint}）：{exc}\n"
            "  ⇒ 常见原因：**远端服务已退出**（例如跑在 ssh 会话里、被网络抖动静默带走），"
            "或 SSH 隧道已断。\n"
            "  ⇒ 检查该端 ready 文件的 `alive_at` 是否还在更新（服务端默认每 5s 心跳一次），"
            "以及远端进程是否还在。")


if __name__ == "__main__":
    raise SystemExit(main())
