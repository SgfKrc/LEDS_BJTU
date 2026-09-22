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
    params.n_ctx = len(prompt) + gen + 8
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
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

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
                                    mode="nextn", n_ctx=len(prompt) + int(args.gen) + 8,
                                    n_threads=int(args.threads),
                                    n_batch=max(512, len(prompt)), n_seq_max=1)
        n_embd = int(upstream.n_embd)
    tail = RelayTcpClient(*_split(args.tail_endpoint), n_embd=n_embd, timeout=180.0)
    middle = (RelayTcpClient(*_split(args.middle_endpoint), n_embd=n_embd, timeout=180.0)
              if args.middle_endpoint else None)

    used_middle = 0
    tokens: list[int] = []
    pos = 0
    started = time.perf_counter()
    try:
        for step in range(int(args.gen)):
            toks = prompt if step == 0 else [tokens[-1]]
            if head_client is not None:
                # ★ P4.5：上游在远端（吃 token 吐 hidden）—— 本机不跑任何模型
                payload = head_client.request_hidden_from_tokens(toks)
                n_tok = len(toks)
            else:
                hidden = upstream.forward_tokens_to_hidden(toks, n_past=pos)
                payload = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
                n_tok = int(hidden.shape[0])
            if middle is not None:
                payload = middle.request_hidden(payload, n_tokens=n_tok)
                used_middle += 1
            token = tail.request_token(payload, n_tokens=n_tok)
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
        "endpoints": {"tail": args.tail_endpoint, "middle": args.middle_endpoint},
        "load": {"prompt": args.prompt, "prefill_tokens": len(prompt), "gen_tokens": int(args.gen),
                 "n_embd": n_embd},
        "tokens_relay": tokens,
        "tokens_baseline": baseline,
        "verdict": {"criterion": "per_token_argmax", "passed": passed,
                    "tokens_match": passed, "matched": matched, "total": len(baseline),
                    "first_mismatch_index": first_bad},
        "timing": {"baseline_s": baseline_s, "relay_s": relay_s,
                   "middle_hops": used_middle},
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


if __name__ == "__main__":
    raise SystemExit(main())
