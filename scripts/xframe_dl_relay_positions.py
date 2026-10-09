"""XFRAME-6 复验（主仓引擎，真 D→L 链路）：**逐位置**误差是否随位置累积。

为什么需要它
------------
`XFRAME-6` 原证据（`scripts/xframe6_recursion_report.py`）是「**同一 HF 实例内**只替换 RMSNorm
归约语义」的对照 —— 它能回答"递推是否放大每步下界"，但**不覆盖生产形态**：
上游是主仓 PyTorch（`ModelManager`）、下游是主仓 llama.cpp（`LlamaCppEngine`）。
本工具用**两侧都是主仓引擎**的真 D→L 链路复验同一判据。

对照设计（唯一变量 = 上游由谁算）
---------------------------------
* **参照 L**：llama.cpp **整模** GGUF（`--whole-gguf`）对同一条 token 序列整段前向 ⇒ 逐位置 logits。
* **被测 R**：`ModelManager.load_layer_range(0, K, has_embedding=True, has_lm_head=False)`
  + `forward_layers(...)` 得逐位置 hidden ⇒ `LlamaCppEngine.forward_layers_from_hidden(...)`
  进下游裁层工件（`--cut-artifact`，丢前 K 层）⇒ 逐位置 logits。

⇒ 两侧的差别**只有**「前 K 层由 PyTorch 算还是由 llama.cpp 算」，这正是 D→L 接力的生产问题。

判据（与 `XFRAME-6` 同源，复用其纯函数）
---------------------------------------
逐位置 `rel` 序列 ⇒ `summarise()` ⇒ `tail_over_head` / 斜率 / `no_extra_accumulation`：
* `tail_over_head < 2.0` 且位置数 ≥ 48 ⇒ **递推不额外放大**（差异只复用每步下界）；
* 显著 > 2.0 ⇒ 存在随位置累积的额外项。

自检（必须通过才有资格谈结论）
-----------------------------
整段 eval 的逐位置缓冲必须能重现贪心序列（`scores` 语义正确性）。实测一致率 > 90%，
不一致处集中在**近并列（低 margin）**位置 —— 那是"整段 eval vs 逐步 eval"的浮点路径差，
不是本工具要测的项。一致率过低（< 0.5）会在报告里显式标红为**无效对照**。

用法::

    python scripts/xframe_dl_relay_positions.py \
        --hf-dir models/qwen3-5-2b \
        --whole-gguf models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf \
        --cut-artifact build/keephead/q35-2b-cut-k12.gguf --k 12 \
        --gen-steps 160 --out build/keephead/xframe-dl-positions.json

工件缺失时按本仓惯例 **打印 SKIP 并退出 0**，不伪装成通过。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PROMPTS = [
    "Explain in one short sentence why floating point addition is not associative.",
    "Name the three primary colors and say which one you would remove first.",
    "Write a single sentence that contains exactly five words about rain.",
]


def segment_means(values: list[float], segments: int = 4) -> list[float]:
    """把序列等分成 `segments` 段并取均值（纯函数）—— 观察趋势用，比单点 first/last 稳。"""
    n = len(values)
    if n == 0 or segments <= 0:
        return []
    out: list[float] = []
    for i in range(segments):
        lo = (i * n) // segments
        hi = ((i + 1) * n) // segments
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def first_divergence(same: list[bool]) -> int | None:
    """首个 `False` 的下标；全 `True` 返回 None（纯函数）。"""
    for i, ok in enumerate(same):
        if not ok:
            return i
    return None


def _position_top1_whole(gguf: str, seq: list[int], n_ctx: int):
    """整模**整段**前向 ⇒ 逐位置 (top1, argmax)。返回 (top1, argmax, selfcheck_ratio)。"""
    import numpy as np
    from llama_cpp import Llama

    llm = Llama(model_path=gguf, n_ctx=n_ctx, logits_all=True, verbose=False)
    llm.reset()
    llm.eval(list(seq))
    scores = np.asarray(llm.scores, dtype=np.float32)
    n = len(seq)
    if scores.shape[0] < n:
        raise RuntimeError(f"整模 scores 行数 {scores.shape[0]} < 序列长度 {n}")
    top1 = [float(np.max(scores[i])) for i in range(n)]
    am = [int(np.argmax(scores[i])) for i in range(n)]
    # ★ 自检：整段缓冲的 argmax 应重现序列的下一步（seq 本身是贪心生成出来的）
    pred = np.argmax(scores[: n - 1], axis=1)
    ok = int((pred == np.asarray(seq[1:], dtype=np.int64)).sum())
    del llm
    return top1, am, (ok / max(1, n - 1))


def _greedy_sequence(gguf: str, pids: list[int], gen_steps: int, n_ctx: int):
    """用整模贪心生成 `gen_steps` 步，返回完整 token 序列（teacher-forcing 的输入）。"""
    import numpy as np
    from llama_cpp import Llama

    llm = Llama(model_path=gguf, n_ctx=n_ctx, logits_all=True, verbose=False)
    llm.reset()
    llm.eval(list(pids))
    eos = llm.token_eos()
    seq = list(pids)
    for _ in range(gen_steps):
        row = np.asarray(llm.scores[len(seq) - 1], dtype=np.float32)
        nxt = int(np.argmax(row))
        if nxt == eos:
            break
        seq.append(nxt)
        llm.eval([nxt])
    del llm
    return seq


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="XFRAME-6 复验：真 D→L 链路的逐位置误差阶")
    ap.add_argument("--hf-dir", required=True, help="HF 模型目录（tokenizer + PyTorch 上游）")
    ap.add_argument("--whole-gguf", required=True, help="整模 GGUF（参照系 L）")
    ap.add_argument("--cut-artifact", required=True, help="下游裁层工件（丢前 K 层）")
    ap.add_argument("--k", type=int, required=True, help="上游层数 K（须与 --cut-artifact 对齐）")
    ap.add_argument("--prompts", help="每行一个 prompt 的文件；缺省用内置 3 条")
    ap.add_argument("--gen-steps", type=int, default=160, help="贪心生成的 token 数（决定位置数）")
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shim", help="keep-head shim 路径（--upstream llama 时的上游通道）")
    ap.add_argument(
        "--upstream", choices=["pytorch", "llama"], default="pytorch",
        help="上游由谁算：pytorch=主仓 `ModelManager` 层段（默认，**跨框架** D→L）；"
             "llama=主仓 `KeepHeadUpstream` 裁层件（**同框架** L→L，需 --head-artifact）",
    )
    ap.add_argument("--head-artifact", help="上游裁层件（保留前 K 层），配合 --upstream llama")
    ap.add_argument("--out", help="把报告写成 JSON")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    hf_dir = pathlib.Path(args.hf_dir)
    whole = pathlib.Path(args.whole_gguf)
    cut = pathlib.Path(args.cut_artifact)
    for path in (hf_dir, whole, cut):
        if not path.exists():
            print(f"SKIP: 缺工件 {path}")
            return 0

    import os

    for _p in (str(_ROOT), str(_ROOT / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import numpy as np
    from transformers import AutoTokenizer

    from xframe6_recursion_report import rel_series, summarise

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [
            ln for ln in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=False)

    # ---- 阶段 A：整模——生成序列 + 整段逐位置参照（用完即释放，避免与 PyTorch 上游争内存） ----
    prepared: list[dict] = []
    for prompt in prompts:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        pids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        seq = _greedy_sequence(str(whole), pids, args.gen_steps, args.n_ctx)
        top1, am, selfcheck = _position_top1_whole(str(whole), seq, args.n_ctx)
        prepared.append({"prompt": prompt, "pids": pids, "seq": seq,
                         "ref_top1": top1, "ref_argmax": am, "selfcheck": selfcheck})
        print(f"[A] {prompt[:36]!r}: seq={len(seq)} 自检一致率={selfcheck:.4f}", flush=True)

    # ---- 阶段 B：主仓引擎——上游（PyTorch 层段 或 llama.cpp 裁层件）→ llama.cpp 下游 ----
    import config as cfg
    import model_module

    cfg.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    cfg.USE_COMPILE = False
    model_module.USE_COMPILE = False
    import torch
    from llama_engine import LlamaCppEngine

    # shim 路径解析：显式参数 > 环境变量 > 仓库默认构建产物（`KeepHeadUpstream` 需要显式路径）
    shim_path = args.shim or os.environ.get("QLH_KEEP_HEAD_SHIM", "").strip()
    if not shim_path:
        shim_path = str(_ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll")
    os.environ["QLH_KEEP_HEAD_SHIM"] = shim_path

    head_path = pathlib.Path(args.head_artifact) if args.head_artifact else None
    mgr = None
    device = None
    if args.upstream == "pytorch":
        mgr = model_module.ModelManager()
        mgr.load_layer_range(0, args.k, has_embedding=True, has_lm_head=False,
                             model_path=str(hf_dir))
        device = mgr.get_device()
    elif head_path is None or not head_path.exists():
        print(f"SKIP: --upstream llama 需要存在 --head-artifact（实得 {args.head_artifact!r}）")
        return 0

    engine = LlamaCppEngine()
    engine.load_model(model_path=str(cut), n_ctx=args.n_ctx, n_threads=args.threads, n_seq_max=1)
    if not engine.is_loaded:
        print(f"SKIP: 下游工件加载失败 {cut}")
        return 0

    report = {"mode": "dl-relay-positions", "hf_dir": str(hf_dir), "k": args.k,
              "upstream": args.upstream, "head_artifact": str(head_path) if head_path else None,
              "whole_gguf": str(whole), "cut_artifact": str(cut),
              "gen_steps": args.gen_steps, "rows": []}

    for item in prepared:
        seq = item["seq"]
        # ★ `forward_layers_from_hidden` 的 KV 位置由调用方管理：同一 context 上换序列重跑
        #   必须清 KV，否则同位置重写 ⇒ `llama_decode rc=-1`（见该方法的 docstring）。
        _ctx = getattr(getattr(engine, "_model", None), "_ctx", None)
        if _ctx is not None and hasattr(_ctx, "kv_cache_clear"):
            _ctx.kv_cache_clear()
        if args.upstream == "pytorch":
            with torch.no_grad():
                out = mgr.forward_layers(
                    input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                    past_key_values=None, use_cache=False,
                )
            hidden = out["hidden_states"]
            hidden = hidden[0] if hidden.ndim == 3 else hidden
            hidden = hidden.to(torch.float32).cpu().numpy()
        else:
            # 同框架 L→L：上游也是 llama.cpp 裁层件（整段喂入；每序列新建实例避免 KV 串味）
            from llama_keep_head import KeepHeadUpstream

            up = KeepHeadUpstream(shim_path, str(head_path), mode="nextn",
                                  n_ctx=args.n_ctx, n_threads=args.threads)
            try:
                hidden = np.asarray(up.forward_tokens_to_hidden(seq, n_past=0),
                                    dtype=np.float32)
            finally:
                up.close()
        logits = engine.forward_layers_from_hidden(hidden, n_past=0, all_logits=True)
        if logits is None:
            print(f"[B] {item['prompt'][:36]!r}: 下游返回 None ⇒ 跳过")
            continue
        logits = np.asarray(logits, dtype=np.float32)
        n = min(len(seq), logits.shape[0])
        alt_top1 = [float(np.max(logits[i])) for i in range(n)]
        alt_argmax = [int(np.argmax(logits[i])) for i in range(n)]

        rel = rel_series(item["ref_top1"][:n], alt_top1)
        same = [a == b for a, b in zip(item["ref_argmax"][:n], alt_argmax)]
        summary = summarise(rel, same)
        summary["selfcheck_scores_argmax_ratio"] = round(item["selfcheck"], 4)
        summary["valid_control"] = bool(item["selfcheck"] >= 0.5)
        summary["segment_mean_rel"] = [round(v, 6) for v in segment_means(rel, 4)]
        summary["first_argmax_divergence"] = first_divergence(same)
        summary["prompt_tokens"] = len(item["pids"])
        summary["seq_len"] = len(seq)
        report["rows"].append({"prompt": item["prompt"][:60], **summary})

        print(f"[B] {item['prompt'][:36]!r}: argmax {summary['argmax_same_positions']}/{n} "
              f"rel_first={summary['first_rel']:.3e} rel_last={summary['last_rel']:.3e} "
              f"tail/head={summary['tail_over_head']:.3f} "
              f"segs={summary['segment_mean_rel']} "
              f"no_extra={summary['no_extra_accumulation']} "
              f"first_div={summary['first_argmax_divergence']}", flush=True)

    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
