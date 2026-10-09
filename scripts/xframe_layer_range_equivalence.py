#!/usr/bin/env python
"""xframe_layer_range_equivalence.py — 主仓引擎复验：**层段切分本身是否数值中性**。

为什么需要它
------------
`XFRAME-1` 的 `--layer-profile` 对照的是「HF 整模 vs **keep-head 上游**（llama.cpp）」，
覆盖的是**跨框架**接口；而生产 D 档的上游走的是**主仓自己的层段路径**：
`ModelManager.load_layer_range(0, K, has_embedding=True, has_lm_head=False)` +
`forward_layers(...)`。这条路径若与整模不等价，D→L 接力在**切分这一步**就引入了偏差 ——
但此前没有任何工具直接量过它（`relay_diag_head_norm.py` 量的是"通道语义"，不是等价性）。

方法（不手写前向：两侧都用各自引擎的正式入口）
----------------------------------------------
对每个切点 K：
  1. **HF 整模**（`AutoModelForCausalLM`，fp32，CPU）跑同一 prompt，取 `hidden_states[K]`；
  2. **主仓层段**（`ModelManager.load_layer_range(0, K, ...)`）跑同一 token 序列，
     取 `forward_layers(...)["hidden_states"]`；
  3. 比较两者（全位置与末位置）⇒ `rel_err` / `cos` / `max_abs`。

判据
----
同一份权重、同一套算子语义 ⇒ 期望**逐元素一致**（`rel_err` 应在 `1e-6` 量级或更低）。
若显著偏大，说明层段路径（mask / RoPE / 层索引 / embedding）与整模有语义差 —— 那就是 D 档上游的真缺陷。

★ 对照口径的**有效上界**：`hidden_states[k]` 是**第 k 层的输入**（= 前 k 层堆叠的输出），
  故 `forward_layers(0, K)` 应与 `hidden_states[K]` 对齐，且 **K 最大只能取 `len(hidden_states)-2`**。
  `hidden_states[-1]` **不是最后一层的输出** —— transformers v5 的通用记录器（`_can_record_outputs`）
  记的是「每个 DecoderLayer 的输入」，而 `Qwen3_5TextModel.forward` 在层循环后先做 final norm 再作为
  `last_hidden_state` 交给记录器（实测 `hidden_states[-1]` 与 HF **内部 hook 抓到的** layer-N 输出
  的 `rel_err = 0.76`，而层段路径与那个 hook 输出**逐比特一致**）⇒ 该口径下 K=num_layers 会给出
  假警报，故本脚本**主动跳过**并打印原因。

内存策略
--------
HF 整模 fp32 与主仓层段**不同时驻留**：第一阶段只把需要的 `hidden_states[K]` 落盘成 `.npz`，
释放模型后再进入第二阶段。避免峰值内存叠加。

工件缺失时按本仓惯例 **打印 SKIP 并退出 0**，不伪装成通过。

用法::

    python scripts/xframe_layer_range_equivalence.py \
        --model-dir models/qwen3-5-2b --layers 12,16 --out build/keephead/xframe-layer-range.json
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="层段切分等价性（主仓 load_layer_range vs 整模）")
    ap.add_argument("--model-dir", required=True, help="HF 模型目录（safetensors）")
    ap.add_argument("--layers", default="12,16", help="逗号分隔的切点 K 列表")
    ap.add_argument("--prompt", default=None, help="单行文本文件；缺省用内置短 prompt")
    ap.add_argument("--out", default=None, help="把结果写成 JSON")
    return ap.parse_args(argv)


DEFAULT_PROMPT = "Name the capital of France, then count from 1 to 5."


def _metrics(reference, actual) -> dict:
    """`rel_err`（范数比）/ `cos` / `max_abs`，与 XFRAME-1 的 `comparison_stats` 同口径。

    ★ 形状检查做在**展平之前**（与 `xframe_divergence_report.comparison_stats` 同源）：
    否则 (2,3) 与 (3,2) 展平后同为 6 元素，会被静默通过。
    """
    import numpy as np

    r0 = np.asarray(reference, dtype=np.float64)
    a0 = np.asarray(actual, dtype=np.float64)
    if r0.shape != a0.shape:
        raise ValueError(f"shape mismatch: {r0.shape} vs {a0.shape}")
    r = r0.reshape(-1)
    a = a0.reshape(-1)
    d = r - a
    nr = float(np.linalg.norm(r))
    na = float(np.linalg.norm(a))
    return {
        "rel_err": float(np.linalg.norm(d) / nr) if nr else 0.0,
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "cos": float(np.dot(r, a) / (nr * na)) if (nr and na) else 1.0,
        "identical": bool(np.array_equal(r, a)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"SKIP: 缺模型目录 {model_dir}")
        return 0
    if not list(model_dir.glob("*.safetensors")):
        print(f"SKIP: {model_dir} 下没有 safetensors")
        return 0

    ks = sorted({int(x) for x in str(args.layers).split(",") if x.strip()})
    prompt = (
        pathlib.Path(args.prompt).read_text(encoding="utf-8").strip()
        if args.prompt else DEFAULT_PROMPT
    )

    import numpy as np
    import torch
    import config as cfg
    import model_module

    cfg.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    cfg.USE_COMPILE = False
    model_module.USE_COMPILE = False

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    print(f"[prompt] {len(ids)} tokens; 切点 K={ks}", flush=True)

    # ---- 阶段 1：HF 整模 -> 保存 hidden_states[K]（唯一一份整模驻留期）----
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="xframe-lr-"))
    ref_paths: dict[int, pathlib.Path] = {}
    try:
        hf = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.float32, trust_remote_code=False,
        )
        hf.eval()
        with torch.no_grad():
            hs = hf(torch.tensor([ids]), output_hidden_states=True).hidden_states
        print(f"[hf] hidden_states 层数（含 embedding）= {len(hs)}", flush=True)
        for k in ks:
            if k >= len(hs) - 1:
                print(f"  [skip] K={k}: 对照口径上界为 len(hidden_states)-2 = {len(hs) - 2}；"
                      f"hidden_states[-1] 是 final-norm 之后的值（不是最后一层输出）⇒ 参见模块 docstring",
                      flush=True)
                continue
            arr = hs[k][0].to(torch.float32).cpu().numpy()
            p = tmpdir / f"hf_k{k}.npy"
            np.save(p, arr)
            ref_paths[k] = p
            print(f"  [hf] K={k} 保存 {arr.shape} rms={float(np.sqrt((arr.astype('float64')**2).mean())):.4f}")
        del hf
        gc.collect()
    finally:
        pass

    # ---- 阶段 2：主仓层段路径 -> 前向 -> 比较（HF 已释放）----
    rows = []
    for k in ks:
        if k not in ref_paths:
            continue
        mgr = model_module.ModelManager()
        mgr.load_layer_range(0, k, has_embedding=True, has_lm_head=False,
                             model_path=str(model_dir))
        device = mgr.get_device()
        with torch.no_grad():
            out = mgr.forward_layers(
                input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                past_key_values=None, use_cache=True,
            )
        got = out["hidden_states"]
        got = got[0] if got.ndim == 3 else got
        got_np = got.to(torch.float32).cpu().numpy()
        ref_np = np.load(ref_paths[k])
        st = _metrics(ref_np, got_np)
        # 末位置单独再看一眼（接力真正传的是每个位置的 hidden，末位置最常用）
        st_last = _metrics(ref_np[-1], got_np[-1])
        rows.append({"K": k, "shape": list(got_np.shape), **st, "last_pos": st_last})
        rms_ref = float(np.sqrt((ref_np.astype("float64") ** 2).mean()))
        rms_got = float(np.sqrt((got_np.astype("float64") ** 2).mean()))
        print(f"  [K={k:3d}] rel_err={st['rel_err']:.3e} cos={st['cos']:.9f} "
              f"max_abs={st['max_abs']:.3e} identical={st['identical']} "
              f"rms_ref={rms_ref:.4f} rms_got={rms_got:.4f} "
              f"| last_pos rel_err={st_last['rel_err']:.3e}", flush=True)
        del mgr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps({"mode": "layer-range-equivalence", "prompt_tokens": len(ids),
                        "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
