"""逐张量比较两个 GGUF 的同名张量（反量化后）—— 用于「反量化实现是否语义等价」的自证。

为什么需要它
------------
跨框架接力曾把 4-bit 通路的残差归因于"PyTorch 侧用的是**第三方反量化**、与 llama.cpp 内部不一致"。
要证伪/证实这一点，必须**直接比权重值**，而不是比模型输出：

    llama-quantize --allow-requantize in-q4km.gguf out-f16.gguf F16   # 走 llama.cpp 的 dequantize_row_q*
    python scripts/gguf_tensor_diff.py --left in-q4km.gguf --right out-f16.gguf --by-qtype

判读：若各量化类型的 `rel` 都落在 **f16 舍入量级**（`~1e-4`～`2e-4`）、且 `F32` 类型为 `0`，
则两套反量化**语义等价**（差异只是把结果写成 f16 的舍入），"归因于反量化"不成立。

用法::

    python scripts/gguf_tensor_diff.py --left a.gguf --right b.gguf [--by-qtype] [--top 5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def compare_arrays(reference, actual) -> dict:
    """两个同形数组的画像：`rel`（分母取 reference 的 L2 范数）/ `max_abs`（纯函数）。"""
    import numpy as np

    r = np.asarray(reference, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if r.shape != a.shape:
        raise ValueError(f"shape mismatch: {r.shape} vs {a.shape}")
    rf = r.reshape(-1)
    af = a.reshape(-1)
    d = rf - af
    nr = float(np.linalg.norm(rf))
    return {
        "rel": float(np.linalg.norm(d) / nr) if nr else 0.0,
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "identical": bool(np.array_equal(rf, af)),
    }


def summarise_by_qtype(rows: list[dict]) -> dict:
    """按 `qtype` 汇总 `rel`（纯函数）：给出每类张量的数量与最大/平均 `rel`。"""
    by: dict[str, list[float]] = {}
    for row in rows:
        by.setdefault(str(row["qtype"]), []).append(float(row["rel"]))
    return {
        q: {"n": len(v), "max_rel": max(v), "mean_rel": sum(v) / len(v)}
        for q, v in sorted(by.items())
    }


def _read(path: pathlib.Path) -> tuple[dict, dict]:
    """读 GGUF ⇒ {张量名: f32 数组} 与 {张量名: qtype}。"""
    import numpy as np
    import gguf
    from gguf import quants

    reader = gguf.GGUFReader(str(path))
    tensors: dict[str, object] = {}
    qtypes: dict[str, str] = {}
    for t in reader.tensors:
        qtypes[t.name] = str(t.tensor_type)
        if str(t.tensor_type) == "0":
            arr = np.asarray(t.data, dtype=np.float32)
        else:
            arr = np.asarray(quants.dequantize(t.data, t.tensor_type), dtype=np.float32)
        tensors[t.name] = np.ascontiguousarray(arr)
    return tensors, qtypes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="两个 GGUF 的逐张量差异（反量化后）")
    ap.add_argument("--left", required=True, help="参照 GGUF")
    ap.add_argument("--right", required=True, help="被测 GGUF")
    ap.add_argument("--by-qtype", action="store_true", help="按 qtype 汇总 rel")
    ap.add_argument("--top", type=int, default=0, help="额外打印 rel 最大的前 N 个张量")
    ap.add_argument("--out", help="把结果写成 JSON")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    left, right = pathlib.Path(args.left), pathlib.Path(args.right)
    for p in (left, right):
        if not p.is_file():
            print(f"SKIP: 缺工件 {p}")
            return 0

    a, qa = _read(left)
    b, _qb = _read(right)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"[warn] 张量集合不同：仅左 {len(only_a)}、仅右 {len(only_b)}（按交集比较）")

    rows = []
    for name in sorted(set(a) & set(b)):
        if a[name].shape != b[name].shape:
            print(f"[skip] {name}: shape {a[name].shape} vs {b[name].shape}")
            continue
        rows.append({"name": name, "qtype": qa.get(name, "?"), **compare_arrays(a[name], b[name])})

    print(f"[info] 比较 {len(rows)} 个张量：{left.name} vs {right.name}")
    print(f"{'tensor':38s} {'qtype':>6s} {'rel':>12s} {'max_abs':>12s}")
    for row in rows[:8]:
        print(f"{row['name']:38s} {row['qtype']:>6s} {row['rel']:12.3e} {row['max_abs']:12.3e}")

    summary = summarise_by_qtype(rows)
    if args.by_qtype:
        print("\n=== 按 qtype 汇总 ===")
        for q, st in summary.items():
            print(f"  qtype={q:>3s} n={st['n']:4d} max_rel={st['max_rel']:.3e} "
                  f"mean_rel={st['mean_rel']:.3e}")

    if args.top:
        print(f"\n=== rel 最大的前 {args.top} 个 ===")
        for row in sorted(rows, key=lambda r: -r["rel"])[: args.top]:
            print(f"  {row['name']:38s} qtype={row['qtype']:>3s} rel={row['rel']:.3e}")

    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"left": str(left), "right": str(right),
                                    "n_compared": len(rows), "by_qtype": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
