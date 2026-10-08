"""XFRAME-4 / XFRAME-5 可复跑工具：归约顺序差异下界，以及「换累加数域」对照。

载体
----
`build/keephead/qlh_shared_rms_norm.c` 编译出的共享库，导出两个同契约函数：

* `qlh_matmul_f32(x, w, y, M, N, K, order)`      —— **f32 累加**（XFRAME-4）
* `qlh_matmul_f32_acc64(x, w, y, M, N, K, order)` —— **f64 累加**（XFRAME-5）

`order` 控制 K 维归约顺序：`0`=正向 / `1`=反向 / `2·4·8`=交错分路。**只改 `order`
只改归约顺序**，其余完全一致 ⇒ 差异全部来自"浮点加法非结合"。

复现的结论（见 `docs/跨框架接力精度结论与生产准入-2026-10-08.md`）
------------------------------------------------------------
* f32 下「只改归约顺序」的差异随 K **近似线性**（阶 `O(K·ε)`；K=2048 时 `rel≈3.5e-04`），
  比算子实现层差异大 ~3 个数量级；
* **f64 累加下同一对照差异为 0** ⇒ 「数学墙」可被"提高累加精度"绕过；但 f64 只作机理
  证明（降吞吐、须两侧同改），生产载体是「整型累加 + 共享量化网格」。

用法
----
```
python scripts/xframe_reduction_order_report.py \
    --lib build/keephead/qlh_shared_rms_norm.dll --out build/keephead/xframe45-report.json
```
若缺少共享库，工具会给出**具名**提示（含构建方式），而不是抛栈。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys

import numpy as np

DEFAULT_ORDERS = (0, 1, 2, 4, 8)
DEFAULT_K = (128, 512, 2048, 8192)


def load_library(lib_path: str) -> ctypes.CDLL:
    """加载共享库并声明两个导出函数的签名（找不到时给具名提示）。"""
    p = pathlib.Path(lib_path)
    if not p.is_file():
        raise SystemExit(
            f"缺少共享库 {p}；先构建："
            "clang -O2 -shared -o build/keephead/qlh_shared_rms_norm.dll "
            "build/keephead/qlh_shared_rms_norm.c"
        )
    lib = ctypes.CDLL(str(p))
    for name in ("qlh_matmul_f32", "qlh_matmul_f32_acc64"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
    return lib


def run_case(fn, x: np.ndarray, w: np.ndarray, m: int, n: int, k: int,
             order: int) -> np.ndarray:
    """调一次 kernel：`y[m,n] = Σ_k x[m·K+k]·w[n·K+k]`，返回 `(M, N)` float32。"""
    x32 = np.ascontiguousarray(x, dtype=np.float32).ravel()
    w32 = np.ascontiguousarray(w, dtype=np.float32).ravel()
    y = np.zeros((m * n,), dtype=np.float32)
    fn(x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       w32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       m, n, k, order)
    return y.reshape(m, n)


def comparison_stats(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """`max_abs` 与相对差（分母取参照的 `max|·|`）。纯函数，可单测。"""
    r = np.asarray(reference, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    max_abs = float(np.max(np.abs(r - a)))
    denom = float(np.max(np.abs(r))) or 1e-30
    return {"max_abs": max_abs, "rel": max_abs / denom}


def build_report(lib: ctypes.CDLL, *, orders=DEFAULT_ORDERS, ks=DEFAULT_K,
                 m: int = 8, n: int = 64, seed: int = 1234) -> dict:
    """跑 `order × K` 网格：f32 与 f64 两种累加域下的"只改归约顺序"差异。"""
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        x = rng.standard_normal((m, k)).astype(np.float32)
        w = rng.standard_normal((n, k)).astype(np.float32)
        base_f32 = run_case(lib.qlh_matmul_f32, x, w, m, n, k, orders[0])
        base_f64 = run_case(lib.qlh_matmul_f32_acc64, x, w, m, n, k, orders[0])
        for order in orders:
            got_f32 = run_case(lib.qlh_matmul_f32, x, w, m, n, k, order)
            got_f64 = run_case(lib.qlh_matmul_f32_acc64, x, w, m, n, k, order)
            rows.append({
                "K": k,
                "order": order,
                "f32_vs_order0": comparison_stats(base_f32, got_f32),
                "acc64_vs_order0": comparison_stats(base_f64, got_f64),
            })
    return {"M": m, "N": n, "seed": seed, "orders": list(orders), "ks": list(ks),
            "rows": rows}


def summarise(report: dict) -> dict:
    """汇总两族判据（纯函数）：
    * f32：`orders != 0` 的最大 rel 随 K 是否单调增长（阶的形态）；
    * acc64：所有 `rel` 是否恒为 0（"墙可绕过"）。
    """
    worst_f32 = {}
    worst_acc64 = {}
    for row in report["rows"]:
        if row["order"] == report["orders"][0]:
            continue
        k = row["K"]
        worst_f32[k] = max(worst_f32.get(k, 0.0), row["f32_vs_order0"]["rel"])
        worst_acc64[k] = max(worst_acc64.get(k, 0.0), row["acc64_vs_order0"]["rel"])
    ks = sorted(worst_f32)
    monotonic = all(
        worst_f32[ks[i]] <= worst_f32[ks[i + 1]]
        for i in range(len(ks) - 1)
    ) if len(ks) > 1 else True
    return {
        "f32_worst_rel_by_K": worst_f32,
        "acc64_worst_rel_by_K": worst_acc64,
        "f32_grows_with_K": monotonic,
        "acc64_all_zero": all(v == 0.0 for v in worst_acc64.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XFRAME-4/5：归约顺序下界 + 换累加数域对照")
    ap.add_argument("--lib", default="build/keephead/qlh_shared_rms_norm.dll",
                    help="qlh_shared_rms_norm 共享库路径")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--out", help="把报告写成 JSON")
    args = ap.parse_args()

    lib = load_library(args.lib)
    report = build_report(lib, m=args.M, n=args.N)
    summary = summarise(report)

    print("[xframe45] f32「只改归约顺序」的最大 rel（按 K）：")
    for k, rel in summary["f32_worst_rel_by_K"].items():
        print(f"  K={k:<6d} rel={rel:.3e}")
    print("[xframe45] 同对照换 f64 累加后：")
    for k, rel in summary["acc64_worst_rel_by_K"].items():
        print(f"  K={k:<6d} rel={rel:.3e}")
    print(f"[xframe45] f32 随 K 增长={summary['f32_grows_with_K']} "
          f"f64 恒为 0={summary['acc64_all_zero']}")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**report, "summary": summary},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[xframe45] written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
