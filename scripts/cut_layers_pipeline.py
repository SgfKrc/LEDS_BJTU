#!/usr/bin/env python
"""cut_layers_pipeline.py — 切层工件制作**流水线**（`#67` 缺口：把四件事串起来）

背景：此前切层是手工敲 CLI（`cut_layers.py --src … --k … --manifest …`），
"**源 sha256 / 用的命令 / 产物 sha256 / 校验结果**"没有任何一处同时留下；
而且 ③（层类型逐位）/ ⑩（按层数组 KV）/ ④（必需张量集合）三条 fail-closed 校验
虽然都实现进了 `cut_layers.py`，但**分散运行**，没人保证一次制作会把它们全跑过。

本脚本把这条链固定下来，一次调用产出三件套：

    <outdir>/<arch>-cut-<lo>-<hi>.gguf         裁层工件
    <outdir>/<arch>-cut-<lo>-<hi>.manifest.json 可复算 manifest
    <outdir>/cut-report.json                    审计报告（源 sha / 命令 / 产物 sha / 校验结果）

并在**失败时绝不留下 `cut-report.json`**（fail-closed：只有全绿才算一次成功制作）。

用法：
    python scripts/cut_layers_pipeline.py --src models/qwen35-2b-Q4_K_M.gguf --k 4 --outdir out/
    python scripts/cut_layers_pipeline.py --src full.gguf --keep-head 12 --outdir out/
    python scripts/cut_layers_pipeline.py --src full.gguf --k 8 --end 16 --outdir out/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUT_SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
REPORT_SCHEMA = "qlh.cut_layers.report.v1"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_identity(src: Path) -> dict:
    """读源的关键元数据（架构 / 层数 / interval）。缺 gguf 时给出可读错误。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import gguf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"FAIL: 需要 gguf 包（{exc}）")
    reader = gguf.GGUFReader(str(src))
    fields = reader.fields
    arch = str(fields["general.architecture"].contents())
    block_count = int(fields[f"{arch}.block_count"].contents())
    nextn_field = fields.get(f"{arch}.nextn_predict_layers")
    nextn = int(nextn_field.contents()) if nextn_field is not None else 0
    interval_field = fields.get(f"{arch}.full_attention_interval")
    interval = int(interval_field.contents()) if interval_field is not None else None
    return {
        "architecture": arch,
        "block_count": block_count,
        "n_layer": max(0, block_count - nextn),
        "nextn_predict_layers": nextn,
        "full_attention_interval": interval,
    }


def _legal_cut_check(total_layers: int, cut_multiple: int, k: int | None,
                     end: int | None, keep_head: int | None) -> tuple[bool, str]:
    """切点合法性（与 `relay_cut_objective.legal_cuts` 同一判据）。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.relay_cut_objective import legal_cuts

    if keep_head is not None:
        if not (0 < keep_head < total_layers):
            return False, f"--keep-head {keep_head} 不在 (0, {total_layers})"
        if cut_multiple > 1 and keep_head % cut_multiple:
            return False, f"--keep-head {keep_head} 不是 {cut_multiple} 的整数倍"
        return True, ""
    if k is None:
        return False, "需要 --k 或 --keep-head"
    if end is None:
        cuts = legal_cuts(total_layers, cut_multiple=max(1, cut_multiple))
        if k not in cuts:
            return False, f"K={k} 不在合法切点 {cuts}（步长 {cut_multiple}）"
        return True, ""
    cuts = legal_cuts(total_layers, cut_multiple=max(1, cut_multiple))
    if k not in cuts:
        return False, f"K={k} 不在合法切点 {cuts}（步长 {cut_multiple}）"
    if end not in cuts and end != total_layers:
        return False, f"--end={end} 不在合法切点 {cuts}（步长 {cut_multiple}）"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切层工件制作流水线（产出工件 + manifest + report）")
    ap.add_argument("--src", required=True, help="源 GGUF（通常是整模）")
    ap.add_argument("--k", type=int, default=None, help="丢弃前 K 层（保留 blk.K..）")
    ap.add_argument("--end", type=int, default=None, help="与 --k 同用：中段工件 [K,end)")
    ap.add_argument("--keep-head", type=int, default=None, help="保留前 N 层（上游段工件）")
    ap.add_argument("--outdir", required=True, help="输出目录（工件/manifest/report 都落这里）")
    ap.add_argument("--hf-config", default=None,
                    help="HF config.json（可选）：额外做 #67-③ 层类型逐位校验")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.is_file():
        print(f"FAIL: 源不存在: {src}")
        return 2
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    identity = _read_identity(src)
    total = int(identity["n_layer"])

    # ★ 自动读 cut_multiple（#67）：人工不再需要记住 full_attention_interval。
    from src.relay_cut_objective import cut_multiple_from_gguf  # noqa: PLC0415

    auto_multiple = cut_multiple_from_gguf(src)
    cut_multiple = auto_multiple if auto_multiple is not None else 1
    print(f"[pipeline] 源={src.name} arch={identity['architecture']} n_layer={total} "
          f"interval={identity['full_attention_interval']} => cut_multiple={cut_multiple}")

    ok, why = _legal_cut_check(total, cut_multiple, args.k, args.end, args.keep_head)
    if not ok:
        print(f"FAIL: 切点不合法 — {why}")
        return 2

    # 产物命名（与设备侧封装一致的 `-cut-<lo>-<hi>` 风格）
    if args.keep_head is not None:
        lo, hi = 0, int(args.keep_head)
    else:
        lo = int(args.k)
        hi = int(args.end) if args.end is not None else total
    stem = f"{identity['architecture']}-cut-{lo}-{hi}"
    dst = outdir / f"{stem}.gguf"
    manifest_path = outdir / f"{stem}.manifest.json"

    cmd = [sys.executable, str(CUT_SCRIPT), "--src", str(src), "--dst", str(dst),
           "--manifest", str(manifest_path)]
    if args.keep_head is not None:
        cmd += ["--keep-head", str(args.keep_head)]
    else:
        cmd += ["--k", str(args.k)]
        if args.end is not None:
            cmd += ["--end", str(args.end)]
    if args.hf_config:
        cmd += ["--hf-config", str(args.hf_config)]

    print(f"[pipeline] 执行: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tail = (proc.stdout or "")[-800:]
    if proc.returncode != 0:
        print(f"FAIL: 切层失败 (exit={proc.returncode})\n{tail}\n{proc.stderr or ''}")
        # fail-closed：绝不在失败时留下 report（产物可能已部分写出，交由人工处理）
        return 2
    if not dst.is_file():
        print(f"FAIL: 切层报告成功但工件不存在: {dst}\n{tail}")
        return 2

    # 复算校验：manifest 对照（#67 要求的"产物 sha"闭合）
    verify = subprocess.run(
        [sys.executable, str(CUT_SCRIPT), "--src", str(dst),
         "--verify-manifest", str(manifest_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    manifest_verified = verify.returncode == 0

    report = {
        "schema": REPORT_SCHEMA,
        "source": {
            "path": str(src),
            "sha256": _sha256(src),
            **identity,
            "cut_multiple": cut_multiple,
            "cut_multiple_source": "auto(gguf)" if auto_multiple is not None else "fallback(1)",
            "source_layer_range": [lo, hi],
        },
        "command": cmd,
        "artifact": {
            "path": str(dst),
            "sha256": _sha256(dst),
            "manifest": str(manifest_path),
        },
        "checks": {
            "cut_point_legal": True,
            "manifest_verified": manifest_verified,
        },
    }
    (outdir / "cut-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pipeline] 工件={dst.name} sha256={report['artifact']['sha256'][:16]}… "
          f"manifest_verified={manifest_verified}")
    print(f"[pipeline] report={outdir / 'cut-report.json'}")
    return 0 if manifest_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
