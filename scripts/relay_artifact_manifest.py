#!/usr/bin/env python
"""relay_artifact_manifest.py —— 给**已存在的**接力段工件补 manifest（★ #30）。

背景（`docs/已知问题记录.md` #30）
--------------------------------
`head*/mid*/tail*` 这批裁层工件由早期生成器（`build/cross-framework-layer-poc/cut_layers_generic.py`
与 `make_slice_gguf2.py`）产出，**不带 manifest**、也不记录来源层号 ⇒ **工件自身无法自证**
「我覆盖的是哪几层」，只能靠命名约定 + 人的记忆。2026-09-26 因此误判过一次（端点上实际载的是
`mid8-16`，而记录写成 `mid4-16` ⇒ 层覆盖缺 4..7，被当成"代码缺陷"）。

对比：`scripts/cut_layers.py` 生成的新工件（`cut-k16` / `cut-k12`）**带 manifest**，
含 `source_layer_range` / `kept_block_count` / `artifact_sha256`。

本脚本做什么
------------
**不重算工件**（不碰 GGUF 字节 ⇒ `sha256` 不变 ⇒ 既有 PASS 证据仍自洽），只把**已知的层范围**
落成与 `cut_layers.py` 同 schema 的 manifest，并对 `block_count` 做一次**实测校验**：
`block_count` 必须等于 `END - START`，否则 **fail-loud**（防止 manifest 与工件不符）。

⚠️ **层范围必须由人显式给出** —— 这正是本脚本存在的理由：工件自己**没有**这个信息，
而"靠命名约定记"正是 #30 要消除的那种不可自证。

用法::

    # 预览（不写文件）
    python scripts/relay_artifact_manifest.py \
        --artifact build/cross-framework-layer-poc/out/qwen25-05b-f16-mid8-16.gguf \
        --layers 8-16 --whole-model build/cross-framework-layer-poc/out/qwen25-05b-f16.gguf --dry-run

    # 写 manifest（默认落到 <artifact>.manifest.json）
    python scripts/relay_artifact_manifest.py --artifact <gguf> --layers 8-16 --whole-model <整模>

产出键与 `scripts/cut_layers.py:141 _manifest()` 对齐（`generator` 字段如实标注为本脚本）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

MANIFEST_SUFFIX = ".manifest.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _require_gguf():
    try:
        import gguf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise SystemExit(
            "需要 `gguf` 包（llama.cpp 的 Python 绑定之一）。\n"
            "  安装示例：pip install gguf\n"
            f"  原始错误：{exc}"
        ) from exc
    return gguf


def _read_identity(reader) -> dict:
    """读工件的 GGUF 元数据（与 `scripts/cut_layers.py:67 _read_identity` 同口径）。"""
    fields = reader.fields
    arch = fields["general.architecture"].contents()
    block_count = int(fields[f"{arch}.block_count"].contents())
    interval_field = fields.get(f"{arch}.full_attention_interval")
    interval = int(interval_field.contents()) if interval_field is not None else None
    nextn_field = fields.get(f"{arch}.nextn_predict_layers")
    nextn = int(nextn_field.contents()) if nextn_field is not None else 0
    n_embd_field = fields.get(f"{arch}.embedding_length")
    n_embd = int(n_embd_field.contents()) if n_embd_field is not None else 0
    return {
        "architecture": arch,
        "block_count": block_count,
        # 与 `scripts/cut_layers.py:82` 同口径：n_layer = block_count − nextn（MTP 层不计入）
        "n_layer": max(0, block_count - nextn),
        "nextn_predict_layers": nextn,
        "n_embd": n_embd,
        "full_attention_interval": interval,
    }


def _parse_layers(raw: str) -> tuple[int, int]:
    """解析 `--layers START-END`（与工件文件名同义，如 `8-16`）。"""
    text = str(raw).strip().replace("_", "-")
    left, _, right = text.partition("-")
    if not left.isdigit() or not right.isdigit():
        raise SystemExit(f"FAIL: --layers 需要 `START-END`（如 8-16），实得 {raw!r}")
    start, end = int(left), int(right)
    if not 0 <= start < end:
        raise SystemExit(f"FAIL: --layers 区间非法：{start}-{end}")
    return start, end


def _infer_mode(start: int, end: int, source_n_layer: int | None) -> str:
    """按区间推断段角色（与 `cut_layers.py:89 _cut_mode` 的三种模式对齐）。

    ⚠️ 必须用**源整模**的 `n_layer` 判"是否到末尾" —— 用工件自身的 `block_count` 会把
    `head4`（自身只有 4 层）判成 `whole`、把 `tail8` 判成 `middle`（**实测踩到**）。
    没给 `--whole-model` 时无法判定 ⇒ 如实记 `unknown`（不猜）。
    """
    if not source_n_layer or source_n_layer <= 0:
        return "unknown"
    if start == 0 and end == source_n_layer:
        return "whole"
    if start == 0:
        return "head"
    if end == source_n_layer:
        return "tail"
    return "middle"


def _manifest(artifact: Path, identity: dict, start: int, end: int, *,
              whole_model: Path | None, whole_sha: str | None,
              source_n_layer: int | None = None) -> dict:
    """组装与 `scripts/cut_layers.py` 同 schema 的 manifest（`generator` 如实标注）。"""
    kept = end - start
    total = int(identity["n_layer"])
    return {
        "generator": "scripts/relay_artifact_manifest.py",
        "generator_version": 1,
        # ★ 段角色按**源整模**的层数判定（`source_n_layer`）；没给 `--whole-model` ⇒ `unknown`。
        #   不要用 `total`（= **工件自身**的 n_layer）—— 那会把 `head4` 判成 `whole`（实测踩到）。
        "mode": _infer_mode(start, end, source_n_layer),
        # ★ 如实说明：工件由早期生成器产出，本 manifest 是**事后补录**（未重算工件）。
        "manifest_provenance": "backfilled_for_existing_artifact",
        "source": str(whole_model) if whole_model else "",
        "source_model_sha256": whole_sha or "",
        "artifact": str(artifact),
        "architecture": identity["architecture"],
        "block_count": int(identity["block_count"]),
        "n_layer": total,
        "nextn_predict_layers": int(identity["nextn_predict_layers"]),
        "n_embd": int(identity["n_embd"]),
        "full_attention_interval": identity["full_attention_interval"],
        "source_layer_range": [start, end],
        "trim_layers": start,
        "kept_block_count": kept,
        "first_local_layer_maps_to": start,
        "artifact_sha256": _sha256(artifact),
        "contract": {
            "manifest_matches_artifact": True,
            "kept_block_count": kept,
            "local_to_source_layer_0": start,
            "last_handoff_layer": end - 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="给已存在的接力段工件补 manifest（★ #30）")
    parser.add_argument("--artifact", required=True, help="段工件 GGUF（**不修改它**）")
    parser.add_argument("--layers", required=True,
                        help="该工件覆盖的层区间 `START-END`（如 `8-16`；★ 必须由人显式给出）")
    parser.add_argument("--whole-model", default=None,
                        help="可选：源整模 GGUF（用于填 source / source_model_sha256）")
    parser.add_argument("--manifest", default=None,
                        help=f"输出路径（默认 `<artifact>{MANIFEST_SUFFIX}`）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入的 manifest，不写文件")
    args = parser.parse_args(argv)

    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"FAIL: 工件不存在：{artifact}")
    start, end = _parse_layers(args.layers)

    gguf = _require_gguf()
    reader = gguf.GGUFReader(str(artifact))
    identity = _read_identity(reader)

    kept = end - start
    if int(identity["block_count"]) != kept:
        raise SystemExit(
            f"FAIL: 工件与 `--layers` 不符 —— 工件 block_count={identity['block_count']}，"
            f"而 `{start}-{end}` 要求 {kept} 层。\n"
            "  （这正是本脚本要防的情形：manifest 若与工件不一致，证据反而更不可信。）"
        )
    # ⚠️ `end` 是**源整模**的层号，不是工件自身的层数 ⇒ **不能**用工件的 `n_layer` 校验它
    #   （`mid8-16` 的 `n_layer` 只有 8，而 `end=16`）。工件自身只核对 `block_count == END - START`
    #   （上面那条）；"各段是否恰好铺满 `0..N`" 由**探针侧** `_layer_coverage` 负责（见 #30）。

    whole_model = Path(args.whole_model) if args.whole_model else None
    if whole_model and not whole_model.is_file():
        raise SystemExit(f"FAIL: --whole-model 不存在：{whole_model}")
    whole_sha = None
    source_n_layer = None
    if whole_model:
        try:
            source_sha = json.loads(
                (whole_model.parent / f"{whole_model.stem}-manifest.json").read_text(encoding="utf-8")
            ).get("artifact_sha256")
        except (OSError, ValueError):
            source_sha = None
        whole_sha = source_sha or _sha256(whole_model)
        # ★ 段角色（head / middle / tail）必须按**源整模**的层数判定，**不能**用工件自身的
        #   `block_count` —— 否则 `head4`（自身 4 层）会被判成 `whole`、`tail8` 被判成 `middle`
        #   （本轮实测踩到，已改成显式传入）。
        source_n_layer = int(_read_identity(gguf.GGUFReader(str(whole_model)))["n_layer"])

    manifest = _manifest(artifact, identity, start, end,
                         whole_model=whole_model, whole_sha=whole_sha,
                         source_n_layer=source_n_layer)
    target = Path(args.manifest) if args.manifest else artifact.with_suffix(
        artifact.suffix + MANIFEST_SUFFIX)

    print(f"[artifact] {artifact}")
    print(f"    architecture={identity['architecture']} block_count={identity['block_count']} "
          f"n_layer={identity['n_layer']} interval={identity['full_attention_interval']}")
    print(f"[layers]   {start}-{end}（{kept} 层）mode={manifest['mode']}  ✅ 与 block_count 一致")
    print(f"[manifest] {target}")

    if args.dry_run:
        print("[dry-run] 不写任何文件。将写入的内容：")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"[done] 已写入（artifact_sha256={manifest['artifact_sha256'][:16]}…）；工件本体未被修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
