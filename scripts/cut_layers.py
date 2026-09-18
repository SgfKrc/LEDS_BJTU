#!/usr/bin/env python
"""cut_layers.py — 离线裁层生成器（主仓正式版）

用途：为「上游跑前 K 层、下游跑其余层」的接力造**下游 GGUF 工件**。
主仓此前只有裁层的**合同与校验**（``src/relay_contract.py`` 的 ``RelayTrimPlan`` /
``RelayHandoff``，以及 ``src/pipeline_node_contract.py`` 的节点校验），没有生成器；
本脚本把实验区的 ``cut_layers_generic.py`` 正规化进来，并与合同对齐。

裁层语义（与 ``relay_contract.RelayTrimPlan`` 一致）
  * 保留 ``blk.K..blk.(N-1)``，并**重命名**为 ``blk.0..blk.(N-1-K)``；
  * ``<arch>.block_count`` 由 ``N`` 改写为 ``N-K``；
  * 非 blk 张量（``token_embd`` / ``output_norm`` / ``output`` 等）**原样保留**；
  * 其余 KV 字段全部复制（``general.architecture`` 与 ``GGUF.*`` 由 writer 维护，不复制）。

约束（2026-09-18 实测）
  * ``0 < K < n_layer``；
  * **K 必须是 ``<arch>.full_attention_interval`` 的整数倍** —— 否则 hybrid 架构（如 Qwen3.5）
    的层类型会错位，llama.cpp 报 ``missing tensor 'blk.x.<...>'`` 而无法加载。

用法
  # 预览（不改任何文件）
  python scripts/cut_layers.py --src full.gguf --k 12 --dry-run

  # 生成裁层工件 + manifest
  python scripts/cut_layers.py --src full.gguf --dst cut.gguf --k 12 \
      --manifest cut.json

  # 校验一个已生成的裁层工件（对照其 manifest）
  python scripts/cut_layers.py --src cut.gguf --verify-manifest cut.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


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


def _read_identity(reader, source_path: Path) -> dict:
    """读取裁层所需的源工件元数据（与 relay_contract.RelayModelIdentity 对齐）。"""
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
        "source": str(source_path),
        "architecture": arch,
        "block_count": block_count,
        "n_layer": max(0, block_count - nextn),
        "nextn_predict_layers": nextn,
        "n_embd": n_embd,
        "full_attention_interval": interval,
    }


def _validate_cut(identity: dict, k: int) -> list[str]:
    problems: list[str] = []
    n_layer = identity["n_layer"]
    if not (0 < k < n_layer):
        problems.append(f"K={k} 不在 (0, {n_layer}) 内")
    interval = identity["full_attention_interval"]
    if interval and k % interval:
        problems.append(
            f"K={k} 不是 full_attention_interval({interval}) 的整数倍 —— "
            "hybrid 架构会层类型错位、无法加载"
        )
    return problems


def _manifest(identity: dict, k: int, dst: Path, kept: int, dropped: int) -> dict:
    """按 relay_contract 的 TrimPlan 语义产出 manifest。"""
    kept_block_count = max(0, identity["block_count"] - k)
    result = {
        "generator": "scripts/cut_layers.py",
        "generator_version": 1,
        "source": identity["source"],
        "artifact": str(dst),
        "architecture": identity["architecture"],
        "block_count": identity["block_count"],
        "n_layer": identity["n_layer"],
        "nextn_predict_layers": identity["nextn_predict_layers"],
        "trim_layers": k,
        "kept_block_count": kept_block_count,
        "tensors_kept": kept,
        "tensors_dropped": dropped,
        "first_local_layer_maps_to": k,
        "artifact_sha256": _sha256(dst) if dst.exists() else "",
    }
    try:  # 与主仓合同字段对齐（缺失时不影响生成）
        from relay_contract import RelayModelIdentity, RelayTrimPlan  # noqa: PLC0415

        src = RelayModelIdentity(
            model_sha256=result.get("artifact_sha256", ""),
            architecture=identity["architecture"],
            block_count=identity["block_count"],
            n_embd=identity["n_embd"],
            nextn_predict_layers=identity["nextn_predict_layers"],
        )
        plan = RelayTrimPlan(trim_layers=k, source=src)
        result["contract"] = {
            "trim_plan_valid": plan.is_valid(),
            "kept_block_count": plan.kept_block_count,
            "local_to_source_layer_0": plan.local_to_source_layer(0),
            "last_handoff_layer": src.last_handoff_layer,
        }
    except Exception as exc:  # noqa: BLE001 - 合同属可选校验
        result["contract"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _copy_kv(gguf, reader, writer, overrides: dict) -> int:
    def infer_type(value):
        if isinstance(value, bool):
            return gguf.GGUFValueType.BOOL
        if isinstance(value, int):
            return gguf.GGUFValueType.INT32
        if isinstance(value, float):
            return gguf.GGUFValueType.FLOAT32
        if isinstance(value, str):
            return gguf.GGUFValueType.STRING
        raise ValueError(f"无法推断类型: {type(value)}")

    count = 0
    for name, field in reader.fields.items():
        # architecture 由 writer 构造时写入；GGUF.version/tensor_count 等由 writer 维护，
        # 复制会造成 "Duplicate GGUF.version" 这类文件损坏。
        if name == "general.architecture" or name.startswith("GGUF."):
            continue
        try:
            value = field.contents()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! KV 跳过 {name}: {exc}")
            continue
        if name in overrides:
            value = overrides[name]
        types = list(getattr(field, "types", []) or [])
        vtype = types[0] if types else None
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = types[1] if len(types) > 1 else infer_type(value[0])
            writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub)
        elif vtype is not None:
            writer.add_key_value(name, value, vtype)
        else:
            raise RuntimeError(f"无法确定 {name} 的类型")
        count += 1
    return count


def _plan_tensors(reader, k: int) -> tuple[list, list]:
    """返回 (保留张量及新名字, 将丢弃的张量名)。不做写入，便于 --dry-run。"""
    keep: list[tuple] = []
    drop: list[str] = []
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            parts = name.split(".", 2)
            index = int(parts[1])
            if index < k:
                drop.append(name)
                continue
            keep.append((tensor, f"blk.{index - k}.{parts[2]}"))
        else:
            keep.append((tensor, name))
    return keep, drop


def main() -> int:
    ap = argparse.ArgumentParser(description="离线裁层生成器（下游工件）")
    ap.add_argument("--src", required=True, help="源 GGUF（通常是整模）")
    ap.add_argument("--dst", help="输出裁层 GGUF")
    ap.add_argument("--k", type=int, help="丢弃前 K 层（保留 blk.K.. 起）")
    ap.add_argument("--manifest", help="输出 manifest JSON 的路径")
    ap.add_argument("--verify-manifest", help="校验模式：对照该 manifest 检查 --src 工件")
    ap.add_argument("--dry-run", action="store_true", help="只列出影响，不写文件")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"FAIL: 源文件不存在: {src}")
        return 2

    gguf = _require_gguf()
    reader = gguf.GGUFReader(str(src))
    identity = _read_identity(reader, src)

    # ---------------- 校验模式 ----------------
    if args.verify_manifest:
        manifest = json.loads(Path(args.verify_manifest).read_text(encoding="utf-8"))
        expected = int(manifest.get("kept_block_count", -1))
        ok = True
        if identity["block_count"] != expected:
            print(f"FAIL: block_count={identity['block_count']} ≠ manifest.kept_block_count={expected}")
            ok = False
        digest = _sha256(src)
        if manifest.get("artifact_sha256") and manifest["artifact_sha256"] != digest:
            print("FAIL: artifact_sha256 不一致（工件与 manifest 不匹配）")
            ok = False
        trim = int(manifest.get("trim_layers", 0))
        if identity["block_count"] and trim:
            print(f"OK: block_count={identity['block_count']}，由 {identity['block_count'] + trim} 裁掉前 {trim} 层")
        else:
            print(f"OK: block_count={identity['block_count']}")
        print(f"     artifact_sha256={digest[:16]}…")
        return 0 if ok else 1

    if args.k is None:
        print("FAIL: 需要 --k（或使用 --verify-manifest）")
        return 2

    problems = _validate_cut(identity, args.k)
    keep, drop = _plan_tensors(reader, args.k)
    kept_block_count = max(0, identity["block_count"] - args.k)
    print(f"[src] {src}")
    print(f"      architecture={identity['architecture']} block_count={identity['block_count']} "
          f"n_layer={identity['n_layer']} nextn={identity['nextn_predict_layers']} "
          f"full_attention_interval={identity['full_attention_interval']}")
    print(f"[plan] 丢弃前 K={args.k} 层 ⇒ 保留 {len(keep)} 张量、丢弃 {len(drop)} 张量；"
          f"block_count: {identity['block_count']} -> {kept_block_count}；"
          f"原 blk.{args.k}.* 将变为 blk.0.*")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 2
    if args.dry_run:
        print("[dry-run] 不写任何文件。将丢弃的 blk 张量示例（最多 8 个）：")
        for name in drop[:8]:
            print(f"    - {name}")
        if len(drop) > 8:
            print(f"    … 共 {len(drop)} 个")
        return 0

    if not args.dst:
        print("FAIL: 非 dry-run 模式需要 --dst")
        return 2

    dst = Path(args.dst)
    writer = gguf.GGUFWriter(str(dst), identity["architecture"])
    bc_name = f"{identity['architecture']}.block_count"
    n_kv = _copy_kv(gguf, reader, writer, {bc_name: kept_block_count})
    print(f"[kv] 复制 {n_kv} 字段，{bc_name}: {identity['block_count']} -> {kept_block_count}")
    for tensor, new_name in keep:
        writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] {dst}：保留 {len(keep)} 张量、丢弃 {len(drop)}")

    manifest = _manifest(identity, args.k, dst, len(keep), len(drop))
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"[manifest] {args.manifest}")
    print(f"[summary] kept_block_count={manifest['kept_block_count']} "
          f"artifact_sha256={manifest['artifact_sha256'][:16]}… "
          f"contract_valid={manifest.get('contract', {}).get('trim_plan_valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
