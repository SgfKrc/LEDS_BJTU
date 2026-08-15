"""G4.4：gemma4 原生独立工件冻结与校验（摆脱 Ollama blobs 依赖）。

受管工件（Hugging Face bartowski 转换，apache-2.0，经 7897 代理下载）：
  - 主模型: bartowski/gemma-4-12B-it-GGUF  gemma-4-12B-it-Q4_K_M.gguf
  - 投影器: bartowski/gemma-4-12B-it-GGUF  mmproj-gemma-4-12B-it-bf16.gguf

用法：
  python scripts/model_tools/gemma4_native_freeze.py --hash        # 全量 SHA-256 校验并冻结
  python scripts/model_tools/gemma4_native_freeze.py --check       # 只读核对（与冻结记录比对）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "gemma4-native"
LOCK = MODEL_DIR / "gemma4-native.lock.json"

# G4.4 最终工件（2026-08-15，独立来源达成）：Hugging Face bartowski Q4_K_M +
# bartowski mmproj-BF16（均 apache-2.0，经 7897 代理下载）。G4.6 调研确认
# Ollama 稳定机制 = llama.cpp b10434 reasoning-budget sampler（思考段超预算
# 强制结束）；47e1de77 上已实现等效（--think-budget 注入结束 tag + 头尾
# 裁剪），HF 工件 + 等效机制 3/3 稳定（sd-001 语义正确）。
ARTIFACTS = {
    "main_gguf": {
        "source": "https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf",
        "license": "apache-2.0",
        "filename": "gemma-4-12B-it-Q4_K_M.gguf",
        "expected_size": 7_662_533_088,
        "expected_sha256": "3962624dcd25b947d889dc9ae1bf275b61db6cd4dbe694057f34fffef1671509",
    },
    "mmproj": {
        "source": "https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/mmproj-gemma-4-12B-it-bf16.gguf",
        "license": "apache-2.0",
        "filename": "mmproj-gemma-4-12B-it-bf16.gguf",
        "expected_size": 175_115_712,
        "expected_sha256": "92de172d87a262e4873a2a1d909b1b6082a76909957648705f00cb9feaa16535",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze() -> dict:
    record = {
        "schema_version": 1,
        "updated_at": "",
        "note": "G4.4 独立 Hugging Face bartowski 转换工件（不依赖 Ollama blobs）",
        "artifacts": {},
    }
    for key, spec in ARTIFACTS.items():
        path = MODEL_DIR / spec["filename"]
        if not path.is_file():
            raise SystemExit(f"缺失工件: {path}")
        size = path.stat().st_size
        if size != spec["expected_size"]:
            raise SystemExit(
                f"工件大小不匹配: {path.name} {size} != {spec['expected_size']}",
            )
        sha = _sha256(path)
        if sha != spec["expected_sha256"]:
            raise SystemExit(f"工件 SHA-256 不匹配: {path.name}")
        print(f"[freeze] {key}: {path.name}  {size/2**30:.2f} GB  sha256={sha[:16]}…")
        record["artifacts"][key] = {
            "filename": spec["filename"],
            "source": spec["source"],
            "license": spec["license"],
            "size_bytes": spec["expected_size"],
            "sha256": spec["expected_sha256"],
        }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(
        json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[freeze] 冻结记录: {LOCK}")
    return record


def _check() -> int:
    if not LOCK.is_file():
        raise SystemExit("无冻结记录：先运行 --hash")
    try:
        record = json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("冻结记录不是有效 JSON") from exc
    entries = record.get("artifacts") if isinstance(record, dict) else None
    if record.get("schema_version") != 1 or not isinstance(entries, dict):
        raise SystemExit("冻结记录 schema 无效")
    if set(entries) != set(ARTIFACTS):
        raise SystemExit("冻结记录工件集合无效")
    failures = 0
    for key, spec in ARTIFACTS.items():
        entry = entries.get(key)
        expected_entry = {
            "filename": spec["filename"],
            "source": spec["source"],
            "license": spec["license"],
            "size_bytes": spec["expected_size"],
            "sha256": spec["expected_sha256"],
        }
        if entry != expected_entry:
            print(f"[FAIL] {key}: 冻结记录与代码内固定身份不一致")
            failures += 1
            continue
        path = MODEL_DIR / entry["filename"]
        if not path.is_file():
            print(f"[FAIL] {key}: 文件缺失 {path}")
            failures += 1
            continue
        if path.stat().st_size != int(entry["size_bytes"]):
            print(f"[FAIL] {key}: 大小不一致 {path.stat().st_size} != {entry['size_bytes']}")
            failures += 1
            continue
        sha = _sha256(path)
        if sha != entry["sha256"]:
            print(f"[FAIL] {key}: SHA-256 不一致")
            failures += 1
            continue
        print(f"[OK]   {key}: {path.name} 大小/SHA-256 与冻结记录一致")
    print(f"[check] {'全部通过' if not failures else f'{failures} 项失败'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash", action="store_true", help="全量 SHA-256 并写入冻结记录")
    parser.add_argument("--check", action="store_true", help="只读核对冻结记录")
    args = parser.parse_args(argv)
    if args.hash:
        _freeze()
        return 0
    return _check()


if __name__ == "__main__":
    raise SystemExit(main())
