"""导出 Python 内置模型注册表（BUILTIN_MODELS）为 catalog seed JSON。

用途：M1 旧源迁移（migration-map.json 的 python_builtin_models 源）。
control-svc 无法读取 Python 进程内存，由本脚本生成静态 seed 文件，
迁移执行器（control/src/data/legacy-migration.ts）读取并灌入 SQLite。

字段映射对齐 schemas/migration-map.json 的 field_map：
  model_id → model_id
  name → name
  model_type → format
  model_path → local_path
  gguf_path → gguf_path
  recommended_vram_gb → min_vram_bytes_approx_gb
  max_context → context_length
  huggingface_id → source.repo_id
  quant_types → quantizations
  is_experimental → is_experimental
  location → origin

用法:
    python scripts/export_model_catalog.py [输出路径]
    默认输出: build/model-fleet/catalog-seed.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_config import BUILTIN_MODELS  # noqa: E402

DEFAULT_OUTPUT = ROOT / "build" / "model-fleet" / "catalog-seed.json"


def export_catalog() -> list[dict]:
    rows = []
    for model in BUILTIN_MODELS:
        rows.append({
            "model_id": model.model_id,
            "name": model.name,
            "format": model.model_type,
            "local_path": model.model_path,
            "gguf_path": model.gguf_path,
            "min_vram_bytes_approx_gb": model.recommended_vram_gb,
            "context_length": model.max_context,
            "source": {"repo_id": model.huggingface_id},
            "quantizations": list(model.quant_types),
            "is_experimental": bool(model.is_experimental),
            "origin": model.location,
        })
    return rows


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    rows = export_catalog()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[catalog] 已导出 {len(rows)} 个内置模型 -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
