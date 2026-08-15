#!/usr/bin/env python3
"""P8 (2026-08-16): 模型导入向导——`model_tools import-model`。

引导式完成 resolve → 下载 → 校验 → 登记 四步：

1. resolve：输入 Hugging Face repo id（或本地已存在路径），解析目标目录
2. download：经 huggingface_hub snapshot_download（继承 QLH_HTTP_PROXY
   代理），失败时可用 ModelScope 镜像路径重试
3. verify：统计下载目录内 safetensors/gguf 的字节数与 SHA-256 摘要；
   提供 --expected-sha256 时严格校验（不匹配 fail-closed）
4. register：登记到主节点 SQLite model_registry（实验模型），后续
   /api/models 可见

安全：不落凭据；交互输入不回显到报告；登记 payload 不含绝对路径之外的
敏感信息（路径为用户指定目录）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from local_store import (
    delete_local_experimental_model,
    save_local_experimental_model,
)

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


def resolve_target(repo_or_path: str, target: str | None, models_root: str = "models") -> Path:
    """Step 1：把 repo id / 路径解析为目标模型目录。"""
    source = Path(repo_or_path)
    if source.is_dir():
        return source.absolute()
    name = repo_or_path.strip("/").split("/")[-1]
    if not name:
        raise ValueError(f"无法从输入解析模型名: {repo_or_path!r}")
    return Path(target or os.path.join(models_root, name)).absolute()


def download_model(repo_or_path: str, target: Path, *, use_modelscope: bool = False) -> list[Path]:
    """Step 2：下载（HF snapshot_download 或 ModelScope），返回权重文件列表。"""
    source = Path(repo_or_path)
    if source.is_dir():
        return _weight_files(target)

    if use_modelscope:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import modelscope, sys; from modelscope import snapshot_download; "
             f"snapshot_download({repo_or_path!r}, local_dir={str(target)!r})"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ModelScope 下载失败: {(result.stderr or result.stdout)[-200:]}")
    else:
        import huggingface_hub
        huggingface_hub.snapshot_download(
            repo_id=repo_or_path,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
    files = _weight_files(target)
    if not files:
        raise RuntimeError("下载完成但未找到权重文件（safetensors/gguf/bin）")
    return files


def _weight_files(target: Path) -> list[Path]:
    return sorted(
        path for path in target.rglob("*")
        if path.is_file() and path.name.lower().endswith(_WEIGHT_SUFFIXES)
    )


def verify_files(files: list[Path], expected_sha256: str | None = None) -> dict[str, Any]:
    """Step 3：字节数 + SHA-256 摘要（可选严格校验）。"""
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    summary = {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }
    if expected_sha256 and digest.hexdigest() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 不匹配: 期望 {expected_sha256.lower()}，实际 {digest.hexdigest()}"
        )
    return summary


def register_model(
    model_id: str,
    target: Path,
    summary: dict[str, Any],
    *,
    gguf_path: str = "",
) -> bool:
    """Step 4：登记到主节点 SQLite model_registry（实验模型）。"""
    config = {
        "model_id": model_id,
        "name": model_id,
        "model_path": str(target) if not gguf_path else "",
        "gguf_path": gguf_path or (str(target / "model.gguf") if (target / "model.gguf").is_file() else ""),
        "quantization": "Q4_K_M" if gguf_path else "fp16",
        "sha256": summary["sha256"],
        "source": "import-model",
    }
    return save_local_experimental_model(model_id, config)


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(description="模型导入向导（resolve→下载→校验→登记）")
    parser.add_argument("source", help="Hugging Face repo id、ModelScope 路径或本地目录")
    parser.add_argument("--target", default="", help="目标模型目录（默认 models/<repo 名>）")
    parser.add_argument("--model-id", default="", help="登记用的 model_id（默认取 repo 名）")
    parser.add_argument("--expected-sha256", default="", help="严格校验期望 SHA-256")
    parser.add_argument("--gguf-path", default="", help="GGUF 登记路径（覆盖自动探测）")
    parser.add_argument("--modelscope", action="store_true", help="使用 ModelScope 下载")
    parser.add_argument("--skip-download", action="store_true", help="仅校验/登记本地目录")
    parser.add_argument("--register", action="store_true", help="完成后登记到主节点 SQLite")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    args = parser.parse_args(argv) if not isinstance(argv, argparse.Namespace) else argv

    try:
        target = resolve_target(args.source, args.target)
        if not args.skip_download and not Path(args.source).is_dir():
            print(f"[1/4] 下载 {args.source} → {target} ...", flush=True)
            files = download_model(args.source, target, use_modelscope=args.modelscope)
        else:
            files = _weight_files(target)
            if not files and not Path(args.source).is_dir():
                raise RuntimeError("目标目录无权重文件")

        print(f"[2/4] 校验 {len(files)} 个权重文件 ...", flush=True)
        summary = verify_files(files, args.expected_sha256 or None)

        model_id = args.model_id or target.name
        print(f"[3/4] 登记准备: model_id={model_id}")

        registered = False
        if args.register:
            registered = register_model(model_id, target, summary, gguf_path=args.gguf_path)
            print(f"[4/4] 登记{'成功' if registered else '失败'}（主节点 SQLite model_registry）")
        else:
            print("[4/4] 未登记（加 --register 写入主节点 SQLite）")

        result = {
            "model_id": model_id,
            "target": str(target),
            "file_count": summary["file_count"],
            "total_bytes": summary["total_bytes"],
            "sha256": summary["sha256"],
            "registered": registered,
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=1))
        else:
            print(f"完成: {result['file_count']} 文件 / {result['total_bytes']} bytes")
            print(f"  SHA-256: {result['sha256']}")
        return 0
    except Exception as exc:
        print(f"[error] 导入失败: {str(exc)[:200]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
