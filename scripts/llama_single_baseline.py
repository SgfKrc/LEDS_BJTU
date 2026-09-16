"""Run a path-free single-host llama.cpp/GGUF baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _model_profile(model_id: str, model_path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from model_config import (
            get_builtin_models,
            get_model_profile_metadata,
            resolve_model_path,
        )

        selected_id = model_id
        if not selected_id:
            absolute_path = str(model_path.resolve())
            for candidate in get_builtin_models():
                if candidate.gguf_path and str(
                    Path(resolve_model_path(candidate.gguf_path)).resolve()
                ) == absolute_path:
                    selected_id = candidate.model_id
                    break
        return selected_id, get_model_profile_metadata(selected_id) if selected_id else {}
    except Exception:
        return model_id, {}


def run_baseline(
    model_path: str,
    *,
    model_id: str = "",
    n_ctx: int = 1024,
    max_tokens: int = 16,
    prompt: str = "Reply with one short sentence: 2 + 2 = ?",
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"GGUF file does not exist: {path}")
    if not 128 <= n_ctx <= 131072:
        raise ValueError("n_ctx must be between 128 and 131072")
    if not 1 <= max_tokens <= 256:
        raise ValueError("max_tokens must be between 1 and 256")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    selected_id, profile = _model_profile(model_id, path)
    from llama_engine import LlamaCppEngine

    report: dict[str, Any] = {
        "schema": "qlh.llama_single_baseline.v1",
        "model_id": selected_id or "unregistered",
        "engine": "llama.cpp",
        "model": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
        "profile": {
            key: profile.get(key)
            for key in ("revision", "template", "thinking", "vision")
            if key in profile
        },
    }
    engine = LlamaCppEngine()
    load_started = time.perf_counter()
    rss_before = _rss_bytes()
    try:
        engine.load_model(
            model_path=str(path),
            n_ctx=n_ctx,
            model_profile=profile,
        )
        load_elapsed = time.perf_counter() - load_started
        rss_after_load = _rss_bytes()
        report["load"] = {
            "elapsed_seconds": round(load_elapsed, 3),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after_load,
            "rss_delta_bytes": rss_after_load - rss_before,
        }
        report["capabilities"] = engine.get_capabilities()

        messages = [{"role": "user", "content": prompt}]
        chat_started = time.perf_counter()
        chat_result = engine.chat(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            show_thinking=False,
        )
        report["chat"] = {
            "elapsed_seconds": round(time.perf_counter() - chat_started, 3),
            "content_chars": len(str(chat_result.get("content") or "")),
            "completion_tokens": int(
                (chat_result.get("usage") or {}).get("completion_tokens", 0)
            ),
            "finish_reason": chat_result.get("finish_reason"),
            "tokens_per_second": chat_result.get("tokens_per_second", 0),
            "nonempty": bool(str(chat_result.get("content") or "").strip()),
        }

        stream_started = time.perf_counter()
        stream_chunks = list(
            engine.chat_stream(
                messages,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                show_thinking=False,
            )
        )
        report["stream"] = {
            "elapsed_seconds": round(time.perf_counter() - stream_started, 3),
            "chunks": len(stream_chunks),
            "content_chars": len("".join(stream_chunks)),
            "nonempty": bool("".join(stream_chunks).strip()),
        }
        report["rss_after_inference_bytes"] = _rss_bytes()
        report["ok"] = bool(
            report["chat"]["nonempty"] and report["stream"]["nonempty"]
        )
    finally:
        engine.unload()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="local GGUF file")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Reply with one short sentence: 2 + 2 = ?")
    args = parser.parse_args(argv)
    print(json.dumps(run_baseline(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
