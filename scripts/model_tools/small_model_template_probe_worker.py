"""Isolated, tokenizer-only worker for the small-model template probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 256 * 1024
THINKING_MARKERS = ("<think>", "</think>", "<|think|>")


def _base() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": "template_probe",
        "valid": False,
        "read_only": True,
        "network_used": False,
        "weights_loaded": False,
        "isolated": True,
        "runtime": {"transformers_version": None},
        "thinking": {"switch_declared": False, "runtime_status": "unknown"},
        "rendering": {},
        "errors": [],
    }


def _has_nonempty_thinking(value: str) -> bool:
    lower = value.lower()
    start = lower.find("<think>")
    end = lower.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return bool(value[start + len("<think>"):end].strip())
    return any(marker in lower for marker in ("<|think|>", "</think>"))


def _render(tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool | None, tokenize: bool) -> tuple[str, list[int]]:
    kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": True}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    value = tokenizer.apply_chat_template(messages, **kwargs)
    if tokenize:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            value = value.get("input_ids", [])
        ids = [int(item) for item in value] if isinstance(value, (list, tuple)) else []
        return "", ids
    return str(value), []


def _render_report(tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool | None) -> dict[str, Any]:
    text, _ = _render(tokenizer, messages, thinking=thinking, tokenize=False)
    _, ids = _render(tokenizer, messages, thinking=thinking, tokenize=True)
    try:
        first_text = tokenizer.decode(ids[:100], skip_special_tokens=False)
    except Exception:
        first_text = ""
    return {
        "status": "passed",
        "length": len(text),
        "token_count": len(ids),
        "thinking_markers": any(marker in text.lower() for marker in THINKING_MARKERS),
        "nonempty_thinking": _has_nonempty_thinking(text),
        "first_100_token_ids": ids[:100],
        "first_100_token_text": first_text[:4096],
        "first_100_tokens": {
            "status": "prompt_render_only",
            "generation_status": "not_run",
            "thinking_markers": any(marker in first_text.lower() for marker in THINKING_MARKERS),
            "nonempty_thinking": _has_nonempty_thinking(first_text),
        },
    }


def execute(request: dict[str, Any]) -> dict[str, Any]:
    result = _base()
    model_path = Path(str(request.get("model_path", ""))).expanduser().absolute().resolve(strict=False)
    if not model_path.is_dir():
        result["errors"].append({"code": "model_path_invalid", "message": "model directory is missing"})
        return result
    controller = Path(str(request.get("controller_python", ""))).absolute().resolve(strict=False)
    result["isolated"] = controller != Path(sys.executable).absolute().resolve(strict=False)
    if not result["isolated"]:
        result["errors"].append({"code": "runtime_not_isolated", "message": "template worker must use an isolated Python environment"})
        return result
    try:
        import transformers

        result["runtime"]["transformers_version"] = str(getattr(transformers, "__version__", "unknown"))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=bool(request.get("trust_remote_code", False)),
        )
        template = str(getattr(tokenizer, "chat_template", "") or "")
        if not template:
            result["errors"].append({"code": "chat_template_missing", "message": "tokenizer chat template is missing"})
            return result
        messages = [{"role": "user", "content": "Reply with the word OK."}]
        result["thinking"] = {
            "switch_declared": "enable_thinking" in template.lower(),
            "runtime_status": "not_declared",
        }
        result["rendering"]["default"] = _render_report(tokenizer, messages, thinking=None)
        if result["thinking"]["switch_declared"]:
            result["rendering"]["thinking_disabled"] = _render_report(tokenizer, messages, thinking=False)
            result["rendering"]["thinking_enabled"] = _render_report(tokenizer, messages, thinking=True)
            result["thinking"]["runtime_status"] = "supported"
            if result["rendering"]["thinking_disabled"]["nonempty_thinking"]:
                result["errors"].append({"code": "thinking_disable_rendered_content", "message": "enable_thinking=false produced non-empty thinking content"})
        else:
            try:
                _render(tokenizer, messages, thinking=False, tokenize=False)
            except TypeError:
                result["thinking"]["runtime_status"] = "unsupported"
            else:
                result["thinking"]["runtime_status"] = "accepted_but_undeclared"
        result["valid"] = not result["errors"]
        return result
    except Exception as exc:
        result["errors"].append({"code": "tokenizer_probe_failed", "message": exc.__class__.__name__})
        return result


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("template probe request exceeds protocol limit")
    request = json.loads(raw.decode("utf-8"))
    if request.get("schema_version") != SCHEMA_VERSION or request.get("operation") != "template_probe":
        raise ValueError("unsupported template probe protocol")
    print(json.dumps(execute(request), ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fallback = _base()
        fallback["errors"] = [{"code": "invalid_request", "message": exc.__class__.__name__}]
        print(json.dumps(fallback, ensure_ascii=True, separators=(",", ":")))
        raise SystemExit(2)
