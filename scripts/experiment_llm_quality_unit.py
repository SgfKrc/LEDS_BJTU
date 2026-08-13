#!/usr/bin/env python3
"""EX-N3 local GGUF objective-quality unit.

This runner loads one explicitly named local GGUF artifact, performs frozen
greedy completions for the plan-pinned rubric subset, and writes only derived
quality counters.  Prompts and completions are never written to the result
file, experiment record, report, or stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core.objective_rubric import (
    RubricError,
    load_objective_rubric,
    score_objective_outputs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prompts(path: Path, ids: set[str]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt_id = item.get("id")
            prompt = item.get("prompt")
            if prompt_id in ids and isinstance(prompt, str):
                prompts[prompt_id] = prompt
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("cannot read objective prompts") from exc
    if set(prompts) != ids:
        raise ValueError("objective prompt set is incomplete")
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one pinned EX-N3 local GGUF quality unit")
    parser.add_argument("--model", required=True, help="local GGUF artifact")
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--prompt-set-dir", required=True)
    parser.add_argument("--prompt-set-sha256", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--expected-rubric-sha256", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--prompt-mode",
        choices=["chatml", "raw"],
        default="chatml",
        help="chatml: legacy chat-format wrapper; raw: feed pre-rendered prompts "
        "verbatim (Qwen3 non-thinking renders produced by the isolated sidecar)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-p", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    prompt_set_dir = Path(args.prompt_set_dir).expanduser().resolve()
    rubric_path = Path(args.rubric).expanduser().resolve()
    result_path = Path(args.result_file).expanduser()

    try:
        if not model_path.is_file():
            raise ValueError("local GGUF model does not exist")
        if _sha256(model_path) != args.expected_model_sha256:
            raise ValueError("local GGUF SHA-256 mismatch")
        prompts_path = prompt_set_dir / "prompts.jsonl"
        if _sha256(prompts_path) != args.prompt_set_sha256:
            raise ValueError("prompt set SHA-256 mismatch")
        rubric = load_objective_rubric(
            rubric_path,
            expected_sha256=args.expected_rubric_sha256,
            expected_prompt_set={
                "id": prompt_set_dir.name,
                "sha256": args.prompt_set_sha256,
            },
        )
        prompt_ids = {str(entry["prompt_id"]) for entry in rubric.entries}
        prompts = _load_prompts(prompts_path, prompt_ids)
    except (OSError, ValueError, RubricError) as exc:
        print(f"EX-N3 input validation failed: {exc}", file=sys.stderr)
        return 2

    try:
        from llama_cpp import Llama
    except ImportError:
        print("EX-N3 requires llama-cpp-python for the local GGUF runner", file=sys.stderr)
        return 2

    outputs: dict[str, str | None] = {}
    truncated: set[str] = set()
    started = time.monotonic()
    try:
        load_kwargs = {
            "model_path": str(model_path),
            "n_ctx": args.n_ctx,
            "n_gpu_layers": args.n_gpu_layers,
            "seed": args.seed,
            "verbose": False,
        }
        if args.prompt_mode == "chatml":
            load_kwargs["chat_format"] = "chatml"
        if args.n_threads > 0:
            load_kwargs["n_threads"] = args.n_threads
        model = Llama(**load_kwargs)
        sample_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        }
        for entry in rubric.entries:
            prompt_id = str(entry["prompt_id"])
            if args.prompt_mode == "raw":
                # Pre-rendered (e.g. Qwen3 enable_thinking=False) prompt: feed verbatim.
                response = model.create_completion(
                    prompts[prompt_id],
                    max_tokens=args.max_new_tokens,
                    stream=False,
                    **sample_kwargs,
                )
                choices = response.get("choices", []) if isinstance(response, dict) else []
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                content = choice.get("text")
            else:
                response = model.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Follow the requested output format exactly. Do not add explanations unless requested.",
                        },
                        {"role": "user", "content": prompts[prompt_id]},
                    ],
                    max_tokens=args.max_new_tokens,
                    stream=False,
                    **sample_kwargs,
                )
                choices = response.get("choices", []) if isinstance(response, dict) else []
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                message = choice.get("message") if isinstance(choice, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
            outputs[prompt_id] = content if isinstance(content, str) else None
            if choice.get("finish_reason") == "length":
                truncated.add(prompt_id)
    except Exception as exc:
        # Do not preserve backend exception text in a structured experiment result.
        print(f"EX-N3 local generation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        model = None

    counters = score_objective_outputs(rubric, outputs, truncated_prompt_ids=truncated)
    result = {
        "quality_completed": 1,
        "quality_duration_s": round(time.monotonic() - started, 3),
        "quality_invalid_count": (
            counters["correctness"]["invalid_count"] + counters["format"]["invalid_count"]
        ),
        "quality_evidence": {
            "llm": {
                "prompt_set_id": rubric.prompt_set["id"],
                "prompt_set_sha256": rubric.prompt_set["sha256"],
                "correctness": {
                    "evaluated_count": counters["correctness"]["evaluated_count"],
                    "passed_count": counters["correctness"]["passed_count"],
                },
                "format": {
                    "evaluated_count": counters["format"]["evaluated_count"],
                    "passed_count": counters["format"]["passed_count"],
                },
            }
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "quality_completed": 1,
        "correctness_evaluated": counters["correctness"]["evaluated_count"],
        "format_evaluated": counters["format"]["evaluated_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
