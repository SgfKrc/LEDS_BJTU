#!/usr/bin/env python3
"""EX-N3 真实 Gemma 判题 runner：SD 图像 → Ollama gemma4:12b 图生文 → 计数证据。

设计（对齐 fixtures/quality_rubrics/gemma-judge-counts-v1.json 契约）：
- 每张图像经 Ollama OpenAI 兼容 /v1/chat/completions 得到描述；
- 匹配为可复现的归一化包含判定（不依赖二次 LLM 判断）：
  - key_element_coverage：描述包含该要素（小写、去空白/标点后子串匹配）即计 1；
  - topic_hit：该图要素命中数 ≥ ceil(要素数/2) 计 1；
- 判题失败（Ollama 错误/超时/空输出）fail-closed：计入 evaluated_count 且
  passed=0，报告只记录错误类型（脱敏）；
- 证据 JSON 严格遵循契约持久化白名单：仅 model / judge_contract_id /
  judge_contract_sha256 / topic_hit 与 key_element_coverage 的计数，
  不保存 prompt、图像、路径、描述或判题解释。

用法:
    python scripts/experiment_gemma_judge_real.py \
        --prompts prompts.json --result-file build/exp/gemma-evidence.json --json
"""

import argparse
import base64
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_MODEL = "gemma4:12b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_JUDGE_PROMPT = "Describe this image in one or two sentences."
DEFAULT_CONTRACT = PROJECT_ROOT / "fixtures" / "quality_rubrics" / "gemma-judge-counts-v1.json"
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 与 G4.2 图像限制一致


def _normalize(text: str) -> str:
    """归一化用于包含匹配：小写、去空白与标点。"""
    return re.sub(r"[\s\W_]+", "", text.lower())


def _image_data_url(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes: {path.name}")
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def _describe_image(
    client: httpx.Client,
    ollama_url: str,
    model: str,
    judge_prompt: str,
    image_data_url: str,
    *,
    max_tokens: int,
    timeout: float,
    reasoning_effort: str | None = "none",
) -> str:
    from multimodal import build_openai_user_content

    content = build_openai_user_content(judge_prompt, [image_data_url])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_tokens": max_tokens,
        # keep_alive 保持模型驻留；reasoning_effort 仅 gemma4 系列需要
        # （gemma4 默认开启 thinking 会占满短输出预算导致 content 为空），
        # qwen3-vl 等模型对该字段会返回空输出（QWVL-J1 实测），须传 ""。
        "keep_alive": "30m",
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    response = client.post(
        f"{ollama_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    text = (choices[0].get("message") or {}).get("content") or ""
    text = text.strip()
    if not text:
        raise ValueError("empty completion")
    return text


def _match_counts(description: str, key_elements: list[str]) -> tuple[int, int, int]:
    """返回 (要素命中数, 要素总数, 主题是否命中)。"""
    normalized = _normalize(description)
    hits = sum(1 for element in key_elements if _normalize(element) in normalized)
    threshold = math.ceil(len(key_elements) / 2) if key_elements else 1
    topic_hit = 1 if hits >= threshold else 0
    return hits, len(key_elements), topic_hit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(contract.get("persistence", {}).get("allowed", []))
    if not allowed:
        raise SystemExit("[error] judge contract 缺少 persistence.allowed")
    return contract, allowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EX-N3 真实 Gemma 判题 runner")
    parser.add_argument("--prompts", required=True, help="[{image, prompt, key_elements: []}]")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT)
    parser.add_argument("--judge-contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help='Ollama reasoning_effort 值；"" 表示不发送该字段（qwen3-vl 等模型'
        "对该字段会返回空输出，实测 QWVL-J1）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="单图判题失败重试次数（qwen3-vl:4b 在 Ollama 下有偶发空输出，"
        "QWVL-J1 实测；重试后仍失败才计 failures）",
    )
    parser.add_argument("--result-file", required=True, help="证据 JSON（白名单字段）")
    parser.add_argument("--report-file", help="脱敏报告 JSON（含失败统计）")
    parser.add_argument("--json", action="store_true", help="stdout 输出汇总 JSON")
    args = parser.parse_args(argv)

    contract, allowed = _load_contract(Path(args.judge_contract))
    contract_sha = _sha256(Path(args.judge_contract))

    items = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise SystemExit("[error] --prompts 必须是非空数组")

    topic_evaluated = topic_passed = 0
    coverage_evaluated = coverage_passed = 0
    failures: list[dict] = []
    started = time.time()

    with httpx.Client() as client:
        for item in items:
            image_path = Path(item["image"]).expanduser().resolve()
            if not image_path.is_file():
                failures.append({"image": image_path.name, "error": "missing_file"})
                topic_evaluated += 1
                coverage_evaluated += 1
                continue
            prompt = str(item.get("prompt") or "")
            key_elements = [str(e) for e in (item.get("key_elements") or [])]
            try:
                description = None
                first_exc: Exception | None = None
                last_exc: Exception | None = None
                for _ in range(max(1, args.retries + 1)):
                    try:
                        description = _describe_image(
                            client,
                            args.ollama_url,
                            args.model,
                            args.judge_prompt,
                            _image_data_url(image_path),
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            reasoning_effort=args.reasoning_effort,
                        )
                        break
                    except Exception as exc:  # 偶发空输出/超时：重试
                        if first_exc is None:
                            first_exc = exc
                        last_exc = exc
                if description is None:
                    # Preserve the original semantic failure.  A later retry
                    # may fail for a secondary reason (for example a mock or
                    # provider response queue being exhausted), which must
                    # not overwrite the reason that triggered retry.
                    raise first_exc or last_exc or ValueError("no description")
                hits, total, topic_hit = _match_counts(description, key_elements)
                topic_evaluated += 1
                topic_passed += topic_hit
                coverage_evaluated += total
                coverage_passed += hits
            except Exception as exc:  # fail-closed：无法判题视为未通过
                failures.append({"image": image_path.name, "error": type(exc).__name__})
                topic_evaluated += 1
                coverage_evaluated += max(1, len(key_elements))

    # 证据严格白名单（契约 persistence.allowed）
    evidence = {
        "model": args.model,
        "judge_contract_id": contract.get("id"),
        "judge_contract_sha256": contract_sha,
        "topic_hit": {
            "evaluated_count": topic_evaluated,
            "passed_count": topic_passed,
        },
        "key_element_coverage": {
            "evaluated_count": coverage_evaluated,
            "passed_count": coverage_passed,
        },
    }
    forbidden = [
        key for key in evidence if key not in allowed
        and key != "topic_hit" and key != "key_element_coverage"
    ]
    if forbidden:
        raise SystemExit(f"[error] 证据字段超出契约白名单: {forbidden}")

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    summary = {
        "model": args.model,
        "judge_contract_id": contract.get("id"),
        "topic_hit_rate": round(topic_passed / topic_evaluated, 4) if topic_evaluated else None,
        "key_element_coverage_rate": (
            round(coverage_passed / coverage_evaluated, 4) if coverage_evaluated else None
        ),
        "items": len(items),
        "failures": len(failures),
        "elapsed_seconds": round(time.time() - started, 2),
        "evidence": str(result_path),
    }
    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {"summary": summary, "failures": failures}, ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
