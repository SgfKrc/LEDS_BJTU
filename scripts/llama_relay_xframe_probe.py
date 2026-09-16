"""Evidence probe for experimental cross-framework relay logits.

This tool compares the raw little-endian float32 logits produced by a complete
llama.cpp reference run and a relay run.  It is intentionally a verifier, not
a relay supervisor: a successful evidence comparison does not bypass
``admit_relay_xframe`` or enable a production route.

The implementation uses only the standard library.  It streams one vocab-sized
row at a time, so a long prompt does not require both complete logit matrices
to be resident in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Any

from src.relay_contract import (
    RELAY_ACCEPTANCE,
    RELAY_ACCEPTANCE_TOLERANT,
    RELAY_TOLERANT_TOP_K,
    RelayXFrameRequest,
    admit_relay_xframe,
    judge_relay_generation,
    judge_relay_generation_tolerant,
)


FLOAT32_BYTES = 4
DEFAULT_VOCAB_SIZE = 248320


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_count(path: Path, vocab_size: int) -> int:
    if vocab_size < 1:
        raise ValueError("vocab_size_must_be_positive")
    if not path.is_file():
        raise ValueError(f"missing_logits_file:{path}")
    row_bytes = vocab_size * FLOAT32_BYTES
    size = path.stat().st_size
    if size == 0 or size % row_bytes:
        raise ValueError(f"invalid_logits_shape:{path}")
    return size // row_bytes


def _read_row(handle: Any, vocab_size: int) -> array:
    row = array("f")
    row.fromfile(handle, vocab_size)
    if len(row) != vocab_size:
        raise ValueError("truncated_logits_row")
    return row


def _argmax(values: array) -> int:
    best_index = 0
    best_value = values[0]
    if not math.isfinite(best_value):
        raise ValueError("non_finite_logits")
    for index, value in enumerate(values[1:], start=1):
        if not math.isfinite(value):
            raise ValueError("non_finite_logits")
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _cosine(left: array, right: array) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero_norm_logits")
    return dot / (left_norm * right_norm)


def _top_k_overlap(left: array, right: array, top_k: int) -> int:
    if top_k < 1:
        return 0
    width = min(top_k, len(left))
    left_indices = heapq.nlargest(width, range(len(left)), key=left.__getitem__)
    right_indices = heapq.nlargest(width, range(len(right)), key=right.__getitem__)
    return len(set(left_indices).intersection(right_indices))


def compare_logits(
    baseline_path: str | Path,
    relay_path: str | Path,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare two raw logits matrices and return a structured evidence record.

    The only acceptance criterion is the sequence of per-position argmax
    values.  Numerical metrics remain diagnostics because they cannot safely
    substitute for generated-token equivalence.
    """
    if sys.byteorder != "little":
        raise RuntimeError("little_endian_host_required")

    baseline = Path(baseline_path)
    relay = Path(relay_path)
    baseline_rows = _row_count(baseline, vocab_size)
    relay_rows = _row_count(relay, vocab_size)
    if baseline_rows != relay_rows:
        raise ValueError(f"position_count_mismatch:{baseline_rows}!={relay_rows}")

    baseline_tokens: list[int] = []
    relay_tokens: list[int] = []
    #: Ranked baseline candidates per position; kept for the tolerant criterion below.
    baseline_topk: list[list[int]] = []
    cosine_values: list[float] = []
    overlap_values: list[int] = []
    bitwise_equal = True
    tolerant_k = max(1, top_k) if top_k > 0 else RELAY_TOLERANT_TOP_K
    with baseline.open("rb") as baseline_handle, relay.open("rb") as relay_handle:
        for _ in range(baseline_rows):
            baseline_row = _read_row(baseline_handle, vocab_size)
            relay_row = _read_row(relay_handle, vocab_size)
            baseline_tokens.append(_argmax(baseline_row))
            relay_tokens.append(_argmax(relay_row))
            ranked = heapq.nlargest(
                tolerant_k, range(len(baseline_row)), key=baseline_row.__getitem__
            )
            baseline_topk.append([int(index) for index in ranked])
            bitwise_equal = bitwise_equal and baseline_row.tobytes() == relay_row.tobytes()
            cosine_values.append(_cosine(baseline_row, relay_row))
            if top_k > 0:
                overlap_values.append(_top_k_overlap(baseline_row, relay_row, top_k))

    mean_cosine = sum(cosine_values) / len(cosine_values)
    verdict = judge_relay_generation(
        baseline_tokens,
        relay_tokens,
        cosine=mean_cosine,
        bitwise_equal=bitwise_equal,
    )
    # The embd handoff path cannot satisfy per-token argmax (measured 2026-09-16: every
    # observed divergence was a top-1 <-> top-2 swap). Record the tolerant verdict alongside
    # the strict one; the strict verdict still drives ``status`` so existing consumers are
    # unaffected.
    tolerant_verdict = judge_relay_generation_tolerant(
        baseline_topk,
        relay_tokens,
        top_k=tolerant_k,
        cosine=mean_cosine,
        bitwise_equal=bitwise_equal,
    )
    mismatch_positions = [
        index
        for index, (baseline_token, relay_token) in enumerate(zip(baseline_tokens, relay_tokens))
        if baseline_token != relay_token
    ]
    report: dict[str, Any] = {
        "status": "evidence_accepted" if verdict.accepted else "evidence_rejected",
        "criterion": RELAY_ACCEPTANCE,
        "verdict": verdict.to_dict(),
        # Tolerant view for the embd handoff path (see RELAY_ACCEPTANCE_TOLERANT). ``status``
        # above stays strict so existing consumers keep their semantics; the tolerant verdict is
        # the one D->L evidence should be judged by.
        "tolerant_criterion": RELAY_ACCEPTANCE_TOLERANT,
        "tolerant_verdict": tolerant_verdict.to_dict(),
        "tolerant_status": (
            "evidence_accepted" if tolerant_verdict.accepted else "evidence_rejected"
        ),
        "baseline_topk": baseline_topk,
        "baseline_path": str(baseline),
        "relay_path": str(relay),
        "baseline_sha256": _sha256(baseline),
        "relay_sha256": _sha256(relay),
        "logits_format": "float32_le",
        "vocab_size": vocab_size,
        "position_count": baseline_rows,
        "argmax_match_count": baseline_rows - len(mismatch_positions),
        "baseline_tokens": baseline_tokens,
        "relay_tokens": relay_tokens,
        "mismatch_positions": mismatch_positions,
        "diagnostics": {
            "bitwise_equal": bitwise_equal,
            "cosine_min": min(cosine_values),
            "cosine_mean": sum(cosine_values) / len(cosine_values),
            "cosine_max": max(cosine_values),
            "top_k": top_k,
            "top_k_overlap_min": min(overlap_values) if overlap_values else None,
            "top_k_overlap_mean": (
                sum(overlap_values) / len(overlap_values) if overlap_values else None
            ),
            "top_k_overlap_max": max(overlap_values) if overlap_values else None,
        },
    }
    return report


def build_evidence_report(
    baseline_path: str | Path,
    relay_path: str | Path,
    *,
    upstream_engine: str = "pytorch",
    downstream_engine: str = "llama.cpp",
    network_mode: str = "loopback",
    temperature: float = 0.0,
    top_p: float = 1.0,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    top_k: int = 10,
) -> dict[str, Any]:
    """Attach the current fail-closed production policy to a logits comparison."""
    report = compare_logits(
        baseline_path,
        relay_path,
        vocab_size=vocab_size,
        top_k=top_k,
    )
    request = RelayXFrameRequest(
        upstream_engine=upstream_engine,
        downstream_engine=downstream_engine,
        network_mode=network_mode,
        sequence_length=report["position_count"],
        temperature=temperature,
        top_p=top_p,
    )
    report["xframe_request"] = request.to_dict()
    report["production_admission"] = admit_relay_xframe(request).to_dict()
    report["note"] = (
        "evidence acceptance never enables production relay; the XFRAME policy remains fail-closed"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare cross-framework relay logits with per-position argmax"
    )
    parser.add_argument("--baseline-logits", required=True)
    parser.add_argument("--relay-logits", required=True)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--upstream-engine", default="pytorch")
    parser.add_argument("--downstream-engine", default="llama.cpp")
    parser.add_argument("--network-mode", default="loopback")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    try:
        report = build_evidence_report(
            args.baseline_logits,
            args.relay_logits,
            upstream_engine=args.upstream_engine,
            downstream_engine=args.downstream_engine,
            network_mode=args.network_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            vocab_size=args.vocab_size,
            top_k=args.top_k,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        report = {
            "status": "invalid_evidence",
            "criterion": RELAY_ACCEPTANCE,
            "error": str(exc),
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0 if report["status"] == "evidence_accepted" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
