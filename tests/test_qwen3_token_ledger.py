"""MM1.26 isolated token-ledger and tokenizer decode regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.model_tools.qwen3_token_ledger import (  # noqa: E402
    run_qwen3_token_ledger_decode,
)
from scripts.model_tools.qwen3_token_ledger_worker import (  # noqa: E402
    execute_request,
)


def _digest(value: dict) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger(tmp_path: Path, *, text: str = "hello"):
    model = tmp_path / "tokenizer-model"
    model.mkdir()
    unsigned = {
        "schema_version": 1,
        "ledger_kind": "qwen3_mm1_generated_token_ledger",
        "ledger_id": "mm1ledger_fixture",
        "text_chain_id": "c" * 64,
        "contract_sha256": "d" * 64,
        "prefill_generation": 22,
        "token_count": 2,
        "stop_reason": "eos",
        "records": [
            {
                "step_index": 1,
                "generation": 23,
                "sequence_length": 69,
                "token_id": 5,
                "artifact_sha256": "a" * 64,
            },
            {
                "step_index": 2,
                "generation": 24,
                "sequence_length": 70,
                "token_id": 2,
                "artifact_sha256": "b" * 64,
            },
        ],
        "full_model_materialized": False,
    }
    ledger = {**unsigned, "ledger_sha256": _digest(unsigned)}
    path = tmp_path / "ledger.json"
    encoded = json.dumps(
        ledger, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(encoded)
    metadata = {
        "ledger_id": ledger["ledger_id"],
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "status": "committed",
        "content_kind": "generated_token_ledger",
        "token_count": 2,
        "stop_reason": "eos",
    }
    request = {
        "schema_version": 1,
        "tool": "qwen3_token_ledger",
        "operation": "qwen3_token_ledger_decode",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model),
        "ledger_path": str(path),
        "ledger_metadata": metadata,
        "text_max_bytes": 64 * 1024,
        "expected_chain_id": "c" * 64,
        "expected_generation": 22,
        "expected_first_sequence": 69,
        "controller_python": str(ROOT / "controller-python"),
    }
    return request, path, metadata


class _FakeTokenizer:
    def __init__(self, value: str = "hello") -> None:
        self.value = value
        self.calls: list[tuple[list[int], bool]] = []

    def decode(self, ids, *, skip_special_tokens):
        self.calls.append((list(ids), bool(skip_special_tokens)))
        return self.value


def _module(tokenizer: _FakeTokenizer):
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert Path(path).is_dir()
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return tokenizer

    return SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)


def test_mm126_worker_decodes_local_ledger_without_exposing_token_ids_or_paths(tmp_path):
    request, _path, _metadata = _ledger(tmp_path)
    tokenizer = _FakeTokenizer("hello")

    result = execute_request(request, module_loader=lambda _name: _module(tokenizer))

    assert result["status"] == "decoded"
    assert result["gate_passed"] is True
    assert result["text"] == "hello"
    assert result["text_sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert result["ledger"]["token_count"] == 2
    assert result["tokenizer"]["local_files_only"] is True
    assert result["tokenizer"]["trust_remote_code"] is False
    assert result["weights_loaded"] is False
    assert tokenizer.calls == [([5, 2], True)]
    encoded = json.dumps(result, ensure_ascii=True).lower()
    assert "token_id" not in encoded
    assert str(ROOT).lower() not in encoded


def test_mm126_worker_allows_empty_decoded_text_and_skips_special_tokens(tmp_path):
    request, _path, _metadata = _ledger(tmp_path)
    tokenizer = _FakeTokenizer("")

    result = execute_request(request, module_loader=lambda _name: _module(tokenizer))

    assert result["status"] == "decoded"
    assert result["text"] == ""
    assert result["text_bytes"] == 0
    assert tokenizer.calls[0][1] is True


def test_mm126_worker_rejects_ledger_file_tamper_before_tokenizer_load(tmp_path):
    request, path, _metadata = _ledger(tmp_path)
    path.write_bytes(b"tampered")
    tokenizer = _FakeTokenizer()

    result = execute_request(request, module_loader=lambda _name: _module(tokenizer))

    assert result["status"] == "decode_failed"
    assert result["errors"][0]["code"] == "tokenizer_decode_failed"
    assert tokenizer.calls == []


def test_mm126_worker_rejects_ledger_digest_tamper_after_evidence_update(tmp_path):
    request, path, metadata = _ledger(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["records"][0]["token_id"] = 99
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(encoded)
    metadata["size_bytes"] = len(encoded)
    metadata["sha256"] = hashlib.sha256(encoded).hexdigest()
    tokenizer = _FakeTokenizer()

    result = execute_request(request, module_loader=lambda _name: _module(tokenizer))

    assert result["status"] == "decode_failed"
    assert tokenizer.calls == []


def test_mm126_worker_rejects_invalid_utf8_surrogate_and_text_overflow(tmp_path):
    request, _path, _metadata = _ledger(tmp_path)
    surrogate = _FakeTokenizer("\ud800")
    invalid = execute_request(request, module_loader=lambda _name: _module(surrogate))
    assert invalid["status"] == "decode_failed"

    request["text_max_bytes"] = 2
    oversized = execute_request(
        request, module_loader=lambda _name: _module(_FakeTokenizer("hello")),
    )
    assert oversized["status"] == "decode_failed"


def test_mm126_controller_validates_metadata_and_keeps_worker_boundary(tmp_path):
    request, path, metadata = _ledger(tmp_path)
    seen: dict = {}

    def runner(value, _timeout):
        seen.update(value)
        return {
            "schema_version": 1,
            "tool": "qwen3_token_ledger",
            "status": "decoded",
            "gate_passed": True,
        }

    result = run_qwen3_token_ledger_decode(
        model=tmp_path / "tokenizer-model",
        ledger=path,
        ledger_metadata=metadata,
        expected_chain_id="c" * 64,
        expected_generation=22,
        expected_first_sequence=69,
        worker_runner=runner,
    )

    assert result["status"] == "decoded"
    assert seen["ledger_metadata"]["ledger_id"] == "mm1ledger_fixture"
    assert seen["expected_first_sequence"] == 69
    assert "token_id" not in json.dumps(result, ensure_ascii=True)
    invalid = run_qwen3_token_ledger_decode(
        model=tmp_path / "tokenizer-model",
        ledger=path,
        ledger_metadata={**metadata, "sha256": "f" * 64},
        worker_runner=runner,
    )
    assert invalid["status"] == "invalid_request"
