from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from qwen3_pipeline_parity import evaluate_qwen3_cpu_parity  # noqa: E402
from scripts.model_tools.qwen3_pipeline_adapter import Qwen3PipelineAdapter  # noqa: E402
from scripts.model_tools.qwen3_pipeline_chain import execute_segment_chain  # noqa: E402


torch = pytest.importorskip("torch")


class _TinyBlock(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **_kwargs):
        hidden = hidden_states * (1 + self.scale)
        present = None
        if use_cache:
            present = hidden.unsqueeze(2)
            if past_key_value is not None:
                present = torch.cat((past_key_value[0], present), dim=1)
            present = (present, present)
        return hidden, present


class _TinySegmentModel(torch.nn.Module):
    def __init__(self, *, layers, embedding, norm, head) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = embedding
        self.model.layers = torch.nn.ModuleList(layers)
        self.model.norm = norm
        self.lm_head = head
        self.config = types.SimpleNamespace(model_type="qwen3", num_hidden_layers=4)


def _synthetic_chain(layout):
    torch.manual_seed(7)
    embedding = torch.nn.Embedding(16, 4)
    layers = [_TinyBlock((index + 1) / 10) for index in range(4)]
    norm = torch.nn.LayerNorm(4)
    head = torch.nn.Linear(4, 8, bias=False)
    segments = []
    adapters = []
    for index, (start, end) in enumerate(layout):
        has_embedding = index == 0
        has_lm_head = index == len(layout) - 1
        model = _TinySegmentModel(
            layers=layers[start:end],
            embedding=embedding if has_embedding else None,
            norm=norm,
            head=head if has_lm_head else None,
        )
        adapters.append(Qwen3PipelineAdapter(
            model, start_layer=start, end_layer=end,
            has_embedding=has_embedding, has_lm_head=has_lm_head,
            total_layers=4,
        ))
        segments.append({
            "layer_range": [start, end],
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
        })
    return embedding, layers, norm, head, adapters, segments


def _reference_forward(embedding, layers, norm, head, input_ids, past=None):
    hidden = embedding(input_ids)
    next_past = []
    for index, layer in enumerate(layers):
        hidden, present = layer(
            hidden,
            past_key_value=past[index] if past is not None else None,
            use_cache=True,
        )
        next_past.append(present)
    return head(norm(hidden)), next_past


def _save_logits(path: Path, values) -> Path:
    torch.save({"logits": torch.tensor(values, dtype=torch.float32)}, path)
    return path


def _evidence(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _phase_artifacts(
    root: Path, phase: str, *, generation: int, final_values, segment_count: int,
):
    artifacts = []
    reports = []
    for index in range(segment_count):
        values = final_values if index == segment_count - 1 else [[float(index), float(index + 1)]]
        path = _save_logits(root / f"candidate-{phase}-{index}.pt", values)
        size, digest = _evidence(path)
        artifacts.append(path)
        reports.append({
            "segment_index": index,
            "execution": {
                "artifact_bytes": size,
                "artifact_sha256": digest,
                "full_model_materialized": False,
                "segment_materialized": True,
            },
            "kv_contract": {
                "segment_index": index,
                "phase": phase,
                "generation": generation,
                "sequence_length": 3 if phase == "prefill" else 4,
            },
            "hidden_handoff": ({
                "from_segment": index,
                "to_segment": index + 1,
                "shape": [1, 3 if phase == "prefill" else 4, 4],
            } if index < segment_count - 1 else None),
        })
    return artifacts, reports


def _case(root: Path, *, candidate_decode=None, segment_count: int = 2):
    root.mkdir()
    reference_prefill = _save_logits(root / "reference-prefill.pt", [[1.0, 2.0]])
    reference_decode = _save_logits(root / "reference-decode.pt", [[3.0, 4.0]])
    prefill_artifacts, prefill_reports = _phase_artifacts(
        root, "prefill", generation=7, final_values=[[1.0, 2.0]],
        segment_count=segment_count,
    )
    decode_artifacts, decode_reports = _phase_artifacts(
        root, "decode", generation=8,
        final_values=candidate_decode if candidate_decode is not None else [[3.0, 4.0]],
        segment_count=segment_count,
    )
    return {
        "artifact_root": root,
        "reference_prefill": reference_prefill,
        "candidate_prefill": prefill_artifacts[-1],
        "reference_decode": reference_decode,
        "candidate_decode": decode_artifacts[-1],
        "prefill_artifacts": prefill_artifacts,
        "prefill_reports": prefill_reports,
        "decode_artifacts": decode_artifacts,
        "decode_reports": decode_reports,
        "segment_count": segment_count,
        "generation": 7,
    }


@pytest.mark.parametrize("segment_count", [2, 3])
def test_cpu_parity_passes_with_metadata_only_evidence(tmp_path, segment_count):
    case = _case(tmp_path / "artifacts", segment_count=segment_count)

    report = evaluate_qwen3_cpu_parity(**case)

    assert report["status"] == "passed"
    assert report["gate_passed"] is True
    assert report["full_model_fallback"] is False
    assert report["full_model_materialized"] is False
    assert report["execution"]["prefill"]["segment_count"] == segment_count
    assert report["execution"]["decode"]["generation"] == 8
    encoded = json.dumps(report, sort_keys=True)
    assert str(case["artifact_root"]) not in encoded
    assert ".pt" not in encoded


def test_cpu_parity_rejects_logit_mismatch_without_full_model_fallback(tmp_path):
    case = _case(tmp_path / "artifacts", candidate_decode=[[30.0, 40.0]])

    report = evaluate_qwen3_cpu_parity(**case)

    assert report["status"] == "rejected"
    assert report["gate_passed"] is False
    assert report["full_model_fallback"] is False
    assert report["errors"][0]["code"] == "qwen3_parity_logits_mismatch"


def test_cpu_parity_rejects_artifact_tampering(tmp_path):
    case = _case(tmp_path / "artifacts")
    _save_logits(case["decode_artifacts"][0], [[9.0, 9.0]])

    report = evaluate_qwen3_cpu_parity(**case)

    assert report["status"] == "rejected"
    assert report["errors"][0]["code"] == "qwen3_parity_evidence_mismatch"


def test_cpu_parity_rejects_generation_mismatch(tmp_path):
    case = _case(tmp_path / "artifacts")
    case["decode_reports"][1]["kv_contract"]["generation"] = 7

    report = evaluate_qwen3_cpu_parity(**case)

    assert report["status"] == "rejected"
    assert report["errors"][0]["code"] == "qwen3_parity_kv_mismatch"


@pytest.mark.parametrize("layout", [[(0, 2), (2, 4)], [(0, 1), (1, 3), (3, 4)]])
def test_cpu_parity_accepts_two_and_three_segment_synthetic_qwen3(tmp_path, layout):
    embedding, layers, norm, head, adapters, segments = _synthetic_chain(layout)
    input_ids = torch.tensor([[1, 2, 3]])
    decode_ids = torch.tensor([[4]])
    reference_prefill, reference_kv = _reference_forward(
        embedding, layers, norm, head, input_ids,
    )
    reference_decode, _ = _reference_forward(
        embedding, layers, norm, head, decode_ids, past=reference_kv,
    )
    candidate = execute_segment_chain(
        adapters, input_ids=input_ids, segments=segments,
        decode_input_ids=decode_ids,
    )
    root = tmp_path / "artifacts"
    root.mkdir()
    reference_prefill_path = _save_logits(root / "reference-prefill.pt", reference_prefill)
    reference_decode_path = _save_logits(root / "reference-decode.pt", reference_decode)
    count = len(layout)
    prefill_artifacts, prefill_reports = _phase_artifacts(
        root, "prefill", generation=0,
        final_values=candidate["prefill"]["logits"], segment_count=count,
    )
    decode_artifacts, decode_reports = _phase_artifacts(
        root, "decode", generation=1,
        final_values=candidate["decode"]["logits"], segment_count=count,
    )

    report = evaluate_qwen3_cpu_parity(
        artifact_root=root,
        reference_prefill=reference_prefill_path,
        candidate_prefill=prefill_artifacts[-1],
        reference_decode=reference_decode_path,
        candidate_decode=decode_artifacts[-1],
        prefill_artifacts=prefill_artifacts,
        prefill_reports=prefill_reports,
        decode_artifacts=decode_artifacts,
        decode_reports=decode_reports,
        segment_count=count,
        generation=0,
    )

    assert candidate["full_model_materialized"] is False
    assert len(candidate["hidden_handoffs"]) == count - 1
    assert all(item["sequence_length"] == 4 for item in candidate["kv_contracts"]["decode"])
    assert report["status"] == "passed"
    assert report["gate_passed"] is True
