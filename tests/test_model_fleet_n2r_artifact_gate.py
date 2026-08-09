"""MF-MEM-N2R artifact admission tests."""

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_fleet_n2r_artifact_gate import TARGET_FAMILY, inspect_artifacts  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gate_does_not_promote_smaller_or_gguf_candidates():
    result = inspect_artifacts(PROJECT_ROOT)

    assert result["ticket"] == "MF-MEM-N2R"
    assert result["target"]["family"] == TARGET_FAMILY
    assert result["target_artifact_available"] == 1
    assert result["resource_rejected"] == 0
    assert result["candidate_quant_artifact_available"] == 1
    assert result["candidate_target_match"] == 0
    assert "7b_gguf_q4_k_m_is_not_target_quantization" in result["reason_codes"]
    assert "available_quant_candidate_is_qwen_1_8b_not_7b" in result["reason_codes"]
    assert "fixed_7b_bf16_weights_with_explicit_runtime_int8_nf4_recipe" in result["reason_codes"]


def test_gate_records_missing_7b_safetensors_weight_files():
    result = inspect_artifacts(PROJECT_ROOT)
    manifest_records = [
        record for record in result["local_artifacts"] if record["format"] == "manifest"
    ]
    assert manifest_records
    assert manifest_records[0]["expected_safetensors"]
    assert manifest_records[0]["all_expected_weights_present"] is True


def test_gate_json_is_serializable():
    result = inspect_artifacts(PROJECT_ROOT)
    json.dumps(result, ensure_ascii=False)
