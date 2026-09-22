from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import relay_cut_plan  # noqa: E402


def _record(*, model: str = "model-a", cut: int = 4) -> dict:
    return {
        "model": model,
        "layers": cut,
        "gen": 32,
        "batch": 1,
        "prefill": 32,
        "git_head": "same-head",
        "timing_ms": {
            "upstream_decode": 10.0 + cut,
            "downstream_decode": 40.0 - cut,
        },
        "whole_model_bytes": 240 * 1024 * 1024,
    }


def _analysis_report() -> dict:
    identity = {
        "model": "model-a",
        "kind": "mainrepo_end_to_end",
        "path": "d2l_mainrepo",
        "commit": "same-head",
        "prefill": 32,
        "gen": 32,
        "batch": 1,
        "warmup": 3,
    }
    return {
        "schema_version": "qlh.relay_cut_model_analysis.v2",
        "total_layers": 24,
        "min_rounds": 3,
        "rounds_per_cut": {"4": 3, "8": 3},
        "experiment_identity": identity,
        "warmup": {"steps": 3, "consistent": True},
        "whole_model_bytes": 240 * 1024 * 1024,
        "record_origin": {
            "kind": "raw_repeated_measurements",
            "source_records": ["r1-k4.json", "r1-k8.json"],
            "rounds": 3,
        },
        "by_cut": [
            {"cut": 4, "rounds": 3, "all_passed": True,
             "upstream_median_ms": 14.0, "downstream_median_ms": 36.0,
             "total_median_ms": 50.0},
            {"cut": 8, "rounds": 3, "all_passed": True,
             "upstream_median_ms": 18.0, "downstream_median_ms": 32.0,
             "total_median_ms": 50.0},
        ],
        "verdict": {"input_valid": True},
    }


def test_extract_accepts_p1_layer_layout_and_metrics():
    record = {
        "experiment_id": "relay-d2l_mainrepo-model-a-k4-p32-g32-b1",
        "kind": "mainrepo_end_to_end",
        "path": "d2l_mainrepo",
        "commit": "same-head",
        "models": {"upstream": {"id": "model-a"}, "whole": {"model_bytes": 240}},
        "layer_layout": {"upstream_layers": 4},
        "load": {"prefill_tokens": 32, "gen_tokens": 32, "batch": 1},
        "metrics": {
            "upstream_decode_ms": {"mean": 14.0},
            "downstream_decode_ms": {"mean": 36.0},
        },
    }
    extracted = relay_cut_plan._extract(record)
    assert extracted is not None
    assert extracted["upstream_layers"] == 4
    assert extracted["model"] == "model-a"
    assert extracted["prefill"] == 32
    assert extracted["gen"] == 32


def test_main_requires_analysis_report(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(_record(cut=4)), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_record(model="model-b", cut=8)), encoding="utf-8")
    code = relay_cut_plan.main([
        "--records", str(tmp_path / "*.json"),
        "--total-layers", "24",
    ])
    assert code == 2
    assert "analysis-report" in capsys.readouterr().out


def test_main_does_not_pass_an_unmeasured_global_prediction(tmp_path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(_analysis_report()), encoding="utf-8")
    code = relay_cut_plan.main([
        "--analysis-report", str(analysis),
        "--total-layers", "24",
        "--json-out", str(tmp_path / "report.json"),
    ])
    assert code == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["verdict"]["passed"] is False
    assert report["verdict"]["reason"] == "predicted_cut_not_measured"


def test_main_rejects_malformed_analysis_report_without_traceback(tmp_path, capsys):
    analysis = tmp_path / "analysis.json"
    payload = _analysis_report()
    payload["rounds_per_cut"] = [3, 3]
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    code = relay_cut_plan.main([
        "--analysis-report", str(analysis), "--total-layers", "24",
    ])
    assert code == 2
    assert "invalid warmup/round-count objects" in capsys.readouterr().out
