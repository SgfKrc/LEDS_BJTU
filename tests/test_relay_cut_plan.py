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


def test_main_rejects_mixed_experiment_identity(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(_record(cut=4)), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_record(model="model-b", cut=8)), encoding="utf-8")
    code = relay_cut_plan.main([
        "--records", str(tmp_path / "*.json"),
        "--total-layers", "24",
    ])
    assert code == 2
    assert "multiple experiment identities" in capsys.readouterr().out


def test_main_does_not_pass_an_unmeasured_global_prediction(tmp_path):
    for cut in (4, 8):
        (tmp_path / f"{cut}.json").write_text(
            json.dumps(_record(cut=cut)), encoding="utf-8")
    code = relay_cut_plan.main([
        "--records", str(tmp_path / "*.json"),
        "--total-layers", "24",
        "--json-out", str(tmp_path / "report.json"),
    ])
    assert code == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["verdict"]["passed"] is False
    assert report["verdict"]["reason"] == "predicted_cut_not_measured"
