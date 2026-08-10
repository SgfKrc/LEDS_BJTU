import json
import sys
from pathlib import Path
from types import SimpleNamespace


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import diagnose


def _integrity_report(*categories: str) -> dict:
    return {
        "ok": False,
        "level": "quick",
        "summary": {"checked": 4, "passed": 1, "failed": len(categories)},
        "manifest": {"app_id": "qlh-edge-inference", "variant": "cuda"},
        "failed": [
            {"path": f"program-{index}.bin", "category": category}
            for index, category in enumerate(categories)
        ],
    }


def test_diagnose_knowledge_base_has_all_ten_local_symptom_classes(tmp_path, monkeypatch):
    root = tmp_path / "app"
    root.mkdir()
    (root / "models").mkdir()
    (root / "models" / "user-owned.gguf").write_bytes(b"do not scan")
    monkeypatch.setattr(
        diagnose.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1, total=10 * diagnose.LOW_DISK_BYTES),
    )
    monkeypatch.setattr(diagnose.os, "access", lambda _path, _mode: False)
    monkeypatch.setattr(
        diagnose,
        "probe_nvidia_smi",
        lambda: {"probed": True, "available": False, "reason": "not_found"},
    )
    raw_error = "DLL CUDA driver network timeout no space permission model token=do-not-export"
    report = diagnose.diagnose_install(
        root,
        error=raw_error,
        integrity_report=_integrity_report("signature", "missing", "version"),
    )

    assert len(diagnose.KNOWLEDGE_BASE) == 10
    assert {item["id"] for item in report["issues"]} == {
        item["id"] for item in diagnose.KNOWLEDGE_BASE
    } - {"antivirus_interference"}
    antivirus = diagnose.diagnose_install(
        root,
        integrity_report=_integrity_report("missing"),
        probe_gpu=False,
    )
    assert "antivirus_interference" in {item["id"] for item in antivirus["issues"]}
    assert raw_error not in json.dumps(report, ensure_ascii=False)
    assert all(item["auto_repair_available"] is False for item in report["issues"])
    assert "models" not in json.dumps(report["integrity"], ensure_ascii=False)


def test_diagnosis_bundle_summary_removes_root_and_file_paths(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    report = diagnose.diagnose_install(
        root,
        integrity_report=_integrity_report("missing"),
        probe_gpu=False,
    )
    bundle = diagnose.diagnosis_bundle_summary(report)
    serialized = json.dumps(bundle, ensure_ascii=False)

    assert "root" not in bundle
    assert "failed_paths" not in serialized
    assert str(root) not in serialized

    output = diagnose.write_diagnosis_report(report, tmp_path / "diagnose.json", bundle_safe=True)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == bundle


def test_diagnose_cli_json_uses_failure_exit_code_without_error_text(tmp_path, monkeypatch, capsys):
    report = {
        "schema_version": 1,
        "command": "diagnose",
        "issues": [{"id": "program_file_integrity"}],
        "ok": False,
    }
    monkeypatch.setattr(diagnose, "diagnose_install", lambda *_args, **_kwargs: report)

    assert diagnose.main(["--root", str(tmp_path), "--error", "token=secret", "--json"]) == 3
    assert json.loads(capsys.readouterr().out) == report
