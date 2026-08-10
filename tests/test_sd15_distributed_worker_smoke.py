import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_sd15_distributed_worker.py"
SPEC = importlib.util.spec_from_file_location("sd15_distributed_worker_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def test_parser_enforces_stage_budget_bounds():
    parser = smoke.build_parser()
    parsed = parser.parse_args([
        "--steps", "4",
        "--width", "512",
        "--height", "512",
    ])
    assert parsed.steps == 4
    assert parsed.width == 512

    with pytest.raises(SystemExit):
        parser.parse_args(["--steps", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--width", "513"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--height", "776"])


def test_worker_environment_is_offline_and_bypasses_proxy_for_loopback():
    env = smoke._offline_worker_environment("x" * 32)

    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["QLH_DIFFUSION_WORKER_EXPERIMENTAL_ENABLED"] == "true"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["QLH_CLUSTER_SECRET"] == "x" * 32


def test_installed_manifest_digest_requires_matching_artifact(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    manifest_path = model_path / ".qlh-sd-asset.json"
    manifest_path.write_text(json.dumps({
        "artifact_sha256": "a" * 64,
        "asset": {"artifact_id": "sd15_test"},
    }), encoding="utf-8")

    assert smoke._artifact_manifest_digest(model_path, "sd15_test") == "a" * 64
    with pytest.raises(smoke.DistributedSD15SmokeError, match="does not match"):
        smoke._artifact_manifest_digest(model_path, "another_artifact")


def test_report_rejects_local_paths_and_credentials(tmp_path):
    model_path = tmp_path / "private-model"
    model_path.mkdir()
    secret = "secret-value-which-must-not-be-reported"
    safe = {
        "status": "passed",
        "artifact": {"manifest_sha256": "a" * 64},
        "result": {"output_file": "distributed-output.png"},
    }
    smoke._assert_report_safe(safe, model_path=model_path, cluster_secret=secret)

    for leaked in (
        {"model_path": str(model_path.resolve())},
        {"secret": secret},
        {"transfer": {"grant": "temporary"}},
        {"transfer": {"lease_id": "temporary"}},
    ):
        with pytest.raises(smoke.DistributedSD15SmokeError, match="forbidden"):
            smoke._assert_report_safe(
                leaked,
                model_path=model_path,
                cluster_secret=secret,
            )


def test_worker_command_uses_current_python_and_explicit_local_artifact(tmp_path):
    args = smoke.build_parser().parse_args([
        "--artifact-id", "sd15_test",
        "--model-path", str(tmp_path / "model"),
        "--profile", "balanced",
        "--node-id", "worker_test",
    ])
    command = smoke._worker_command(
        args,
        tcp_port=12345,
        http_port=12346,
        worker_state=tmp_path / "worker-state",
        ready_file=tmp_path / "ready.json",
        completion_file=tmp_path / "complete",
    )

    assert command[0] == sys.executable
    assert "--worker" in command
    assert command[command.index("--artifact-id") + 1] == "sd15_test"
    assert command[command.index("--node-id") + 1] == "worker_test"
    assert "--model-path" in command
