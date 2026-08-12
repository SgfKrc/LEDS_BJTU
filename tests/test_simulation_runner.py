import json

from scripts import run_simulation


def test_simulation_runner_profiles_are_fixed_and_layered():
    assert run_simulation.targets_for_profile("quick") == (
        "tests/test_task_graph_simulation.py",
        "tests/test_task_worker_control_simulation.py",
        "tests/test_diffusion_data_plane_simulation.py",
        "tests/test_mixed_workflow_simulation.py",
        "tests/test_capacity_simulation.py",
    )
    assert run_simulation.targets_for_profile("extended")[:5] == (
        run_simulation.targets_for_profile("quick")
    )
    assert run_simulation.targets_for_profile("full") == ("tests",)


def test_simulation_runner_quick_profile_emits_safe_json(tmp_path):
    output = tmp_path / "simulation-summary.json"

    assert run_simulation.main([
        "--profile", "quick",
        "--json-output", str(output),
    ]) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "qlh.simulation_runner.v1"
    assert summary["profile"] == "quick"
    assert summary["outcome"] == "passed"
    assert summary["pytest"]["exit_code"] == 0
    assert summary["pytest"]["counts"]["passed"] >= 41
    assert [item["family"] for item in summary["evidence"]] == [
        "task_graph",
        "task_worker_control",
        "diffusion_data_plane",
        "mixed_workflow",
        "capacity",
    ]
    assert summary["acceptance_scope"] == {
        "physical_nodes": "not_established",
        "real_models": "not_established",
        "real_network": "not_established",
        "performance": "not_established",
        "installation_package": "not_established",
    }
    serialized = json.dumps(summary, sort_keys=True)
    for field in (
        "authorization",
        "root_input",
        "blob_id",
        "content",
        "prompt",
        "transfer_plan",
        "secret",
        "token",
        "url",
    ):
        assert f'"{field}"' not in serialized
    assert "simulated visual prompt" not in serialized


def test_simulation_runner_default_stdout_is_parseable_json(capsys, monkeypatch):
    monkeypatch.setattr(run_simulation, "_run_pytest", lambda _targets: (0, "1 passed"))
    monkeypatch.setattr(run_simulation, "collect_evidence", lambda: [])

    assert run_simulation.main([]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["profile"] == "quick"
    assert summary["pytest"]["counts"]["passed"] == 1
