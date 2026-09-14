from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_SURFACES = (
    ROOT / "src" / "api_server.py",
    ROOT / "src" / "inference_service" / "routes.py",
    ROOT / "src" / "inference_service" / "protocol.py",
    ROOT / "src" / "scheduler.py",
    ROOT / "src" / "task_worker_protocol.py",
    ROOT / "schemas" / "artifact-manifest.schema.json",
    ROOT / "schemas" / "experiment-record.schema.json",
    ROOT / "frontend_cybergothic" / "src" / "app" / "routes.tsx",
    ROOT / "frontend_cybergothic" / "src" / "data" / "api.ts",
    ROOT / "frontend_cybergothic" / "src" / "data" / "types.ts",
)
FORBIDDEN_MAIN_MARKERS = (
    "diffusion",
    "sd15",
    "image_generate",
    "image_edit",
    "image_grid",
    "image_prompt",
    "diffusers",
)


def test_main_project_has_no_image_generation_surface():
    assert not list((ROOT / "src" / "diffusion").glob("*.py"))
    assert not (ROOT / "packaging" / "requirements-sd15.txt").exists()
    assert all(
        marker not in path.read_text(encoding="utf-8").lower()
        for path in MAIN_SURFACES
        for marker in FORBIDDEN_MAIN_MARKERS
    )


def test_koakumix_owns_image_generation_surface():
    app = (ROOT / "harness_workbench" / "api_layer" / "app.py").read_text(encoding="utf-8")
    builtin = (ROOT / "harness_workbench" / "mcp_server" / "builtin.py").read_text(encoding="utf-8")
    assert "/v1/images/generations" in app
    assert "image_generate" in builtin
    assert not (ROOT / "harness_workbench" / "image_workbench" / "remote_qlh.py").exists()


def test_defense_demo_has_no_main_image_generation_scenario():
    storyline = (ROOT / "scripts" / "demo" / "storyline.json").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "demo" / "defense_demo.py").read_text(encoding="utf-8")
    scenarios = (ROOT / "scripts" / "demo" / "scenarios.json").read_text(encoding="utf-8")

    assert "fixture-image" not in storyline
    assert "three fixed scenarios" not in storyline
    assert '"kind": "image"' not in scenarios
    assert 'choices=("dialog", "image", "topology")' not in launcher
