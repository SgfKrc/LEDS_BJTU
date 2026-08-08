import sys
from pathlib import Path


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import qlh_launcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_development_launcher_finds_legacy_application_entry():
    command = qlh_launcher.app_command("ui")
    assert command[-2].endswith("packaging\\launcher.py") or command[-2].endswith("packaging/launcher.py")
    assert command[-1] == "--ui"


def test_controller_variant_is_used_for_check(monkeypatch):
    captured = {}

    def fake_check(sources, **kwargs):
        captured.update(kwargs)
        return {"current_version": "0.1.8.1", "latest_version": "0.1.8.1"}

    monkeypatch.setattr(qlh_launcher, "check_updates", fake_check)
    monkeypatch.setattr(qlh_launcher, "installed_app_root", lambda *_args: Path("."))
    monkeypatch.setattr(qlh_launcher, "detect_current_version", lambda *_args, **_kwargs: "0.1.8.1")
    controller = qlh_launcher.LauncherController(
        ["https://example.invalid/latest.json"], variant_override="cpu",
    )

    controller.check_update()

    assert captured["profile"]["variant"] == "cpu"


def test_headless_launch_returns_success_code(monkeypatch):
    launched = []
    monkeypatch.setattr(qlh_launcher, "launch_app", lambda mode, _variant=None: launched.append(mode))
    assert qlh_launcher.main(["--headless"]) == 0
    assert launched == ["headless"]


def test_missing_application_is_reported_without_crashing(monkeypatch):
    monkeypatch.setattr(
        qlh_launcher, "launch_app",
        lambda _mode, _variant=None: (_ for _ in ()).throw(FileNotFoundError("missing app")),
    )
    controller = qlh_launcher.LauncherController([])
    assert controller.start_gui() == 2
    assert controller.last_error == "missing app"


def test_direct_app_command_does_not_open_bootstrap_menu(monkeypatch):
    controller = qlh_launcher.LauncherController([], variant_override="cpu")
    monkeypatch.setattr(controller, "start_gui", lambda: 17)
    monkeypatch.setattr(qlh_launcher, "LauncherController", lambda *_args: controller)
    assert qlh_launcher.main(["app-ui", "--variant", "cpu"]) == 17


def test_windows_application_discovery_prefers_requested_variant(tmp_path, monkeypatch):
    cpu = tmp_path / "QLH-Edge-Inference"
    cuda = tmp_path / "QLH-Edge-Inference-CUDA"
    cpu.mkdir()
    cuda.mkdir()
    (cpu / "QLH-Edge-Inference.exe").write_bytes(b"cpu")
    (cuda / "QLH-Edge-Inference.exe").write_bytes(b"cuda")

    monkeypatch.setattr(
        qlh_launcher, "_candidate_app_roots",
        lambda preferred: [cuda, cpu] if preferred == "cuda" else [cpu, cuda],
    )

    assert qlh_launcher.installed_app_root("cpu") == cpu
    assert qlh_launcher.installed_app_root("cuda") == cuda
    assert qlh_launcher.app_command("ui", "cuda") == [
        str(cuda / "QLH-Edge-Inference.exe"), "--ui",
    ]


def test_linux_deb_build_keeps_bootstrap_and_venv_fallback_contract():
    script = (PROJECT_ROOT / "packaging" / "linux" / "build-deb.sh").read_text(
        encoding="utf-8"
    )

    assert '"$PACKAGING_DIR/qlh_launcher.py"' in script
    assert '"$PACKAGING_DIR/update_core.py"' in script
    assert '"$PACKAGING_DIR/updater.py"' in script
    assert '"$PACKAGING_DIR/signing.py"' in script
    assert '"$PACKAGING_DIR/pubkeys"' in script
    assert "python3 -c 'import venv'" in script
    assert "curl -fsSL --retry 3 https://bootstrap.pypa.io/get-pip.py" in script


def test_linux_application_command_prefers_packaged_venv(tmp_path, monkeypatch):
    app_root = tmp_path / "qlh-edge-inference"
    app = app_root / "bin" / "qlh-app"
    venv_python = app_root / "venv" / "bin" / "python"
    app.parent.mkdir(parents=True)
    venv_python.parent.mkdir(parents=True)
    app.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(qlh_launcher, "installed_app_root", lambda *_args: app_root)
    monkeypatch.setattr(qlh_launcher.os, "name", "posix")

    assert qlh_launcher.app_command("tui") == [
        str(venv_python), str(app), "--tui",
    ]
