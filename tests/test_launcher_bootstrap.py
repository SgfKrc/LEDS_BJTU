import sys
from pathlib import Path


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import qlh_launcher


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
