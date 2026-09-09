from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "packaging", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import launcher  # noqa: E402


def test_model_asset_check_is_nonblocking_when_no_model_is_installed(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "model_downloader",
        SimpleNamespace(ensure_model_or_warn=lambda: False),
    )

    assert launcher._check_model_assets_for_startup() is False

