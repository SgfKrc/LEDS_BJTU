import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
# ★ 2026-09-20：`tools/reasonix-codex-bridge` **不再是子模块**，改为**工作区独立目录**
#   （见 .gitignore 与 .gitmodules 的说明）。因此**新克隆 / CI 上不保证存在** ⇒
#   依赖它的用例在缺失时**跳过**（本机存在则照常真跑，不降低既有覆盖）。
BRIDGE_DIR = ROOT / "tools" / "reasonix-codex-bridge"
_bridge_present = pytest.mark.skipif(
    not BRIDGE_DIR.is_dir(),
    reason="reasonix-codex-bridge 是工作区独立目录（非子模块），未克隆时跳过",
)


@_bridge_present
def test_bridge_preset_examples_use_current_v41_api_ref():
    presets = json.loads(
        (BRIDGE_DIR / "presets.example.json").read_text(encoding="utf-8")
    )
    docagent_env = (ROOT / "tools" / "docagent" / ".env.docagent.example").read_text(encoding="utf-8")

    refs = [item["modelRef"] for item in presets["presets"]]
    assert refs
    assert all(ref.endswith("/deepseek-flash") for ref in refs)
    assert not any("deepseek-v4-flash" in ref for ref in refs)
    assert "DOCAGENT_DEEPSEEK_MODEL=deepseek-flash" in docagent_env
    assert "DOCAGENT_DEEPSEEK_MODEL=deepseek-v4-flash" not in docagent_env


@_bridge_present
def test_main_docs_distinguish_current_ref_from_legacy_aliases():
    cache_plan = (
        ROOT
        / "docs"
        / "archive"
        / "runtime"
        / "\u7f13\u5b58\u673a\u5236\u4e13\u9879\u8ba1\u5212-2026-09-13.md"
    ).read_text(encoding="utf-8")
    bridge_config = (BRIDGE_DIR / "config.example.toml").read_text(encoding="utf-8")

    assert "deepseek-flash" in cache_plan
    assert "deepseek-flash" in bridge_config
    assert "deepseek-v4-flash" not in bridge_config
    assert "\u65e7\u540d\u4ec5\u4f5c\u517c\u5bb9\u8bf4\u660e" in cache_plan


def test_readme_does_not_present_legacy_deepseek_api_ref():
    for name in ("README.md", "docs/README.en.md"):
        content = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "deepseek-v4-flash" not in content
