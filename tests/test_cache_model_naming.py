import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bridge_preset_examples_use_current_v41_api_ref():
    presets = json.loads(
        (ROOT / "tools" / "reasonix-codex-bridge" / "presets.example.json").read_text(encoding="utf-8")
    )
    docagent_env = (ROOT / "tools" / "docagent" / ".env.docagent.example").read_text(encoding="utf-8")

    refs = [item["modelRef"] for item in presets["presets"]]
    assert refs
    assert all(ref.endswith("/deepseek-flash") for ref in refs)
    assert not any("deepseek-v4-flash" in ref for ref in refs)
    assert "DOCAGENT_DEEPSEEK_MODEL=deepseek-flash" in docagent_env
    assert "DOCAGENT_DEEPSEEK_MODEL=deepseek-v4-flash" not in docagent_env


def test_main_docs_distinguish_current_ref_from_legacy_aliases():
    cache_plan = (ROOT / "docs" / "缓存机制专项计划-2026-09-13.md").read_text(encoding="utf-8")
    bridge_checklist = (ROOT / "docs" / "reasonix-codex-bridge主仓接线清单-2026-09-12.md").read_text(encoding="utf-8")

    assert "deepseek-flash" in cache_plan
    assert "deepseek-flash" in cache_plan
    assert "modelRef = opencode-go-2ae…/deepseek-flash" in bridge_checklist
    assert "旧名仅作兼容说明" in cache_plan


def test_readme_does_not_present_legacy_deepseek_api_ref():
    for name in ("README.md", "docs/README.en.md"):
        content = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "deepseek-v4-flash" not in content
