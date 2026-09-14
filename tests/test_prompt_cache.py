"""CACHE-01 prompt prefix stability checks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from harness_workbench.adaptation.builtin import builtin_adaptation_profiles
from harness_workbench.adaptation.cache_policy import find_prompt_cache_violations


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_prompt_cache.py"
BRIDGE_PROMPTS = ROOT / "tools" / "reasonix-codex-bridge" / "prompts"


def test_dynamic_prompt_values_are_reported_without_echoing_content() -> None:
    secret_path = r"C:\private\prompt.txt"
    prompt = f"policy\nrequest_id=abc123\ncreated=2026-09-14T12:30:00Z\npath={secret_path}\n"
    violations = find_prompt_cache_violations(prompt)
    assert {item.rule for item in violations} == {"runtime_id_value", "timestamp", "windows_absolute_path"}
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--text", prompt],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "VIOLATION <text>:2 runtime_id_value" in result.stdout
    assert secret_path not in result.stdout


def test_bridge_and_builtin_harness_prompts_are_stable() -> None:
    prompt_paths = [
        BRIDGE_PROMPTS / "deepseek-worker-prompt.md",
        BRIDGE_PROMPTS / "deepseek-worker-write-prompt.md",
    ]
    for prompt_path in prompt_paths:
        assert find_prompt_cache_violations(prompt_path.read_text(encoding="utf-8")) == ()
    prompts, _, _ = builtin_adaptation_profiles("QW1.8B")
    assert all(find_prompt_cache_violations(prompt.system_prompt) == () for prompt in prompts)


def test_checker_accepts_all_cache_safe_prompt_files() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER), *(str(path) for path in (
            BRIDGE_PROMPTS / "deepseek-worker-prompt.md",
            BRIDGE_PROMPTS / "deepseek-worker-write-prompt.md",
        ))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("OK ") == 2
