import json

import pytest

from harness_workbench.tools import PROMPT_LAB_SCHEMA, run_prompt_lab
from harness_workbench.tools.prompt_lab import (
    PromptLabCase,
    build_parser,
    builtin_prompt_lab_cases,
    load_prompt_cases,
    load_prompt_profiles,
    main,
)
from harness_workbench.adaptation import PromptProfile
from harness_workbench.context_engine import ContextMessage


def test_prompt_lab_default_matrix_and_diff_are_valid():
    report = run_prompt_lab()

    assert report.schema == PROMPT_LAB_SCHEMA
    assert report.model_family == "QW1.8B"
    assert report.profile_count == 2
    assert report.case_count == 3
    assert report.render_count == 6
    assert report.valid is True
    assert all(report.checks.values())
    assert report.comparisons[0].changed_fields == ("system_prompt", "structured_output")
    assert {item.estimated_tokens_delta for item in report.comparisons[0].case_deltas} == {3}


def test_prompt_lab_render_digest_is_stable_and_payload_free():
    first = run_prompt_lab()
    second = run_prompt_lab()

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    encoded = json.dumps(first.as_dict(), ensure_ascii=False)
    assert "Answer clearly and briefly" not in encoded
    assert "Name the primary color" not in encoded
    assert '"content"' not in encoded
    assert "A/B prompt render report" in first.to_markdown()


def test_prompt_lab_filters_profiles_and_cases_without_changing_render_contract():
    report = run_prompt_lab(
        selected_profiles=["qwen_chat_v1-structured", "qwen_chat_v1-minimal"],
        selected_cases=["structured-json-v1"],
    )
    assert report.profile_count == 2
    assert report.case_count == 1
    assert report.render_count == 2
    assert report.selected_profile_ids == (
        "qwen_chat_v1-structured",
        "qwen_chat_v1-minimal",
    )
    assert report.comparisons[0].profile_a == "qwen_chat_v1-structured"
    assert report.valid is True

    with pytest.raises(ValueError, match="at least two"):
        run_prompt_lab(selected_profiles=["qwen_chat_v1-minimal"])
    with pytest.raises(ValueError, match="unknown prompt profile"):
        run_prompt_lab(selected_profiles=["missing", "qwen_chat_v1-minimal"])
    with pytest.raises(ValueError, match="unique"):
        run_prompt_lab(selected_profiles=["qwen_chat_v1-minimal", "qwen_chat_v1-minimal"])


def test_prompt_lab_supports_gemma_builtin_family():
    report = run_prompt_lab(model_family="Gemma-small")
    assert {profile.family for profile in report.profiles} == {"gemma_chat_v1"}
    assert report.valid is True


def test_prompt_lab_loads_custom_profiles_and_cases(tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "qlh.prompt_lab.v1",
                "profiles": [
                    {
                        "id": "a",
                        "family": "fixture",
                        "system_prompt": "Be concise.",
                        "structured_output": "json_repair",
                    },
                    {
                        "id": "b",
                        "family": "fixture",
                        "system_prompt": "Be concise and format exactly.",
                        "structured_output": "grammar_first",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    case_path = tmp_path / "cases.json"
    case_path.write_text(
        json.dumps(
            {
                "schema": "qlh.prompt_lab.v1",
                "cases": [
                    {
                        "case_id": "custom",
                        "messages": [{"role": "user", "content": "Say hello."}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    profiles = load_prompt_profiles(profile_path)
    cases = load_prompt_cases(case_path)
    report = run_prompt_lab(profile_file=profile_path, case_file=case_path)
    assert len(profiles) == 2
    assert cases[0].case_id == "custom"
    assert report.runner_kind == "input"
    assert report.render_count == 2
    assert report.valid is True


def test_prompt_lab_rejects_bad_schema_paths_and_duplicate_ids(tmp_path):
    bad_schema = tmp_path / "bad-schema.json"
    bad_schema.write_text(
        json.dumps({"schema": "wrong.v1", "cases": [{"messages": [{"role": "user", "content": "x"}]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported prompt lab input schema"):
        load_prompt_cases(bad_schema)

    bad_path = tmp_path / "bad-path.json"
    bad_path.write_text(
        json.dumps({"cases": [{"case_id": "x", "description": "C:\\secret" , "messages": [{"role": "user", "content": "x"}]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absolute paths"):
        load_prompt_cases(bad_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {
                "profiles": [
                    {"id": "same", "family": "x", "system_prompt": "a"},
                    {"id": "same", "family": "x", "system_prompt": "b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="profile ids"):
        load_prompt_profiles(duplicate)


def test_prompt_lab_cli_lists_and_writes_reports(tmp_path, capsys):
    assert main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert "qwen_chat_v1-minimal" in listing
    assert "factual-short-v1" in listing

    json_path = tmp_path / "prompt-lab.json"
    markdown_path = tmp_path / "prompt-lab.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == PROMPT_LAB_SCHEMA
    assert payload["render_count"] == 6
    assert "Name the primary color" not in json_path.read_text(encoding="utf-8")
    assert "Answer clearly and briefly" not in json_path.read_text(encoding="utf-8")
    assert "Length comparison" in markdown_path.read_text(encoding="utf-8")


def test_prompt_lab_parser_and_case_value_contracts():
    args = build_parser().parse_args(["--profile", "a", "--profile", "b", "--case", "x"])
    assert args.profile_ids == ["a", "b"]
    assert args.case_ids == ["x"]
    case = PromptLabCase("case", (ContextMessage("user", "hello"),))
    assert case.messages_digest
    with pytest.raises(ValueError):
        PromptLabCase("", (ContextMessage("user", "hello"),))


def test_prompt_lab_accepts_direct_prompt_profiles():
    profiles = (
        PromptProfile("a", "fixture", "one"),
        PromptProfile("b", "fixture", "two"),
    )
    report = run_prompt_lab(profiles=profiles, cases=builtin_prompt_lab_cases()[:1])
    assert report.profile_count == 2
    assert report.case_count == 1
    assert report.valid is True
