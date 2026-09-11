import json

import pytest

from harness_workbench.tools import FUN_CLI_SCHEMA, QuoteCard, build_quote_report, build_say_report, run_fun_cli
from harness_workbench.tools.fun_cli import (
    FUN_QUOTES_INPUT_SCHEMA,
    FunCliError,
    builtin_quote_cards,
    load_quote_cards,
    main,
    render_banner,
    render_progress,
    render_quote_cards,
)


def test_say_report_is_deterministic_ascii_and_fixture_only():
    first = build_say_report()
    second = build_say_report()

    assert first.schema == FUN_CLI_SCHEMA
    assert first.valid is True
    assert first.as_dict() == second.as_dict()
    assert all(line.isascii() for line in first.lines)
    assert first.model_invoked is False
    assert first.network_used is False
    assert first.weights_loaded is False
    assert first.lines[-1].endswith("100%")


def test_say_supports_theme_and_progress_style_switches():
    assert render_banner("cyber") != render_banner("classic")
    assert render_banner("minimal")[0] == "QLH HARN"
    outputs = {style: render_progress(style, 2, 4) for style in ("bar", "blocks", "dots", "steps", "none")}
    assert len(set(outputs.values())) == 5
    assert all("50%" in value for value in outputs.values())


def test_say_redacts_message_in_terminal_output_and_keeps_digest_bound():
    report = build_say_report("token=sk-1234567890 https://example.invalid/x", theme="minimal", progress_style="none", steps=1)

    assert "sk-1234567890" not in "\n".join(report.lines)
    assert "example.invalid" not in "\n".join(report.lines)
    assert report.message_chars > 0
    assert report.message_digest != build_say_report("other", steps=1).message_digest


def test_progress_and_theme_boundaries_fail_closed():
    with pytest.raises(FunCliError, match="unsupported theme"):
        render_banner("rainbow")
    with pytest.raises(FunCliError, match="between zero and total"):
        render_progress("bar", 5, 4)
    with pytest.raises(FunCliError, match="between 8 and 80"):
        render_progress("bar", 1, 2, width=4)
    with pytest.raises(FunCliError, match="between 1 and 100"):
        build_say_report(steps=0)


def test_builtin_quote_report_is_fixture_only_and_deterministic():
    first = build_quote_report()
    second = build_quote_report()

    assert first.valid is True
    assert first.digest == second.digest
    assert len(first.cards) == 3
    assert all(card.source == "fixture" for card in first.cards)
    assert all(card.claim_scope == "fixture_only" for card in first.cards)
    assert all(card.as_dict()["model_invoked"] is False for card in first.cards)
    assert "model invoked: false" in render_quote_cards(first)


def test_quote_markdown_contains_sanitized_cards_not_quality_claims():
    report = build_quote_report()
    markdown = report.to_markdown()

    assert "FUN-CLI-01 model quotes" in markdown
    assert "QW1.8B" in markdown
    assert "model-quality claim" in markdown
    assert "95d383d78a712fd9" in render_quote_cards(report)


def test_custom_quote_input_is_schema_checked_and_redacted(tmp_path):
    path = tmp_path / "quotes.json"
    path.write_text(
        json.dumps(
            {
                "schema": FUN_QUOTES_INPUT_SCHEMA,
                "cards": [
                    {
                        "card_id": "custom",
                        "model_id": "fixture-model",
                        "prompt": "Say hello",
                        "quote": "contact https://example.invalid with sk-1234567890",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cards = load_quote_cards(path)
    report = build_quote_report(cards)

    assert report.valid is True
    rendered = render_quote_cards(report)
    assert "example.invalid" not in rendered
    assert "sk-1234567890" not in rendered
    assert "<redacted" in rendered


def test_quote_input_rejects_wrong_schema_duplicates_and_non_fixture_source(tmp_path):
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": "wrong", "cards": []}), encoding="utf-8")
    with pytest.raises(FunCliError, match="qlh.fun_quotes.v1"):
        load_quote_cards(wrong)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {
                "schema": FUN_QUOTES_INPUT_SCHEMA,
                "cards": [
                    {"card_id": "same", "model_id": "a", "prompt": "p", "quote": "q"},
                    {"card_id": "same", "model_id": "b", "prompt": "p", "quote": "q"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FunCliError, match="unique"):
        load_quote_cards(duplicate)

    with pytest.raises(FunCliError, match="fixture-only"):
        QuoteCard("unsafe", "model", "prompt", "quote", source="live")


def test_fun_cli_command_writes_json_and_markdown(tmp_path):
    say_json = tmp_path / "say.json"
    say_md = tmp_path / "say.md"
    quote_json = tmp_path / "quotes.json"

    assert main(["say", "--progress", "steps", "--json", str(say_json), "--markdown", str(say_md)]) == 0
    assert main(["quotes", "--model", "QW1.8B", "--json", str(quote_json)]) == 0
    assert json.loads(say_json.read_text(encoding="utf-8"))["valid"] is True
    assert "progress" in say_md.read_text(encoding="utf-8")
    quote_payload = json.loads(quote_json.read_text(encoding="utf-8"))
    assert quote_payload["summary"]["card_count"] == 1
    assert quote_payload["network_used"] is False


def test_fun_cli_package_exports_are_lazy_and_run_alias_is_fixture_only():
    report = run_fun_cli("say", theme="classic", progress_style="dots", steps=2)

    assert report.valid is True
    assert report.theme == "classic"
    assert report.progress_style == "dots"

