"""Fixture-only fun CLI for ``FUN-CLI-01``.

The commands are deliberately presentation helpers, not model clients.  They
render a small ASCII banner/progress demo and deterministic quote cards.  No
model, network, filesystem artifact, or external dependency is required.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FUN_CLI_SCHEMA = "qlh.harness.fun_cli.v1"
FUN_QUOTES_INPUT_SCHEMA = "qlh.fun_quotes.v1"

_THEMES = ("cyber", "classic", "minimal")
_PROGRESS_STYLES = ("bar", "blocks", "dots", "steps", "none")
_SECRET_TEXT = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+|(?:api[-_]?key|access[-_]?token|refresh[-_]?token|password)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
_URL = re.compile(r"\b(?:https?|wss?)://\S+", re.IGNORECASE)
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
_IP_TEXT = re.compile(r"(?<![0-9A-Fa-f:.])(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){2,})(?![0-9A-Fa-f:.])")


class FunCliError(ValueError):
    """Stable user-facing error without leaking input content."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _redact_text(value: str) -> str:
    """Keep card text useful while removing common secret/address forms."""

    value = _SECRET_TEXT.sub("<redacted>", value)
    value = _URL.sub("<redacted-url>", value)
    value = re.sub(r"(?<!\S)(?:[A-Za-z]:[\\/][^\s]+|/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)", "<redacted-path>", value)
    for candidate in _IP_TEXT.findall(value):
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        value = value.replace(candidate, "<redacted-address>")
    return value


def _ascii_text(value: str) -> str:
    return value.encode("ascii", "backslashreplace").decode("ascii")


def _validate_text(value: Any, field_name: str, *, allow_newlines: bool = True) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FunCliError("invalid_text", f"{field_name} must be non-empty text")
    if len(value) > 4000:
        raise FunCliError("text_too_long", f"{field_name} is too long")
    if not allow_newlines and any(char in value for char in "\r\n"):
        raise FunCliError("invalid_text", f"{field_name} cannot contain newlines")
    return value


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe_ratio(completed: int, total: int) -> float:
    if isinstance(completed, bool) or isinstance(total, bool) or not isinstance(completed, int) or not isinstance(total, int):
        raise FunCliError("invalid_progress", "progress values must be integers")
    if total < 1 or completed < 0 or completed > total:
        raise FunCliError("invalid_progress", "completed must be between zero and total")
    return completed / total


def render_banner(theme: str = "cyber") -> tuple[str, ...]:
    """Return an ASCII-only banner for the requested presentation theme."""

    if theme not in _THEMES:
        raise FunCliError("invalid_theme", f"unsupported theme: {theme}")
    if theme == "cyber":
        return (
            "+--------------------------------------+",
            "| QLH // SMALL MODEL WORKBENCH        |",
            "| fixture mode :: no model :: no net   |",
            "+--------------------------------------+",
        )
    if theme == "classic":
        return (
            "========================================",
            "        Q L H   H A R N E S S",
            "        small tools / safe demo",
            "========================================",
        )
    return ("QLH HARN", "------------", "fixture-only demo")


def render_progress(style: str, completed: int, total: int, *, width: int = 20) -> str:
    """Render one deterministic progress sample using ASCII characters only."""

    ratio = _safe_ratio(completed, total)
    if isinstance(width, bool) or not isinstance(width, int) or not 8 <= width <= 80:
        raise FunCliError("invalid_progress", "progress width must be between 8 and 80")
    if style not in _PROGRESS_STYLES:
        raise FunCliError("invalid_progress", f"unsupported progress style: {style}")
    percent = int(round(ratio * 100))
    if style == "none":
        return f"progress {completed}/{total} ({percent}%)"
    filled = min(width, int(round(width * ratio)))
    if style == "bar":
        track = "#" * filled + "." * (width - filled)
        return f"[{track}] {percent:3d}%"
    if style == "blocks":
        track = "=" * filled + "-" * (width - filled)
        return f"<{track}> {percent:3d}%"
    if style == "dots":
        track = "." * filled + " " * (width - filled)
        return f"{track} {percent:3d}%"
    return f"step {completed:02d}/{total:02d} " + ("#" * filled + "." * (width - filled)) + f" {percent:3d}%"


@dataclass(frozen=True, slots=True)
class SayReport:
    theme: str
    progress_style: str
    total_steps: int
    message_digest: str
    message_chars: int
    lines: tuple[str, ...]
    schema: str = FUN_CLI_SCHEMA
    network_used: bool = False
    weights_loaded: bool = False
    model_invoked: bool = False

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "ascii_output": all(line.isascii() for line in self.lines),
            "lines_present": bool(self.lines),
            "progress_bounded": self.total_steps > 0,
            "fixture_boundary": not self.network_used and not self.weights_loaded and not self.model_invoked,
        }

    @property
    def valid(self) -> bool:
        return self.schema == FUN_CLI_SCHEMA and all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "theme": self.theme,
            "progress_style": self.progress_style,
            "total_steps": self.total_steps,
            "message_digest": self.message_digest,
            "message_chars": self.message_chars,
            "lines": list(self.lines),
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
            "model_invoked": self.model_invoked,
            "checks": self.checks,
        }

    def to_markdown(self) -> str:
        lines = [
            "# FUN-CLI-01 qlh_say",
            "",
            f"- Valid: `{str(self.valid).lower()}`; theme: `{self.theme}`; progress: `{self.progress_style}`; steps: `{self.total_steps}`",
            f"- Message digest: `{self.message_digest}`; chars: `{self.message_chars}`; model invoked: `{str(self.model_invoked).lower()}`",
            "",
            "```text",
            *self.lines,
            "```",
            "",
            "## Checks",
            "",
        ]
        lines.extend(f"- `{name}`: **{'passed' if passed else 'failed'}**" for name, passed in self.checks.items())
        return "\n".join(lines) + "\n"


def build_say_report(
    message: str = "fixture mode: no model call, no network",
    *,
    theme: str = "cyber",
    progress_style: str = "bar",
    steps: int = 4,
    width: int = 20,
) -> SayReport:
    message = _validate_text(message, "message", allow_newlines=False)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise FunCliError("invalid_progress", "steps must be between 1 and 100")
    banner = render_banner(theme)
    samples = tuple(dict.fromkeys((0, max(1, steps // 2), steps)))
    progress = tuple(render_progress(progress_style, step, steps, width=width) for step in samples)
    safe_message = _ascii_text(_redact_text(message))
    lines = banner + (f"message: {safe_message}",) + progress
    return SayReport(theme, progress_style, steps, _digest(message), len(message), lines)


@dataclass(frozen=True, slots=True)
class QuoteCard:
    card_id: str
    model_id: str
    prompt: str
    quote: str
    source: str = "fixture"
    claim_scope: str = "fixture_only"

    def __post_init__(self) -> None:
        _validate_text(self.card_id, "card_id", allow_newlines=False)
        _validate_text(self.model_id, "model_id", allow_newlines=False)
        _validate_text(self.prompt, "prompt")
        _validate_text(self.quote, "quote")
        if self.source != "fixture" or self.claim_scope != "fixture_only":
            raise FunCliError("unsafe_quote_source", "quote cards must remain fixture-only")

    @property
    def prompt_digest(self) -> str:
        return _digest(self.prompt)

    @property
    def quote_digest(self) -> str:
        return _digest(self.quote)

    def as_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "model_id": self.model_id,
            "prompt": _redact_text(self.prompt),
            "quote": _redact_text(self.quote),
            "prompt_digest": self.prompt_digest,
            "quote_digest": self.quote_digest,
            "source": self.source,
            "claim_scope": self.claim_scope,
            "model_invoked": False,
        }


def builtin_quote_cards() -> tuple[QuoteCard, ...]:
    """Return deterministic cards; these strings are not model observations."""

    return (
        QuoteCard(
            "qw18b-context-tight",
            "QW1.8B",
            "What is your debugging superpower?",
            "I keep the context small enough that every clue still has a seat.",
        ),
        QuoteCard(
            "qwen3-fixture-pause",
            "Qwen3-0.6B-fixture",
            "Write a one-line motto for an offline harness.",
            "No network, no drama: make the boundary observable.",
        ),
        QuoteCard(
            "gemma-fixture-review",
            "Gemma-small-fixture",
            "Describe a careful release in six words.",
            "Ship the evidence, then ship the feature.",
        ),
    )


def _cards_from_payload(payload: Mapping[str, Any]) -> tuple[QuoteCard, ...]:
    if payload.get("schema") != FUN_QUOTES_INPUT_SCHEMA or not isinstance(payload.get("cards"), list):
        raise FunCliError("invalid_schema", "quote input must use qlh.fun_quotes.v1 with cards")
    cards: list[QuoteCard] = []
    for index, raw in enumerate(payload["cards"]):
        if not isinstance(raw, Mapping):
            raise FunCliError("invalid_card", f"quote card {index + 1} must be an object")

        def text_field(name: str) -> str:
            value = raw.get(name, "")
            if not isinstance(value, str):
                raise FunCliError("invalid_card", f"quote card {index + 1} field {name} must be text")
            return value

        cards.append(
            QuoteCard(
                text_field("card_id"),
                text_field("model_id"),
                text_field("prompt"),
                text_field("quote"),
                text_field("source") if "source" in raw else "fixture",
                text_field("claim_scope") if "claim_scope" in raw else "fixture_only",
            )
        )
    if not cards:
        raise FunCliError("empty_quotes", "at least one quote card is required")
    if len({card.card_id for card in cards}) != len(cards):
        raise FunCliError("duplicate_card", "quote card IDs must be unique")
    return tuple(cards)


def load_quote_cards(path: str | Path) -> tuple[QuoteCard, ...]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FunCliError("invalid_input", "unable to read quote input") from exc
    if not isinstance(payload, Mapping):
        raise FunCliError("invalid_schema", "quote input must be an object")
    return _cards_from_payload(payload)


@dataclass(frozen=True, slots=True)
class QuoteReport:
    cards: tuple[QuoteCard, ...]
    schema: str = FUN_CLI_SCHEMA
    network_used: bool = False
    weights_loaded: bool = False
    model_invoked: bool = False

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "cards_present": bool(self.cards),
            "unique_card_ids": len({card.card_id for card in self.cards}) == len(self.cards),
            "fixture_sources": all(card.source == "fixture" and card.claim_scope == "fixture_only" for card in self.cards),
            "redaction_applied": all(card.as_dict()["quote"] == _redact_text(card.quote) for card in self.cards),
            "fixture_boundary": not self.network_used and not self.weights_loaded and not self.model_invoked,
        }

    @property
    def valid(self) -> bool:
        return self.schema == FUN_CLI_SCHEMA and all(self.checks.values())

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
            "model_invoked": self.model_invoked,
            "cards": [card.as_dict() for card in self.cards],
            "summary": {"card_count": len(self.cards), "models": sorted({card.model_id for card in self.cards})},
            "checks": self.checks,
        }
        if include_digest:
            result["report_digest"] = self.digest
        return result

    def to_markdown(self) -> str:
        lines = [
            "# FUN-CLI-01 model quotes",
            "",
            f"- Valid: `{str(self.valid).lower()}`; cards: `{len(self.cards)}`; model invoked: `{str(self.model_invoked).lower()}`; report digest: `{self.digest}`",
            "- Scope: fixture-only presentation cards; quote text is redacted before rendering and is not a model-quality claim.",
            "",
            "## Quote cards",
            "",
        ]
        for card in self.cards:
            lines.extend((f"### {card.model_id} / {card.card_id}", "", f"- Prompt: {_redact_text(card.prompt)}", f"- Quote: {_redact_text(card.quote)}", f"- Quote digest: `{card.quote_digest}`", ""))
        lines.extend(("## Checks", ""))
        lines.extend(f"- `{name}`: **{'passed' if passed else 'failed'}**" for name, passed in self.checks.items())
        return "\n".join(lines) + "\n"


def build_quote_report(cards: Sequence[QuoteCard] | None = None) -> QuoteReport:
    selected = tuple(cards) if cards is not None else builtin_quote_cards()
    if not selected:
        raise FunCliError("empty_quotes", "at least one quote card is required")
    if not all(isinstance(card, QuoteCard) for card in selected):
        raise FunCliError("invalid_card", "quote reports require QuoteCard values")
    return QuoteReport(selected)


def run_fun_cli(command: str = "say", **kwargs: Any) -> SayReport | QuoteReport:
    """Programmatic fixture entry point used by demos and tests."""

    if command == "say":
        return build_say_report(
            kwargs.pop("message", "fixture mode: no model call, no network"),
            theme=kwargs.pop("theme", "cyber"),
            progress_style=kwargs.pop("progress_style", "bar"),
            steps=kwargs.pop("steps", 4),
            width=kwargs.pop("width", 20),
        )
    if command == "quotes":
        return build_quote_report(kwargs.pop("cards", None))
    raise FunCliError("invalid_command", f"unsupported fun command: {command}")


def render_quote_cards(report: QuoteReport) -> str:
    lines = ["QLH MODEL QUOTES // FIXTURE ONLY", "=" * 36, ""]
    for card in report.cards:
        lines.extend((f"[{card.model_id}] {card.card_id}", f"prompt: {_redact_text(card.prompt)}", f"quote : {_redact_text(card.quote)}", f"digest: {card.quote_digest[:16]}", ""))
    lines.append("model invoked: false | network: false | weights: false")
    return "\n".join(lines) + "\n"


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fun-cli", description="Fixture-only QLH presentation helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    say = subparsers.add_parser("say", help="render an ASCII banner and progress demo")
    say.add_argument("--theme", choices=_THEMES, default="cyber")
    say.add_argument("--progress", choices=_PROGRESS_STYLES, default="bar")
    say.add_argument("--steps", type=int, default=4)
    say.add_argument("--width", type=int, default=20)
    say.add_argument("--message", default="fixture mode: no model call, no network")
    _add_output_args(say)
    quotes = subparsers.add_parser("quotes", help="render deterministic fixture quote cards")
    quotes.add_argument("--input", metavar="PATH", help="qlh.fun_quotes.v1 JSON fixture")
    quotes.add_argument("--model", help="filter cards by model id")
    _add_output_args(quotes)
    return parser


def _emit_report(report: SayReport | QuoteReport, *, json_path: str, markdown_path: str, text: str) -> None:
    outputs = 0
    if json_path:
        _write_text(json_path, json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n")
        outputs += 1
    if markdown_path:
        _write_text(markdown_path, report.to_markdown())
        outputs += 1
    if not outputs:
        print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "say":
            report = build_say_report(args.message, theme=args.theme, progress_style=args.progress, steps=args.steps, width=args.width)
            _emit_report(report, json_path=args.json_path, markdown_path=args.markdown_path, text="\n".join(report.lines) + "\n")
        else:
            cards = load_quote_cards(args.input) if args.input else builtin_quote_cards()
            if args.model:
                cards = tuple(card for card in cards if card.model_id == args.model)
                if not cards:
                    raise FunCliError("unknown_model", "no quote card matched model")
            report = build_quote_report(cards)
            _emit_report(report, json_path=args.json_path, markdown_path=args.markdown_path, text=render_quote_cards(report))
    except (OSError, FunCliError, ValueError) as exc:
        parser.error(str(exc))
    return 0 if report.valid else 1


__all__ = [
    "FUN_CLI_SCHEMA",
    "FUN_QUOTES_INPUT_SCHEMA",
    "FunCliError",
    "QuoteCard",
    "QuoteReport",
    "SayReport",
    "build_parser",
    "build_quote_report",
    "build_say_report",
    "builtin_quote_cards",
    "load_quote_cards",
    "main",
    "render_banner",
    "render_progress",
    "render_quote_cards",
    "run_fun_cli",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
