from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "agent_tool"))

from docagent_rules import (  # noqa: E402
    RULES_SCHEMA_VERSION,
    RulesConfigError,
    get_rule,
    load_rules,
    rules_fingerprint,
    validate_rules,
)


def test_default_rules_load_with_all_initial_rule_ids():
    rules = load_rules()

    assert rules["schema_version"] == RULES_SCHEMA_VERSION
    assert {rule["id"] for rule in rules["rules"]} == {"R1", "R2", "R3", "R4", "R5"}
    assert get_rule("R4", rules)["level"] == "warn"
    assert len(rules_fingerprint(rules)) == 64
    assert rules_fingerprint(rules) == rules_fingerprint(load_rules())


def test_rules_loader_rejects_unknown_schema_and_missing_rule(tmp_path: Path):
    rules = load_rules()

    unknown = dict(rules)
    unknown["schema_version"] = "qlh.docagent.rules.v999"
    with pytest.raises(RulesConfigError, match="unsupported rules schema"):
        validate_rules(unknown)

    missing = dict(rules)
    missing["rules"] = [rule for rule in rules["rules"] if rule["id"] != "R5"]
    path = tmp_path / "rules.yaml"
    path.write_text(json.dumps(missing, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RulesConfigError, match="missing required IDs"):
        load_rules(path)


def test_rules_loader_rejects_duplicate_and_unknown_rule_ids():
    rules = load_rules()
    duplicate = dict(rules)
    duplicate["rules"] = [*rules["rules"], dict(rules["rules"][0])]
    with pytest.raises(RulesConfigError, match="duplicate rule id"):
        validate_rules(duplicate)

    unknown = dict(rules)
    unknown["rules"] = [dict(rule) for rule in rules["rules"]]
    unknown["rules"][0]["id"] = "R9"
    with pytest.raises(RulesConfigError, match="unknown rule id"):
        validate_rules(unknown)


def test_rules_loader_rejects_incomplete_rule_parameters():
    rules = load_rules()
    malformed = json.loads(json.dumps(rules))
    r1 = next(rule for rule in malformed["rules"] if rule["id"] == "R1")
    del r1["parameters"]["done_markers"]

    with pytest.raises(RulesConfigError, match="R1.done_markers"):
        validate_rules(malformed)
