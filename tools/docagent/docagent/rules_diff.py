"""Structural diff for validated docagent rulesets."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .rules import rules_fingerprint, validate_rules


def _changes(old: object, new: object, path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    changed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in sorted(set(old) | set(new), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                added.append({"path": child, "value": copy.deepcopy(new[key])})
            elif key not in new:
                removed.append({"path": child, "value": copy.deepcopy(old[key])})
            else:
                child_changed, child_added, child_removed = _changes(old[key], new[key], child)
                changed.extend(child_changed)
                added.extend(child_added)
                removed.extend(child_removed)
        return changed, added, removed
    if path == "rules" and isinstance(old, list) and isinstance(new, list):
        old_by_id = {item["id"]: item for item in old if isinstance(item, Mapping) and "id" in item}
        new_by_id = {item["id"]: item for item in new if isinstance(item, Mapping) and "id" in item}
        if len(old_by_id) == len(old) and len(new_by_id) == len(new):
            for rule_id in sorted(set(old_by_id) | set(new_by_id), key=str):
                child = f"{path}.{rule_id}"
                if rule_id not in old_by_id:
                    added.append({"path": child, "value": copy.deepcopy(new_by_id[rule_id])})
                elif rule_id not in new_by_id:
                    removed.append({"path": child, "value": copy.deepcopy(old_by_id[rule_id])})
                else:
                    child_changed, child_added, child_removed = _changes(
                        old_by_id[rule_id], new_by_id[rule_id], child,
                    )
                    changed.extend(child_changed)
                    added.extend(child_added)
                    removed.extend(child_removed)
            return changed, added, removed
    if old != new:
        changed.append({"path": path, "old": copy.deepcopy(old), "new": copy.deepcopy(new)})
    return changed, added, removed


def diff_rules(old: Mapping[str, Any], new: Mapping[str, Any], *, old_source: str = "old", new_source: str = "new") -> dict[str, Any]:
    # Candidate rulesets may add/remove IDs here; the runtime scanner remains strict.
    old = validate_rules(old, allow_rule_set_changes=True)
    new = validate_rules(new, allow_rule_set_changes=True)
    changed, added, removed = _changes(old, new, "")
    return {
        "schema_version": "qlh.docagent.rules-diff.v1",
        "old": {
            "source": old_source,
            "ruleset_version": old["ruleset_version"],
            "fingerprint": rules_fingerprint(old, allow_rule_set_changes=True),
        },
        "new": {
            "source": new_source,
            "ruleset_version": new["ruleset_version"],
            "fingerprint": rules_fingerprint(new, allow_rule_set_changes=True),
        },
        "changed": changed,
        "added": added,
        "removed": removed,
        "summary": {
            "changed": len(changed),
            "added": len(added),
            "removed": len(removed),
            "identical": not changed and not added and not removed,
        },
    }


__all__ = ["diff_rules"]
