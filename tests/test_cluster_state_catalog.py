from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import validate

sys.path.insert(0, "src")

from cluster_state_catalog import (  # noqa: E402
    CATALOG_SCHEMA_VERSION,
    StateCatalog,
    StateCatalogError,
    StateCatalogEntry,
    build_state_catalog,
    validate_state_catalog,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_catalog_is_schema_valid_and_deterministic():
    catalog = build_state_catalog()
    payload = catalog.to_dict()
    schema = json.loads(
        (ROOT / "schemas" / "cluster-state-catalog-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate(payload, schema)
    assert payload == build_state_catalog().to_dict()
    assert payload["schema_version"] == CATALOG_SCHEMA_VERSION
    assert len(payload["entries"]) >= 15
    assert len({entry["state_id"] for entry in payload["entries"]}) == len(payload["entries"])


def test_catalog_covers_all_four_recovery_classes_and_sensitive_boundaries():
    entries = {entry.state_id: entry for entry in build_state_catalog().entries}
    assert {entry.state_class for entry in entries.values()} == {
        "must_replicate", "rebuildable", "audit_only", "must_not_migrate",
    }
    assert entries["control.quorum_ledger"].state_class == "must_replicate"
    assert entries["control.quorum_ledger"].current_boundary == "durable_local"
    assert entries["task_graph.snapshots"].state_class == "rebuildable"
    assert entries["audit.control_events"].state_class == "audit_only"
    assert entries["cluster.node_private_keys"].recovery_strategy == "never_copy"
    assert entries["models.loaded_runtime_and_kv"].recovery_strategy == "never_copy"
    assert entries["auth.identities_and_totp"].sensitive is True


def test_catalog_roundtrip_rejects_duplicates_and_noncanonical_fields():
    payload = build_state_catalog().to_dict()
    assert StateCatalog.from_dict(payload) == build_state_catalog()

    duplicate = json.loads(json.dumps(payload))
    duplicate["entries"].append(duplicate["entries"][0])
    with pytest.raises(StateCatalogError, match="duplicate state_id"):
        validate_state_catalog(duplicate)

    noncanonical = json.loads(json.dumps(payload))
    noncanonical["entries"][0]["unexpected"] = True
    with pytest.raises(StateCatalogError, match="entry fields are not canonical"):
        validate_state_catalog(noncanonical)


def test_forbidden_state_cannot_claim_a_copy_strategy():
    with pytest.raises(StateCatalogError, match="must use never_copy"):
        StateCatalogEntry(
            state_id="bad.secret",
            owner="test",
            source="test",
            state_class="must_not_migrate",
            authority="node",
            current_boundary="memory_only",
            recovery_strategy="rebuild_from_local_state",
            handoff_policy="invalidate_on_term_change",
            sensitive=True,
            notes="bad",
        )


def test_catalog_cli_emits_metadata_only_json():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cluster_state_catalog.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["document_type"] == "cluster_state_catalog"
    text = result.stdout.lower()
    assert '"private_key"' not in text
    assert '"model_bytes"' not in text
