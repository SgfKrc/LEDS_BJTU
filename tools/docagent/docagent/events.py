"""Optional M3-compatible SQLite audit event adapter.

The event database is a derived index. Reports, rules, and Git remain the
authoritative sources, so the standalone scanner does not require this module.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .gate import GateError, validate_report


EVENTS_SCHEMA_VERSION = "qlh.docagent.events.v1"
_EVENT_KINDS = frozenset({"scan", "gate"})


class EventStoreError(ValueError):
    """Raised when the optional event index cannot be updated safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS docagent_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('scan', 'gate')),
  run_ts TEXT NOT NULL,
  rules_fingerprint TEXT NOT NULL,
  report_fingerprint TEXT NOT NULL,
  evolution_fingerprint TEXT,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docagent_events_ts
  ON docagent_events(ts DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_docagent_events_report
  ON docagent_events(report_fingerprint);
"""


class DocEventStore:
    """Small, path-free event adapter for the existing M3 SQLite workflow."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.path)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        except (OSError, sqlite3.Error) as exc:
            raise EventStoreError(f"cannot open event store: {self.path}") from exc

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DocEventStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def record(
        self,
        report: Mapping[str, Any],
        *,
        kind: str = "scan",
        gate: Mapping[str, Any] | None = None,
        evolution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in _EVENT_KINDS:
            raise EventStoreError(f"unsupported event kind: {kind!r}")
        try:
            selected = validate_report(report, require_fingerprint=True)
        except GateError as exc:
            raise EventStoreError(str(exc)) from exc
        gate_summary = None
        if gate is not None:
            gate_summary = {
                "state": gate.get("state"),
                "gate_fingerprint": gate.get("gate_fingerprint"),
            }
        evolution_summary = None
        if evolution is not None:
            evolution_summary = {
                "state": evolution.get("state"),
                "evolution_fingerprint": evolution.get("evolution_fingerprint"),
            }
        payload = {
            "schema_version": EVENTS_SCHEMA_VERSION,
            "run_ts": selected["run_ts"],
            "profile": selected["profile"],
            "document_count": len(selected["docs"]),
            "finding_count": sum(len(doc["findings"]) for doc in selected["docs"]),
            "rules": {
                "ruleset_version": selected["ruleset_version"],
                "fingerprint": selected["rules_fingerprint"],
            },
            "gate": gate_summary,
            "evolution": evolution_summary,
        }
        evolution_fingerprint = evolution_summary["evolution_fingerprint"] if evolution_summary else None
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    INSERT INTO docagent_events (
                      ts, kind, run_ts, rules_fingerprint, report_fingerprint,
                      evolution_fingerprint, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _now(), kind, selected["run_ts"], selected["rules_fingerprint"],
                        selected["report_fingerprint"], evolution_fingerprint, _json(payload),
                    ),
                )
        except sqlite3.Error as exc:
            raise EventStoreError("cannot write event store") from exc
        return {
            "event_id": int(cursor.lastrowid),
            "kind": kind,
            "run_ts": selected["run_ts"],
            "rules_fingerprint": selected["rules_fingerprint"],
            "report_fingerprint": selected["report_fingerprint"],
            "evolution_fingerprint": evolution_fingerprint,
        }

    def index_snapshot(
        self,
        audit: Mapping[str, Any],
        repo_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compatibility name for M3 callers; ``repo_root`` is intentionally unused."""
        del repo_root
        return self.record(audit, kind="scan")

    def recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise EventStoreError("limit must be an integer from 1 to 1000")
        rows = self.conn.execute(
            """
            SELECT event_id, ts, kind, run_ts, rules_fingerprint,
                   report_fingerprint, evolution_fingerprint, payload
            FROM docagent_events
            ORDER BY event_id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result


__all__ = ["EVENTS_SCHEMA_VERSION", "DocEventStore", "EventStoreError"]
