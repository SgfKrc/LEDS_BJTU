"""文档维护 Agent M3.1：本地事件库与当前快照索引。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_events import DocEventStore, rebuild_event_database  # noqa: E402


def _repo(tmp_path: Path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text(
        "# X title\n\n> 状态：规划\n> 更新日期：2026-08-17\n\nsecret body text\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "docs/x.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "initial"], check=True)
    return repo


def _two_doc_repo(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "docs" / "y.md").write_text(
        "# Y title\n\nTask graph fusion scheduler reference.\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(repo), "add", "docs/y.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "add y"], check=True)
    return repo


def _second_commit(repo: Path):
    doc = repo / "docs" / "x.md"
    doc.write_text(doc.read_text(encoding="utf-8") + "\nsecond edit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "docs/x.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "second"], check=True)


def _audit(*, llm=False):
    value = {
        "run_ts": "2026-08-17T10:00:00",
        "rules": {"R1": "完成未收口"},
        "docs": [{
            "doc": "docs/x.md", "status_line": "> 状态：规划",
            "updated_at": "2026-08-17", "sha256": "a" * 64,
            "findings": [{"rule": "R1", "level": "warn", "message": "疑似"}],
        }],
    }
    if llm:
        value["llm"] = {"judgements": [{
            "doc": "docs/x.md", "judgement": "needs_review", "confidence": 0.4,
            "suggestion": "人工核对", "source": "llm",
            "provider": "opencode", "model": "test",
        }]}
    return value


def test_snapshot_initializes_schema_metadata_and_scan_event(tmp_path):
    repo = _repo(tmp_path)
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        result = store.index_snapshot(_audit(), repo)
        meta = store.get_doc_meta("docs/x.md")
        events = store.recent_events("docs/x.md")
        assert result["docs_indexed"] == 1
        assert meta is not None
        assert meta["title"] == "X title"
        assert len(meta["last_commit"]) == 40
        assert events[0]["kind"] == "scan"
        # 当前快照索引不保存正文。
        database_text = " ".join(
            str(value) for row in store.conn.execute("SELECT * FROM doc_meta") for value in row
        )
        assert "secret body text" not in database_text


def test_repeated_snapshot_updates_meta_and_appends_runs(tmp_path):
    repo = _repo(tmp_path)
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        first = store.index_snapshot(_audit(), repo)
        changed = _audit()
        changed["docs"][0]["sha256"] = "b" * 64
        second = store.index_snapshot(changed, repo)
        assert second["run_id"] == first["run_id"] + 1
        assert store.get_doc_meta("docs/x.md")["sha256"] == "b" * 64
        assert len(store.recent_events("docs/x.md")) == 2
        assert store.conn.execute("SELECT COUNT(*) FROM check_runs").fetchone()[0] == 2


def test_llm_judgement_adds_separate_event(tmp_path):
    repo = _repo(tmp_path)
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        result = store.index_snapshot(_audit(llm=True), repo)
        assert result["llm_events"] == 1
        assert [event["kind"] for event in store.recent_events("docs/x.md")] == [
            "llm_suggestion", "scan",
        ]


def test_manual_decisions_update_existing_run_only(tmp_path):
    repo = _repo(tmp_path)
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        result = store.index_snapshot(_audit(), repo)
        store.record_decisions(result["run_id"], {"docs/x.md": "accurate"})
        raw = store.conn.execute(
            "SELECT decisions FROM check_runs WHERE run_id = ?", (result["run_id"],)
        ).fetchone()[0]
        assert json.loads(raw) == {"docs/x.md": "accurate"}
        with pytest.raises(KeyError):
            store.record_decisions(999, {})


def test_document_escape_is_rejected_atomically(tmp_path):
    repo = _repo(tmp_path)
    audit = _audit()
    audit["docs"][0]["doc"] = "../outside.md"
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        with pytest.raises(ValueError, match="escapes docs root"):
            store.index_snapshot(audit, repo)
        assert store.conn.execute("SELECT COUNT(*) FROM check_runs").fetchone()[0] == 0


def test_git_history_replay_records_commits_in_order(tmp_path):
    repo = _repo(tmp_path)
    _second_commit(repo)
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        result = store.replay_git_history(repo)
        events = list(store.conn.execute(
            "SELECT payload FROM doc_events ORDER BY event_id"
        ))
    assert result == {"commits_replayed": 2, "git_events": 2}
    assert [json.loads(row[0])["subject"] for row in events] == ["initial", "second"]


def test_rebuild_backs_up_old_database_and_atomically_replaces_it(tmp_path):
    repo = _repo(tmp_path)
    _second_commit(repo)
    database = repo / "build" / "events.sqlite"
    with DocEventStore(database) as store:
        store.index_snapshot(_audit(), repo)
    result = rebuild_event_database(database, _audit(), repo)
    assert result["commits_replayed"] == 2
    assert result["git_events"] == 2
    assert result["scan_events"] == 1
    assert result["backup"] is not None
    assert (repo / result["backup"]).is_file()
    with DocEventStore(database) as store:
        kinds = [row[0] for row in store.conn.execute(
            "SELECT kind FROM doc_events ORDER BY event_id"
        )]
        assert kinds == ["manual_edit", "manual_edit", "scan"]


def test_failed_rebuild_preserves_existing_database(tmp_path):
    repo = _repo(tmp_path)
    database = repo / "build" / "events.sqlite"
    with DocEventStore(database) as store:
        store.index_snapshot(_audit(), repo)
    bad_audit = _audit()
    bad_audit["docs"][0]["doc"] = "../outside.md"
    with pytest.raises(ValueError, match="escapes docs root"):
        rebuild_event_database(database, bad_audit, repo)
    with DocEventStore(database) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM check_runs").fetchone()[0] == 1


def test_chunk_index_search_and_incremental_skip(tmp_path):
    repo = _two_doc_repo(tmp_path)
    audit = _audit()
    audit["docs"].append({
        "doc": "docs/y.md", "status_line": "", "updated_at": None,
        "sha256": "b" * 64, "findings": [],
    })
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        first = store.index_chunks(audit, repo)
        second = store.index_chunks(audit, repo)
        results = store.search_chunks("task graph fusion", limit=3)
    assert first["documents_indexed"] == 2
    assert second == {"documents_indexed": 0, "documents_unchanged": 2, "chunks": 0}
    assert results[0]["doc_id"] == "docs/y.md"
    assert "[Task]" in results[0]["snippet"]


def test_chunk_index_replaces_changed_document_and_safe_query(tmp_path):
    repo = _two_doc_repo(tmp_path)
    audit = _audit()
    audit["docs"].append({
        "doc": "docs/y.md", "status_line": "", "updated_at": None,
        "sha256": "b" * 64, "findings": [],
    })
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        store.index_chunks(audit, repo)
        audit["docs"][1]["sha256"] = "c" * 64
        (repo / "docs" / "y.md").write_text("# Y\n\nunique needle\n", encoding="utf-8")
        result = store.index_chunks(audit, repo)
        assert result["documents_indexed"] == 1
        assert store.search_chunks("unique needle")[0]["doc_id"] == "docs/y.md"
        assert store.search_chunks('" OR *', limit=5) == []
