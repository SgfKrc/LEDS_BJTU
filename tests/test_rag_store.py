"""RAG-S0 local boundary, privacy, and recovery tests."""
from __future__ import annotations

import sqlite3

import pytest

from src.rag_store import RagStore, RagStoreError


def _store(tmp_path):
    return RagStore(tmp_path / "qlh-rag.sqlite3", max_chunk_chars=256)


def _ingest(store, **overrides):
    values = {
        "source_id": "docs-plan",
        "relative_ref": "docs/plan.md",
        "sha256": None,
        "mime": "text/markdown",
        "title": "Plan",
        "text": "alpha beta gamma delta epsilon",
        "revision": "r1",
    }
    values.update(overrides)
    return store.ingest_document(**values)


def test_initialize_uses_separate_wal_full_store_and_health(tmp_path):
    store = _store(tmp_path)
    path = store.initialize()
    assert path.name == "qlh-rag.sqlite3"
    health = store.health()
    assert health["status"] == "ok"
    assert health["journal_mode"] == "wal"
    assert health["source_count"] == 0
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT value FROM rag_meta WHERE key='schema_version'").fetchone()[0] == "2"
    finally:
        connection.close()


def test_ingest_is_revision_idempotent_and_supersedes_old_chunks(tmp_path):
    store = _store(tmp_path)
    first = _ingest(store)
    assert first.status == "ingested"
    duplicate = _ingest(store)
    assert duplicate.status == "duplicate"
    assert duplicate.document_id == first.document_id
    second = _ingest(store, revision="r2", text="new revision", title="Plan v2")
    assert second.status == "ingested"
    assert len(store.list_sources()) == 1
    connection = sqlite3.connect(store.path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM rag_documents WHERE status='active'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM rag_documents WHERE status='superseded'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM rag_chunks WHERE document_id=?", (second.document_id,)).fetchone()[0] == 1
    finally:
        connection.close()
    with pytest.raises(RagStoreError, match="different text") as exc:
        _ingest(store, revision="r2", text="tampered revision")
    assert exc.value.code == "revision_conflict"


@pytest.mark.parametrize("field,value,code", [
    ("relative_ref", ".env", "sensitive_source_rejected"),
    ("relative_ref", "../outside.md", "source_ref_invalid"),
    ("relative_ref", "models/weights.safetensors", "mime_rejected"),
    ("owner_scope", "remote", "owner_scope_invalid"),
    ("access_scope", "public", "access_scope_invalid"),
    ("text", "-----BEGIN PRIVATE KEY-----", "sensitive_content_rejected"),
    ("text", "token: abc123", "sensitive_content_rejected"),
])
def test_sensitive_and_out_of_boundary_material_is_rejected(tmp_path, field, value, code):
    store = _store(tmp_path)
    kwargs = {field: value}
    if field == "relative_ref" and value.endswith("safetensors"):
        kwargs["relative_ref"] = "models/weights.safetensors"
        kwargs["mime"] = "application/octet-stream"
    with pytest.raises(RagStoreError) as exc:
        _ingest(store, **kwargs)
    assert exc.value.code == code
    assert store.health()["source_count"] == 0


def test_delete_is_atomic_and_rebuild_removes_materialized_data(tmp_path):
    store = _store(tmp_path)
    _ingest(store)
    _ingest(store, source_id="notes", relative_ref="notes/a.txt", revision="r1", text="private note")
    assert store.delete_source("docs-plan") is True
    assert store.delete_source("docs-plan") is False
    assert [row["source_id"] for row in store.list_sources()] == ["notes"]
    counts = store.reset_index()
    assert counts["documents"] == 1
    assert counts["chunks"] >= 1
    assert store.health()["document_count"] == 0
    assert store.list_sources()[0]["status"] == "pending"


def test_failed_ingest_does_not_leave_partial_revision(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(RagStoreError, match="secret-like"):
        _ingest(store, text="ok\npassword: hidden", revision="bad")
    assert store.list_sources() == []


def test_fts_search_filters_active_revision_and_audits_without_raw_query(tmp_path):
    store = _store(tmp_path)
    first = _ingest(store, text="old alpha", access_scope="owner")
    _ingest(store, revision="r2", text="new alpha", access_scope="owner")
    _ingest(store, source_id="project-doc", relative_ref="docs/project.md", text="project alpha", access_scope="project")
    results = store.search("alpha", access_scope="owner")
    assert [row["source_id"] for row in results] == ["docs-plan"]
    assert results[0]["revision"] == "r2"
    assert results[0]["text_content"] == "new alpha"
    project_results = store.search("alpha", access_scope="project")
    assert [row["source_id"] for row in project_results] == ["project-doc"]
    connection = sqlite3.connect(store.path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM rag_chunks_fts").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM rag_query_events").fetchone()[0] == 2
        event = connection.execute("SELECT query_digest, filters_json FROM rag_query_events ORDER BY created_at LIMIT 1").fetchone()
        assert event[0] != "alpha"
        assert "alpha" not in event[1]
    finally:
        connection.close()
    assert first.document_id != results[0]["document_id"]


def test_fts_delete_rebuild_and_invalid_query_are_atomic(tmp_path):
    store = _store(tmp_path)
    _ingest(store)
    assert store.search("alpha")
    assert store.delete_source("docs-plan") is True
    assert store.search("alpha") == []
    _ingest(store, text="restored alpha", revision="r3")
    assert store.rebuild_fts() == 1
    assert store.search("restored")[0]["revision"] == "r3"
    with pytest.raises(RagStoreError) as exc:
        store.search('alpha OR "')
    assert exc.value.code == "query_invalid"
    with pytest.raises(RagStoreError) as exc:
        store.search('\u4e3b\u8282\u70b9 "')
    assert exc.value.code == "query_invalid"
    with pytest.raises(RagStoreError) as exc:
        store.search("\u3000")
    assert exc.value.code == "query_invalid"


def test_cjk_index_fallback_normalizes_queries_without_relaxing_fts_errors(tmp_path):
    store = _store(tmp_path)
    _ingest(
        store,
        text="\u4e3b\u8282\u70b9\u672c\u5730 SQLite WAL \u548c FULL \u540c\u6b65\u7b56\u7565\uff0c\u77e5\u8bc6\u5e93\u68c0\u7d22\u4e0e\u5411\u91cf\u6a21\u578b\u3002",
    )
    assert store.search("\u4e3b\u8282\u70b9")
    assert store.search("  \uff33\uff31\uff2c\uff49\uff54\uff45  ")
    with pytest.raises(RagStoreError) as exc:
        store.search('alpha OR "')
    assert exc.value.code == "query_invalid"


def test_chunk_boundaries_overlap_and_preserve_offsets(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256, chunk_overlap_chars=32)
    text = ("paragraph one. " * 30) + ("paragraph two. " * 30)
    result = _ingest(store, text=text)
    assert result.chunk_count > 1
    connection = sqlite3.connect(store.path)
    try:
        rows = connection.execute(
            "SELECT start_offset, end_offset, text_content FROM rag_chunks ORDER BY ordinal"
        ).fetchall()
    finally:
        connection.close()
    assert rows[0][0] == 0
    assert all(0 <= start < end <= len(text) for start, end, _ in rows)
    assert all(rows[index][0] < rows[index - 1][1] for index in range(1, len(rows)))
    assert all(text[start:end] == chunk for start, end, chunk in rows)


def test_bounded_vector_search_and_hybrid_fts_fallback(tmp_path):
    store = _store(tmp_path)
    _ingest(store, text="alpha lexical document")
    _ingest(store, source_id="notes", relative_ref="notes/a.txt", text="semantic neighbor", revision="r1")
    connection = sqlite3.connect(store.path)
    try:
        chunk_ids = dict(connection.execute(
            "SELECT d.source_id, c.chunk_id FROM rag_chunks c "
            "JOIN rag_documents d ON d.document_id=c.document_id WHERE d.status='active'"
        ))
    finally:
        connection.close()
    model_digest = "a" * 64
    assert store.upsert_embeddings(
        provider="ollama", model_id="nomic-embed-text:latest", model_sha256=model_digest,
        dimensions=2, vectors={chunk_ids["docs-plan"]: [1.0, 0.0], chunk_ids["notes"]: [0.0, 1.0]},
    ) == 2
    semantic = store.semantic_search(
        [0.0, 1.0], provider="ollama", model_id="nomic-embed-text:latest",
        model_sha256=model_digest, dimensions=2,
    )
    assert semantic[0]["source_id"] == "notes"
    hybrid = store.hybrid_search(
        "alpha", [1.0, 0.0], provider="ollama", model_id="nomic-embed-text:latest",
        model_sha256=model_digest, dimensions=2, max_scan=1,
    )
    assert hybrid[0]["source_id"] == "docs-plan"
    assert all(row["hybrid_mode"] == "fts_fallback" for row in hybrid)
    assert all(row["vector_reason_code"] == "vector_scan_budget_exceeded" for row in hybrid)
    assert store.health()["embedding_count"] == 2


def test_embeddings_reject_superseded_chunks_and_reset_with_index(tmp_path):
    store = _store(tmp_path)
    first = _ingest(store)
    connection = sqlite3.connect(store.path)
    try:
        old_chunk_id = connection.execute(
            "SELECT chunk_id FROM rag_chunks WHERE document_id=?", (first.document_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    _ingest(store, revision="r2", text="replacement")
    with pytest.raises(RagStoreError) as exc:
        store.upsert_embeddings(
            provider="ollama", model_id="nomic-embed-text:latest", model_sha256="b" * 64,
            dimensions=2, vectors={old_chunk_id: [1.0, 0.0]},
        )
    assert exc.value.code == "embedding_chunk_invalid"
    connection = sqlite3.connect(store.path)
    try:
        active_chunk_id = connection.execute(
            "SELECT c.chunk_id FROM rag_chunks c JOIN rag_documents d ON d.document_id=c.document_id WHERE d.status='active'"
        ).fetchone()[0]
    finally:
        connection.close()
    store.upsert_embeddings(
        provider="ollama", model_id="nomic-embed-text:latest", model_sha256="b" * 64,
        dimensions=2, vectors={active_chunk_id: [1.0, 0.0]},
    )
    assert store.reset_index()["embeddings"] == 1
    assert store.health()["embedding_count"] == 0


def test_index_active_embeddings_batches_provider_without_storing_provider_inputs(tmp_path):
    store = _store(tmp_path)
    _ingest(store, text="a" * 300)

    class ProviderResult:
        provider = "ollama"
        model_id = "nomic-embed-text:latest"
        dimensions = 2

        def __init__(self, vectors):
            self.vectors = vectors

    class Provider:
        calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return ProviderResult([[1.0, 0.0] for _ in texts])

    provider = Provider()
    assert store.index_active_embeddings(provider, model_sha256="c" * 64, batch_size=1) == 2
    assert len(provider.calls) == 2
    assert store.health()["embedding_count"] == 2
    assert store.index_active_embeddings(provider, model_sha256="c" * 64, source_id="docs-plan") == 2
    with pytest.raises(RagStoreError) as exc:
        store.index_active_embeddings(provider, model_sha256="invalid")
    assert exc.value.code == "embedding_model_digest_invalid"


def test_resumable_embedding_job_commits_batches_and_resumes(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    store.ingest_document(
        source_id="job-source", relative_ref="docs/job.md", sha256=None,
        mime="text/markdown", title="Job", text=("alpha beta gamma delta " * 40), revision="r1",
    )

    class Provider:
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return type("Result", (), {
                "provider": "ollama",
                "model_id": "nomic-embed-text:latest",
                "dimensions": 2,
                "vectors": [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
            })()

    provider = Provider()
    job = store.create_embedding_job(
        provider="ollama", model_sha256="d" * 64, source_id="job-source", batch_size=1,
    )
    first = store.run_embedding_job(job["job_id"], provider, max_batches=1)
    assert first["state"] == "paused"
    assert first["cursor"]["position"] == 1
    assert first["cursor"]["indexed"] == 1
    final = store.run_embedding_job(job["job_id"], provider)
    assert final["state"] == "completed"
    assert final["cursor"]["indexed"] == 4
    assert len(provider.calls) == 4
    assert store.health()["embedding_count"] == 4


def test_embedding_capacity_is_bounded_and_reports_vector_budget(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    store.ingest_document(
        source_id="capacity", relative_ref="docs/capacity.md", sha256=None,
        mime="text/markdown", title="Capacity", text="one two", revision="r1",
    )
    estimate = store.embedding_capacity(dimensions=768)
    assert estimate["active_chunk_count"] == 1
    assert estimate["estimated_vector_bytes"] == 768 * 4
    assert estimate["max_index_chunks"] == 10_000


def test_embedding_job_lease_blocks_second_worker_until_first_releases(tmp_path):
    import threading

    store_a = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    store_b = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    store_a.ingest_document(
        source_id="lease-source", relative_ref="docs/lease.md", sha256=None,
        mime="text/markdown", title="Lease", text="lease test", revision="r1",
    )
    job = store_a.create_embedding_job(provider="ollama", model_sha256="f" * 64, batch_size=1)
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider:
        def embed(self, texts):
            started.set()
            assert release.wait(5)
            return type("Result", (), {
                "provider": "ollama", "model_id": "nomic-embed-text:latest",
                "dimensions": 2, "vectors": [[1.0, 2.0]],
            })()

    errors = []

    def run_first():
        try:
            store_a.run_embedding_job(job["job_id"], BlockingProvider(), max_batches=1, lease_seconds=30)
        except Exception as exc:  # pragma: no cover - assertion below captures unexpected worker failures
            errors.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert started.wait(5)
    class FastProvider:
        def embed(self, texts):
            return type("Result", (), {
                "provider": "ollama", "model_id": "nomic-embed-text:latest",
                "dimensions": 2, "vectors": [[1.0, 2.0]],
            })()
    with pytest.raises(RagStoreError) as exc:
        store_b.run_embedding_job(job["job_id"], FastProvider(), max_batches=1, lease_seconds=30)
    assert exc.value.code == "rag_job_leased"
    release.set()
    worker.join(timeout=5)
    assert errors == []


def test_retryable_embedding_failure_can_resume_after_bounded_retry(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    store.ingest_document(
        source_id="retry-source", relative_ref="docs/retry.md", sha256=None,
        mime="text/markdown", title="Retry", text="retry test", revision="r1",
    )
    job = store.create_embedding_job(provider="ollama", model_sha256="a" * 64, batch_size=1)

    class RetryableError(RuntimeError):
        retryable = True
        code = "provider_timeout"

    class Provider:
        calls = 0
        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RetryableError("temporary")
            return type("Result", (), {
                "provider": "ollama", "model_id": "nomic-embed-text:latest",
                "dimensions": 2, "vectors": [[1.0, 2.0]],
            })()

    provider = Provider()
    result = store.run_embedding_job(job["job_id"], provider, max_batches=1, max_retries=1)
    assert result["state"] == "completed"
    assert provider.calls == 2
    assert result["attempts"] == 1


def test_s5a_job_schema_is_migrated_in_place(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE rag_jobs (job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, "
        "cursor TEXT NOT NULL, error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    )
    connection.close()
    store = RagStore(path)
    store.initialize()
    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(rag_jobs)").fetchall()}
    assert {"lease_owner", "lease_expires_at", "attempts"} <= columns
