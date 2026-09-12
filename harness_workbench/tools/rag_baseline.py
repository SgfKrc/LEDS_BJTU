"""Frozen, model-free retrieval baseline for the main project and harness stores.

The baseline deliberately exercises only the existing SQLite FTS5 contracts. A
single local fixture corpus and a 30-case question set are ingested into both
stores, then compared using source-reference hit@k and mean reciprocal rank.
Reports contain digests and ranks, never query text, document text, or temp
paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness_workbench.rag.chunking import CHUNK_STRATEGIES
from harness_workbench.rag.query import rewrite_query as harness_rewrite_query


RAG_BASELINE_INPUT_SCHEMA = "qlh.rag_baseline_input.v1"
RAG_BASELINE_SCHEMA = "qlh.rag_baseline.v1"
RAG_CHUNK_COMPARISON_SCHEMA = "qlh.rag_chunk_comparison.v1"
RAG_QUERY_COMPARISON_SCHEMA = "qlh.rag_query_comparison.v1"
BASELINE_CASE_COUNT = 30
_MAX_TEXT_CHARS = 16_384
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_REF_PARTS = frozenset({".git", ".ssh", "credentials", "private", "secrets"})


class RagBaselineError(ValueError):
    """Stable validation error for the frozen baseline contract."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _query_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_ref(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagBaselineError("source_ref_invalid", "source_ref is required")
    ref = value.strip().replace("\\", "/")
    if ref.startswith("/") or re.match(r"^[A-Za-z]:", ref) or "//" in ref or "://" in ref:
        raise RagBaselineError("source_ref_invalid", "source_ref must be repository-relative")
    parts = [part for part in ref.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise RagBaselineError("source_ref_invalid", "source_ref contains traversal")
    lowered = {part.lower() for part in parts}
    if lowered & _SENSITIVE_REF_PARTS or any(part.lower().endswith((".key", ".pem", ".p12", ".pfx")) for part in parts):
        raise RagBaselineError("sensitive_source", "source_ref points to sensitive material")
    return "/".join(parts)


def _validate_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RagBaselineError("identifier_invalid", f"{field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class RagBaselineDocument:
    document_id: str
    source_ref: str
    title: str
    text: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RagBaselineDocument":
        if not isinstance(value, Mapping):
            raise RagBaselineError("document_invalid", "document must be an object")
        document_id = _validate_identifier(value.get("document_id"), field="document_id")
        source_ref = _validate_ref(value.get("source_ref"))
        title = value.get("title")
        text = value.get("text")
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 256:
            raise RagBaselineError("document_invalid", "document title is invalid")
        if not isinstance(text, str) or not text.strip() or len(text) > _MAX_TEXT_CHARS or "\x00" in text:
            raise RagBaselineError("document_invalid", "document text is invalid")
        if re.search(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|(?:api[_ -]?key|authorization|password)\s*[:=]", text, re.I):
            raise RagBaselineError("sensitive_content", "document text looks like secret material")
        return cls(document_id, source_ref, title.strip(), text)

    def as_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "source_ref": self.source_ref,
            "title": self.title,
            "text": self.text,
        }

    def __post_init__(self) -> None:
        _validate_identifier(self.document_id, field="document_id")
        _validate_ref(self.source_ref)
        if not isinstance(self.title, str) or not self.title.strip() or len(self.title.strip()) > 256:
            raise RagBaselineError("document_invalid", "document title is invalid")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > _MAX_TEXT_CHARS or "\x00" in self.text:
            raise RagBaselineError("document_invalid", "document text is invalid")
        if re.search(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|(?:api[_ -]?key|authorization|password)\s*[:=]", self.text, re.I):
            raise RagBaselineError("sensitive_content", "document text looks like secret material")


@dataclass(frozen=True, slots=True)
class RagBaselineCase:
    case_id: str
    query: str
    target_source_refs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RagBaselineCase":
        if not isinstance(value, Mapping):
            raise RagBaselineError("case_invalid", "case must be an object")
        case_id = _validate_identifier(value.get("case_id"), field="case_id")
        query = value.get("query")
        targets = value.get("target_source_refs")
        if not isinstance(query, str) or not query.strip() or len(query) > 512 or "\x00" in query:
            raise RagBaselineError("case_invalid", "query is invalid")
        if not isinstance(targets, (list, tuple)) or not targets:
            raise RagBaselineError("case_invalid", "case needs target source references")
        normalized = tuple(_validate_ref(item) for item in targets)
        if len(set(normalized)) != len(normalized):
            raise RagBaselineError("case_invalid", "target source references must be unique")
        return cls(case_id, query.strip(), normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "target_source_refs": list(self.target_source_refs),
        }

    def __post_init__(self) -> None:
        _validate_identifier(self.case_id, field="case_id")
        if not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 512 or "\x00" in self.query:
            raise RagBaselineError("case_invalid", "query is invalid")
        if not isinstance(self.target_source_refs, tuple) or not self.target_source_refs:
            raise RagBaselineError("case_invalid", "case needs target source references")
        normalized = tuple(_validate_ref(item) for item in self.target_source_refs)
        if len(set(normalized)) != len(normalized):
            raise RagBaselineError("case_invalid", "target source references must be unique")


def _validate_fixture(documents: Sequence[RagBaselineDocument], cases: Sequence[RagBaselineCase]) -> tuple[tuple[RagBaselineDocument, ...], tuple[RagBaselineCase, ...]]:
    docs = tuple(documents)
    queries = tuple(cases)
    if not docs or len(docs) > 100:
        raise RagBaselineError("document_count_invalid", "baseline needs a bounded non-empty document set")
    if len(queries) != BASELINE_CASE_COUNT:
        raise RagBaselineError("case_count_mismatch", f"baseline requires exactly {BASELINE_CASE_COUNT} cases")
    if len({doc.document_id for doc in docs}) != len(docs) or len({doc.source_ref for doc in docs}) != len(docs):
        raise RagBaselineError("document_duplicate", "document ids and source refs must be unique")
    if len({case.case_id for case in queries}) != len(queries):
        raise RagBaselineError("case_duplicate", "case ids must be unique")
    refs = {doc.source_ref for doc in docs}
    if any(target not in refs for case in queries for target in case.target_source_refs):
        raise RagBaselineError("target_unknown", "a case references a document outside the corpus")
    return docs, queries


def builtin_rag_baseline_documents() -> tuple[RagBaselineDocument, ...]:
    specs = (
        ("ragbase_doc_01", "fixtures/rag_base/01-alpha.md", "RAG baseline alpha", "Offline retrieval fixture for the alpha lane. Marker ragbaseanchoralpha identifies this document and its source boundary."),
        ("ragbase_doc_02", "fixtures/rag_base/02-bravo.md", "RAG baseline bravo", "Offline retrieval fixture for the bravo lane. Marker ragbaseanchorbravo identifies this document and its source boundary."),
        ("ragbase_doc_03", "fixtures/rag_base/03-charlie.md", "RAG baseline charlie", "Offline retrieval fixture for the charlie lane. Marker ragbaseanchorcharlie identifies this document and its source boundary."),
        ("ragbase_doc_04", "fixtures/rag_base/04-delta.md", "RAG baseline delta", "Offline retrieval fixture for the delta lane. Marker ragbaseanchordelta identifies this document and its source boundary."),
        ("ragbase_doc_05", "fixtures/rag_base/05-echo.md", "RAG baseline echo", "Offline retrieval fixture for the echo lane. Marker ragbaseanchorecho identifies this document and its source boundary."),
        ("ragbase_doc_06", "fixtures/rag_base/06-foxtrot.md", "RAG baseline foxtrot", "Offline retrieval fixture for the foxtrot lane. Marker ragbaseanchorfoxtrot identifies this document and its source boundary."),
    )
    return tuple(RagBaselineDocument(*item) for item in specs)


def builtin_rag_baseline_cases() -> tuple[RagBaselineCase, ...]:
    docs = builtin_rag_baseline_documents()
    cases: list[RagBaselineCase] = []
    for index in range(BASELINE_CASE_COUNT):
        doc = docs[index % len(docs)]
        marker = doc.text.split("Marker ", 1)[1].split(" ", 1)[0]
        cases.append(RagBaselineCase(f"ragbase_case_{index + 1:02d}", marker, (doc.source_ref,)))
    return tuple(cases)


def load_rag_baseline_input(path: str | Path) -> tuple[tuple[RagBaselineDocument, ...], tuple[RagBaselineCase, ...]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RagBaselineError("input_unreadable", "baseline input is unreadable") from exc
    if not isinstance(raw, Mapping) or raw.get("schema", RAG_BASELINE_INPUT_SCHEMA) != RAG_BASELINE_INPUT_SCHEMA:
        raise RagBaselineError("input_schema_invalid", "baseline input schema is invalid")
    raw_documents = raw.get("documents")
    raw_cases = raw.get("cases")
    if not isinstance(raw_documents, list) or not isinstance(raw_cases, list):
        raise RagBaselineError("input_invalid", "baseline input needs documents and cases lists")
    documents = tuple(RagBaselineDocument.from_mapping(item) for item in raw_documents)
    cases = tuple(RagBaselineCase.from_mapping(item) for item in raw_cases)
    return _validate_fixture(documents, cases)


def _evaluate(
    cases: Sequence[RagBaselineCase],
    search: Callable[[str, int], Sequence[Mapping[str, Any]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    hit_count = 0
    reciprocal_total = 0.0
    for case in cases:
        rows = list(search(case.query, top_k))[:top_k]
        targets = set(case.target_source_refs)
        rank: int | None = None
        for index, row in enumerate(rows, start=1):
            ref = str(row.get("source_ref", ""))
            if ref in targets:
                rank = index
                break
        if rank is not None:
            hit_count += 1
            reciprocal_total += 1.0 / rank
        details.append({
            "case_id": case.case_id,
            "query_sha256": _query_digest(case.query),
            "target_source_refs": list(case.target_source_refs),
            "hit": rank is not None,
            "rank": rank,
            "returned_count": len(rows),
        })
    count = len(cases)
    return {
        "case_count": count,
        "top_k": top_k,
        "hit_at_k": round(hit_count / count if count else 0.0, 6),
        "mean_reciprocal_rank": round(reciprocal_total / count if count else 0.0, 6),
        "details": details,
    }


@dataclass(frozen=True, slots=True)
class RagBaselineReport:
    corpus_digest: str
    case_set_digest: str
    document_count: int
    case_count: int
    top_k: int
    sides: Mapping[str, Mapping[str, Any]]
    details_match: bool
    schema: str = RAG_BASELINE_SCHEMA
    model_used: bool = False
    network_used: bool = False

    @property
    def valid(self) -> bool:
        expected_sides = {"main_project", "harness"}
        side_metrics_valid = all(
            0.0 <= float(side.get("hit_at_k", -1.0)) <= 1.0
            and 0.0 <= float(side.get("mean_reciprocal_rank", -1.0)) <= 1.0
            and int(side.get("case_count", -1)) == self.case_count
            for side in self.sides.values()
        )
        return (
            self.schema == RAG_BASELINE_SCHEMA
            and self.document_count > 0
            and self.case_count == BASELINE_CASE_COUNT
            and self.details_match
            and set(self.sides) == expected_sides
            and side_metrics_valid
            and not self.model_used
            and not self.network_used
            and all(float(side.get("hit_at_k", 0.0)) >= 0.0 for side in self.sides.values())
        )

    @property
    def report_digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "status": "passed" if self.valid else "failed",
            "valid": self.valid,
            "corpus_digest": self.corpus_digest,
            "case_set_digest": self.case_set_digest,
            "document_count": self.document_count,
            "case_count": self.case_count,
            "top_k": self.top_k,
            "sides": {name: dict(side) for name, side in sorted(self.sides.items())},
            "comparability": {"details_match": self.details_match, "same_corpus": True, "same_case_set": True},
            "model_used": self.model_used,
            "network_used": self.network_used,
        }
        if include_digest:
            value["report_digest"] = self.report_digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# RAG baseline",
            "",
            f"- Status: `{self.as_dict(include_digest=False)['status']}`; documents: `{self.document_count}`; cases: `{self.case_count}`; top-k: `{self.top_k}`",
            f"- Corpus digest: `{self.corpus_digest}`; case-set digest: `{self.case_set_digest}`",
            "- Mode: local SQLite FTS5 only; model used: `false`; network used: `false`",
            "",
            "| side | hit@k | MRR | cases |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, side in sorted(self.sides.items()):
            lines.append(f"| `{name}` | {side['hit_at_k']:.6f} | {side['mean_reciprocal_rank']:.6f} | {side['case_count']} |")
        lines.extend(("", f"Details identical: `{str(self.details_match).lower()}`; report digest: `{self.report_digest}`", ""))
        return "\n".join(lines)


def run_rag_baseline(
    *,
    documents: Sequence[RagBaselineDocument] | None = None,
    cases: Sequence[RagBaselineCase] | None = None,
    top_k: int = 5,
) -> RagBaselineReport:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise RagBaselineError("config_invalid", "top_k must be between 1 and 100")
    raw_documents = documents if documents is not None else builtin_rag_baseline_documents()
    raw_cases = cases if cases is not None else builtin_rag_baseline_cases()
    docs, queries = _validate_fixture(
        tuple(item if isinstance(item, RagBaselineDocument) else RagBaselineDocument.from_mapping(item) for item in raw_documents),
        tuple(item if isinstance(item, RagBaselineCase) else RagBaselineCase.from_mapping(item) for item in raw_cases),
    )
    corpus_digest = _digest([doc.as_dict() for doc in docs])
    case_set_digest = _digest([case.as_dict() for case in queries])
    with tempfile.TemporaryDirectory(prefix="qlh-rag-base-", ignore_cleanup_errors=True) as root:
        root_path = Path(root)
        from src.rag_store import RagStore as MainRagStore
        from harness_workbench.rag.store import RagStore as HarnessRagStore

        main_store = MainRagStore(root_path / "main.sqlite", max_chunk_chars=1024, chunk_overlap_chars=80)
        main_store.initialize()
        harness_store = HarnessRagStore(root_path / "harness.sqlite")
        harness_source_ids: dict[str, str] = {}
        for doc in docs:
            harness_result = harness_store.add_document(
                source_ref=doc.source_ref, title=doc.title, text=doc.text,
                owner_scope="project", max_chars=1024, overlap_chars=80, strategy="fixed",
            )
            harness_source_ids[doc.source_ref] = str(harness_result["source_id"])
            main_store.ingest_document(
                source_id=doc.document_id, relative_ref=doc.source_ref, sha256=None,
                mime="text/markdown", title=doc.title, text=doc.text, revision="baseline-v1",
                language="en", owner_scope="project", access_scope="project",
                metadata={"fixture": "rag-base-v1"}, strategy="fixed",
            )

        def main_search(query: str, limit: int) -> list[dict[str, Any]]:
            return [
                {"source_ref": str(row["relative_ref"]), "chunk_id": str(row["chunk_id"])}
                for row in main_store.search(query, access_scope="project", limit=limit)
            ]

        def harness_search(query: str, limit: int) -> list[dict[str, Any]]:
            rows = harness_store.search(query, owner_scope="project", limit=limit)
            return [
                {"source_ref": next(ref for ref, source_id in harness_source_ids.items() if source_id == hit.source_id), "chunk_id": hit.chunk_id}
                for hit in rows
            ]

        main_result = _evaluate(queries, main_search, top_k=top_k)
        harness_result = _evaluate(queries, harness_search, top_k=top_k)
        del main_search, harness_search, main_store, harness_store
        gc.collect()
    detail_signature = lambda item: [(entry["case_id"], entry["query_sha256"], entry["target_source_refs"], entry["hit"], entry["rank"], entry["returned_count"]) for entry in item["details"]]
    details_match = detail_signature(main_result) == detail_signature(harness_result)
    return RagBaselineReport(
        corpus_digest=corpus_digest, case_set_digest=case_set_digest,
        document_count=len(docs), case_count=len(queries), top_k=top_k,
        sides={"main_project": main_result, "harness": harness_result}, details_match=details_match,
    )


def run_rag_chunk_comparison(
    *,
    documents: Sequence[RagBaselineDocument] | None = None,
    cases: Sequence[RagBaselineCase] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate every deterministic chunk strategy with the frozen case set."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise RagBaselineError("config_invalid", "top_k must be between 1 and 100")
    raw_documents = documents if documents is not None else builtin_rag_baseline_documents()
    raw_cases = cases if cases is not None else builtin_rag_baseline_cases()
    docs, queries = _validate_fixture(
        tuple(item if isinstance(item, RagBaselineDocument) else RagBaselineDocument.from_mapping(item) for item in raw_documents),
        tuple(item if isinstance(item, RagBaselineCase) else RagBaselineCase.from_mapping(item) for item in raw_cases),
    )
    corpus_digest = _digest([doc.as_dict() for doc in docs])
    case_set_digest = _digest([case.as_dict() for case in queries])
    comparisons: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="qlh-rag-chunk-") as root:
        root_path = Path(root)
        from src.rag_store import RagStore as MainRagStore
        from harness_workbench.rag.store import RagStore as HarnessRagStore

        for strategy in sorted(CHUNK_STRATEGIES):
            main_store = MainRagStore(root_path / f"main-{strategy}.sqlite", max_chunk_chars=1024, chunk_overlap_chars=80)
            main_store.initialize()
            harness_store = HarnessRagStore(root_path / f"harness-{strategy}.sqlite")
            harness_source_ids: dict[str, str] = {}
            for doc in docs:
                harness_result = harness_store.add_document(
                    source_ref=doc.source_ref, title=doc.title, text=doc.text,
                    owner_scope="project", max_chars=1024, overlap_chars=80, strategy=strategy,
                )
                harness_source_ids[doc.source_ref] = str(harness_result["source_id"])
                main_store.ingest_document(
                    source_id=doc.document_id, relative_ref=doc.source_ref, sha256=None,
                    mime="text/markdown", title=doc.title, text=doc.text, revision="baseline-v1",
                    language="en", owner_scope="project", access_scope="project",
                    metadata={"fixture": "rag-base-v1"}, strategy=strategy,
                )

            def main_search(query: str, limit: int) -> list[dict[str, Any]]:
                return [
                    {"source_ref": str(row["relative_ref"]), "chunk_id": str(row["chunk_id"])}
                    for row in main_store.search(query, access_scope="project", limit=limit)
                ]

            def harness_search(query: str, limit: int) -> list[dict[str, Any]]:
                rows = harness_store.search(query, owner_scope="project", limit=limit)
                return [
                    {"source_ref": next(ref for ref, source_id in harness_source_ids.items() if source_id == hit.source_id), "chunk_id": hit.chunk_id}
                    for hit in rows
                ]

            main_result = _evaluate(queries, main_search, top_k=top_k)
            harness_result = _evaluate(queries, harness_search, top_k=top_k)
            detail_signature = lambda item: [
                (entry["case_id"], entry["query_sha256"], entry["target_source_refs"], entry["hit"], entry["rank"], entry["returned_count"])
                for entry in item["details"]
            ]
            comparisons[strategy] = {
                "main_project": main_result,
                "harness": harness_result,
                "details_match": detail_signature(main_result) == detail_signature(harness_result),
            }
            del main_search, harness_search, main_store, harness_store
            gc.collect()
    return {
        "schema": RAG_CHUNK_COMPARISON_SCHEMA,
        "status": "passed" if all(item["details_match"] for item in comparisons.values()) else "failed",
        "valid": all(item["details_match"] for item in comparisons.values()),
        "corpus_digest": corpus_digest,
        "case_set_digest": case_set_digest,
        "document_count": len(docs),
        "case_count": len(queries),
        "top_k": top_k,
        "strategies": comparisons,
        "model_used": False,
        "network_used": False,
    }


def run_rag_query_comparison(
    *,
    documents: Sequence[RagBaselineDocument] | None = None,
    cases: Sequence[RagBaselineCase] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare original and deterministic rewritten queries on both stores."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise RagBaselineError("config_invalid", "top_k must be between 1 and 100")
    raw_documents = documents if documents is not None else builtin_rag_baseline_documents()
    raw_cases = cases if cases is not None else builtin_rag_baseline_cases()
    docs, queries = _validate_fixture(
        tuple(item if isinstance(item, RagBaselineDocument) else RagBaselineDocument.from_mapping(item) for item in raw_documents),
        tuple(item if isinstance(item, RagBaselineCase) else RagBaselineCase.from_mapping(item) for item in raw_cases),
    )
    corpus_digest = _digest([doc.as_dict() for doc in docs])
    case_set_digest = _digest([case.as_dict() for case in queries])
    with tempfile.TemporaryDirectory(prefix="qlh-rag-query-") as root:
        root_path = Path(root)
        from src.rag_store import RagStore as MainRagStore
        from harness_workbench.rag.store import RagStore as HarnessRagStore

        main_store = MainRagStore(root_path / "main.sqlite", max_chunk_chars=1024, chunk_overlap_chars=80)
        main_store.initialize()
        harness_store = HarnessRagStore(root_path / "harness.sqlite")
        harness_source_refs: dict[str, str] = {}
        for doc in docs:
            harness_result = harness_store.add_document(
                source_ref=doc.source_ref, title=doc.title, text=doc.text,
                owner_scope="project", max_chars=1024, overlap_chars=80, strategy="fixed",
            )
            harness_source_refs[str(harness_result["source_id"])] = doc.source_ref
            main_store.ingest_document(
                source_id=doc.document_id, relative_ref=doc.source_ref, sha256=None,
                mime="text/markdown", title=doc.title, text=doc.text, revision="baseline-v1",
                language="en", owner_scope="project", access_scope="project",
                metadata={"fixture": "rag-base-v1"}, strategy="fixed",
            )

        def main_search(query: str, limit: int, rewritten: bool) -> list[dict[str, Any]]:
            variants = harness_rewrite_query(query).variants if rewritten else (query,)
            merged: dict[str, dict[str, Any]] = {}
            for variant in variants:
                for row in main_store.search(variant, access_scope="project", limit=limit):
                    merged.setdefault(str(row["chunk_id"]), {"source_ref": str(row["relative_ref"]), "chunk_id": str(row["chunk_id"])})
            return list(merged.values())[:limit]

        def harness_search(query: str, limit: int, rewritten: bool) -> list[dict[str, Any]]:
            variants = harness_rewrite_query(query).variants if rewritten else (query,)
            merged: dict[str, dict[str, Any]] = {}
            for variant in variants:
                for hit in harness_store.search(variant, owner_scope="project", limit=limit):
                    merged.setdefault(hit.chunk_id, {"source_ref": harness_source_refs[hit.source_id], "chunk_id": hit.chunk_id})
            return list(merged.values())[:limit]

        sides: dict[str, Any] = {}
        for name, searcher in (("main_project", main_search), ("harness", harness_search)):
            baseline = _evaluate(queries, lambda query, limit: searcher(query, limit, False), top_k=top_k)
            rewritten = _evaluate(queries, lambda query, limit: searcher(query, limit, True), top_k=top_k)
            sides[name] = {
                "baseline": baseline,
                "rewritten": rewritten,
                "delta": {
                    "hit_at_k": round(float(rewritten["hit_at_k"]) - float(baseline["hit_at_k"]), 6),
                    "mean_reciprocal_rank": round(float(rewritten["mean_reciprocal_rank"]) - float(baseline["mean_reciprocal_rank"]), 6),
                },
                "details_match": [
                    (entry["case_id"], entry["query_sha256"], entry["target_source_refs"])
                    for entry in baseline["details"]
                ] == [
                    (entry["case_id"], entry["query_sha256"], entry["target_source_refs"])
                    for entry in rewritten["details"]
                ],
            }
        del main_search, harness_search, main_store, harness_store
        gc.collect()
    return {
        "schema": RAG_QUERY_COMPARISON_SCHEMA,
        "status": "passed" if all(side["details_match"] for side in sides.values()) else "failed",
        "valid": all(side["details_match"] for side in sides.values()),
        "corpus_digest": corpus_digest,
        "case_set_digest": case_set_digest,
        "document_count": len(docs),
        "case_count": len(queries),
        "top_k": top_k,
        "sides": sides,
        "model_used": False,
        "network_used": False,
    }


def _write_report(report: RagBaselineReport, json_path: Path | None, markdown_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report.to_markdown(), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline dual-side RAG baseline")
    parser.add_argument("--input", type=Path, help="optional frozen baseline input JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", type=Path, help="write redacted JSON report")
    parser.add_argument("--markdown", type=Path, help="write redacted Markdown report")
    args = parser.parse_args(argv)
    documents, cases = load_rag_baseline_input(args.input) if args.input else (None, None)
    report = run_rag_baseline(documents=documents, cases=cases, top_k=args.top_k)
    _write_report(report, args.json, args.markdown)
    if args.json is None and args.markdown is None:
        print(report.to_markdown())
    return 0 if report.valid else 1


__all__ = [
    "BASELINE_CASE_COUNT", "RAG_BASELINE_INPUT_SCHEMA", "RAG_BASELINE_SCHEMA", "RAG_CHUNK_COMPARISON_SCHEMA", "RAG_QUERY_COMPARISON_SCHEMA",
    "RagBaselineCase", "RagBaselineDocument", "RagBaselineError", "RagBaselineReport",
    "builtin_rag_baseline_cases", "builtin_rag_baseline_documents", "load_rag_baseline_input",
    "run_rag_baseline",
    "run_rag_chunk_comparison", "run_rag_query_comparison",
]


if __name__ == "__main__":
    raise SystemExit(main())
