#!/usr/bin/env python3
"""文档维护机械化扫描器（M1）——只读，不修改任何文档。

按《文档维护 Agent 工具设计》§3 实现 5 条矛盾检测规则：

  R1 完成未收口：状态行含"进行中/规划/待/Candidate"等，但正文含完成标记
  R2 未提交登记：当前文档在 git status 中有未提交改动
  R3 状态行滞后：文档"更新日期"早于粗关联源代码的最近提交日期
  R4 链接失效：docs/ 内部相对链接指向不存在的文件
  R5 状态行缺失：前 12 行找不到"状态"行（非豁免文档）

输出：build/doc-audit/audit.json（结构化）+ 核对清单（markdown，--list）。
入口示例：
  python docs/agent_tool/doc_maintenance_audit.py
  python docs/agent_tool/doc_maintenance_audit.py --json
  python docs/agent_tool/doc_maintenance_audit.py --since 7d
  python docs/agent_tool/doc_maintenance_audit.py --fail-on R4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
OUT_DIR = REPO_ROOT / "build" / "doc-audit"

# R1 命中词：状态行含这些词 → "疑似未收口"
# 注意：① 不用"待"（"待办/待验"是正常状态描述）；② 不用"进行中/实施中/In Progress/
# Open"（进行中的文档有部分完成记录是正常的）；③ 状态行自身已含完成标记 → 不命中
STALE_HINTS = ("规划", "Candidate", "Blocked")
# R1/R5 豁免：状态行含这些词 → 故意不更新的文档
EXEMPT_HINTS = (
    "历史参考", "待拆分", "不作为当前能力", "已废弃", "废弃", "冻结", "历史记录",
)
# 正文完成标记（前 200 行内）
DONE_MARKS = re.compile(
    r"✅|Completed|已完成|已关闭|已验收|开发门完成|阶段 2 完成|完成（", re.IGNORECASE
)
# R5：状态行
STATUS_RE = re.compile(r"(?:\*\*)?状态(?:\*\*)?[:：]")
LIFECYCLE_RE = re.compile(r"文档生命周期")
GIT_LOG_MARKER = "@@DOCAGENT_COMMIT@@"
TOPIC_STOP_WORDS = {
    "agent", "audit", "candidate", "completed", "design", "development",
    "document", "implementation", "maintenance", "model", "plan", "planning",
    "project", "status", "support", "tool", "update",
}


def _git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    return r.stdout.strip()


def _configure_text_stream(stream) -> None:
    """让 Windows/重定向输出稳定使用 UTF-8；不支持 reconfigure 时保持原样。"""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        return


def _dirty_doc_paths(repo_root: Path) -> dict[str, str]:
    """返回有工作区改动的 docs 路径及 porcelain 状态。"""
    output = _git([
        "-c", "core.quotepath=false", "status", "--short",
        "--untracked-files=all", "--", "docs/",
    ], repo_root)
    dirty: dict[str, str] = {}
    for line in output.splitlines():
        if len(line) < 4:
            continue
        state = line[:2]
        path_field = line[3:]
        # rename/copy 记录同时登记旧、新路径；普通含空格路径不会被拆分。
        for path in path_field.split(" -> "):
            normalized = path.strip().strip('"').replace("\\", "/")
            if normalized.startswith("docs/") and normalized.endswith(".md"):
                dirty[normalized] = state
    return dirty


def _source_changes(repo_root: Path, since_date: str) -> list[dict]:
    """一次读取 since_date 后的 src/ 提交，供全库 R3 粗关联复用。"""
    output = _git([
        "log", f"--since={since_date}T00:00:00", "--date=short",
        f"--format={GIT_LOG_MARKER}%H%x09%cs%x09%s", "--name-only", "--", "src/",
    ], repo_root)
    changes: list[dict] = []
    current: dict | None = None
    for line in output.splitlines():
        if line.startswith(GIT_LOG_MARKER):
            fields = line[len(GIT_LOG_MARKER):].split("\t", 2)
            if len(fields) != 3:
                current = None
                continue
            current = {
                "commit": fields[0], "date": fields[1], "subject": fields[2],
                "paths": [],
            }
            changes.append(current)
        elif current is not None and line.strip():
            current["paths"].append(line.strip().replace("\\", "/"))
    return changes


def _doc_topics(doc: Path, text: str) -> tuple[set[str], set[str]]:
    """从文件名、首个标题和显式 src/ 引用提取保守的 R3 关联线索。"""
    heading = next(
        (line.lstrip("# ").strip() for line in text.splitlines()
         if line.startswith("#")),
        "",
    )
    source_refs = {
        match.rstrip(".,;:)]}\"").lower()
        for match in re.findall(r"src/[A-Za-z0-9_./-]+", text)
    }
    token_text = " ".join((doc.stem, heading, *source_refs)).lower()
    tokens = {
        token for token in re.findall(r"[a-z][a-z0-9-]{3,}", token_text.replace("_", " "))
        if token not in TOPIC_STOP_WORDS
    }
    return tokens, source_refs


def _related_source_change(doc: Path, text: str, updated: str,
                           changes: list[dict]) -> tuple[dict, str] | None:
    tokens, source_refs = _doc_topics(doc, text)
    if not tokens and not source_refs:
        return None
    for change in changes:  # git log 为新到旧，首个匹配即最近提交
        if change["date"] <= updated:
            continue
        paths = [path.lower() for path in change["paths"]]
        for ref in source_refs:
            ref_prefix = ref.rstrip("/")
            matched = next((path for path in paths if path.startswith(ref_prefix)), None)
            if matched:
                return change, matched
        # M1 的关联仅使用 src 路径/文件名前缀；提交标题只作为输出证据，
        # 不参与匹配，避免标题里出现通用词时把无关文档一并标记。
        haystack = " ".join(paths)
        matched_token = next((token for token in sorted(tokens) if token in haystack), None)
        if matched_token:
            matched_path = next((path for path in paths if matched_token in path), "src/")
            return change, matched_path
    return None


def _status_line(text: str) -> str:
    """取前 12 行的状态行原文（跳过文档生命周期行）。"""
    for ln in text.splitlines()[:12]:
        if STATUS_RE.search(ln) and not LIFECYCLE_RE.search(ln):
            return ln.strip()
    return ""


def _updated_at(text: str) -> str | None:
    """取前 12 行的 更新日期/更新 字段（YYYY-MM-DD）。"""
    for ln in text.splitlines()[:12]:
        m = re.search(r"更新(?:日期|时间)?[:：]\s*(\d{4}-\d{2}-\d{2})", ln)
        if m:
            return m.group(1)
    return None


def _extract_links(text: str) -> list[tuple[str, str]]:
    """提取 markdown 相对链接 [text](path)，返回 (text, path)。"""
    links = []
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        href = m.group(2).strip()
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        # 去掉锚点
        href = href.split("#")[0]
        if href:
            links.append((m.group(1), href))
    return links


def _check_link(href: str, docs_dir: Path | None = None) -> bool:
    """docs/ 内相对链接存在性；../README.md 或 docs/ 同级。"""
    base = docs_dir or DOCS_DIR
    href = unquote(href)  # 处理 %20 等 URL 编码（markdown 链接常见）
    if href.startswith("../"):
        target = (base.parent / href[3:]).resolve()
    else:
        target = (base / href).resolve()
    return target.is_file()


def scan_doc(doc: Path, repo_root: Path, since: datetime | None,
             *, dirty_docs: dict[str, str] | None = None,
             source_changes: list[dict] | None = None) -> dict:
    """对单份文档执行 5 条规则，返回命中列表。"""
    text = doc.read_text(encoding="utf-8", errors="replace")
    rel = doc.relative_to(repo_root).as_posix()
    findings: list[dict] = []

    status = _status_line(text)
    updated = _updated_at(text)
    status_lower = status.lower()
    exempt = any(h.lower() in status_lower for h in EXEMPT_HINTS)

    # R5 状态行缺失（非豁免）
    if not status:
        if not exempt:
            findings.append({"rule": "R5", "level": "info",
                             "message": "前 12 行无状态行"})
    else:
        # R1 完成未收口（非豁免；只检查状态行主干=去括号内容，避免
        # "（…仍为规划…）"类括号说明误报；状态行自身已含完成标记 → 不命中）
        status_stem = re.sub(r"[（(][^（）()]*[）)]", "", status)
        stem_lower = status_stem.lower()
        if not exempt and not DONE_MARKS.search(status) and any(
                h.lower() in stem_lower for h in STALE_HINTS):
            body_head = "\n".join(text.splitlines()[:200])
            if DONE_MARKS.search(body_head):
                findings.append({
                    "rule": "R1", "level": "warn",
                    "message": f"状态行含未收口词但正文含完成标记：{status[:60]}",
                })

    # R3 状态行滞后：关联源代码提交晚于文档更新日期（保守粗关联，需人工确认）。
    if updated:
        changes = source_changes
        if changes is None:
            changes = _source_changes(repo_root, updated)
        related = _related_source_change(doc, text, updated, changes)
        if related:
            change, matched_path = related
            findings.append({
                "rule": "R3", "level": "info",
                "message": (
                    f"更新日期 {updated} 早于关联代码提交 "
                    f"{change['commit'][:12]} ({change['date']}, {matched_path})"
                ),
            })

    # R4 链接失效（以文档所在目录为基）
    for text_label, href in _extract_links(text):
        if not _check_link(href, doc.parent):
            findings.append({
                "rule": "R4", "level": "warn",
                "message": f"失效链接 [{text_label}]({href})",
            })

    # R2 未提交登记：仅给实际改动的文档命中，避免每份文档重复同一全局告警。
    dirty = dirty_docs if dirty_docs is not None else _dirty_doc_paths(repo_root)
    if rel in dirty:
        findings.append({
            "rule": "R2", "level": "warn",
            "message": f"当前文档有未提交改动（git status: {dirty[rel]}）",
        })

    return {
        "doc": rel,
        "status_line": status[:120],
        "updated_at": updated,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "findings": findings,
    }


def scan_all(since: timedelta | None, fail_level: str,
             out_dir: Path | None = None) -> dict:
    since_date = datetime.now() - since if since else None

    docs = []
    update_dates: list[str] = []
    for doc in sorted(DOCS_DIR.glob("*.md")):
        if since_date:
            mtime = datetime.fromtimestamp(doc.stat().st_mtime)
            if mtime < since_date:
                continue
        docs.append(doc)
        updated = _updated_at(doc.read_text(encoding="utf-8", errors="replace"))
        if updated:
            update_dates.append(updated)

    dirty_docs = _dirty_doc_paths(REPO_ROOT)
    source_changes = _source_changes(REPO_ROOT, min(update_dates)) if update_dates else []
    results = []
    for doc in docs:
        results.append(scan_doc(
            doc, REPO_ROOT, since_date,
            dirty_docs=dirty_docs, source_changes=source_changes,
        ))

    out = {
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "rules": {
            "R1": "完成未收口", "R2": "未提交登记", "R3": "状态行滞后",
            "R4": "链接失效", "R5": "状态行缺失",
        },
        "docs": results,
    }
    _write_outputs(out, out_dir or OUT_DIR)
    return out


def _write_outputs(out: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 核对清单 markdown
    lines = [
        "# 文档维护核对清单",
        "",
        f"> 生成时间：{out['run_ts']}（M1 机械化扫描，只读）",
        "",
        "## 命中汇总",
        "",
        "| 文档 | 规则 | 级别 | 证据 |",
        "|---|---|---|---|",
    ]
    for doc in out["docs"]:
        for f in doc["findings"]:
            lines.append(
                f"| {doc['doc']} | {f['rule']} | {f['level']} | "
                f"{f['message'].replace('|', '\\|').replace(chr(10), ' ')} |")
    (out_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _configure_text_stream(sys.stdout)
    _configure_text_stream(sys.stderr)
    ap = argparse.ArgumentParser(description="文档维护机械化扫描器（M1，只读）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 到 stdout")
    ap.add_argument("--since", metavar="DUR", help="只看近 N 天有改动的文档，如 7d")
    ap.add_argument("--llm", action="store_true", help="对 M1 疑似项执行可选 LLM 判定")
    ap.add_argument("--provider", choices=("opencode", "deepseek", "ollama"),
                    help="覆盖 .env.docagent 的 provider 选择（仅配合 --llm）")
    ap.add_argument("--cost", action="store_true", help="显示本次 --llm 缓存与调用计数")
    ap.add_argument("--apply", action="store_true",
                    help="只生成 suggestions.patch，不修改文档或执行 git apply")
    ap.add_argument("--index", action="store_true",
                    help="将本次快照写入本地 build/docagent-events.sqlite")
    ap.add_argument("--rebuild", action="store_true",
                    help="备份旧库并从 git 历史重建本地文档事件索引")
    ap.add_argument("--index-chunks", action="store_true",
                    help="建立或增量更新本地 FTS 文档分块索引")
    ap.add_argument("--search", metavar="QUERY", help="查询本地 FTS 文档候选")
    ap.add_argument("--search-limit", type=int, default=5, metavar="N",
                    help="--search 返回的候选数（默认 5）")
    ap.add_argument("--embed", action="store_true",
                    help="使用本地 Ollama embedding 增量建立 doc_embeddings")
    ap.add_argument("--semantic-search", metavar="QUERY",
                    help="使用本地 embedding 查询语义候选")
    ap.add_argument("--embedding-model", metavar="MODEL",
                    help="覆盖 DOCAGENT_OLLAMA_MODEL（仅 embedding 使用）")
    ap.add_argument(
        "--fail-on", choices=("none", "warn", "error", "R1", "R2", "R3", "R4", "R5"),
        default="error", help="命中该级别及以上或指定规则时返回非零",
    )
    args = ap.parse_args(argv)
    if args.apply and not args.llm:
        ap.error("--apply requires --llm and only generates a review patch")
    if args.index and args.rebuild:
        ap.error("--index and --rebuild are mutually exclusive")
    if args.search_limit <= 0:
        ap.error("--search-limit must be positive")

    since = None
    if args.since:
        m = re.fullmatch(r"(\d+)d", args.since)
        if not m:
            print(f"--since 格式错误: {args.since}（示例 7d）", file=sys.stderr)
            return 2
        since = timedelta(days=int(m.group(1)))

    out = scan_all(since, args.fail_on)
    if args.llm:
        from doc_maintenance_llm import load_docagent_config
        from doc_maintenance_runtime import run_llm_judgements

        try:
            config = load_docagent_config(REPO_ROOT / ".env.docagent", args.provider)
            out["llm"] = run_llm_judgements(
                out, REPO_ROOT, config, REPO_ROOT / "build" / "docagent-cache.sqlite",
            )
            if args.apply:
                from doc_maintenance_suggestions import generate_suggestion_diff

                out["suggestion_diff"] = generate_suggestion_diff(
                    out["llm"], REPO_ROOT, OUT_DIR,
                    confidence_floor=config.confidence_floor,
                )
            _write_outputs(out, OUT_DIR)
        except ValueError as exc:
            # 机械化扫描已经完成；不要让本地配置错误中断其结果。
            out["llm"] = {
                "enabled": False, "warnings": [f"invalid_docagent_config: {exc}"],
                "judgements": [],
            }
            _write_outputs(out, OUT_DIR)
    if args.index or args.rebuild or args.index_chunks or args.search or args.embed or args.semantic_search:
        from doc_maintenance_events import DocEventStore, rebuild_event_database

        database = REPO_ROOT / "build" / "docagent-events.sqlite"
        if args.rebuild:
            out["rebuild"] = rebuild_event_database(database, out, REPO_ROOT)
        else:
            with DocEventStore(database) as store:
                if args.index:
                    out["index"] = store.index_snapshot(out, REPO_ROOT)
                if args.index_chunks:
                    out["chunk_index"] = store.index_chunks(out, REPO_ROOT)
                if args.search:
                    out["search_results"] = store.search_chunks(args.search, args.search_limit)
                if args.embed or args.semantic_search:
                    from doc_maintenance_embeddings import (
                        EmbeddingUnavailable, OllamaEmbeddingProvider,
                    )
                    from doc_maintenance_llm import load_docagent_config

                    try:
                        config = load_docagent_config(REPO_ROOT / ".env.docagent")
                        provider = OllamaEmbeddingProvider(
                            base_url=config.ollama_base_url,
                            model=args.embedding_model or config.ollama_model,
                        )
                        if args.embed:
                            if "chunk_index" not in out:
                                out["chunk_index"] = store.index_chunks(out, REPO_ROOT)
                            out["embedding"] = store.index_embeddings(out, provider, REPO_ROOT)
                        if args.semantic_search:
                            out["semantic_results"] = store.semantic_search(
                                args.semantic_search, provider, args.search_limit,
                            )
                    except (EmbeddingUnavailable, ValueError):
                        out["embedding"] = {
                            "enabled": False,
                            "error": "embedding_provider_unavailable_or_invalid",
                        }
                        if args.semantic_search:
                            out["semantic_results"] = []
        _write_outputs(out, OUT_DIR)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        n = sum(len(d["findings"]) for d in out["docs"])
        print(f"扫描 {len(out['docs'])} 份文档，命中 {n} 项")
        for doc in out["docs"]:
            for f in doc["findings"]:
                print(f"  [{f['level']}] {doc['doc']} :: {f['rule']} {f['message']}")
        if args.cost and out.get("llm"):
            cost = out["llm"].get("cost")
            if cost:
                print(
                    "LLM: cache_hits={cache_hits}, provider_calls={provider_calls}, "
                    "needs_review={needs_review}".format(**cost)
                )
        print(f"清单: {OUT_DIR / 'audit.md'}")
    findings = [finding for doc in out["docs"] for finding in doc["findings"]]
    if args.fail_on.startswith("R"):
        return 1 if any(finding["rule"] == args.fail_on for finding in findings) else 0
    if args.fail_on == "warn":
        return 1 if any(finding["level"] in {"warn", "error"} for finding in findings) else 0
    if args.fail_on == "error":
        return 1 if any(finding["level"] == "error" for finding in findings) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
