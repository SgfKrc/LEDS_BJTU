#!/usr/bin/env python
"""edge_import_audit.py —— Edge 入口的 **import 闭包审计**（★ P0-3：动态导入审计的静态半边）。

为什么需要它（`docs/边缘最小发行包审计清单-2026-09-26.md` 的 P0 第 3 条）
----------------------------------------------------------------------------
SLIM/Edge 发行包**排除了 torch 系**（`qlh-slim.spec` 的 `excludes`），而 SLIM 实际跑的是
**源码** `uvicorn src.api_server:app`（不是打包 exe）⇒ **不能只靠 PyInstaller `excludes`**。
排除得不彻底时，**顶层 import 立即 ImportError**；排除得过头时，函数内的 torch 分支走到才崩。

既有覆盖的局限（本脚本要补的正是这些）
* `tests/test_qlh_edge.py::test_edge_source_has_no_forbidden_top_level_imports` —— 只扫
  **`src/qlh_edge.py` 一个文件**、且**只看 `tree.body`**（顶层）⇒ 传递依赖与函数内 import 全漏；
* `scripts/edge_preflight.py` —— **运行时**探针（在目标 venv 里 `import qlh_edge` 后查
  `sys.modules`），只覆盖 **`qlh_edge`** 一个入口，而 **SLIM 的入口是 `src.api_server`**。

本脚本的口径（**两遍 BFS**，这是关键）
--------------------------------------
* ``must_load``：从入口出发**只沿「无条件顶层 import」**传播 ⇒ 这些模块**导入入口时必然被加载**
  ⇒ 其中若有无条件顶层 `torch` 系 import ⇒ **导入即 ImportError** ⇒ **硬失败**。
* ``may_load``：再沿「`try/except ImportError` 兜住的顶层 import」与「函数内 import」传播
  ⇒ 多出来的部分是**延迟加载**（走到那条分支才加载）⇒ 记 **deferred**，不算硬失败。
* ``try/except ImportError`` 包住的 forbidden import ⇒ 归 **guarded**（有兜底，安全）。

⚠️ 教训（写进代码，避免后来者踩）：**只判"是不是顶层"不够** ——
`if` / `try` 块内的 import 在 AST 里也挂在模块体下；而"模块 B 被顶层 import"与
"模块 B 只被某函数的 import 引用"是**完全不同**的性质（前者导入即崩，后者延迟）。
本脚本用两遍 BFS 把这两者分开，就是为了避免把"延迟隐患"误报成"真阻断"。

用法::

    python scripts/edge_import_audit.py                       # 默认审 qlh_edge + src.api_server
    python scripts/edge_import_audit.py --entry qlh_edge --strict
    python scripts/edge_import_audit.py --json
    python scripts/edge_import_audit.py --list-modules        # 只列闭包，便于人工核对

退出码：``must_load`` 里出现无条件顶层 forbidden ⇒ 1；``--strict`` 下任何命中 ⇒ 1；否则 0。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (ROOT, ROOT / "src")

#: 与 `scripts/edge_preflight.py::FORBIDDEN_MODULES` 保持一致（两边都改，别只改一边）。
DEFAULT_FORBIDDEN = ("torch", "transformers", "accelerate", "bitsandbytes",
                     "pandas", "einops", "tiktoken")

#: 默认审计的入口：`qlh_edge` 是 Edge 最小入口（要求最严）；`src.api_server` 是 SLIM 真实入口。
DEFAULT_ENTRIES = ("qlh_edge", "src.api_server")


def module_file(name: str) -> Path | None:
    """把模块名解析成本地文件（只认本仓，不认 site-packages —— 那不属于我们的代码）。"""
    if not name:
        return None
    parts = name.split(".")
    for base in SEARCH_ROOTS:
        candidate = base.joinpath(*parts)
        if candidate.with_suffix(".py").is_file():
            return candidate.with_suffix(".py")
        if (candidate / "__init__.py").is_file():
            return candidate / "__init__.py"
    return None


class _Collector(ast.NodeVisitor):
    """按「加载性质」分开收集一个模块的 import。

    * `top`     —— 无条件顶层（导入模块即加载）
    * `guarded` —— 顶层、但被 `try/except ImportError` 兜住（尝试加载，失败不崩）
    * `inner`   —— 函数/方法内（延迟加载）
    """

    def __init__(self) -> None:
        self.top: list[tuple[str, int]] = []
        self.guarded: list[tuple[str, int]] = []
        self.inner: list[tuple[str, int]] = []
        self._fn_depth = 0
        self._guard_depth = 0

    def _record(self, raw: str, lineno: int) -> None:
        if self._fn_depth:
            self.inner.append((raw, lineno))
        elif self._guard_depth:
            self.guarded.append((raw, lineno))
        else:
            self.top.append((raw, lineno))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._record("." * node.level + node.module, node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn_depth += 1
        self.generic_visit(node)
        self._fn_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Try(self, node: ast.Try) -> None:
        catches_import_error = False
        for handler in node.handlers:
            exc = handler.type
            if exc is None:                      # 裸 except
                catches_import_error = True
                continue
            names: list[str] = []
            if isinstance(exc, ast.Name):
                names = [exc.id]
            elif isinstance(exc, ast.Tuple):
                names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
            if {"ImportError", "ModuleNotFoundError"} & set(names):
                catches_import_error = True
        if catches_import_error:
            self._guard_depth += 1
        for child in node.body:
            self.visit(child)
        if catches_import_error:
            self._guard_depth -= 1
        for handler in node.handlers:            # except 体按普通顶层处理
            for child in handler.body:
                self.visit(child)
        for child in (*node.orelse, *node.finalbody):
            self.visit(child)


def imports_of(path: Path) -> _Collector:
    collector = _Collector()
    collector.visit(ast.parse(path.read_text(encoding="utf-8")))
    return collector


def root_name(raw: str) -> str:
    return raw.lstrip(".").split(".")[0]


def _walk(entry: str, forbidden: set[str], *, follow) -> tuple[set[str], list[tuple[str, str, int]]]:
    """闭包 BFS。`follow` 决定从哪些 bucket 继续传播（这就是"两遍"的差别）。"""
    seen: set[str] = set()
    queue = [entry]
    hits: list[tuple[str, str, int]] = []
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = module_file(name)
        if path is None:
            continue
        collected = imports_of(path)
        for bucket in follow(collected):
            for raw, lineno in bucket:
                root = root_name(raw)
                if root in forbidden:
                    hits.append((name, root, lineno))
                elif module_file(root) is not None:
                    queue.append(root)
    return seen, hits


def audit(entry: str, forbidden: set[str]) -> dict:
    """返回审计结果：`must_load` / `may_load` / 三类命中 / 各自的证据行。"""
    must, must_hits = _walk(entry, forbidden, follow=lambda c: (c.top,))
    may, may_hits = _walk(entry, forbidden, follow=lambda c: (c.top, c.guarded, c.inner))

    # 归类：只被"延迟路径"引入的模块（may - must）里的命中算 deferred；其余按 top/guarded 分。
    top_hits: list[tuple[str, str, int, str]] = []      # (module, forbidden, lineno, 性质)
    for module, bad, lineno in must_hits:
        top_hits.append((module, bad, lineno, "must_load"))
    for module, bad, lineno in may_hits:
        if module in must:
            continue
        top_hits.append((module, bad, lineno, "deferred"))

    guarded_hits: list[tuple[str, str, int]] = []
    inner_hits: list[tuple[str, str, int]] = []
    for module in sorted(may):
        path = module_file(module)
        if path is None:
            continue
        collected = imports_of(path)
        for raw, lineno in collected.guarded:
            if root_name(raw) in forbidden:
                guarded_hits.append((module, root_name(raw), lineno))
        for raw, lineno in collected.inner:
            if root_name(raw) in forbidden:
                inner_hits.append((module, root_name(raw), lineno))

    return {
        "entry": entry,
        "must_load": sorted(must),
        "may_load": sorted(may),
        "blocking": [h for h in top_hits if h[3] == "must_load"],
        "deferred": [h for h in top_hits if h[3] == "deferred"],
        "guarded": guarded_hits,
        "inner": inner_hits,
    }


#: `--summary` 时每类命中最多带出多少条（够定位问题，又不至于撑爆调用方的尾部保留窗口）。
SUMMARY_HIT_CAP = 20


def _summarize(report: dict, cap: int = SUMMARY_HIT_CAP) -> dict:
    """把完整报告压成摘要：模块**数**代替全量列表，命中列表截断到 `cap` 条。

    存在的理由：`startup_matrix._run_subprocess` 只保留 stdout 的**尾部** `OUTPUT_TAIL` 字符，
    而 `must_load` / `may_load` 是整个闭包的模块名列表（实测 86 个）⇒ 全量 JSON 必然被截断。
    """
    return {
        "entry": report["entry"],
        "must_load": len(report["must_load"]),
        "may_load": len(report["may_load"]),
        "blocking": [list(hit) for hit in report["blocking"][:cap]],
        "blocking_count": len(report["blocking"]),
        "deferred": [list(hit) for hit in report["deferred"][:cap]],
        "deferred_count": len(report["deferred"]),
        "guarded": [list(hit) for hit in report["guarded"][:cap]],
        "guarded_count": len(report["guarded"]),
        "inner": [list(hit) for hit in report["inner"][:cap]],
        "inner_count": len(report["inner"]),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Edge 入口的 import 闭包审计（P0-3）")
    parser.add_argument("--entry", action="append", default=None,
                        help=f"入口模块（可多次）；默认 {list(DEFAULT_ENTRIES)}")
    parser.add_argument("--forbidden", action="append", default=None,
                        help=f"禁止的顶层模块名（可多次）；默认 {list(DEFAULT_FORBIDDEN)}")
    parser.add_argument("--strict", action="store_true",
                        help="任何命中（含 guarded / 函数内 / deferred）都算失败；用于最小入口")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--summary", action="store_true",
                        help="JSON 用紧凑摘要（模块数代替全量列表、命中截断到 20 条）—— "
                             "供 startup_matrix 等只保留尾部输出的调用方使用，避免被 OUTPUT_TAIL 截断")
    parser.add_argument("--list-modules", action="store_true", help="列出闭包模块，便于人工核对")
    args = parser.parse_args(argv)

    entries = tuple(args.entry) if args.entry else DEFAULT_ENTRIES
    forbidden = set(args.forbidden) if args.forbidden else set(DEFAULT_FORBIDDEN)

    reports: list[dict] = []
    failed = False
    for entry in entries:
        if module_file(entry) is None:
            print(f"[skip] 入口模块不存在：{entry}", file=sys.stderr)
            continue
        report = audit(entry, forbidden)
        reports.append(report)
        if report["blocking"] or (args.strict and (report["guarded"] or report["inner"]
                                                  or report["deferred"])):
            failed = True

    if args.json:
        payload = {"forbidden": sorted(forbidden)}
        if args.summary:
            payload["entries"] = [_summarize(report) for report in reports]
        else:
            payload["reports"] = reports
        # ★ 紧凑分隔符（不 indent）：`startup_matrix` 只保留 stdout 的**尾部** OUTPUT_TAIL 字符，
        #   缩进后的 JSON 会膨胀数倍而**从中间被截断**（实测踩到 ⇒ 表现为"JSON 非法"的假象）。
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return 1 if failed else 0

    for report in reports:
        print(f"\n=== {report['entry']} ===")
        print(f"  must_load（导入入口即加载）= {len(report['must_load'])} 个模块")
        print(f"  may_load（含延迟路径）      = {len(report['may_load'])} 个模块")
        if args.list_modules:
            for name in report["must_load"]:
                print(f"      [must] {name}")
            for name in report["may_load"]:
                if name not in report["must_load"]:
                    print(f"      [late] {name}")
        print(f"  ★ blocking（must_load 内无条件顶层命中）= {len(report['blocking'])}")
        for module, bad, lineno, _ in report["blocking"]:
            print(f"      {module}:{lineno} -> {bad}")
        print(f"  deferred（只在延迟路径上命中）        = {len(report['deferred'])}")
        for module, bad, lineno, _ in report["deferred"][:10]:
            print(f"      {module}:{lineno} -> {bad}")
        print(f"  guarded（try/except ImportError 兜住） = {len(report['guarded'])}")
        for module, bad, lineno in report["guarded"]:
            print(f"      {module}:{lineno} -> {bad}")
        print(f"  inner（函数内命中，延迟加载）         = {len(report['inner'])}")
        for module, bad, lineno in report["inner"][:10]:
            print(f"      {module}:{lineno} -> {bad}")
        if len(report["inner"]) > 10:
            print(f"      … 另 {len(report['inner']) - 10} 处")

    print(f"\n[verdict] {'FAIL' if failed else 'PASS'} "
          f"（{'--strict' if args.strict else '非 strict'}；"
          f"判据 = must_load 内不得有无条件顶层 forbidden import）")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
