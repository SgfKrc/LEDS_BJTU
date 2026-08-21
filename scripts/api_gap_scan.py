# -*- mode: python ; coding: utf-8 -*-
"""接口契约对齐审计：一键导出前后端接口清单并机械比对缺口（静态、离线）。

用法：
    python scripts/api_gap_scan.py            # 默认：单体后端 × 两端前端
    python scripts/api_gap_scan.py --include-gateway   # 加 TS 网关面
    python scripts/api_gap_scan.py --json -o api-gap.json
    python scripts/api_gap_scan.py --only chat          # 过滤子串

产出（schema v1）：
    inventory: {backend: [{method, path, source, file, line}...],
                frontend: [{method, path, source, file, line, kind}...]}
    gaps:
      frontend_without_backend (缺口 A：前端调、后端未定义 → 404 风险)
      backend_without_frontend (缺口 B：后端定义、前端未消费 → 功能缺口)
      method_mismatch         (软缺口：同路径、方法不一致)
    summary: 计数与基线

退出码：0=对齐（无 A/B）；1=有缺口 A；2=有缺口 B；3=A+B；10=解析中途失败。
method 推断：前端缺省 GET；`?` 表示无法静态确定（比对时匹配任意后端方法）。
边界：只扫字面量/模板字面量，动态拼路径不计入；只读，不改任何源文件。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
HTTP_METHODS_TS = {"Get", "Post", "Put", "Delete", "Patch", "Head", "Options"}


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """归一化：/api 前缀统一、去 query、参数占位统一、去尾斜杠。

    模板 **查询拼接**（``sessions${query}``、``recent${q?'?'+q:''}``）在参数
    占位后紧贴上一段（无 ``/`` 分隔），整体丢弃；真路径参数 ``/sessions/${id}``
    有 ``/`` 分隔保留为 ``{param}``。
    """
    p = path.strip()
    p = re.sub(r"\$\{[^}]*\}", "{param}", p)                 # ${...} → {param}
    p = re.sub(r"\{[^}]*\}", "{param}", p)                    # {id} → {param}
    p = re.sub(r"/:([A-Za-z_][A-Za-z0-9_]*)", "/{param}", p)  # /:id → /{param}
    p = p.split("?", 1)[0].strip()                            # 去 query（须在 ${} 之后）
    if not p.startswith("/"):
        p = "/" + p
    p = re.sub(r"(?<!/)\{param\}$", "", p)                    # query 拼接尾段丢弃
    p = re.sub(r"/+", "/", p)
    while len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p


def ensure_api_prefix(path: str) -> str:
    """前端引用统一补 /api 前缀（旧前端用 /auth/...，cybergothic 用 BASE='/api'+path）。"""
    p = normalize_path(path)
    if p == "/api" or p.startswith("/api/"):
        return p
    return "/api" + p


# --------------------------------------------------------------------------
# 后端导出：FastAPI（AST 静态提取）
# --------------------------------------------------------------------------

@dataclass
class BackendEntry:
    method: str
    path: str
    source: str
    file: str
    line: int


_FRAMEWORK_INSTANCES = {"FastAPI", "APIRouter"}


def _collect_fastapi_instances(tree: ast.Module) -> dict[str, str]:
    """模块级 FastAPI/APIRouter 实例名 → 路径前缀。

    ``APIRouter(prefix="/v1")`` 会把装饰器路径整体加前缀（如 inference_service
    的 ``/v1`` 面）；提取时静态展开，避免伪缺口。
    """
    out: dict[str, str] = {}

    def _prefix_of(call: ast.Call) -> str:
        for kw in call.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                return kw.value.value
        return ""

    for node in tree.body:
        if isinstance(node, ast.Assign) and node.value and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id in _FRAMEWORK_INSTANCES:
            prefix = _prefix_of(node.value)
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = prefix
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id in _FRAMEWORK_INSTANCES \
                and isinstance(node.target, ast.Name):
            out[node.target.id] = _prefix_of(node.value)
    return out


def extract_fastapi(file: Path) -> list[BackendEntry]:
    text = file.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    instances = _collect_fastapi_instances(tree)
    if not instances:
        instances = {"app": "", "router": ""}
    out: list[BackendEntry] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if dec.func.attr not in HTTP_METHODS:
                continue
            owner = dec.func.value
            if not isinstance(owner, ast.Name) or owner.id not in instances:
                continue
            if not dec.args or not isinstance(dec.args[0], ast.Constant) \
                    or not isinstance(dec.args[0].value, str):
                continue
            prefix = (instances[owner.id] or "").rstrip("/")
            raw = dec.args[0].value
            path = prefix + ("/" + raw.lstrip("/") if raw else "")
            out.append(BackendEntry(
                method=dec.func.attr.upper(),
                path=path,
                source=file.name,
                file=_rel_or_abs(file),
                line=node.lineno,
            ))
    return out


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def extract_backend_python() -> list[BackendEntry]:
    src = REPO_ROOT / "src"
    out: list[BackendEntry] = []
    for py in sorted(src.rglob("*.py")):
        if "__pycache__" in py.parts or py.name.startswith("test_"):
            continue
        try:
            out.extend(extract_fastapi(py))
        except (SyntaxError, OSError) as exc:
            print(f"[warn] 无法解析 {py}: {exc}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# 后端导出：NestJS 网关（正则，仅 --include-gateway）
# --------------------------------------------------------------------------

_CONTROLLER_RE = re.compile(r"@Controller\s*\(\s*['\"]([^'\"]*)['\"]")
_METHOD_RE = re.compile(r"@(Get|Post|Put|Delete|Patch|Head|Options)\(([^)]*)\)")
_PATH_ARG_RE = re.compile(r"['\"]([^'\"]*)['\"]")


def extract_ts_controllers(root: Path) -> list[BackendEntry]:
    out: list[BackendEntry] = []
    if not root.is_dir():
        return out
    for ts in sorted(root.rglob("*.controller.ts")):
        prefix = ""
        for lineno, raw in enumerate(
                ts.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            line = raw.strip()
            m = _CONTROLLER_RE.search(line)
            if m and line.startswith("@"):
                prefix = m.group(1).strip("/")
                continue
            m = _METHOD_RE.search(line)
            if m and line.startswith("@"):
                tail = ""
                pm = _PATH_ARG_RE.search(m.group(2))
                if pm:
                    tail = pm.group(1).strip("/")
                parts = [x for x in (prefix, tail) if x]
                path = "/api" + ("/" + "/".join(parts) if parts else "")
                out.append(BackendEntry(
                    method=m.group(1).upper(), path=path, source=ts.name,
                    file=_rel_or_abs(ts), line=lineno,
                ))
    return out


# --------------------------------------------------------------------------
# 前端导出（正则：request()/fetch() 字面量与模板）
# --------------------------------------------------------------------------

@dataclass
class FrontendEntry:
    method: str
    path: str
    source: str
    file: str
    line: int
    kind: str


_CALL_RE = re.compile(r"\b(request|fetch)\s*(?:<[^>\n]*>)?\s*\(\s*([`'\"])(.*?)\2", re.S)
_METHOD_IN_CALL_RE = re.compile(r"method\s*:\s*['\"]([A-Z]+)['\"]")
_BASE_VAR_RE = re.compile(r"const\s+BASE\s*=\s*['\"]/api['\"]")
# const path = '/chat/stream' → 变量随后交给 request(path)，method 无法静态确定标 '?'
_PATH_VAR_RE = re.compile(r"\bconst\s+path\s*=\s*(['\"`])(/.*?)\1")


def _call_window(text: str, start: int, maxlen: int = 400) -> str:
    """从调用起点截取双目配平后的完整调用文本（模板内嵌括号不截断）。"""
    depth = 0
    i = start
    while i < len(text) and i - start < maxlen:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return text[start:i]


def extract_frontend(root: Path, label: str) -> list[FrontendEntry]:
    out: list[FrontendEntry] = []
    for ext in ("js", "jsx", "ts", "tsx"):
        for f in sorted(root.rglob(f"*.{ext}")):
            if "node_modules" in f.parts or "dist" in f.parts:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _CALL_RE.finditer(text):
                path_token = m.group(3).strip()
                if not path_token.startswith(("/", "${BASE}")):
                    continue
                # 通用包装器形态（fetch(`${BASE}${path}`)）无法静态解析，跳过
                if path_token.startswith("${BASE}${"):
                    continue
                # 嵌套反引号/未闭合 ${ 的模板（query 拼接截断）→ 取第一个 ${ 前缀为净路径
                if "${" in path_token and not re.search(r"\$\{[^}]*\}", path_token):
                    path_token = path_token.split("${", 1)[0].rstrip()
                    if not path_token.startswith(("/", "${BASE}")):
                        continue
                mm = _METHOD_IN_CALL_RE.search(_call_window(text, m.start()))
                method = mm.group(1) if mm else "GET"
                path = path_token.replace("${BASE}", "/api")
                out.append(FrontendEntry(
                    method=method,
                    path=ensure_api_prefix(path),
                    source=f"{label}/{f.name}",
                    file=_rel_or_abs(f),
                    line=text.count("\n", 0, m.start()) + 1,
                    kind="template" if "`" in m.group(2) else "literal",
                ))
            # const path = '/…' 形态（request(path) 变量传参）
            for m in _PATH_VAR_RE.finditer(text):
                path = m.group(2).strip()
                if path.startswith("${BASE}"):
                    path = path.replace("${BASE}", "/api")
                out.append(FrontendEntry(
                    method="?",
                    path=ensure_api_prefix(path),
                    source=f"{label}/{f.name}",
                    file=_rel_or_abs(f),
                    line=text.count("\n", 0, m.start()) + 1,
                    kind="path-var",
                ))
    return out


# --------------------------------------------------------------------------
# 比对与报告
# --------------------------------------------------------------------------

@dataclass
class GapReport:
    backend: list[BackendEntry]
    frontend: list[FrontendEntry]
    a: list[tuple[str, str, list[FrontendEntry]]] = field(default_factory=list)
    b: list[tuple[str, str, list[BackendEntry]]] = field(default_factory=list)
    soft: list[tuple[str, str, str, list[FrontendEntry]]] = field(default_factory=list)


def compare(backend: list[BackendEntry], frontend: list[FrontendEntry]) -> GapReport:
    report = GapReport(backend=backend, frontend=frontend)
    backend_by_path: dict[str, list[BackendEntry]] = {}
    for e in backend:
        backend_by_path.setdefault(normalize_path(e.path), []).append(e)
    frontend_by_path: dict[str, list[FrontendEntry]] = {}
    for e in frontend:
        frontend_by_path.setdefault(normalize_path(e.path), []).append(e)

    for norm, entries in sorted(frontend_by_path.items()):
        be = backend_by_path.get(norm)
        if not be:
            report.a.append((norm, entries[0].method, entries))
            continue
        matched = any(
            e.method == "?" or e.method == b.method for e in entries for b in be
        )
        if matched:
            continue
        report.soft.append((norm, entries[0].method,
                            ",".join(sorted({b.method for b in be})), entries))

    for norm, entries in sorted(backend_by_path.items()):
        if norm not in frontend_by_path:
            report.b.append((norm, entries[0].method, entries))
    return report


# 疑似内部/节点面（前端 UI 通常不应消费）的路径提示，用于缺口 B 降噪标注
_INTERNAL_HINTS = (
    "bootstrap", "android", "/register", "heartbeat", "log-aggregate",
    "spare-master", "/activate", "probe", "sidecar", "smoke", "worker",
)


def is_internal(norm: str) -> bool:
    """非 /api 前缀，或命中内部面关键词 → 判为内部/节点面（前端不调属正常）。"""
    if not norm.startswith("/api/"):
        return True
    return any(h in norm for h in _INTERNAL_HINTS)


def exit_code(report: GapReport) -> int:
    """0=对齐；1=有缺口 A；2=有缺口 B；3=A+B。"""
    code = 0
    if report.a:
        code |= 1
    if report.b:
        code |= 2
    return code


def render(report: GapReport, only: str = "") -> str:
    a = [(p, m, es) for p, m, es in report.a if not only or only in p]
    b = [(p, m, es) for p, m, es in report.b if not only or only in p]
    soft = [(p, fm, bm, es) for p, fm, bm, es in report.soft if not only or only in p]
    lines = [
        "═" * 72,
        "QLH 接口契约对齐审计（静态面扫描）",
        "═" * 72,
        f"后端定义: {len(report.backend)} 条 · 前端引用: {len(report.frontend)} 条",
        "",
        f"── 缺口 A：前端调用但后端未定义（404 风险）—— {len(a)} 条",
    ]
    for p, _m, es in a:
        for e in es:
            lines.append(f"  [{e.method:<7}] {p:<52} {e.file}:{e.line}  ({e.source})")
    lines += [
        "",
        f"── 缺口 B：后端已定义但前端未消费 —— {len(b)} 条"
        f"（内部/节点面 {sum(1 for p, _m, _es in b if is_internal(p))} 条，标 [i]）",
    ]
    for p, m, es in b:
        tag = " [i]" if is_internal(p) else ""
        for e in es:
            lines.append(f"  [{m:<7}] {p:<52} {e.file}:{e.line}{tag}")
    lines += [
        "",
        f"── 软缺口：同路径、方法不一致 —— {len(soft)} 条",
    ]
    for p, fm, bm, es in soft:
        for e in es:
            lines.append(f"  [前端 {fm} / 后端 {bm}] {p:<40} {e.file}:{e.line}")
    lines += ["", "（对齐的接口不逐条列出；`--json` 输出全量 inventory）"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="api-gap-scan", description=__doc__)
    parser.add_argument("--include-gateway", action="store_true",
                        help="纳入 TS 网关（gateway/src）controller 面")
    parser.add_argument("--include-control", action="store_true",
                        help="纳入 control-svc（control/src）controller 面")
    parser.add_argument("--json", action="store_true", help="输出 JSON 到 stdout")
    parser.add_argument("-o", "--output", default="", help="写报告/JSON 到文件")
    parser.add_argument("--only", default="", help="只输出含该子串的路径")
    args = parser.parse_args(argv)

    try:
        backend = extract_backend_python()
        if args.include_gateway:
            backend += extract_ts_controllers(REPO_ROOT / "gateway" / "src")
        if args.include_control:
            backend += extract_ts_controllers(REPO_ROOT / "control" / "src")
        frontend = (extract_frontend(REPO_ROOT / "frontend" / "src", "frontend")
                    + extract_frontend(REPO_ROOT / "frontend_cybergothic" / "src", "cybergothic"))
    except Exception as exc:  # noqa: BLE001
        print(f"提取失败: {exc}", file=sys.stderr)
        return 10

    report = compare(backend, frontend)
    if args.json:
        payload = {
            "schema": "qlh.api_gap_scan.v1",
            "summary": {
                "backend": len(report.backend), "frontend": len(report.frontend),
                "gap_a": len(report.a), "gap_b": len(report.b), "soft": len(report.soft),
            },
            "inventory": {
                "backend": [e.__dict__ for e in report.backend],
                "frontend": [e.__dict__ for e in report.frontend],
            },
            "gaps": {
                "frontend_without_backend": [
                    {"path": p, "frontend": [e.__dict__ for e in es]}
                    for p, _m, es in report.a
                ],
                "backend_without_frontend": [
                    {"path": p, "method": m, "internal": is_internal(p),
                     "backend": [e.__dict__ for e in es]}
                    for p, m, es in report.b
                ],
                "method_mismatch": [
                    {"path": p, "frontend_method": fm, "backend_methods": bm,
                     "frontend": [e.__dict__ for e in es]}
                    for p, fm, bm, es in report.soft
                ],
            },
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = render(report, only=args.only)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[ok] 报告已写入 {args.output}")
    else:
        print(text)

    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())