"""scripts/api_gap_scan.py 单元测试（mini fixture，不依赖真实仓库面）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import api_gap_scan as gap  # noqa: E402


# ---------- 归一化 ----------

def test_normalize_path():
    assert gap.normalize_path("/api/alpha?x=1") == "/api/alpha"
    assert gap.normalize_path("api//alpha/") == "/api/alpha"
    assert gap.normalize_path("/api/beta/123") == "/api/beta/123"  # 具体字面量路径保持
    assert gap.normalize_path("/api/beta/{id}") == "/api/beta/{param}"
    assert gap.normalize_path("/api/beta/:id") == "/api/beta/{param}"
    assert gap.normalize_path("${x}/sessions/1") == "/{param}/sessions/1"


def test_ensure_api_prefix():
    assert gap.ensure_api_prefix("/api/alpha") == "/api/alpha"
    assert gap.ensure_api_prefix("/auth/login") == "/api/auth/login"
    assert gap.ensure_api_prefix("users") == "/api/users"


# ---------- 后端提取（FastAPI AST） ----------

_BACKEND_FIXTURE = '''
"""doc"""
from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter(prefix="/v1")

@app.get("/api/alpha")
def alpha():
    return {}

@app.post("/api/beta/{id}")
async def beta(id: str):
    return {}

@router.get("/models")
def gamma():
    return {}

@app.get  # 动态路径不收录（非字符串常量）
def dynamic():
    return {}

def not_a_route():
    @app.get
    def inner(): ...
'''


def test_extract_fastapi(tmp_path):
    f = tmp_path / "be.py"
    f.write_text(_BACKEND_FIXTURE, encoding="utf-8")
    entries = gap.extract_fastapi(f)
    by_path = {(e.method, gap.normalize_path(e.path)) for e in entries}
    assert ("GET", "/api/alpha") in by_path
    assert ("POST", "/api/beta/{param}") in by_path
    assert ("GET", "/v1/models") in by_path          # router prefix 静态展开拼入
    assert all("nested" not in p for _m, p in by_path)  # 缺路径常量不收录
    assert all(e.line >= 1 for e in entries)


# ---------- 前端提取 ----------

_FRONTEND_FIXTURE = '''
export function api(oldClient, newClient) {
  return request('/api/alpha');                                  // GET
  return request('/api/beta/42');                                // 模板参数之外的字面量
  return request('/auth/logout', { method: 'POST', auth: true }); // 无 /api 前缀 → 补前缀
}

const BASE = '/api';
export function sessions() {
  return request(`${BASE}/sessions/${encodeURIComponent('s1')}`, { method: 'DELETE' });
}
export function direct() {
  return fetch('/api/health', { method: 'GET' });
}
export function genericPost() {
  return request<Result<{ ok: boolean }>>('/api/generic-post', { method: 'POST' });
}
// 非路径字面量不算
export function ignore() {
  return request('not-a-path', { method: 'POST' });
}
// 嵌套反引号 query 拼接 + const path 变量形态
export function nested() {
  return request(`/logs/recent${q ? `?${q}` : ''}`, { headers: {} });
}
const path = '/chat/stream';
export function streamed() {
  return request(path, { method: 'POST' });
}
'''


def test_extract_frontend(tmp_path):
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "api.ts"
    f.write_text(_FRONTEND_FIXTURE, encoding="utf-8")
    entries = gap.extract_frontend(tmp_path / "src", "dummy")
    by_path = {(e.method, e.path) for e in entries}
    assert ("GET", "/api/alpha") in by_path
    assert ("GET", "/api/beta/42") in by_path
    assert ("POST", "/api/auth/logout") in by_path      # 补 /api 前缀 + method 推断
    assert ("DELETE", "/api/sessions/{param}") in by_path  # 模板 ${BASE} + ${id} 归一化
    assert ("GET", "/api/health") in by_path            # fetch 直调
    assert ("POST", "/api/generic-post") in by_path   # 泛型中的 {} 不截断 method 检测
    assert all("/api/" in e.path for e in entries)
    kinds = {e.kind for e in entries}
    assert "template" in kinds and "literal" in kinds and "path-var" in kinds
    # 嵌套反引号 query 模板 → 净路径；const path → /api 前缀 + method='?'
    assert ("GET", "/api/logs/recent") in by_path
    assert ("?", "/api/chat/stream") in by_path


def test_frontend_scope_and_surface(tmp_path):
    data = tmp_path / "frontend_cybergothic" / "src" / "data"
    pages = tmp_path / "frontend_cybergothic" / "src" / "pages"
    data.mkdir(parents=True)
    pages.mkdir(parents=True)
    (data / "api.ts").write_text("export const x = () => request('/api/client');", encoding="utf-8")
    (pages / "Home.tsx").write_text("export const x = () => fetch('/api/page');", encoding="utf-8")
    entries = gap.extract_frontend(
        tmp_path / "frontend_cybergothic" / "src", "cybergothic", scope="canonical"
    )
    by_path = {e.path: e for e in entries}
    assert by_path["/api/client"].scope == "canonical"
    assert by_path["/api/client"].surface == "shared_client"
    assert by_path["/api/page"].surface == "page"


def test_collect_frontend_defaults_to_canonical(tmp_path, monkeypatch):
    canonical = tmp_path / "frontend_cybergothic" / "src"
    legacy = tmp_path / "frontend" / "src"
    canonical.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (canonical / "api.ts").write_text("request('/api/canonical');", encoding="utf-8")
    (legacy / "client.js").write_text("request('/api/legacy');", encoding="utf-8")
    monkeypatch.setattr(gap, "REPO_ROOT", tmp_path)

    default = gap.collect_frontend()
    assert {e.path for e in default} == {"/api/canonical"}
    assert {e.scope for e in default} == {"canonical"}

    with_legacy = gap.collect_frontend(include_legacy=True)
    assert {e.path for e in with_legacy} == {"/api/canonical", "/api/legacy"}
    assert {e.scope for e in with_legacy} == {"canonical", "legacy"}


def test_symbol_consumer_requires_api_import(tmp_path):
    root = tmp_path / "frontend_cybergothic" / "src"
    data = root / "data"
    pages = root / "pages"
    data.mkdir(parents=True)
    pages.mkdir(parents=True)
    (data / "api.ts").write_text(
        "export const fetchThing = () => request('/api/thing');", encoding="utf-8"
    )
    (pages / "Home.tsx").write_text(
        "import * as api from '../data/api';\nexport function Home() { api.fetchThing(); }",
        encoding="utf-8",
    )
    entries = gap.extract_frontend(root, "cy", scope="canonical")
    entries.extend(gap._extract_symbol_consumers(root, entries, scope="canonical"))
    consumers = [e for e in entries if e.kind == "symbol-call"]
    assert len(consumers) == 1
    assert consumers[0].surface == "page"
    assert consumers[0].path == "/api/thing"


# ---------- 比对 ----------

def _be(method, path):
    return gap.BackendEntry(method=method, path=path, source="be.py",
                            file="src/be.py", line=1)


def _fe(method, path, kind="literal"):
    return gap.FrontendEntry(method=method, path=path, source="fe.ts",
                             file="frontend_cybergothic/src/fe.ts", line=1, kind=kind)


def test_compare_aligned_and_gaps():
    backend = [_be("GET", "/api/alpha"), _be("POST", "/api/beta/{id}"),
               _be("GET", "/api/unused")]
    frontend = [_fe("GET", "/api/alpha"), _fe("POST", "/api/beta/${id}"),
                _fe("GET", "/api/ghost")]                  # 缺口 A
    report = gap.compare(backend, frontend)
    a_paths = {p for p, _m, _es in report.a}
    assert a_paths == {"/api/ghost"}
    b_paths = {p for p, _m, _es in report.b}
    assert b_paths == {"/api/unused"}                       # 缺口 B
    assert report.soft == []                                # 方法一致无软缺口


def test_compare_method_mismatch_soft():
    backend = [_be("POST", "/api/alpha")]
    frontend = [_fe("GET", "/api/alpha")]
    report = gap.compare(backend, frontend)
    assert not report.a and not report.b
    assert len(report.soft) == 1
    p, fm, bm, _es = report.soft[0]
    assert (p, fm, bm) == ("/api/alpha", "GET", "POST")


def test_is_internal_classification():
    assert gap.is_internal("/activate")            # 非 /api 前缀
    assert gap.is_internal("/api/bootstrap/info")  # 内部面关键词
    assert gap.is_internal("/api/cluster/android/heartbeat")
    assert not gap.is_internal("/api/tasks")
    assert not gap.is_internal("/api/models/current")


def test_coverage_classification():
    backend = [_be("GET", "/api/page"), _be("GET", "/api/client"),
               _be("GET", "/api/bootstrap/info"), _be("GET", "/api/unconsumed")]
    frontend = [gap.FrontendEntry("GET", "/api/page", "fe", "src/pages/Home.tsx", 1,
                                  "literal", surface="page"),
                gap.FrontendEntry("GET", "/api/client", "fe", "src/data/api.ts", 1,
                                  "literal", surface="shared_client")]
    report = gap.compare(backend, frontend)
    counts = gap.coverage_summary(report)
    assert counts["page_consumed"] == 1
    assert counts["client_declared"] == 1
    assert counts["internal_protocol"] == 1
    assert counts["unconsumed"] == 1


def test_ownership_separates_domain_from_frontend_status():
    backend = [
        _be("GET", "/api/page"),
        _be("POST", "/api/experimental/speculative"),
        _be("GET", "/api/cluster/android/heartbeat"),
        _be("GET", "/api/unconsumed"),
    ]
    frontend = [
        gap.FrontendEntry("GET", "/api/page", "page", "frontend_cybergothic/src/pages/Home.tsx", 1,
                          "literal", surface="page"),
        gap.FrontendEntry("POST", "/api/experimental/speculative", "api", "frontend_cybergothic/src/data/api.ts", 1,
                          "literal", surface="shared_client"),
        gap.FrontendEntry("GET", "/api/cluster/android/heartbeat", "hook", "frontend_cybergothic/src/data/hooks.ts", 1,
                          "literal", surface="hook"),
    ]
    report = gap.compare(backend, frontend)
    rows = {row["path"]: row for row in gap.build_ownership(report)}
    assert rows["/api/page"]["frontend_status"] == "page_consumed"
    assert rows["/api/page"]["domain"] == "product"
    assert rows["/api/experimental/speculative"]["frontend_status"] == "client_declared"
    assert rows["/api/experimental/speculative"]["domain"] == "experimental"
    assert rows["/api/cluster/android/heartbeat"]["domain"] == "internal_protocol"
    assert rows["/api/cluster/android/heartbeat"]["frontend_status"] == "hook_consumed"
    assert rows["/api/unconsumed"]["frontend_status"] == "unconsumed"


def test_ownership_is_deterministic_and_json_ready():
    backend = [_be("GET", "/api/b"), _be("GET", "/api/a")]
    report = gap.compare(backend, [])
    rows = gap.build_ownership(report)
    assert [row["path"] for row in rows] == ["/api/a", "/api/b"]
    assert gap.ownership_summary(rows) == {
        "domain:product": 2,
        "frontend:unconsumed": 2,
    }


def test_json_cli_exposes_versioned_ownership(monkeypatch, capsys):
    monkeypatch.setattr(gap, "extract_backend_python", lambda: [_be("GET", "/api/health")])
    monkeypatch.setattr(gap, "collect_frontend", lambda **_kwargs: [])
    assert gap.main(["--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ownership_schema"] == "qlh.api_ownership.v1"
    assert payload["summary"]["ownership_counts"]["domain:product"] == 1
    assert payload["ownership"][0]["path"] == "/api/health"


def test_exit_code_mask():
    b = [_be("GET", "/api/x")]
    assert gap.exit_code(gap.compare(b, [_fe("GET", "/api/x")])) == 0                      # 对齐
    assert gap.exit_code(gap.compare(b, [_fe("GET", "/api/x"), _fe("GET", "/api/z")])) == 1  # 仅 A
    assert gap.exit_code(gap.compare([_be("GET", "/api/x"), _be("GET", "/api/y")],
                                     [_fe("GET", "/api/x")])) == 2                          # 仅 B
    assert gap.exit_code(gap.compare(b, [_fe("GET", "/api/z")])) == 3                      # A+B
