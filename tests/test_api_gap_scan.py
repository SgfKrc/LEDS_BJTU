"""scripts/api_gap_scan.py 单元测试（mini fixture，不依赖真实仓库面）。"""

from __future__ import annotations

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
    assert all("/api/" in e.path for e in entries)
    kinds = {e.kind for e in entries}
    assert "template" in kinds and "literal" in kinds and "path-var" in kinds
    # 嵌套反引号 query 模板 → 净路径；const path → /api 前缀 + method='?'
    assert ("GET", "/api/logs/recent") in by_path
    assert ("?", "/api/chat/stream") in by_path


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


def test_exit_code_mask():
    b = [_be("GET", "/api/x")]
    assert gap.exit_code(gap.compare(b, [_fe("GET", "/api/x")])) == 0                      # 对齐
    assert gap.exit_code(gap.compare(b, [_fe("GET", "/api/x"), _fe("GET", "/api/z")])) == 1  # 仅 A
    assert gap.exit_code(gap.compare([_be("GET", "/api/x"), _be("GET", "/api/y")],
                                     [_fe("GET", "/api/x")])) == 2                          # 仅 B
    assert gap.exit_code(gap.compare(b, [_fe("GET", "/api/z")])) == 3                      # A+B