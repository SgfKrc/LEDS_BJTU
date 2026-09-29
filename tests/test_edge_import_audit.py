"""★ P0-3：Edge 入口 **import 闭包审计** 的回归（`scripts/edge_import_audit.py`）。

两类用例，缺一不可：

* **判据本身**（合成模块树，`tmp_path` + monkeypatch `SEARCH_ROOTS`）——
  验证它真的能区分「**导入即崩**」与「**延迟加载**」。
  ⚠️ 没有这类用例，审计脚本会退化成"永远 PASS 的摆设"（本仓对此有明确纪律：
  负向用例必须证明判据会红）。
* **真入口**（钉住现状）—— `qlh_edge`（Edge 最小入口）与 `src.api_server`（SLIM 真实入口）。

为什么要审 `src.api_server` 而不只是 `qlh_edge`：SLIM 实际跑的是**源码**
`uvicorn src.api_server:app`（`packaging/packaging/qlh_launcher.py:229`）⇒ 它才是运行时载荷；
而既有的 `test_edge_source_has_no_forbidden_top_level_imports` **只扫 `qlh_edge.py` 一个文件**、
且**只看 `tree.body`**（顶层），传递依赖与函数内 import 全漏。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "edge_import_audit.py"

FORBIDDEN = {"torch", "transformers"}


def _load():
    spec = importlib.util.spec_from_file_location("edge_import_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


# ── 判据本身：合成模块树（该红必须红 / 不该红不能红）────────────────────────────


def test_unconditional_top_level_import_is_blocking(tmp_path, monkeypatch) -> None:
    """★ 入口**无条件顶层** import torch ⇒ 必须判 `blocking`（**该红必须红**）。"""
    module = _load()
    _tree(tmp_path, {"entry.py": "import torch\n", "torch.py": ""})
    monkeypatch.setattr(module, "SEARCH_ROOTS", (tmp_path,))

    report = module.audit("entry", FORBIDDEN)

    assert [hit[1] for hit in report["blocking"]] == ["torch"]
    assert report["deferred"] == []


def test_function_level_import_is_deferred_not_blocking(tmp_path, monkeypatch) -> None:
    """★ 函数内 import torch ⇒ 只能判 `deferred`（**反向守卫**：别把延迟误报成阻断）。

    这正是本脚本 v1 踩过的坑：把"经函数内 import 到达的模块"也当成"导入即崩"，
    于是 `model_module` 被误报 —— 而它其实只被 `model_host` 的函数内 import 引用。
    """
    module = _load()
    _tree(tmp_path, {
        "entry.py": "def go():\n    import torch\n    return torch\n",
        "torch.py": "",
    })
    monkeypatch.setattr(module, "SEARCH_ROOTS", (tmp_path,))

    report = module.audit("entry", FORBIDDEN)

    assert report["blocking"] == []
    assert [hit[1] for hit in report["inner"]] == ["torch"]


def test_guarded_top_level_import_is_not_blocking(tmp_path, monkeypatch) -> None:
    """★ `try: import torch / except ImportError:` ⇒ 归 `guarded`，**不是** blocking。

    Edge 运行时本来就没装 torch；这种写法是**有意**的可选依赖（`config.py` / `tcp_comm.py` 就是）。
    """
    module = _load()
    _tree(tmp_path, {
        "entry.py": "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
        "torch.py": "",
    })
    monkeypatch.setattr(module, "SEARCH_ROOTS", (tmp_path,))

    report = module.audit("entry", FORBIDDEN)

    assert report["blocking"] == []
    assert [hit[1] for hit in report["guarded"]] == ["torch"]


def test_transitive_top_level_import_is_followed(tmp_path, monkeypatch) -> None:
    """★ **传递依赖**：入口 → helper（顶层）→ torch ⇒ 必须抓到。

    既有的 `test_edge_source_has_no_forbidden_top_level_imports` **只扫单个文件的 `tree.body`**
    ⇒ 这类"隔一层"的命中它会**完全漏掉**，本用例把这条补上。
    """
    module = _load()
    _tree(tmp_path, {
        "entry.py": "import helper\n",
        "helper.py": "import torch\n",
        "torch.py": "",
    })
    monkeypatch.setattr(module, "SEARCH_ROOTS", (tmp_path,))

    report = module.audit("entry", FORBIDDEN)

    assert "helper" in report["must_load"]
    assert [(hit[0], hit[1]) for hit in report["blocking"]] == [("helper", "torch")]


def test_deferred_module_is_not_in_must_load(tmp_path, monkeypatch) -> None:
    """★ 只经函数内 import 到达的模块，**不得**出现在 `must_load` 里（两个集合必须分开）。"""
    module = _load()
    _tree(tmp_path, {
        "entry.py": "def go():\n    from heavy import thing\n    return thing\n",
        "heavy.py": "import torch\n",
        "torch.py": "",
    })
    monkeypatch.setattr(module, "SEARCH_ROOTS", (tmp_path,))

    report = module.audit("entry", FORBIDDEN)

    assert "heavy" in report["may_load"]
    assert "heavy" not in report["must_load"]
    assert report["blocking"] == []                     # heavy 的顶层 torch 只是"延迟隐患"
    assert any(hit[0] == "heavy" for hit in report["deferred"])


# ── 真入口：钉住现状 ─────────────────────────────────────────────────────────


def test_edge_minimal_entry_closure_is_clean() -> None:
    """`qlh_edge` 是 Edge **最小**入口（docstring 承诺只依赖白名单）⇒ 三类命中必须全 0。"""
    module = _load()

    report = module.audit("qlh_edge", set(module.DEFAULT_FORBIDDEN))

    assert report["blocking"] == []
    assert report["deferred"] == []
    assert report["guarded"] == []
    assert report["inner"] == []


def test_slim_entry_has_no_must_load_forbidden_import() -> None:
    """★ `src.api_server` 是 SLIM 的**真实入口** ⇒ `must_load` 内不得有无条件顶层 forbidden。

    依据：SLIM 以 `uvicorn src.api_server:app` 跑**源码**、且 `qlh-slim.spec` 排除 torch 系
    ⇒ 顶层闭包里一旦出现无条件 `import torch`，就是**导入即 ImportError**。
    （实测现状：blocking = 0；`model_module` 的顶层 torch 走的是函数内路径 ⇒ 归 deferred。）
    """
    module = _load()

    report = module.audit("src.api_server", set(module.DEFAULT_FORBIDDEN))

    assert report["blocking"] == []
    assert "model_module" in report["may_load"]          # 它确实在闭包里（走延迟路径）


def test_cli_reports_pass_on_real_entries() -> None:
    """CLI 端到端：默认两个入口 ⇒ exit 0，且 JSON 里没有任何 blocking。"""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT, check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["reports"], "应当至少审计到一个入口"
    assert all(not report["blocking"] for report in payload["reports"])
