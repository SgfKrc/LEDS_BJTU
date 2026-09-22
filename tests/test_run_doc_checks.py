"""`scripts/run_doc_checks.py` 的守卫测试。

这个入口是 CI 与本地 pre-push 钩子的**唯一**检查清单来源。它自己出错（清单漏项、路径写错、
输出在 GBK 控制台崩掉）会让两层保护同时失效，所以每一条都要钉住。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_doc_checks", ROOT / "scripts" / "run_doc_checks.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_list_targets_exist() -> None:
    """★ 清单守卫：每条检查指向的脚本必须真实存在。

    清单是 CI 与钩子共用的；若某条路径写错，两端会"通过"却什么都没检查 —— 比不检查更危险。
    """
    module = _load()
    names = [name for name, _ in module.CHECKS]
    assert names == ["doc_links", "readme_l10n"]
    for name, relative in module.CHECKS:
        assert (ROOT / relative).is_file(), f"{name} 指向的脚本不存在：{relative}"


def test_missing_script_is_reported_not_crashed() -> None:
    module = _load()
    result = module.run_check("nope", "scripts/definitely_missing.py")
    assert result["ok"] is False
    assert "不存在" in str(result["output"])


def test_main_passes_and_reports_json(capsys) -> None:
    """真跑一遍两条检查（都只用标准库，秒级）⇒ 退出码 0 且 JSON 可解析。"""
    module = _load()
    assert module.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["passed"] is True
    assert [check["name"] for check in payload["checks"]] == ["doc_links", "readme_l10n"]
    assert all(check["ok"] for check in payload["checks"])


def test_main_fails_when_a_check_fails(monkeypatch, capsys) -> None:
    """任一条失败 ⇒ 退出码 1（钩子与 CI 都靠它判断）。"""
    module = _load()
    monkeypatch.setattr(module, "CHECKS", (("broken", "scripts/definitely_missing.py"),))
    assert module.main([]) == 1
    assert "存在失败项" in capsys.readouterr().out


def test_print_lines_are_gbk_encodable() -> None:
    """★ 回归守卫：print 的内容必须能在 GBK 控制台编码。

    `relay_health.py` 曾因输出 `⇒`(U+21D2) 在 cp936 下抛 `UnicodeEncodeError`，把检查结果变成
    traceback。检查工具自伤会让"有检查"比"没检查"更糟，所以这里对每个 print 行做编码断言。
    """
    source = (ROOT / "scripts" / "run_doc_checks.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if "print(" in line:
            line.encode("gbk")


def test_hooks_and_ci_use_the_shared_entry() -> None:
    """★ 两侧必须调用同一个入口（一处定义、两处复用），否则清单会漂移。"""
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "checks.yml").read_text(encoding="utf-8")
    assert "scripts/run_doc_checks.py" in hook
    assert "scripts/run_doc_checks.py" in workflow
    # 钩子在找不到 python 时必须放行而不是卡死 push
    assert "WARN" in hook and "exit 1" in hook
    # 钩子必须转发 Git LFS，否则启用 hooksPath 后推送大文件会静默漏传
    assert "git lfs pre-push" in hook


def test_hooks_are_lf_pinned_in_gitattributes() -> None:
    """★ `.githooks/*` 必须是 LF：CRLF 会让 `#!/bin/sh` 脚本在 Windows 检出后执行失败。"""
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert ".githooks/* text eol=lf" in attributes


def test_hook_scripts_have_shebang_and_no_crlf() -> None:
    for name in ("pre-push", "post-commit", "post-checkout", "post-merge", "install.sh"):
        raw = (ROOT / ".githooks" / name).read_bytes()
        assert raw.startswith(b"#!/bin/sh"), f"{name} 缺少 shebang"
        assert b"\r\n" not in raw, f"{name} 含 CRLF（会使 shebang 失效）"


def test_main_requires_nothing_and_defaults_to_plain_output(capsys) -> None:
    module = _load()
    assert module.main([]) == 0
    out = capsys.readouterr().out
    assert "[PASS]" in out and "[verdict]" in out
