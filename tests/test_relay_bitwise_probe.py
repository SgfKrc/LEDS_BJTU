"""`scripts/relay_bitwise_probe.py` 的判据测试（不需要 shim / 不需要模型）。

覆盖：
- `gen-input` 的确定性（同 seed → 逐字节相同，跨进程也相同）；
- `compare` 的两种判定：逐比特相同 / 有差异（并给出 rel 与差异元素数）；
- `compare` 的形状不一致要 fail-loud，而不是"自适应"成能比就比。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relay_bitwise_probe.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=ROOT)


def _write_npy(path: Path, data) -> None:
    np = pytest.importorskip("numpy")
    np.save(path, data)


def test_gen_input_is_deterministic(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    for target in (a, b):
        done = _run("gen-input", "--n-tokens", "4", "--n-embd", "8", "--out", str(target))
        assert done.returncode == 0, done.stderr
    assert a.read_bytes() == b.read_bytes()
    assert np.load(a).shape == (4, 8)


def test_compare_reports_bitwise_equality(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    base = np.arange(24, dtype=np.float32).reshape(3, 8)
    p1, p2, report = tmp_path / "p1.npy", tmp_path / "p2.npy", tmp_path / "r.json"
    np.save(p1, base)
    np.save(p2, base.copy())
    done = _run("compare", "--case", f"a={p1}", "--case", f"b={p2}", "--out", str(report))
    assert done.returncode == 0, done.stderr
    assert "bitwise_identical_across_platforms=True" in done.stdout
    row = json.loads(report.read_text(encoding="utf-8"))["pairs"][0]
    assert row["bitwise_equal"] is True
    assert row["differing_elements"] == 0
    assert row["max_abs"] == 0.0


def test_compare_reports_differences(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    base = np.ones((2, 4), dtype=np.float32)
    other = base.copy()
    other[0, 0] = 1.0 + 1e-3          # 相对 1.0 的 1e-3 偏差
    p1, p2, report = tmp_path / "p1.npy", tmp_path / "p2.npy", tmp_path / "r.json"
    np.save(p1, base)
    np.save(p2, other)
    done = _run("compare", "--case", f"a={p1}", "--case", f"b={p2}", "--out", str(report))
    assert done.returncode == 0, done.stderr
    assert "bitwise_identical_across_platforms=False" in done.stdout
    row = json.loads(report.read_text(encoding="utf-8"))["pairs"][0]
    assert row["bitwise_equal"] is False
    assert row["differing_elements"] == 1
    assert row["total_elements"] == 8
    assert 0 < row["rel"] < 1e-2


def test_compare_rejects_shape_mismatch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    p1, p2 = tmp_path / "p1.npy", tmp_path / "p2.npy"
    np.save(p1, np.zeros((2, 4), dtype=np.float32))
    np.save(p2, np.zeros((2, 5), dtype=np.float32))
    done = _run("compare", "--case", f"a={p1}", "--case", f"b={p2}")
    assert done.returncode != 0
    assert "形状不一致" in (done.stdout + done.stderr)


def test_compare_needs_two_cases(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    p1 = tmp_path / "p1.npy"
    np.save(p1, np.zeros((2, 4), dtype=np.float32))
    done = _run("compare", "--case", f"a={p1}")
    assert done.returncode != 0
    assert "至少给两个" in (done.stdout + done.stderr)
