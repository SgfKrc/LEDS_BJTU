"""`scripts/relay_cut_model_analysis.py`（P2 遗留②：噪声量化 + 内部最优判定）的测试。

覆盖：
- 文件名解析：只有 `r<round>-k<cut>` 形式才算多轮记录，其它一律 skip 并提示；
- 置信区间：单点给 0 半宽、多点用 max−min 半宽（样本少时的保守口径）；
- 线性拟合：完美线性 r²=1；离散数据 r²<1；
- 端到端判据：**内部最优不显著**与**显著**两种数据要给出相反的 verdict；
- 缺指标（upstream/downstream 为 null）的记录不参与统计。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relay_cut_model_analysis.py"


def _write(path: Path, up: float | None, down: float | None, *, passed: bool = True) -> None:
    path.write_text(json.dumps({
        "schema_version": "qlh.relay_experiment.v1",
        "experiment_id": "exp-model-a",
        "kind": "mainrepo_end_to_end",
        "path": "d2l_mainrepo",
        "commit": "same-head",
        "models": {"upstream": {"id": "model-a"},
                   "whole": {"model_bytes": 240 * 1024 * 1024}},
        "load": {"prefill_tokens": 32, "gen_tokens": 32, "batch": 1, "warmup": 3},
        "record_origin": {"kind": "raw_measurement", "rounds": 1},
        "metrics": {
            "upstream_decode_ms": {"mean": up},
            "downstream_decode_ms": {"mean": down},
        },
        "verdict": {"passed": passed},
    }, ensure_ascii=False), encoding="utf-8")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=ROOT)


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cut_model_analysis_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_requires_round_cut_name(tmp_path: Path) -> None:
    module = _load_module()
    ok = tmp_path / "r2-k8.json"
    _write(ok, 10.0, 20.0)
    item = module._extract(ok)
    assert item is not None and item["round"] == 2 and item["cut"] == 8
    assert item["total_ms"] == 30.0

    bad = tmp_path / "qwen25-05b-k8-b1-p32-g32.json"
    _write(bad, 10.0, 20.0)
    assert module._extract(bad) is None


def test_ci95_single_and_multi_point() -> None:
    module = _load_module()
    assert module._ci95([5.0]) == (0.0, 0.0)
    median, half = module._ci95([1.0, 2.0, 3.0])
    assert median == 2.0 and half == 1.0


def test_total_median_is_additive_stage_median(tmp_path: Path) -> None:
    module = _load_module()
    for round_no, (up, down) in enumerate(((1.0, 100.0), (2.0, 100.0),
                                            (100.0, 1.0)), 1):
        _write(tmp_path / f"r{round_no}-k4.json", up, down)
    rows = [module._extract(tmp_path / f"r{round_no}-k4.json")
            for round_no in (1, 2, 3)]
    assert all(row is not None for row in rows)

    done = _run("--records", str(tmp_path / "r*-k*.json"),
                "--total-layers", "24", "--out", str(tmp_path / "report.json"))
    assert done.returncode == 0, done.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    entry = report["by_cut"][0]
    assert entry["total_median_ms"] == 102.0
    assert entry["median_of_totals_ms"] == 101.0


def test_repeated_scan_requires_balanced_three_rounds(tmp_path: Path) -> None:
    for round_no in (1, 2):
        _write(tmp_path / f"r{round_no}-k4.json", 10.0, 20.0)
        _write(tmp_path / f"r{round_no}-k20.json", 20.0, 10.0)
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24")
    assert done.returncode == 2
    assert "at least 3" in done.stderr


def test_repeated_scan_rejects_correctness_failure(tmp_path: Path) -> None:
    for round_no in (1, 2, 3):
        _write(tmp_path / f"r{round_no}-k4.json", 10.0, 20.0,
               passed=(round_no != 2))
        _write(tmp_path / f"r{round_no}-k20.json", 20.0, 10.0)
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24")
    assert done.returncode == 2
    assert "correctness verdict failed" in done.stderr


def test_linear_fit_perfect_and_noisy() -> None:
    module = _load_module()
    perfect = module._linear_fit([1.0, 2.0, 3.0], [3.0, 5.0, 7.0])
    assert perfect is not None and abs(perfect["r2"] - 1.0) < 1e-9
    noisy = module._linear_fit([1.0, 2.0, 3.0], [3.0, 9.0, 4.0])
    assert noisy is not None and noisy["r2"] < 1.0


def test_no_interior_optimum_is_not_significant(tmp_path: Path) -> None:
    """端点最优 + 有噪声 ⇒ 不能宣称内部最优。"""
    for round_no in (1, 2, 3):
        # K=4 最慢、K=20 最快；中间点都在两端之间，且抖动在 ±1ms 内
        _write(tmp_path / f"r{round_no}-k4.json", 40.0 + round_no, 20.0)
        _write(tmp_path / f"r{round_no}-k12.json", 30.0 + round_no, 18.0)
        _write(tmp_path / f"r{round_no}-k20.json", 20.0 + round_no, 16.0)
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24")
    assert done.returncode == 0, done.stderr
    assert "显著内部最优：False" in done.stdout


def test_significant_interior_optimum_is_detected(tmp_path: Path) -> None:
    """内部点稳定地优于两端（远超噪声）⇒ 必须报 True，提示补非线性项。"""
    for round_no in (1, 2, 3):
        _write(tmp_path / f"r{round_no}-k4.json", 30.0 + 0.1 * round_no, 20.0)
        _write(tmp_path / f"r{round_no}-k12.json", 5.0 + 0.1 * round_no, 5.0)   # 内部显著更优
        _write(tmp_path / f"r{round_no}-k20.json", 30.0 + 0.1 * round_no, 25.0)
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24")
    assert done.returncode == 0, done.stderr
    assert "显著内部最优：True" in done.stdout


def test_skips_records_without_metrics(tmp_path: Path) -> None:
    _write(tmp_path / "r1-k4.json", 10.0, 20.0)
    _write(tmp_path / "r2-k4.json", None, None)      # 缺指标 ⇒ 不参与
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24",
                "--min-rounds", "1")
    assert done.returncode == 0, done.stderr
    assert "无法解析或缺少指标" in (done.stdout + done.stderr)


def test_fails_loud_without_records(tmp_path: Path) -> None:
    done = _run("--records", str(tmp_path / "nothing-*.json"), "--total-layers", "24")
    assert done.returncode != 0
    assert "没有可解析的记录" in (done.stdout + done.stderr)


def test_repeated_scan_rejects_duplicate_round_cut(tmp_path: Path) -> None:
    _write(tmp_path / "r1-k4.json", 10.0, 20.0)
    duplicate_dir = tmp_path / "alias"
    duplicate_dir.mkdir()
    duplicate = duplicate_dir / "r1-k4.json"
    duplicate.write_text((tmp_path / "r1-k4.json").read_text(encoding="utf-8"), encoding="utf-8")
    done = _run("--records", str(tmp_path / "r*-k*.json"),
                "--records", str(duplicate_dir / "r*-k*.json"),
                "--total-layers", "24", "--min-rounds", "1")
    assert done.returncode == 2
    assert "duplicate experiment record" in done.stderr


def test_repeated_scan_rejects_median_input(tmp_path: Path) -> None:
    _write(tmp_path / "r1-k4.json", 10.0, 20.0)
    data = json.loads((tmp_path / "r1-k4.json").read_text(encoding="utf-8"))
    data["record_origin"] = {"kind": "median_of_repeats", "rounds": 5}
    (tmp_path / "r1-k4.json").write_text(json.dumps(data), encoding="utf-8")
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24", "--min-rounds", "1")
    assert done.returncode == 2
    assert "not raw input" in done.stderr


def test_repeated_scan_rejects_missing_origin(tmp_path: Path) -> None:
    _write(tmp_path / "r1-k4.json", 10.0, 20.0)
    data = json.loads((tmp_path / "r1-k4.json").read_text(encoding="utf-8"))
    data.pop("record_origin", None)
    (tmp_path / "r1-k4.json").write_text(json.dumps(data), encoding="utf-8")
    done = _run("--records", str(tmp_path / "r*-k*.json"), "--total-layers", "24", "--min-rounds", "1")
    assert done.returncode == 2
    assert "not raw input" in done.stderr
