"""`scripts/relay_quant_budget.py`（P4 精度预算表）的判据测试。

覆盖：
- 文件名解析：prompt 名里含 `-`（如 `natural-short`）也必须解析正确；
- 档位聚合：翻转数、最坏 margin、**基线无余量子集**要分开统计；
- 判定分层：对照档不能被"基线本身就很低"的 prompt 拉成 tight；
- 无余量 prompt 全占满时给 `no-headroom`，而不是假装 safe；
- 无法解析的文件名只跳过并提示，不静默算错。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relay_quant_budget.py"


def _write_record(path: Path, *, prompt: str, passed: bool, relay: float, base: float,
                  hidden: str = "f16", upstream: str = "fp16") -> None:
    path.write_text(json.dumps({
        "metrics": {
            "logit_margin": {"mean": relay},
            "baseline_logit_margin": {"mean": base},
            "resident_weight_bytes": {"upstream": 510_000_000},
        },
        "device_profile": {"hidden_wire_bytes_per_token": 3584},
        "verdict": {"passed": passed, "matched_runs": 1 if passed else 0, "total_runs": 1},
    }, ensure_ascii=False), encoding="utf-8")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=ROOT)


def test_parse_case_handles_hyphenated_prompt() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("budget_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._parse_case("p4b-natural-short-fp16-int4_block128") == (
        "p4b-natural-short", "fp16", "int4_block128")
    assert module._parse_case("p4x-four-seg-f16") is None          # 不是档位命名
    assert module._parse_case("nohyphen") is None


def test_budget_separates_no_headroom_prompts(tmp_path: Path) -> None:
    """对照档（fp16×f16）不应被"基线本身就低"的 prompt 判成 tight。"""
    # code: 基线 3.76（低于闸门 5.0）⇒ 无余量；natural-short: 基线 6.15 ⇒ 有余量
    _write_record(tmp_path / "p4b-code-fp16-f16.json", prompt="code",
                  passed=True, relay=3.7611, base=3.7555)
    _write_record(tmp_path / "p4b-natural-short-fp16-f16.json", prompt="natural-short",
                  passed=True, relay=6.1583, base=6.1491)
    done = _run("--records", str(tmp_path / "p4b-*.json"))
    assert done.returncode == 0, done.stderr
    assert "safe" in done.stdout
    assert "no_hd" in done.stdout


def test_budget_marks_unsafe_when_headroom_prompts_flip(tmp_path: Path) -> None:
    """有余量的 prompt 上翻转 ⇒ 该档必须判 tight（不能被无余量 prompt 掩盖）。"""
    _write_record(tmp_path / "p4b-natural-short-int4-int4_block128.json",
                  prompt="natural-short", passed=False, relay=4.0134, base=6.1491)
    _write_record(tmp_path / "p4b-natural-long-int4-int4_block128.json",
                  prompt="natural-long", passed=False, relay=3.8284, base=5.8149)
    done = _run("--records", str(tmp_path / "p4b-*.json"))
    assert done.returncode == 0, done.stderr
    assert "tight" in done.stdout
    assert "没有" in done.stdout          # 无 safe 档时给明确建议


def test_budget_reports_no_headroom_when_all_baselines_low(tmp_path: Path) -> None:
    _write_record(tmp_path / "p4b-code-fp16-f16.json", prompt="code",
                  passed=True, relay=3.7611, base=3.7555)
    done = _run("--records", str(tmp_path / "p4b-*.json"))
    assert done.returncode == 0, done.stderr
    assert "no-headroom" in done.stdout


def test_budget_skips_unparseable_filenames(tmp_path: Path) -> None:
    _write_record(tmp_path / "p4b-natural-short-fp16-f16.json", prompt="natural-short",
                  passed=True, relay=6.1583, base=6.1491)
    _write_record(tmp_path / "unrelated-name.json", prompt="x", passed=True,
                  relay=1.0, base=1.0)
    done = _run("--records", str(tmp_path / "*.json"))
    assert done.returncode == 0, done.stderr
    assert "无法从文件名解析档位" in (done.stdout + done.stderr)


def test_budget_fails_loud_on_empty_records(tmp_path: Path) -> None:
    done = _run("--records", str(tmp_path / "nothing-*.json"))
    assert done.returncode != 0
    assert "没有可解析的记录" in (done.stdout + done.stderr)
