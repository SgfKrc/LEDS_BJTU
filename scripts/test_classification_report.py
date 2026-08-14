"""按测试分类运行 pytest 并生成 JSON + 摘要报告（T-4）。

分类规则（文件名启发式 + marker 覆盖；不改动既有文件命名）：
  - 质量门/回归: 文件名含 quality_gate（-m quality_gate）
  - 契约/协议:   文件名含 contract / protocol / worker_adapter
  - 集成/组件:   文件名含 integration / tcp_comm / scheduler / engine_host
  - 仿真/压力:   tests/simulation/ 或文件名含 simulation / parallel / stress
  - 专项/其他:   其余（单元/逻辑、系统端到端归入此类的按需细分）

用法：
  python scripts/test_classification_report.py            # 全部分类（串行，RAM 安全）
  python scripts/test_classification_report.py --class quality_gate
  python scripts/test_classification_report.py --json build/test-classification-report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 分类 → (文件名匹配正则, 附加 marker 过滤)
CLASSIFIERS: dict[str, dict] = {
    "quality_gate": {"pattern": r"quality_gate", "marker": "quality_gate"},
    "contract": {"pattern": r"contract|protocol|worker_adapter", "marker": None},
    "integration": {"pattern": r"integration|tcp_comm|scheduler|engine_host", "marker": None},
    "simulation": {"pattern": r"simulation|parallel|stress", "marker": None},
    "unit": {"pattern": r".*", "marker": None},  # 兜底：其余全部
}

# 稳定输出顺序
CLASS_ORDER = ["quality_gate", "contract", "integration", "simulation", "unit"]


def classify_test_files(tests_dir: Path) -> dict[str, list[str]]:
    """按文件名启发式把 tests/ 下的 test_*.py 分到各分类（不移动文件）。"""
    result: dict[str, list[str]] = {name: [] for name in CLASS_ORDER}
    matched: set[str] = set()
    for path in sorted(tests_dir.glob("test_*.py")):
        name = path.name
        assigned = False
        for cls in CLASS_ORDER[:-1]:  # 前四类按规则，unit 兜底
            info = CLASSIFIERS[cls]
            if re.search(info["pattern"], name):
                result[cls].append(name)
                matched.add(name)
                assigned = True
                break
        if not assigned:
            result["unit"].append(name)
    # 子目录（simulation/ 等）
    for sub in sorted(tests_dir.iterdir()):
        if sub.is_dir() and (sub / "conftest.py").exists() or sub.is_dir() and sub.name == "simulation":
            for path in sorted(sub.glob("test_*.py")):
                cls = "simulation" if sub.name == "simulation" else "unit"
                result[cls].append(f"{sub.name}/{path.name}")
                matched.add(path.name)
    return result


def run_class(cls: str, files: list[str], *, venv_python: str, workers: int = 1) -> dict:
    """对某一分类运行 pytest（只跑该类文件），返回统计。"""
    if not files:
        return {"class": cls, "files": 0, "tests": 0,
                "passed": 0, "failed": 0, "skipped": 0, "duration_s": 0.0, "rc": 0}
    # 传具体文件路径（分类精确），quality_gate 文件自带 marker 无需 -m
    cmd = [
        venv_python, "-m", "pytest",
        *[f"tests/{name}" for name in files],
        "-q", "-n", str(workers),
        "--maxfail=20",
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
    duration = time.time() - started
    # 解析最后一行 "N passed, M skipped" / "N failed, ..."
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    m = re.search(r"(\d+) passed", tail)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) skipped", tail)
    skipped = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", tail)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) error", tail)
    errors = int(m.group(1)) if m else 0
    return {
        "class": cls,
        "files": len(files),
        "tests": passed + skipped + failed + errors,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "duration_s": round(duration, 2),
        "rc": proc.returncode,
    }


def _venv_python() -> str:
    """优先 .venv-test（隔离测试环境），回退当前解释器。"""
    candidate = ROOT / ".venv-test" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", choices=CLASS_ORDER,
                        help="只运行指定分类")
    parser.add_argument("--json", default="", help="报告输出路径（默认 build/test-classification-report.json）")
    parser.add_argument("--workers", type=int, default=1, help="每分类 pytest worker 数（默认 1，RAM 安全）")
    args = parser.parse_args(argv)

    files_by_class = classify_test_files(ROOT / "tests")
    results = []
    for cls in CLASS_ORDER:
        if args.class_name and cls != args.class_name:
            continue
        print(f"[report] 运行分类: {cls}（{len(files_by_class[cls])} 文件）", flush=True)
        results.append(run_class(cls, files_by_class[cls], venv_python=_venv_python(),
                                 workers=args.workers))

    total = {
        "passed": sum(r["passed"] for r in results),
        "failed": sum(r["failed"] for r in results),
        "skipped": sum(r["skipped"] for r in results),
        "errors": sum(r["errors"] for r in results),
        "tests": sum(r["tests"] for r in results),
        "duration_s": round(sum(r["duration_s"] for r in results), 2),
    }
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "total": total, "by_class": results}
    out = Path(args.json) if args.json else ROOT / "build" / "test-classification-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 测试分类报告摘要 =====")
    for r in results:
        status = "FAIL" if (r["failed"] or r["errors"]) else "PASS"
        print(f"  [{status}] {r['class']:<12} tests={r['tests']:>4} "
              f"passed={r['passed']:>4} failed={r['failed']:>3} "
              f"skipped={r['skipped']:>3} {r['duration_s']:>7.1f}s")
    print(f"  TOTAL {total['tests']} tests: {total['passed']} passed / "
          f"{total['failed']} failed / {total['skipped']} skipped / "
          f"{total['errors']} errors（{total['duration_s']}s）")
    print(f"  报告: {out}")
    return 1 if (total["failed"] or total["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
