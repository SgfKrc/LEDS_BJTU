#!/usr/bin/env python
"""文档检查的统一入口：一处定义、两处复用（GitHub Actions + 本地 pre-push 钩子）。

## 为什么需要它

`check_doc_links.py` 与 `check_readme_l10n.py` 本来各自独立，接 CI 和接钩子会变成**两份清单**，
迟早漂移（加了一条检查、只改了其中一处，另一个入口就形同虚设）。所以把它们收敛到这一个命令里，
CI 与钩子都只调用它。

## 覆盖（2026-09-22 约定）

| 检查 | 抓什么 |
| --- | --- |
| `check_doc_links.py` | 相对链接死链。**实测有效**：一次就抓出 20 处"文件已归档但仍按旧路径引用" |
| `check_readme_l10n.py` | `README.md` 与 `docs/README.en.md` 的双语结构是否同步 |

两条都是**纯静态、只用标准库** ⇒ 无需安装任何依赖，可秒级跑完。

## 用法

    python scripts/run_doc_checks.py            # 人读输出
    python scripts/run_doc_checks.py --json     # 机器可读（CI 用）

退出码：0 = 全通过；1 = 有检查失败。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS: tuple[tuple[str, str], ...] = (
    ("doc_links", "scripts/check_doc_links.py"),
    ("readme_l10n", "scripts/check_readme_l10n.py"),
)


def _enable_utf8_stdout() -> None:
    """GBK 控制台下不要把检查结果变成 traceback（同一个坑在 relay_health 上踩过）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def run_check(name: str, relative: str) -> dict[str, object]:
    script = ROOT / relative
    if not script.exists():
        return {"name": name, "script": relative, "ok": False,
                "output": f"脚本不存在：{relative}"}
    done = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)
    output = ((done.stdout or "") + (done.stderr or "")).strip()
    return {"name": name, "script": relative, "ok": done.returncode == 0,
            "returncode": done.returncode, "output": output}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="文档检查统一入口（CI 与本地钩子共用）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)
    _enable_utf8_stdout()

    results = [run_check(name, relative) for name, relative in CHECKS]
    passed = all(result["ok"] for result in results)

    if args.json:
        print(json.dumps({"passed": passed, "checks": results}, ensure_ascii=False))
    else:
        for result in results:
            mark = "PASS" if result["ok"] else "FAIL"
            print(f"[{mark}] {result['name']}（{result['script']}）")
            for line in str(result["output"]).splitlines():
                print(f"    {line}")
        print(f"[verdict] {'文档检查全部通过' if passed else '文档检查存在失败项'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
