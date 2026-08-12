#!/usr/bin/env python3
"""README 中英双语同步检查（trivial 级别）。

断言：
1. 中英 README 顶部互链存在；
2. 中文 README 的每个 H2 章节都有英文映射（新增章节未补映射时报告）；
3. 英文 README 的每个 H2 都能反查到中文来源（防止英文版新增孤儿章节）。

中文 README 是唯一事实源；英文版允许对细节章节标 `Translation pending`，
但章节标题必须成对。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZH = ROOT / "README.md"
EN = ROOT / "docs" / "README.en.md"

# 章节映射（中文 H2 -> 英文 H2）。新增中文章节必须在此登记并同步英文版。
SECTION_MAP = {
    "📋 项目简介": "📋 Project Introduction",
    "🌐 Tailscale 组网（重要）": "🌐 Tailscale Networking (Important)",
    "🏗️ 项目架构": "🏗️ Project Architecture",
    "📦 环境依赖": "📦 Environment Dependencies",
    "🤖 模型下载": "🤖 Model Download",
    "🚀 快速开始": "🚀 Quick Start",
    "📊 量化效果": "📊 Quantization Results",
    "🧪 对照实验组": "🧪 Comparative Experiments",
    "📊 核心评判指标": "📊 Core Metrics",
    "👥 团队分工": "👥 Team",
    "📚 文档索引": "📚 Documentation Index",
    "📄 许可证": "📄 License",
}


def h2_titles(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^## (.+)$", text, re.M))


def main() -> int:
    errors: list[str] = []
    zh_text = ZH.read_text(encoding="utf-8")
    en_text = EN.read_text(encoding="utf-8")

    # 1) 互链
    if "docs/README.en.md" not in zh_text:
        errors.append("README.md 缺少指向 docs/README.en.md 的语言链接")
    if "README.en.md" not in en_text or "../README.md" not in en_text:
        errors.append("docs/README.en.md 缺少回指中文 README 的语言链接")

    zh_h2 = h2_titles(ZH)
    en_h2 = h2_titles(EN)

    # 2) 中文每个 H2 必须有映射，且映射目标存在于英文版
    for section in sorted(zh_h2):
        target = SECTION_MAP.get(section)
        if target is None:
            errors.append(f"中文章节未登记英文映射（请在 SECTION_MAP 与 README.en.md 同步补上）: {section}")
            continue
        if target not in en_h2:
            errors.append(f"英文版缺少章节标题: {target}（中文: {section}）")

    # 3) 英文每个 H2 必须能反查来源
    zh_by_en = {v: k for k, v in SECTION_MAP.items()}
    for section in sorted(en_h2):
        if section not in zh_by_en:
            errors.append(f"英文章节无中文来源（孤儿章节或映射缺失）: {section}")

    if errors:
        print("\n".join(errors))
        return 1
    print("README 双语同步检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
