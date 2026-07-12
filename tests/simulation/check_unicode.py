#!/usr/bin/env python3
"""
检查测试文件中的 Unicode 符号

统计每个文件中 ✓、✗、⚠ 的数量
"""

import os
from pathlib import Path


def count_unicode_symbols(file_path: str) -> dict:
    """统计文件中的 Unicode 符号数量

    Args:
        file_path: 文件路径

    Returns:
        符号统计字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 统计 Unicode 符号
    counts = {
        'check_mark': content.count('\u2713'),  # ✓
        'cross_mark': content.count('\u2717'),  # ✗
        'warning': content.count('\u26A0'),     # ⚠
    }

    return counts


def main():
    """主函数"""
    # 获取当前目录
    current_dir = Path(__file__).parent

    # 查找所有 Python 文件
    python_files = list(current_dir.glob('*.py'))

    print("Unicode 符号统计报告")
    print("=" * 60)
    print()

    # 统计每个文件中的 Unicode 符号
    total_counts = {
        'check_mark': 0,
        'cross_mark': 0,
        'warning': 0,
    }

    for file_path in python_files:
        counts = count_unicode_symbols(str(file_path))
        total = sum(counts.values())

        if total > 0:
            print(f"{file_path.name}:")
            print(f"  check_mark (U+2713): {counts['check_mark']}")
            print(f"  cross_mark (U+2717): {counts['cross_mark']}")
            print(f"  warning (U+26A0): {counts['warning']}")
            print(f"  Total: {total}")
            print()

            # 累加总数
            for key in total_counts:
                total_counts[key] += counts[key]

    print("=" * 60)
    print("总计:")
    print(f"  check_mark (U+2713): {total_counts['check_mark']}")
    print(f"  cross_mark (U+2717): {total_counts['cross_mark']}")
    print(f"  warning (U+26A0): {total_counts['warning']}")
    print(f"  Total: {sum(total_counts.values())}")


if __name__ == "__main__":
    main()
