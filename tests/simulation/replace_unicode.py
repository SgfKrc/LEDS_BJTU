#!/usr/bin/env python3
"""
批量替换测试文件中的 Unicode 符号

将所有 [OK] 替换为 [OK]
将所有 [FAIL] 替换为 [FAIL]
将所有 [WARN] 替换为 [WARN]
"""

import os
from pathlib import Path


def replace_unicode_symbols(file_path: str) -> int:
    """替换文件中的 Unicode 符号

    Args:
        file_path: 文件路径

    Returns:
        替换次数
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 替换 Unicode 符号
    replacements = [
        ('[OK]', '[OK]'),
        ('[FAIL]', '[FAIL]'),
        ('[WARN]', '[WARN]'),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # 统计替换次数
        count = sum(original_content.count(old) for old, _ in replacements)
        return count

    return 0


def main():
    """主函数"""
    # 获取当前目录
    current_dir = Path(__file__).parent

    # 查找所有 Python 文件
    python_files = list(current_dir.glob('*.py'))

    print(f"找到 {len(python_files)} 个 Python 文件")
    print()

    # 替换每个文件中的 Unicode 符号
    total_replacements = 0
    for file_path in python_files:
        count = replace_unicode_symbols(str(file_path))
        if count > 0:
            print(f"[OK] {file_path.name}: 替换了 {count} 个符号")
            total_replacements += count

    print()
    print(f"总计替换了 {total_replacements} 个 Unicode 符号")


if __name__ == "__main__":
    main()
