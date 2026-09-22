# -*- coding: utf-8 -*-
"""P1-6 文档收口：检查仓库内 Markdown 相对链接是否指向存在文件。

覆盖范围：docs/、README.md、tests/simulation/README.md 等主仓文档。
- 提取 [text](url) 与 [text](url "title") 形式链接
- 跳过 http(s)://、mailto:、# 锚点、<...> 自动链接
- 解码 URL 编码，去掉 #锚点 后按相对路径解析
- 报告：目标不存在 / 指向目录 / 其他问题
"""
import os
import re
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LINK_RE = re.compile(r'\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

SKIP_PREFIX = ('http://', 'https://', 'mailto:', 'ftp://', '#')

_IGNORED_CACHE: dict[str, bool] = {}


def _is_gitignored(rel_path: str) -> bool:
    """目标路径是否被 .gitignore 忽略（= 不在版本控制里 ⇒ CI 上必然不存在）。

    为什么必须单独查这一点：本机会有**工作区独立目录**（`tools/`、`docs/agent_tool/` 等已被
    主仓裁撤的路径），所以「文件存在」在本机恒真 —— 本地检查全绿、CI 却红。
    主仓文档本就不应链接已从主仓移除的东西，因此这里**报错**而不是放过。
    """
    if rel_path in _IGNORED_CACHE:
        return _IGNORED_CACHE[rel_path]
    try:
        done = subprocess.run(['git', 'check-ignore', '-q', rel_path], cwd=ROOT,
                              capture_output=True, check=False)
        result = done.returncode == 0
    except (OSError, ValueError):
        result = False          # 环境里没有 git ⇒ 不应用这条规则，其余检查照旧
    _IGNORED_CACHE[rel_path] = result
    return result


def iter_md_files():
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, 'docs')):
        # Archived documents keep historical references and are outside the
        # active link quality gate after their one-time migration repair.
        dirs[:] = [name for name in dirs if name != 'archive']
        for f in files:
            if f.endswith('.md'):
                yield os.path.join(dirpath, f)
    for f in ['README.md', 'tests/simulation/README.md']:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            yield p


def check_file(md_path):
    """返回 (ok, problems)"""
    problems = []
    base = os.path.dirname(md_path)
    with open(md_path, encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            for m in LINK_RE.finditer(line):
                raw = m.group(1).strip()
                if raw.startswith(SKIP_PREFIX):
                    continue
                # 解码 + 去锚点
                target = urllib.parse.unquote(raw)
                target = target.split('#')[0]
                if not target:
                    continue
                # 相对路径解析（相对当前文件所在目录）
                abs_target = os.path.normpath(os.path.join(base, target))
                # 仓库外的相对路径（如 `../../qlh-release/docs/…`）是**本地工作区路径说明**，
                # 不是可校验的仓库内链接：它们指向与主仓并列的独立仓库/工作区目录，
                # 换一台机器或 CI 上必然不存在。这类引用不计入链接质量门。
                try:
                    inside_repo = not os.path.relpath(abs_target, ROOT).startswith('..')
                except ValueError:
                    # Windows 上跨盘符时 relpath 会抛 ValueError ⇒ 目标必然在仓库外。
                    # 不接住它，检查工具自己就会崩（比漏报一条链接更糟）。
                    inside_repo = False
                if not inside_repo:
                    continue
                rel_to_root = os.path.relpath(abs_target, ROOT).replace(os.sep, '/')
                # 仓库内但**未入库**的目标（工作区独立目录、已裁撤的工具目录）：
                # 本机存在、CI 上不存在 ⇒ 本地会全绿而 CI 红。主仓文档不应链接这类路径。
                if _is_gitignored(rel_to_root):
                    problems.append((lineno, raw, '目标未入库（CI 上不存在）'))
                    continue
                if not os.path.exists(abs_target):
                    problems.append((lineno, raw, '目标不存在'))
                elif os.path.isdir(abs_target):
                    problems.append((lineno, raw, '指向目录而非文件'))
    return problems


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    total_links = 0
    all_problems = []
    for md in iter_md_files():
        rel = os.path.relpath(md, ROOT)
        problems = check_file(md)
        n = 0
        with open(md, encoding='utf-8') as fh:
            for line in fh:
                n += len(LINK_RE.findall(line))
        total_links += n
        for lineno, raw, why in problems:
            all_problems.append((rel, lineno, raw, why))
    print(f'扫描文件数: {len(list(iter_md_files()))}  链接总数: {total_links}')
    if not all_problems:
        print('✅ 全部相对链接目标存在')
        return 0
    print(f'❌ 发现 {len(all_problems)} 个问题:')
    for rel, lineno, raw, why in all_problems:
        print(f'  {rel}:{lineno}  [{raw}]  ->  {why}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
