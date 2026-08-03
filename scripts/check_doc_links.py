# -*- coding: utf-8 -*-
"""P1-6 文档收口：检查仓库内 Markdown 相对链接是否指向存在文件。

覆盖范围：docs/、README.md、packaging/README.md、gateway/README.md 等。
- 提取 [text](url) 与 [text](url "title") 形式链接
- 跳过 http(s)://、mailto:、# 锚点、<...> 自动链接
- 解码 URL 编码，去掉 #锚点 后按相对路径解析
- 报告：目标不存在 / 指向目录 / 其他问题
"""
import os
import re
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LINK_RE = re.compile(r'\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

SKIP_PREFIX = ('http://', 'https://', 'mailto:', 'ftp://', '#')


def iter_md_files():
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, 'docs')):
        for f in files:
            if f.endswith('.md'):
                yield os.path.join(dirpath, f)
    for f in ['README.md', 'packaging/README.md', 'gateway/README.md',
              'tests/simulation/README.md']:
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
