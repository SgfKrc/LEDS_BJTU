#!/usr/bin/env python3
"""ModelScope 权重下载器（H 盘友好）——MSYS curl 写 H 盘失败，Python 原生路径无此问题。

支持断点续传（Range）、已存在跳过、失败重试。
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://modelscope.cn/api/v1/models/{org}/{name}/repo?Revision=master&FilePath={file}"
RETRIES = 3


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(RETRIES):
        try:
            headers = {}
            if dest.exists() and dest.stat().st_size > 0:
                headers["Range"] = f"bytes={dest.stat().st_size}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as f:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            if total == 0 and dest.stat().st_size == 0:
                # 空响应重试（可能限流）
                time.sleep(3)
                continue
            print(f"  OK {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
            return True
        except Exception as exc:
            print(f"  重试 {attempt + 1}/{RETRIES}: {exc}", flush=True)
            time.sleep(5)
    return False


def main() -> int:
    # 参数: org/name dest_dir file1 file2 ...
    org, name, dest_dir, *files = sys.argv[1:]
    dest = Path(dest_dir)
    ok = True
    for f in files:
        if f.startswith("@"):
            # 文件列表文件
            for line in Path(f[1:]).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    ok = download(BASE.format(org=org, name=name, file=line),
                                  dest / line) and ok
        else:
            ok = download(BASE.format(org=org, name=name, file=f), dest / f) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
