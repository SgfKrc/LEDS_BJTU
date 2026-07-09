#!/usr/bin/env python3
"""
QLH 边缘推理系统 — Linux 启动包装器
用法: qlh-launcher [--headless] [--check-only]
"""
import os
import sys
import site

# 确保应用源码在 path 中
APP_DIR = "/opt/qlh-edge-inference"
SRC_DIR = os.path.join(APP_DIR, "src")
sys.path.insert(0, SRC_DIR)

# 添加 venv site-packages（兼容 Python 3.4+，替代已废弃的 activate_this.py）
VENV_DIR = os.path.join(APP_DIR, "venv")
if os.path.isdir(VENV_DIR):
    for d in os.listdir(os.path.join(VENV_DIR, "lib")):
        site_packages = os.path.join(VENV_DIR, "lib", d, "site-packages")
        if os.path.isdir(site_packages):
            site.addsitedir(site_packages)
            break

# 将控制权交给启动器
if __name__ == "__main__":
    # launcher.py 在 SRC_DIR 的父级
    launcher_path = os.path.join(APP_DIR, "bin", "__launcher_main__.py")
    if os.path.isfile(launcher_path):
        exec(open(launcher_path).read())
    else:
        # 直接导入并运行 main
        from launcher import main
        main()
