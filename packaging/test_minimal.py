"""最小化测试 — 验证 PyInstaller 打包本身是否正常"""
import sys
import os

print("=" * 50)
print("MINIMAL TEST — PyInstaller 启动正常")
print(f"Python: {sys.version}")
print(f"cwd: {os.getcwd()}")
print(f"__file__: {__file__}")
print("=" * 50)

# 尝试导入关键模块
print("\n测试导入...")
try:
    import webview
    print(f"  webview: OK ({webview})")
except Exception as e:
    print(f"  webview: FAIL — {e}")

try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  torch: OK (CUDA={cuda})")
except Exception as e:
    print(f"  torch: FAIL — {e}")

try:
    import uvicorn
    print(f"  uvicorn: OK")
except Exception as e:
    print(f"  uvicorn: FAIL — {e}")

print("\n测试完成！")
input("按 Enter 退出...")
