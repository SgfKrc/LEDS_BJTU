# QLH 边缘推理系统 — 打包说明

## 版本策略

本项目提供两种 Windows 安装包：

| 版本 | PyTorch | 大小 | 适用设备 |
|------|---------|------|---------|
| **集显版** (CPU-only) | torch CPU | ~800 MB | 无 NVIDIA 独显的笔记本/台式机 |
| **独显版** (CUDA) | torch CUDA | ~3 GB | 有 NVIDIA 独显的设备 |

当前目录包含集显版的打包配置。独显版可基于主 `requirements.txt` 类比构建。

## 集显版构建步骤

### 前置条件

- Python 3.10+
- Node.js 18+ (前端构建)
- Windows 10/11

### 步骤

```bash
# 1. 安装 CPU-only 依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r packaging/requirements-cpu.txt

# 2. 构建前端
cd frontend
npm install
npx vite build
cd ..

# 3. PyInstaller 打包
pip install pyinstaller
cd packaging
pyinstaller qlh-cpu.spec --noconfirm
```

输出: `packaging/dist/QLH-Dist/QLH-Edge-Inference.exe`

或一键构建:
```bat
packaging\build-cpu.bat
```

## 模型文件说明

模型文件 (~3.6GB) **不包含在安装包内**。首次启动时会自动弹出下载引导：
- 推荐：网盘下载（百度网盘）
- 备用：命令行下载（ModelScope / HuggingFace）

模型文件需放入 `models/qwen-1_8b-chat/` 目录。

## 独显版 (CUDA) 构建

如需构建带 CUDA 支持的版本，类比修改：

1. 使用主 `requirements.txt`（CUDA PyTorch）
2. 复制 `qlh-cpu.spec` → `qlh-cuda.spec`，移除 `excludes` 中的 CUDA 排除
3. 运行 `pyinstaller qlh-cuda.spec`

## 目录结构

```
packaging/
├── launcher.py           # 打包版启动器（模型检查 → 启动服务器）
├── qlh-cpu.spec          # PyInstaller spec（集显版）
├── requirements-cpu.txt  # CPU-only 依赖
├── build-cpu.bat         # 一键构建脚本
└── README.md             # 本文件
```
