# QLH 边缘推理系统 — 打包文档

## 目录总览

```
项目根目录/
├── dist/                          # ★ PyInstaller 输出（安装包源文件）
│   └── QLH-Edge-Inference/
│       ├── QLH-Edge-Inference.exe # 主程序入口
│       └── _internal/             # 运行库 + Python 模块
│
├── packaging/                     # 打包配置与脚本
│   ├── launcher.py               # 打包版启动器（Tailscale 检查 → 模型下载 → 启动服务）
│   ├── qlh-cpu.spec              # PyInstaller 规格文件（集显版）
│   ├── setup.iss                 # Inno Setup 安装包脚本
│   ├── requirements-cpu.txt      # CPU-only 依赖清单
│   ├── build-cpu.bat             # 一键 PyInstaller 打包
│   ├── build-installer.bat       # 一键编译安装包（需要 Inno Setup 6）
│   ├── serve.py                  # 简易 HTTP 文件服务器（内网分发安装包）
│   ├── leds.ico                  # 程序图标
│   ├── scripts/
│   │   └── convert_to_gguf.py    # Safetensors → GGUF 转换工具
│   ├── dist/                     # Inno Setup 输出（安装包 .exe）
│   └── README.md                 # 本文件
│
├── .venv-packaging/              # 打包专用虚拟环境（torch CPU + PyInstaller）
├── frontend/dist/                # React 前端构建产物（PyInstaller 打包进 EXE）
└── src/                          # Python 源码（PyInstaller 从 launcher.py 追踪导入）
```

## 两大产物

| 步骤 | 工具 | 输入 | 输出 |
|------|------|------|------|
| **1. 程序打包** | PyInstaller | `qlh-cpu.spec` + `launcher.py` + `src/` + `frontend/dist/` | `dist/QLH-Edge-Inference/` |
| **2. 安装包编译** | Inno Setup 6 | `setup.iss` + `dist/QLH-Edge-Inference/` | `packaging/dist/QLH-Edge-Inference-Setup-vX.X.X.exe` |

> **关键路径约定**：PyInstaller 始终从**项目根目录**运行，输出到根目录 `dist/`。Inno Setup 从 `packaging/setup.iss` 通过 `..\dist\QLH-Edge-Inference` 引用它。不要再从 `packaging/` 目录运行 PyInstaller，否则会导致路径混乱。

## 快速开始

### 前置条件

- Python 3.10+（推荐 3.12）
- Node.js 18+（前端构建）
- Inno Setup 6（仅编译安装包时需要）
- Windows 10/11 64-bit

### 方案 A：一键脚本

```bat
cd packaging

:: 步骤 1：PyInstaller 打包（自动安装 CPU 依赖 + 构建前端）
build-cpu.bat

:: 步骤 2：编译安装包
build-installer.bat
```

### 方案 B：手动分步

```bash
# 0. 创建并激活打包虚拟环境
python -m venv .venv-packaging
.venv-packaging\Scripts\activate

# 1. 安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2. 构建前端
cd frontend && npm install && npx vite build && cd ..

# 3. PyInstaller 打包（★ 从项目根目录运行）
pyinstaller packaging/qlh-cpu.spec --noconfirm
# 输出: dist/QLH-Edge-Inference/

# 4. 编译 Inno Setup 安装包（需要 Inno Setup 6）
cd packaging
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
# 输出: packaging/dist/QLH-Edge-Inference-Setup-vX.X.X.exe
```

## 文件说明

### `qlh-cpu.spec` — PyInstaller 规格文件

定义了打包的入口、依赖、排除项：

| 配置项 | 说明 |
|--------|------|
| `Analysis` | 入口 `launcher.py`，搜索路径含 `src/` |
| `binaries` | llama.cpp 原生 DLL |
| `datas` | React 前端 `frontend/dist/` → `_internal/frontend/dist/` |
| `hiddenimports` | 动态导入模块（`llama_cpp`, `graph_orchestrator`, `local_store`, `psycopg2` 等） |
| `excludes` | 排除 `tkinter`, `test`, `pydoc` |
| `strip=True` (EXE) | 去除主程序调试符号，减少杀软误报。**COLLECT 不用 strip**，否则会损坏第三方 DLL |
| `console=True` | 保留控制台窗口（Tailscale 首次引导需要） |

### `setup.iss` — Inno Setup 安装脚本

| 配置项 | 说明 |
|--------|------|
| `MyAppSourceDir` | `..\dist\QLH-Edge-Inference`（指向根目录 PyInstaller 输出） |
| `MyAppVersion` | 版本号，应与 `src/__init__.py` 一致 |
| `restartreplace` | 覆盖安装时锁定文件标记为重启后替换 |
| `InitializeSetup` | 安装前检测旧版本，提示自动卸载 |
| 压缩 | LZMA2/max，SolidCompression |

### `launcher.py` — 打包版启动器

与开发模式不同，打包版启动器负责：

1. Tailscale 组网检查（首次引导加入）
2. 模型文件检测（缺失则弹出下载引导）
3. 引擎选择（llama.cpp vs PyTorch）
4. 后台启动 FastAPI（端口 8000）
5. pywebview 原生窗口加载前端

### `serve.py` — 安装包分发服务器

通过 Tailscale 组网，让局域网/虚拟网内的其他节点无需 U 盘即可下载安装包。首页会同时列出 Windows PC 安装包、Android APK/AAB、`models.7z`：

```bash
cd packaging
python serve.py
# 默认端口 9090，启动后其他节点浏览器访问 http://<本机Tailscale IP>:9090 即可下载

# 如需指定端口：
python serve.py 9999
```

Android 安装包会从 `android/app/build/outputs/**/*.apk` / `*.aab` 自动扫描；若首页未显示，请先运行 `android/gradlew.bat assembleDebug`。

## 版本号更新清单

每次发新版本时，以下文件中的版本号需要同步更新：

| 文件 | 字段 | 示例 |
|------|------|------|
| `src/__init__.py` | `__version__` | `"0.1.5"` |
| `src/api_server.py` | `version=` | `"0.1.5"` |
| `packaging/setup.iss` | `MyAppVersion` | `"0.1.5"` |
| `packaging/build-installer.bat` | 安装包文件名 | `v0.1.5.exe` |

## 杀软误报处理

PyInstaller 打包的 EXE 可能被 Windows Defender 或第三方杀软误报。已采取的措施：

1. **`strip=True`** — 去除 EXE 和捆绑 DLL 的调试符号
2. **`upx=False`** — 不使用 UPX 压缩（UPX 壳是杀软常见误报源）
3. **launcher 使用 `socket` 代替 `netstat`** — 避免触发 BITS 行为检测
4. **launcher 使用 `shutil.which` 代替 `subprocess`** — 减少可疑进程调用

如果仍然被报毒，将安装目录 `C:\Program Files\QLH-Edge-Inference` 加入杀软白名单。

## 独显版 (CUDA) 构建

如需构建带 CUDA 支持的版本：

1. 使用项目根目录 `requirements.txt`（含 CUDA PyTorch）
2. 复制 `qlh-cpu.spec` → `qlh-cuda.spec`
3. 移除 `hiddenimports` 中 CPU-only 相关的排除
4. 确保 `torch` 为 CUDA 版本
5. `pyinstaller packaging/qlh-cuda.spec --noconfirm`

## 常见问题

**Q: 安装包大小为什么 ~950MB（压缩后 ~377MB）？**

A: 包含了完整的 Python 运行环境 + torch + transformers + llama.cpp + pywebview。代码本身只有几百 KB。

**Q: 模型文件在安装包里吗？**

A: 不包含。首次启动会自动检测并弹出下载引导（百度网盘 / ModelScope）。模型需放入 `models/` 目录。

**Q: 卸载时会删除 `models/` 目录吗？**

A: 默认不会。卸载程序会弹出确认框，默认选择「否」以保留模型文件；只有用户明确选择「是」时才会同时删除 `models/` 目录。

**Q: 安装后运行报「数据库密码错误」？**

A: 先确认卸载了旧版并手动删除了安装目录，再重新安装。旧版 `_internal/` 中可能残留了旧密码的缓存文件。

**Q: 为什么 `setup.iss` 用 `..\dist` 而不是 `dist`？**

A: `setup.iss` 位于 `packaging/`，`..\dist` 指向项目根目录的 `dist/`（PyInstaller 输出位置）。之前写 `dist\`（相对路径）会错误地指向 `packaging/dist/`（旧版残留）。
