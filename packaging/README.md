# QLH 边缘推理系统 — 打包文档

> 状态：现行工程文档；产物大小和依赖版本以实际构建为准

## 目录总览

```
项目根目录/
├── dist/                          # ★ PyInstaller 输出（安装包源文件）
│   ├── QLH-Edge-Inference/        #   集显版（CPU-only torch）
│   └── QLH-Edge-Inference-CUDA/   #   独显版（CUDA torch）
│
├── packaging/                     # 打包配置 + 分发服务器（不再在此目录执行打包命令）
│   ├── launcher.py               # 主应用启动载荷（Tailscale 检查 → 模型下载 → 启动服务）
│   ├── qlh_launcher.py           # 独立 Bootstrap（GUI/TUI/更新，不导入推理依赖）
│   ├── updater.py                # 更新 CLI
│   ├── update_core.py            # 清单、版本、下载与 SHA-256 核心
│   ├── qlh-launcher.spec         # 独立 Launcher PyInstaller 规格
│   ├── setup-launcher.iss        # 独立 Launcher Setup
│   ├── build-launcher.bat        # 独立 Launcher 构建脚本
│   ├── qlh-cpu.spec              # PyInstaller 规格文件（集显版）
│   ├── qlh-cuda.spec             # PyInstaller 规格文件（独显版）
│   ├── setup.iss                 # Inno Setup 安装脚本 — 集显版
│   ├── setup-cuda.iss            # Inno Setup 安装脚本 — 独显版
│   ├── requirements-cpu.txt      # CPU-only 依赖清单（两个版本共用，torch 由 venv 决定）
│   ├── build-cpu.bat             # [旧] 一键脚本，已不推荐，请用下方 venv 方案
│   ├── build-cuda.bat            # [旧] 一键脚本，已不推荐，请用下方 venv 方案
│   ├── build-installer.bat       # Inno Setup 编译辅助脚本
│   ├── serve.py                  # ★ 极简 HTTP 文件分发服务器
│   ├── leds.ico                  # 程序图标
│   └── dist/                     # Inno Setup 输出（最终安装包 .exe）
│
├── .venv-packaging/              # 集显版打包专用 venv（torch CPU + PyInstaller）
├── .venv-packaging-cuda/         # 独显版打包专用 venv（torch CUDA + PyInstaller）
├── frontend/dist/                # React 前端构建产物（PyInstaller 打包进 EXE）
└── src/                          # Python 源码（PyInstaller 从 launcher.py 追踪导入）
```

> **★ 关键变化（v0.1.6+）**：`packaging/` 目录不再用于执行打包命令。打包命令从**项目根目录**运行，
> 使用根目录的两个独立 venv（`.venv-packaging/` 和 `.venv-packaging-cuda/`）。
> `packaging/` 仅维护配置文件（spec、iss、requirements）和分发服务器（serve.py）。

---

## 三类产物

| 步骤 | 工具 | 输入 | 输出 |
|------|------|------|------|
| **1. 程序打包** | PyInstaller | `qlh-cpu.spec` / `qlh-cuda.spec` + `launcher.py` + `src/` + `frontend/dist/` | `dist/QLH-Edge-Inference/` 或 `dist/QLH-Edge-Inference-CUDA/` |
| **2. 安装包编译** | Inno Setup 6 | `setup.iss` / `setup-cuda.iss` + `dist/` 中的 PyInstaller 输出 | `packaging/dist/QLH-Edge-Inference-Setup-vX.X.X.exe` |
| **3. 独立引导器** | PyInstaller + Inno Setup 6 | `qlh-launcher.spec` + `setup-launcher.iss` | `dist/QLH-Launcher/` + `packaging/dist/QLH-Launcher-Setup-vX.X.X.exe` |

> PyInstaller 始终从**项目根目录**运行，输出到根目录 `dist/`。Inno Setup 从 `packaging/` 运行对应的 `.iss` 文件，通过 `..\dist\` 引用 PyInstaller 输出。

独立引导器可单独构建，不需要 CUDA 环境，也不会修改项目全局解释器：

```powershell
packaging\build-launcher.bat
```

---

## 两个版本的打包环境

| | 集显版 (CPU) | 独显版 (CUDA) |
|---|---|---|
| **venv** | 项目根 `.venv-packaging/` | 项目根 `.venv-packaging-cuda/` |
| **torch** | CPU-only (`--index-url ...whl/cpu`) | CUDA 12.x（默认 `pip install torch`） |
| **PyInstaller spec** | `packaging/qlh-cpu.spec` | `packaging/qlh-cuda.spec` |
| **输出目录** | `dist/QLH-Edge-Inference/` | `dist/QLH-Edge-Inference-CUDA/` |
| **Inno Setup 脚本** | `packaging/setup.iss` | `packaging/setup-cuda.iss` |
| **安装包文件名** | `QLH-Edge-Inference-Setup-vX.X.X.exe` | `QLH-Edge-Inference-Setup-vX.X.X-CUDA.exe` |
| **安装包大小** | ~180 MB | ~1.7 GB |

---

## 快速开始

### 前置条件

- Python 3.10+（推荐 3.12）
- Node.js 18+（前端构建）
- Inno Setup 6（仅编译安装包时需要）
- Windows 10/11 64-bit

### 集显版 (CPU) 打包

```bash
# 0. 创建并激活集显版 venv（仅首次）
python -m venv .venv-packaging
.venv-packaging\Scripts\activate

# 1. 安装依赖（仅首次，或 requirements 变更时）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2. 构建前端（★ 从项目根目录）
cd frontend && npm install && npx vite build && cd ..

# 3. PyInstaller 打包（★ 从项目根目录，使用集显版 venv）
pyinstaller packaging/qlh-cpu.spec --noconfirm
# 输出: dist/QLH-Edge-Inference/

# 4. 编译 Inno Setup 安装包（需要 Inno Setup 6）
cd packaging
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
# 输出: packaging/dist/QLH-Edge-Inference-Setup-vX.X.X.exe
```

### 独显版 (CUDA) 打包

```bash
# 0. 创建并激活独显版 venv（仅首次）
python -m venv .venv-packaging-cuda
.venv-packaging-cuda\Scripts\activate

# 1. 安装依赖（仅首次，或 requirements 变更时）
pip install torch                         # ★ CUDA 版 torch（默认带 CUDA 12.x DLL）
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2. 构建前端（★ 从项目根目录，如已构建可跳过）
cd frontend && npm install && npx vite build && cd ..

# 3. PyInstaller 打包（★ 从项目根目录，使用独显版 venv）
pyinstaller packaging/qlh-cuda.spec --noconfirm
# 输出: dist/QLH-Edge-Inference-CUDA/

# 4. 编译 Inno Setup 安装包（需要 Inno Setup 6）
cd packaging
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup-cuda.iss
# 输出: packaging/dist/QLH-Edge-Inference-Setup-vX.X.X-CUDA.exe
```

> ⚠️ **关键**：两个 venv 不能混用。集显版 venv 安装 CPU-only torch，独显版 venv 安装 CUDA torch。
> 如果装错，PyInstaller 会打进错误的 torch 版本（集显版装了 CUDA torch → 体积从 180MB 膨胀到 1.8GB）。

---

## 文件说明

### PyInstaller Spec 文件

| 文件 | 用途 | torch 版本 |
|------|------|-----------|
| `packaging/qlh-cpu.spec` | 集显版 | CPU-only（~200 MB） |
| `packaging/qlh-cuda.spec` | 独显版 | CUDA 12.x（~3.5 GB，含 CUDA DLL） |

两个 spec 的 `hiddenimports` 相同，区别仅在于 venv 中安装的 torch 版本不同。
PyInstaller 会自动追踪 venv 中的 torch，不需要修改 spec 来切换 CPU/CUDA。

### Inno Setup 安装脚本

| 文件 | 对应版本 | 安装路径 | AppId |
|------|---------|---------|-------|
| `packaging/setup.iss` | 集显版 | `QLH-Edge-Inference` | `F1A3B5C7-...B2C` |
| `packaging/setup-cuda.iss` | 独显版 | `QLH-Edge-Inference-CUDA` | `F1A3B5C7-...B2D` |

两个版本使用不同的 AppId 和安装路径，可以在同一台机器上共存。

### `qlh_launcher.py` — 独立 Bootstrap Launcher

该入口只依赖 Python 标准库和 Windows 自带 Tk（Linux 无 GUI 时回退 TUI），不加载 torch、transformers、FastAPI 或模型。它发现已安装的主应用后启动 `QLH-Edge-Inference.exe` / `qlh-app`，也提供 `check`、`download`、`install` 更新命令。

```powershell
python packaging/qlh_launcher.py --gui
python packaging/qlh_launcher.py --tui
python packaging/qlh_launcher.py check --json
python packaging/qlh_launcher.py download --variant cpu
```

`packaging/setup-launcher.iss` 生成独立的 `QLH-Launcher-Setup-v0.1.8.1.exe`。它与 CPU/CUDA 主应用 Setup 使用不同 AppId，可单独安装、升级和卸载。

### `launcher.py` — 主应用启动载荷

与开发模式不同，打包版启动器负责：

1. Tailscale 组网检查（首次引导加入）
2. 模型文件检测（缺失则弹出下载引导）
3. 引擎选择（llama.cpp vs PyTorch）
4. 后台启动 FastAPI（端口 8000）
5. pywebview 原生窗口加载前端

### `serve.py` — 安装包分发服务器

通过 Tailscale 组网，让局域网/虚拟网内的其他节点无需 U 盘即可下载安装包：

```bash
cd packaging
python serve.py
# 默认端口 9090，浏览器访问 http://<本机Tailscale IP>:9090/
```

首页会列出：
- Windows PC 安装包（集显版 + 独显版）
- Android APK（Full / Lite，Debug / Release）
- PC 模型压缩包 `models_pc.7z`
- Android 模型压缩包 `models_android.7z`（仅包含 GGUF 模型）

分发服务器同时提供 `GET /latest.json`。清单只包含当前 `src.__version__` 对应的 PC/Android 资产，并生成大小与 SHA-256；Launcher 默认并行查询多个源，单源超时不会串行拖慢启动器。

## 版本号更新清单

每次发新版本时，以下文件中的版本号需要同步更新：

| 文件 | 字段 | 示例 |
|------|------|------|
| `src/__init__.py` | `__version__` | `"0.1.8.1"` |
| `src/api_server.py` | `version=` | `"0.1.8.1"` |
| `packaging/setup.iss` | `MyAppVersion` | `"0.1.8.1"` |
| `packaging/setup-cuda.iss` | `MyAppVersion` | `"0.1.8.1"` |
| `packaging/setup-launcher.iss` | `MyAppVersion` | `"0.1.8.1"` |
| `packaging/qlh_launcher.py` / `updater.py` | `LAUNCHER_VERSION` | `"0.1.8.1"` |
| `packaging/build-installer.bat` | 安装包文件名 | `v0.1.8.1` |
| `packaging/version.txt` | Launcher/清单默认版本 | `0.1.8.1` |
| `packaging/linux/build-deb.sh` | `VERSION=` | `0.1.8.1` |
| `packaging/linux/control-cpu` | `Version:` | `0.1.8.1` |
| `packaging/linux/control-cuda` | `Version:` | `0.1.8.1` |
| `android/app/build.gradle.kts` | `versionName` / `versionCode` | `"0.1.8.1"` / `5` |

## 杀软误报处理

PyInstaller 打包的 EXE 可能被 Windows Defender 或第三方杀软误报。已采取的措施：

1. **`strip=True`** — 去除 EXE 和捆绑 DLL 的调试符号
2. **`upx=False`** — 不使用 UPX 压缩（UPX 壳是杀软常见误报源）
3. **launcher 使用 `socket` 代替 `netstat`** — 避免触发 BITS 行为检测
4. **launcher 使用 `shutil.which` 代替 `subprocess`** — 减少可疑进程调用

如果仍然被报毒，将安装目录 `C:\Program Files\QLH-Edge-Inference` 加入杀软白名单。

## 常见问题

**Q: 为什么需要两个独立的 venv？**

A: 集显版需要 CPU-only torch（~200 MB），独显版需要 CUDA torch（~3.5 GB）。如果在同一个 venv 中切换，容易装错导致集显版膨胀到 1.8 GB。

**Q: 集显版安装包大小为什么 ~180 MB？**

A: 包含了 Python 运行环境 + CPU-only torch + transformers + llama.cpp + pywebview。代码本身只有几百 KB。

**Q: 独显版安装包大小为什么 ~1.7 GB？**

A: CUDA torch 自带 ~3.5 GB 的 CUDA DLL（`torch/lib/*.dll`），压缩后约 1.7 GB。

**Q: 模型文件在安装包里吗？**

A: 不包含。首次启动会自动检测并弹出下载引导（百度网盘 / ModelScope）。模型需放入 `models/` 目录。

**Q: 卸载时会删除 `models/` 目录吗？**

A: 默认不会。卸载程序会弹出确认框，默认选择「否」以保留模型文件。

**Q: 安装后运行报「数据库密码错误」？**

A: 先确认卸载了旧版并手动删除了安装目录，再重新安装。旧版 `_internal/` 中可能残留了旧密码的缓存文件。

---

## Ubuntu Linux .deb 打包（v0.1.8.1）

### 目录结构

```
packaging/linux/
├── build-deb.sh                  ← 一键构建脚本
├── control-cpu                   ← dpkg 控制文件 — 集显版
├── control-cuda                  ← dpkg 控制文件 — 独显版
├── postinst                      ← 安装后脚本
├── prerm                         ← 卸载前脚本
├── postrm                        ← 卸载后脚本
├── qlh-edge-inference.desktop    ← 桌面快捷方式
├── qlh-edge-inference.service    ← systemd 用户服务
├── launcher.sh                   ← /usr/local/bin 包装器
└── qlh.png                       ← 应用图标 (需从 leds.ico 转换)
```

### 安装目录布局 (FHS)

```
/opt/qlh-edge-inference/
├── bin/
│   ├── qlh-launcher              ← Python 3 包装器
│   ├── qlh-app                   ← 主应用启动载荷
│   ├── update_core.py / updater.py ← 独立更新核心与 CLI
│   ├── bjtu                      ← BJTU 统一入口（launcher/ui/tui/update/version）
│   └── __launcher_main__.py      ← 旧主应用启动模块
├── src/                          ← Python 源码
├── frontend/dist/                ← React 构建产物
├── models/                       ← 模型目录 (postinst 创建, 755)
├── logs/                         ← 日志目录 (postinst 创建, 1777 sticky)
├── venv/                         ← Python 虚拟环境 (pip 依赖)
/usr/share/applications/qlh-edge-inference.desktop
/usr/share/icons/hicolor/256x256/apps/qlh.png
/usr/lib/systemd/user/qlh-edge-inference.service
/usr/local/bin/qlh-launcher → ../../opt/qlh-edge-inference/bin/qlh-launcher
/usr/local/bin/bjtu → ../../opt/qlh-edge-inference/bin/bjtu
```

### 快速开始

**前置条件** (Ubuntu 22.04/24.04):
```bash
sudo apt install python3 python3-venv python3-pip python3-tk dpkg-dev zenity
```

**构建集显版 .deb**:
```bash
cd packaging/linux
chmod +x build-deb.sh
./build-deb.sh cpu
# 输出: qlh-edge-inference-cpu_0.1.8.1_amd64.deb
```

**构建独显版 .deb**:
```bash
./build-deb.sh cuda
# 输出: qlh-edge-inference-cuda_0.1.8.1_amd64.deb
```

### 安装与卸载

```bash
# 安装
sudo dpkg -i qlh-edge-inference-cpu_0.1.8.1_amd64.deb
sudo apt-get install -f   # 修复可能未满足的依赖

# 运行独立图形 Launcher
qlh-launcher --gui

# 直接运行普通界面
qlh-launcher app-ui

# 运行 (无头模式)
qlh-launcher --headless

# BJTU 统一入口
bjtu launcher
bjtu ui
bjtu tui

# 可选: 开机自启
sudo systemctl enable --now qlh-edge-inference

# 卸载 (保留模型文件)
sudo dpkg -r qlh-edge-inference-cpu

# 完全卸载 (包括模型)
sudo dpkg -r qlh-edge-inference-cpu && sudo rm -rf /opt/qlh-edge-inference
```

### 图标

Linux .deb 需要 PNG 格式图标。从 `packaging/leds.ico` 转换为 `packaging/linux/qlh.png` (256×256):

```bash
# 如果安装了 ImageMagick:
convert packaging/leds.ico packaging/linux/qlh.png
```

### 与 Windows 版本的区别

| | Windows | Linux |
|---|---|---|
| UI | pywebview (Edge WebView2) | 系统浏览器 (xdg-open) |
| 安装路径 | `C:\Program Files\QLH-Edge-Inference` | `/opt/qlh-edge-inference` |
| 打包工具 | PyInstaller + Inno Setup | dpkg-deb |
| 服务管理 | 手动 | systemd (可选) |
| Tailscale IP | 状态 JSON + ip -4 + 网卡 | 状态 JSON + ip -4 + 网卡 (相同) |
| 配置目录 | `%LOCALAPPDATA%\QLH-Edge-Inference` | `~/.config/qlh` |
| 对话框 | Win32 MessageBox | zenity → CLI 回退 |
