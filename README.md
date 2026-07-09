# 🧠 轻量化大模型分布式边缘推理优化系统

**基于分层流水线的边缘分布式大模型推理系统**

模型量化 · 算子融合 · 分页KV缓存 · 图算法智能编排 · 多终端协同推理 · 可视化监控

**v0.1.7**

---

## 📋 项目简介

针对边缘设备（普通笔记本）显存不足、算力有限的痛点，将 Qwen-1.8B 大语言模型按 Transformer 层拆分，部署到多台边缘设备组成分布式推理集群。配合 INT4/INT8 模型量化、算子融合、轻量化分页KV缓存三大单机优化，以及**图算法智能编排**（最大带宽生成树 + DFS 路径搜索），实现低开销、低时延的边缘端大模型推理。

现已支持 **Windows PC + Linux + Android 三端**：PC 端（Windows/Linux）完整支持分布式层间拆分流水线，Android 端以 llama.cpp 本地完整推理或作为薄客户端将请求转发给 PC 集群。

### 软件版本分级（四级四种）

项目按硬件能力和使用场景划分为四种软件版本：

| 级别 | 软件版本 | 目标设备 | 核心能力 | 不包含/不推荐 |
|------|----------|----------|----------|---------------|
| 1 | **PC 集显版** | Windows / Linux 无 NVIDIA 独显的 PC | llama.cpp + GGUF CPU/集显推理、可作为轻量节点、基础分布式能力 | 重模型实验、CUDA 专属能力 |
| 2 | **PC 独显版** | Windows / Linux NVIDIA GPU 主节点 / 实验 PC | PyTorch + CUDA + bitsandbytes、支持 CPU 回退、后续支持多模型/重模型实验 | Android 极简化策略 |
| 3 | **Android 普通版** | Android 手机/平板 | 全有模式本地 GGUF 推理、全无模式转发 PC、SAF 模型目录、较完整设置、后续可研究完整任务 Worker | Transformer 层间拆分、重模型实验 |
| 4 | **Android 极简版** | 普通手机轻量入口 | 极简聊天、尽量压缩 APK/缓存/模型存储占用、单一推荐小模型/INT4 路线 | 完整 models 目录、日志管理、Worker 接收任务、高级控制面板 |

> Android 普通版和极简版的区别：普通版面向“完整移动客户端”，极简版面向“尽量小、尽量少设置、尽量低存储占用”的手机轻量入口。

### 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **智能编排** | 节点数 > 5 时自动启用图算法（最大带宽生成树 + DFS），替代纯算力权重分配 → [详见分布式资源调度系统](docs/分布式资源调度系统.md) |
| 🔗 **链式拓扑流水线** | 按最优路径排序节点，hidden states 逐节点传递，支持 KV Cache 增量解码 |
| 🔄 **双引擎架构** | PyTorch + bitsandbytes (CUDA) / llama.cpp + GGUF (CPU/集显)，自动切换 |
| 📋 **MLFQ 请求队列** | 三级反馈队列管理并发推理请求，短交互优先 + 老化防饥饿 + FIFO 兼容 → [详见调度文档](docs/分布式资源调度系统.md) |
| 🗄️ **多会话管理** | 本地 JSON 存储 + 云 PostgreSQL 双轨，断网自动降级 |
| 🌐 **Tailscale 组网** | 跨子网设备互联，首次启动自动引导加入 |
| 📦 **一键安装包** | PC 集显版 (~180 MB) / PC 独显版 (~1.7 GB) / Linux .deb (~200 MB) / Android 普通版 APK，含 Tailscale 检查 + 模型下载引导 + pywebview 原生窗口 |
| 🎛️ **管理面板** | 节点注册/注销、分层覆盖、角色转让、备用主节点、TCP 连接状态监控 |
| 📱 **Android 客户端** | 普通版支持全有模式（本地 GGUF 推理）/ 全无模式（转发给 PC 集群），极简版后续主打小体积轻量聊天 |

**应用场景**：智能终端 · 物联网 · 边缘计算 · 教育科研

---

## 🌐 Tailscale 组网（重要）

分布式推理模式依赖 **Tailscale** 实现跨子网设备互联。所有参与推理的节点（PC、Android）建议先安装 Tailscale 并加入同一网络。

### 安装 Tailscale

**PC 端**（Windows / macOS / Linux）：

> 🔗 https://tailscale.com/download

安装后用同一账号登录即可自动组网。

**Android 端**：

> 🔗 Google Play 搜索 "Tailscale" 安装，或从 APK Mirror 侧载

**验证组网**：

打开 Tailscale 控制台 https://login.tailscale.com/admin/machines ，确认所有节点均在线且分配了 `100.x.x.x` 地址。

### 为什么需要 Tailscale？

- 校园网 / 家庭网络通常不分配公网 IP，设备间无法直接互访
- Tailscale 基于 WireGuard 创建虚拟局域网，每个设备获得一个固定的 `100.x.x.x` 地址
- Windows 打包版启动器会自动检查 Tailscale 是否已安装并登录

---

## 🏗️ 项目架构

```
项目根目录
├── docs/                          # 项目文档
│   ├── 整体架构.md                 # 项目总览、拓扑架构、技术栈
│   ├── 核心技术原理.md              # 算子融合、QKV、量化、流水线、TCP协议
│   ├── 模块接口说明.md              # 模块功能职责与对外接口
│   ├── 运行流程&异常处理.md         # 全局配置、初始化流程、推理全链路、异常处理
│   ├── 测试与评判标准.md            # 对照实验组、评判指标、可视化形式
│   ├── 图算法.md                   # 最大带宽生成树 + DFS 路径搜索算法设计
│   ├── 分布式资源调度系统.md          # MLFQ 三级反馈队列 + 图算法层编排（原理与关系）
│   ├── 分布式推理流水线实施计划.md    # 链式拓扑、LAYER_FORWARD 协议、KV Cache 方案
│   ├── Android版本远期计划.md       # Android 端方案评估与规划
│   ├── Android SAF模型存储方案.md   # Android SAF 外部模型目录方案
│   └── 纯Embedding节点处理+前端连接状态列计划.md
├── src/                           # Python 源代码（PC 端）
│   ├── config.py                  # 全局配置（网络/模型/KV/分层/运行模式/图算法阈值）
│   ├── model_module.py            # 模型加载、量化、算子融合、层级拆分、前向推理
│   ├── llama_engine.py            # llama.cpp 引擎封装（CPU/集显 GGUF 推理）
│   ├── paged_kv_cache.py          # 轻量化分页KV缓存（内存页管理、动态分配）
│   ├── tcp_comm.py                # TCP主从通信（长连接、心跳、封包解包、张量序列化）
│   ├── scheduler.py               # 任务调度（节点管理、层分配、流水线控制、请求队列）
│   ├── graph_orchestrator.py      # ★ 图算法智能编排（最大带宽生成树 + DFS 路径搜索）
│   ├── device_profiler.py         # 设备画像采集（CPU/GPU/RAM/网络）
│   ├── api_server.py              # FastAPI 服务端（REST API + WebSocket）
│   ├── db.py                      # PostgreSQL 数据库连接池
│   ├── local_store.py             # 本地 JSON 存储（DB 不可用时自动降级）
│   └── model_downloader.py        # 模型下载引导（HuggingFace/ModelScope/百度网盘）
├── android/                       # Android 客户端（Kotlin + Jetpack Compose）
│   ├── app/
│   │   ├── build.gradle.kts       # Gradle 构建脚本（含 release 签名配置）
│   │   └── src/main/java/com/qlh/inference/
│   │       ├── data/              # Room 数据库 + DataStore 设置持久化
│   │       ├── network/           # OkHttp API 客户端 + ChatRepository
│   │       ├── service/           # InferenceService 前台 Service + ModelManager + LocalInferenceEngine
│   │       └── ui/                # ChatScreen / SettingsScreen / SessionListScreen
│   ├── keystore.properties        # release 签名配置（Git 忽略，需本地生成）
│   ├── qlh-release.jks            # release 签名密钥库（Git 忽略）
│   └── gradlew / gradlew.bat      # Gradle Wrapper（无需 Android Studio）
├── .venv-packaging/               # 集显版打包专用 venv（torch CPU + PyInstaller）
├── .venv-packaging-cuda/          # 独显版打包专用 venv（torch CUDA + PyInstaller）
├── packaging/                     # 打包配置 + 分发服务器（不含构建产物）
│   ├── launcher.py                # 打包版启动器（Tailscale → 模型检查 → 引擎选择 → 启动）
│   ├── serve.py                   # ★ 极简 HTTP 文件分发服务器（PC + Android + Linux 安装包）
│   ├── qlh-cpu.spec               # PyInstaller 规格文件（集显版）
│   ├── qlh-cuda.spec              # PyInstaller 规格文件（独显版，CUDA + CPU 回退）
│   ├── setup.iss                  # Inno Setup 安装脚本 集显版
│   ├── setup-cuda.iss             # Inno Setup 安装脚本 独显版
│   ├── requirements-cpu.txt       # CPU-only 依赖清单
│   ├── linux/                     # Linux .deb 打包
│   │   ├── build-deb.sh           # deb 构建脚本
│   │   ├── launcher.py            # Linux 跨平台启动器
│   │   ├── control-cpu / control-cuda  # dpkg 元数据
│   │   ├── postinst / prerm / postrm   # 安装/卸载脚本
│   │   ├── qlh-edge-inference.service  # systemd 服务单元
│   │   └── qlh-edge-inference.desktop  # 桌面入口
│   ├── dist/                      # ★ 最终安装包输出目录（Git 忽略）
│   └── README.md                  # 打包文档
├── frontend/                      # React 前端（Vite + FastAPI 后端代理）
│   └── src/
│       ├── App.jsx                # 主布局 & 设置状态管理
│       ├── api/client.js          # API 客户端封装
│       └── components/            # ChatPanel / AdminPanel / DevicePanel / SettingsModal 等
├── tests/                         # 单元测试（442 个）
├── scripts/                       # 工具脚本
│   ├── quantize_model.py          # 模型准备与量化验证
│   ├── benchmark_all.py           # 全量化档位基准测试
│   ├── benchmark_compile.py       # torch.compile 融合测试
│   └── convert_to_gguf.py         # Safetensors → GGUF 转换
├── models/                        # 模型文件存放目录（需自行下载）
│   ├── qwen-1_8b-chat/            # PC: Safetensors 格式
│   └── qwen-1_8b-chat-Q4_K_M.gguf # PC: GGUF 格式（llama.cpp 引擎）
├── logs/                          # 运行日志目录
├── requirements.txt               # Python 依赖清单
└── README.md                      # 本文件
```

### 硬件拓扑（3 节点流水线示例）

```
用户输入 → 主节点(Master)  → TCP → 从节点1(Client) → TCP → 从节点2(Client) → 结果回传
          Embed + L0-3          L4-14              L15-23 + LM Head
          独显主节点参与首段计算，不再仅协调调度
```

### Android 客户端双模式

```
┌──────────────────────────────┬──────────────────────────────┐
│  全有模式 (本地推理)           │  全无模式 (远程推理)           │
│                              │                              │
│  Android 本地 llama.cpp      │  Android 聊天 UI             │
│  GGUF Q4_K_M (~1.16 GB)      │  HTTP → PC 主节点             │
│  离线可用，不依赖网络          │  PC 集群分布式推理            │
└──────────────────────────────┴──────────────────────────────┘
```

### 软件分层架构

| 层级 | 功能 | 技术 |
|------|------|------|
| 应用层 | 可视化交互 & 节点管理 & 性能监控 | React + Jetpack Compose (Android) |
| 调度层 | 任务调度、指令分发、状态管理、请求队列 | Python threading + 图算法 |
| 通信层 | TCP长连接、粘包处理、心跳、张量序列化 | Python socket + struct |
| 推理层 | 双引擎：模型加载、量化、融合、KV缓存 | PyTorch (CUDA) / llama.cpp (CPU / Android) |
| 存储层 | 对话持久化、节点注册、配置管理 | PostgreSQL + 本地 JSON 降级 + Room (Android) |
| 基础层 | 运行环境 | Python / CUDA / bitsandbytes / llama.cpp |

---

## 📦 环境依赖

### 核心框架

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 开发环境: 3.12.10 |
| PyTorch | ≥ 2.2.0 | CUDA 版本用于独显；CPU 版本用于集成显卡 |
| **transformers** | **≥ 4.45, < 5.0** | ⚠️ 必须保持 4.x！5.x 移除了 `load_in_4bit`/`load_in_8bit` |
| accelerate | ≥ 1.0.0 | 模型加载加速（bitsandbytes 依赖） |

### 模型量化

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| bitsandbytes | ≥ 0.45.0 | INT4/INT8 量化（独显必装，集显可选） |

### CPU/集显推理引擎

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| llama-cpp-python | ≥ 0.3.0 | CPU 优化 GGUF 推理，3-5x 快于 PyTorch CPU |

### Web 可视化

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| fastapi | ≥ 0.110.0 | API 后端框架 |
| uvicorn[standard] | ≥ 0.29.0 | ASGI 服务器 |
| pywebview | ≥ 5.0 | 打包版原生窗口（替代浏览器） |
| python-multipart | ≥ 0.0.12 | 文件上传支持 |

### 数据库

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| psycopg2-binary | ≥ 2.9 | PostgreSQL 客户端（云端同步用，可选） |

### 网络（分布式模式必装）

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| **Tailscale** | 最新版 | 跨子网虚拟组网，所有分布式节点必须安装 |

> 🔗 下载: https://tailscale.com/download

### 工具

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| tqdm | ≥ 4.65.0 | 进度条 |
| psutil | ≥ 5.9.0 | 系统资源监控 |

### 前端

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Node.js | ≥ 18 | 前端构建 |
| npm | — | 包管理器 |

### Android 客户端

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Android SDK | API 34+ | 编译目标 |
| Gradle | 8.11+ | Wrapper 已内置，无需单独安装 |
| Kotlin | 2.1.0 | 通过 Gradle 自动下载 |
| Java | JDK 17 | 编译必需 |

> Android 客户端**不需要 Android Studio**，有 JDK + Android SDK 命令行工具即可通过 `gradlew.bat` 构建。

### 一键安装

```bash
# Python 依赖（PostgreSQL 客户端可选，不装也不影响单机模式）
pip install -r requirements.txt

# 可选：PostgreSQL 数据库驱动（分布式集群节点注册/配置同步）
pip install psycopg2-binary

# 前端依赖
cd frontend && npm install && cd ..
```

---

## 🤖 模型下载

本项目使用 **Qwen-1.8B-Chat** 模型。支持两种格式：

| 格式 | 引擎 | 大小 | 适用场景 |
|------|------|------|---------|
| **Safetensors** | PyTorch (CUDA) | ~3.5 GB | 独显推理、分布式流水线 |
| **GGUF Q4_K_M** | llama.cpp (CPU / Android) | ~1.16 GB | 集显/CPU、单机推理、Android 本地推理 |

### Safetensors 格式（PyTorch / 分布式）

**方式一：ModelScope（推荐，国内更快）**

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
```

**方式二：Hugging Face**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen-1.8B-Chat --local-dir models/qwen-1_8b-chat
```

**方式三：百度网盘**

> 🔗 https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3 | 提取码：vtp3

### GGUF 格式（llama.cpp / PC CPU 引擎）

```bash
# 下载推荐版本 Q4_K_M (~1.16 GB)
huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf \
  Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/
```

| 量化 | 大小 | 说明 |
|------|------|------|
| Q3_K_M | ~0.94 GB | 低质量，极限内存 |
| **Q4_K_M** ⭐ | **~1.16 GB** | **推荐 — 速度/质量最佳平衡** |
| Q5_K_M | ~1.31 GB | 更高质量 |
| Q8_0 | ~1.82 GB | 近无损 |

### GGUF 格式（Android 本地推理）

Android 全有模式下，模型需放在**用户选择的外部目录**中（SAF `ACTION_OPEN_DOCUMENT_TREE`），**不放在应用内部存储**，这样卸载 APK 时模型会默认保留。

**Android 模型存放位置**：

| 推荐位置 | 说明 |
|----------|------|
| `Download/QLH/models/` | 手机内置的下载目录，卸载 APK 不会删除 |
| 用户自选的外部 SD 卡目录 | 通过 SAF 授权的任意目录 |

**获取方式**：

1. **PC 分发**：在 PC 上启动分发服务器，Android 浏览器下载后移动到 SAF 模型目录

   ```bash
   cd packaging
   python serve.py
   ```

2. **直接下载**：Android 浏览器访问 Hugging Face 或通过 USB 传文件

3. **后续**：应用内会提供从 PC 主节点直接下载到 SAF 目录的功能

**操作流程**：

```text
打开应用 → 设置 → 切换"全有模式" → 模型管理 → 选择目录
  → 选择包含 .gguf 的目录 → 扫描 → 选中模型 → 完成
```

> 详细方案参见 [Android SAF 模型存储方案](docs/Android SAF模型存储方案.md)

---

## 🚀 快速开始

### 开发模式（PC）

```bash
# 终端 1：启动 Python 后端（从项目根目录运行）
python src/api_server.py

# 终端 2：启动前端开发服务器（可选，后端已内置前端构建产物）
cd frontend && npm run dev
```

后端就绪后：
- **后端直连**：`http://localhost:8000`（含前端，`npm run build` 后）
- **开发前端**：`http://localhost:5173`（Vite 热更新，代理到 8000）

### 单机模式（PC）

修改 `src/config.py`：`RUN_MODE = "single"`，然后：

```bash
python src/api_server.py
```

### 分布式模式（PC）

> ⚠️ 前提：所有参与节点已安装 Tailscale 并用同一账号登录。

**主节点**：

```bash
python src/api_server.py
# 在管理面板启用"分布式推理"，配置 Tailscale 组网
```

**从节点**：

```bash
python src/api_server.py
# 在管理面板输入主节点 Tailscale IP，点击"连接主节点"
```

> 系统会自动完成：节点注册 → 设备画像上报 → 层分配计算 → 分层配置推送。

### 打包版（Windows 安装包）

提供两个版本，按需选择：

| 版本 | 安装包 | 典型大小 | 适用场景 |
|------|--------|---------|---------|
| **集显版** | `QLH-Edge-Inference-Setup-vX.X.X.exe` | ~180 MB | CPU / 集成显卡节点（从节点） |
| **独显版** | `QLH-Edge-Inference-Setup-vX.X.X-CUDA.exe` | ~1.7 GB | NVIDIA GPU 节点（主节点），无 GPU 时自动回退 CPU |

**集显版 (CPU) 构建**：

```bash
# 0. 创建并激活集显版 venv（仅首次）
python -m venv .venv-packaging
.venv-packaging\Scripts\activate

# 1. 安装依赖（仅首次）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2. 构建前端
cd frontend && npm install && npx vite build && cd ..

# 3. PyInstaller 打包（★ 从项目根目录运行）
pyinstaller packaging/qlh-cpu.spec --noconfirm

# 4. Inno Setup 安装包编译
cd packaging
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
```

**独显版 (CUDA) 构建**（需另一独立 venv）：

```bash
# 0. 创建并激活独显版 venv（仅首次）
python -m venv .venv-packaging-cuda
.venv-packaging-cuda\Scripts\activate

# 1. 安装依赖（仅首次，先 torch 后共享依赖，不会互相覆盖）
pip install torch                        # ★ CUDA 12.x（默认），不是 CPU 版
pip install -r packaging/requirements-cpu.txt
pip install pyinstaller

# 2-4. 同集显版，但 spec 和 iss 分别用 qlh-cuda.spec / setup-cuda.iss
pyinstaller packaging/qlh-cuda.spec --noconfirm
cd packaging && "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup-cuda.iss
```

> ⚠️ **关键**：两个版本使用**不同的独立 venv**（`.venv-packaging/` vs `.venv-packaging-cuda/`）。
> 不能混用——集显版 venv 必须装 CPU-only torch，独显版 venv 必须装 CUDA torch。
> 装错会导致集显版体积从 180 MB 膨胀到 1.8 GB。
>
> 安装后双击桌面快捷方式即可启动，无需配置 Python 环境。卸载时会询问是否同时删除 `models/` 目录，默认保留模型文件。
>
> 详细打包流程参见 [packaging/README.md](packaging/README.md)。

### Linux 安装包 (.deb)

提供与 Windows 集显版对应的 `.deb` 安装包，适用于 Ubuntu 22.04+ / Debian 12+：

| 版本 | 安装包 | 典型大小 | 适用场景 |
|------|--------|---------|---------|
| **CPU 版** | `qlh-edge-inference-cpu_0.1.7_amd64.deb` | ~200 MB | CPU / 集成显卡节点 |
| **CUDA 版** | `qlh-edge-inference-cuda_0.1.7_amd64.deb` | ~1.8 GB | NVIDIA GPU 节点 |

**构建**（需 Ubuntu/Debian 环境）：

```bash
cd packaging/linux
bash build-deb.sh cpu     # 集显版
bash build-deb.sh cuda    # 独显版
```

**安装**：

```bash
sudo dpkg -i qlh-edge-inference-cpu_0.1.7_amd64.deb
# 安装后自动注册 systemd 服务、桌面入口和 /usr/local/bin/qlh-launcher
```

**使用**：

```bash
qlh-launcher              # 桌面模式（浏览器打开前端）
qlh-launcher --headless   # 无头模式（仅 API，适合服务器）
sudo systemctl enable --now qlh-edge-inference  # 开机自启
```

> 前置依赖：`python3` (≥ 3.10)、`python3-venv`、`tailscale`（分布式模式）。安装包内置独立 venv，不污染系统 Python。

### Android 客户端

> 前提：已安装 JDK 17 + Android SDK（API 34+），SDK 路径配置在 `android/local.properties`

**编译**（无需 Android Studio）：

```bash
cd android

# Debug APK（未压缩，开发用）
./gradlew.bat assembleDebug

# Release APK（R8 压缩 + 签名，分发用）
./gradlew.bat assembleRelease
```

产物：

| 产物 | 路径 | 典型大小 | 说明 |
|------|------|---------|------|
| Full Debug | `android/app/build/outputs/apk/full/debug/app-full-debug.apk` | ~29 MB | 含 llama.cpp native 后端 |
| Full Release | `android/app/build/outputs/apk/full/release/app-full-release.apk` | **~6.7 MB** | R8 + native strip |
| Lite Release | `android/app/build/outputs/apk/lite/release/app-lite-release.apk` | **~1.5 MB** | 纯薄客户端，不含 native 库 |

**安装**：

```bash
adb install android/app/build/outputs/apk/full/release/app-full-release.apk
```

**使用**：

1. 启动 App → 底部导航选择「设置」
2. 全无模式：输入 PC 主节点 Tailscale IP 和端口 → 测试连接 → 开始对话
3. 全有模式：切换模式 → 选择包含 `.gguf` 的 SAF 外部目录 → 扫描并选中模型 → 离线推理

### 安装包分发服务器

在同一 Tailscale 网络内分发安装包，让其他设备浏览器直接下载：

```bash
cd packaging
python serve.py
# 默认端口 9090，浏览器访问 http://<本机Tailscale IP>:9090/
```

首页会列出：

- Windows PC 安装包 (.exe)
- Linux 安装包 (.deb)
- Android Full / Lite APK
- 模型压缩包 `models.7z`

> 其他设备（包括 Android 手机）直接浏览器打开链接即可下载。

---

## 📊 量化效果

### CUDA 独显（PyTorch + bitsandbytes）

> 测试环境: NVIDIA RTX GPU + CUDA 12.6 + PyTorch 2.12.0 + Qwen-1.8B-Chat (24层)

| 配置 | GPU 显存 | 推理速度 | 备注 |
|------|---------|----------|------|
| FP16 | 3.47 GB | 53.2 tok/s | 基线对照组 |
| FP16 + compile | 3.47 GB | 55.1 tok/s | 算子融合 +3.6% |
| INT8 | 2.30 GB | 9.8 tok/s | 省显存但速度损失大 |
| **INT4** ⭐ | **1.75 GB** | **28.7 tok/s** | **推荐边缘设备：显存减半** |

### CPU / 集显（llama.cpp + GGUF）

> 测试环境: Intel i5-12400F / AMD R5 5600 + 16GB RAM + Windows 11

| 引擎 | 量化 | 内存 | 推理速度 | 备注 |
|------|------|------|----------|------|
| PyTorch CPU | FP16 | ~3.5 GB | ~3 tok/s | 无 CUDA 回退 |
| llama.cpp | Q4_K_M | ~1.2 GB | **~12 tok/s** | **推荐 CPU/集显** |

> llama.cpp 相比 PyTorch CPU：内存 **-65%**，速度 **+300%（3-5x）**

### Android 本地推理（预估）

| 芯片 | 等级 | Q4_K_M tok/s | 峰值 RAM |
|------|------|-------------|----------|
| 骁龙 8 Gen 3 | 旗舰 | 12-18 | 1.8 GB |
| 骁龙 8+ Gen 1 | 次旗舰 | 8-12 | 1.8 GB |
| 骁龙 865 | 中端 | 5-8 | 1.8 GB |

---

## 🧪 对照实验组

| 实验组 | 量化 | 算子融合 | KV缓存 | 编排策略 | 部署模式 |
|--------|------|----------|--------|----------|----------|
| 基线组 | FP16 | 无 | 传统KV | — | 单机 |
| 实验组1 | INT4 | 无 | 传统KV | — | 单机 |
| 实验组2 | INT4 | 融合 | 传统KV | — | 单机 |
| 实验组3 | INT4 | 融合 | 分页KV | — | 单机 |
| 实验组4 | INT4 | 融合 | 分页KV | 简单权重 | 分布式(3节点) |
| 实验组5 | INT4 | 融合 | 分页KV | 🧠 图算法 | 分布式(>5节点) |

---

## 📊 核心评判指标

- **显存占用**：量化、分页KV优化效果
- **推理时延 / Token生成速度**：算子融合、流水线延迟
- **网络带宽利用率**：图算法编排 vs 简单权重分配
- **CPU负载 / 网络延迟**：分布式通信开销
- **对话通顺度**：量化精度损失评估
- **长时间运行稳定性**：断线重连、心跳恢复、缓存清理

---

## 👥 团队分工

| 小组 | 职责 |
|------|------|
| 模型优化组 | 文献调研、模型量化、算子融合、KV缓存优化 |
| 分布式架构组 | 分布式架构设计、通信协议开发、多机调度逻辑 |
| 前端与文档组 | Web可视化平台、性能监控模块、文档与演示材料 |

**指导教师**：高博 副教授（北京交通大学软件学院）

---

## 📚 文档索引

### 设计文档

- [整体架构](docs/整体架构.md)
- [核心技术原理](docs/核心技术原理.md)
- [模块接口说明](docs/模块接口说明.md)
- [运行流程&异常处理](docs/运行流程&异常处理.md)
- [测试与评判标准](docs/测试与评判标准.md)

### 专项文档

- [图算法智能编排](docs/图算法.md) — 最大带宽生成树 + DFS 路径搜索
- [分布式推理流水线实施计划](docs/分布式推理流水线实施计划.md) — 链式拓扑、LAYER_FORWARD 协议、KV Cache
- [Android 版本远期计划](docs/Android版本远期计划.md) — Android 端方案评估、全有/全无模式
- [Android SAF 模型存储方案](docs/Android SAF模型存储方案.md) — SAF 外部目录、`/proc/self/fd` 加载、缓存副本 fallback
- [PC 与 Android 端交互体验优化计划](docs/PC与Android端交互体验优化计划.md) — 四级版本分级、交互优化、日志、Worker 与远期任务链规划
- [纯 Embedding/LM Head 节点处理计划](docs/纯Embedding节点处理+前端连接状态列计划.md)

### 工程文档

- [打包说明](packaging/README.md) — PyInstaller + Inno Setup 打包流程

---

## 📄 许可证

本项目为北京交通大学 2026 年大学生创新创业训练计划项目。

---

© 2026 北京交通大学 · 项目团队
