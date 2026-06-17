# 🧠 轻量化大模型分布式边缘推理优化系统

**基于分层流水线的边缘分布式大模型推理系统**

模型量化 · 算子融合 · 分页KV缓存 · 图算法智能编排 · 多终端协同推理 · 可视化监控

**v0.1.4**

---

## 📋 项目简介

针对边缘设备（普通笔记本）显存不足、算力有限的痛点，将 Qwen-1.8B 大语言模型按 Transformer 层拆分，部署到多台边缘设备组成分布式推理集群。配合 INT4/INT8 模型量化、算子融合、轻量化分页KV缓存三大单机优化，以及**图算法智能编排**（最大带宽生成树 + DFS 路径搜索），实现低开销、低时延的边缘端大模型推理。

### 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **智能编排** | 节点数 > 5 时自动启用图算法（最大带宽生成树 + DFS），替代纯算力权重分配 |
| 🔗 **链式拓扑流水线** | 按最优路径排序节点，hidden states 逐节点传递，支持 KV Cache 增量解码 |
| 🔄 **双引擎架构** | PyTorch + bitsandbytes (CUDA) / llama.cpp + GGUF (CPU/集显)，自动切换 |
| 📋 **请求队列** | FIFO 队列管理并发推理请求，支持队列深度监控 |
| 🗄️ **多会话管理** | 本地 JSON 存储 + 云 PostgreSQL 双轨，断网自动降级 |
| 🌐 **Tailscale 组网** | 跨子网设备互联，首次启动自动引导加入 |
| 📦 **一键安装包** | Windows 安装包（Inno Setup），含 Tailscale 检查 + 模型下载引导 + pywebview 原生窗口 |
| 🎛️ **管理面板** | 节点注册/注销、分层覆盖、角色转让、备用主节点、TCP 连接状态监控 |

**应用场景**：智能终端 · 物联网 · 边缘计算 · 教育科研

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
│   ├── 分布式推理流水线实施计划.md    # 链式拓扑、LAYER_FORWARD 协议、KV Cache 方案
│   ├── Android版本远期计划.md       # Android 端方案评估与规划
│   └── 纯Embedding节点处理+前端连接状态列计划.md
├── src/                           # Python 源代码
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
├── packaging/                     # 打包配置与脚本
│   ├── launcher.py                # 打包版启动器（Tailscale → 模型检查 → 引擎选择 → 启动）
│   ├── qlh-cpu.spec               # PyInstaller 规格文件
│   ├── setup.iss                  # Inno Setup 安装脚本
│   ├── build-cpu.bat              # 一键 PyInstaller 打包
│   ├── build-installer.bat        # 一键编译安装包
│   └── README.md                  # 打包文档
├── frontend/                      # React 前端（Vite + FastAPI 后端代理）
│   └── src/
│       ├── App.jsx                # 主布局 & 设置状态管理
│       ├── api/client.js          # API 客户端封装
│       └── components/            # ChatPanel / AdminPanel / DevicePanel / SettingsModal 等
├── tests/                         # 单元测试（272 个）
├── scripts/                       # 工具脚本
│   ├── quantize_model.py          # 模型准备与量化验证
│   ├── benchmark_all.py           # 全量化档位基准测试
│   ├── benchmark_compile.py       # torch.compile 融合测试
│   └── convert_to_gguf.py         # Safetensors → GGUF 转换
├── models/                        # 模型文件存放目录（需自行下载）
├── logs/                          # 运行日志目录
├── requirements.txt               # Python 依赖清单
└── README.md                      # 本文件
```

### 硬件拓扑（3 节点流水线示例）

```
用户输入 → 主节点(Master) → TCP → 从节点1(Client) → TCP → 从节点2(Client) → 结果回传
          Embed + L0-7           L8-15                L16-23 + LM Head
```

### 软件分层架构

| 层级 | 功能 | 技术 |
|------|------|------|
| 应用层 | 可视化交互 & 节点管理 & 性能监控 | React + FastAPI |
| 调度层 | 任务调度、指令分发、状态管理、请求队列 | Python threading + 图算法 |
| 通信层 | TCP长连接、粘包处理、心跳、张量序列化 | Python socket + struct |
| 推理层 | 双引擎：模型加载、量化、融合、KV缓存 | PyTorch (CUDA) / llama.cpp (CPU) |
| 存储层 | 对话持久化、节点注册、配置管理 | PostgreSQL + 本地 JSON 降级 |
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

### 一键安装

```bash
# Python 依赖
pip install -r requirements.txt

# 前端依赖
cd frontend && npm install && cd ..
```

---

## 🤖 模型下载

本项目使用 **Qwen-1.8B-Chat** 模型。支持两种格式：

| 格式 | 引擎 | 大小 | 适用场景 |
|------|------|------|---------|
| **Safetensors** | PyTorch (CUDA) | ~3.5 GB | 独显推理、分布式流水线 |
| **GGUF Q4_K_M** | llama.cpp (CPU) | ~1.16 GB | 集显/CPU、单机推理 |

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

### GGUF 格式（llama.cpp / CPU）

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

---

## 🚀 快速开始

### 开发模式

```bash
# 终端 1：启动 Python 后端（从项目根目录运行）
python src/api_server.py

# 终端 2：启动前端开发服务器（可选，后端已内置前端构建产物）
cd frontend && npm run dev
```

后端就绪后：
- **后端直连**：`http://localhost:8000`（含前端，`npm run build` 后）
- **开发前端**：`http://localhost:5173`（Vite 热更新，代理到 8000）

### 单机模式

修改 `src/config.py`：`RUN_MODE = "single"`，然后：

```bash
python src/api_server.py
```

### 分布式模式

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

参见 [packaging/README.md](packaging/README.md)。安装后双击桌面快捷方式即可启动，无需配置 Python 环境。

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

| 成员 | 学号 | 职责 |
|------|------|------|
| **杨睿涵** | 23301053 | 项目负责人 — 文献调研、模型量化、算子融合、KV缓存优化 |
| **张禄政** | 23301056 | 分布式架构设计、通信协议开发、多机调度逻辑 |
| **王泽远** | 23301077 | Web可视化平台、性能监控模块、文档与演示材料 |

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
- [Android 版本远期计划](docs/Android版本远期计划.md) — Android 端方案评估
- [纯 Embedding/LM Head 节点处理计划](docs/纯Embedding节点处理+前端连接状态列计划.md)

### 工程文档

- [打包说明](packaging/README.md) — PyInstaller + Inno Setup 打包流程

---

## 📄 许可证

本项目为北京交通大学 2026 年大学生创新创业训练计划项目。

---

© 2026 北京交通大学 · 杨睿涵 · 张禄政 · 王泽远
