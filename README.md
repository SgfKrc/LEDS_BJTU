# 🧠 轻量化大模型分布式边缘推理优化系统

**基于分层流水线的边缘分布式大模型推理系统**

模型量化 · 算子融合 · 分页KV缓存 · 多终端协同推理 · 可视化监控

---

## 📋 项目简介

针对边缘设备（普通笔记本）显存不足、算力有限的痛点，将 Qwen-1.8B 大语言模型按 Transformer 层拆分，部署到多台边缘设备组成分布式推理集群。配合 INT4/INT8 模型量化、算子融合、轻量化分页KV缓存三大单机优化，实现低开销、低时延的边缘端大模型推理。

**应用场景**：智能终端 · 物联网 · 边缘计算 · 教育科研

---

## 🏗️ 项目架构

```
项目根目录
├── docs/                    # 项目开发文档
│   ├── 整体架构.md           # 项目总览、拓扑架构、技术栈
│   ├── 核心技术原理.md        # 算子融合、QKV、量化、流水线、TCP协议
│   ├── 模块接口说明.md        # 5大模块功能职责与对外接口
│   ├── 运行流程&异常处理.md   # 全局配置、初始化流程、推理全链路、异常处理
│   └── 测试与评判标准.md      # 对照实验组、评判指标、可视化形式
├── src/                     # 源代码
│   ├── config.py            # 全局配置（网络/模型/KV/分层/运行模式）
│   ├── model_module.py      # 模型加载、量化、算子融合、层级拆分、前向推理
│   ├── paged_kv_cache.py    # 轻量化分页KV缓存（内存页管理、动态分配）
│   ├── tcp_comm.py          # TCP主从通信（长连接、心跳、封包解包、张量序列化）
│   ├── scheduler.py         # 任务调度（节点状态管理、推理分发、流水线控制）
│   └── web_ui.py            # Streamlit可视化平台（对话+监控+图表+对照实验）
├── frontend/                # React 前端（Vite + FastAPI 后端）
│   └── src/
│       ├── App.jsx          # 主布局 & 设置状态管理
│       └── components/      # ChatPanel / DevicePanel / SettingsModal 等
├── scripts/                 # 工具脚本
│   ├── quantize_model.py    # 模型准备与量化验证
│   ├── benchmark_all.py     # 全量化档位基准测试
│   └── benchmark_compile.py # torch.compile 融合测试
├── models/                  # 模型文件存放目录（需自行下载）
│   └── qwen-1_8b-chat/      # Qwen-1.8B-Chat
├── logs/                    # 运行日志目录
├── requirements.txt         # Python 依赖清单
└── README.md               # 本文件
```

### 硬件拓扑（3台边缘设备）

```
前端交互 → 主节点(Server) → TCP → 从节点1(Client) → TCP → 从节点2(Client) → 结果回传
```

### 软件分层架构

| 层级 | 功能 | 技术 |
|------|------|------|
| 应用层 | 可视化交互 & 性能监控 | React + FastAPI |
| 调度层 | 任务调度、指令分发、状态管理 | Python threading |
| 通信层 | TCP长连接、粘包处理、心跳、序列化 | Python socket + torch.save |
| 推理层 | 模型加载、量化、融合、KV缓存、前向计算 | PyTorch + Transformers |
| 基础层 | 运行环境 | Python / CUDA / bitsandbytes |

---

## 📦 环境依赖

### 核心框架

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 开发环境: 3.12.10 |
| PyTorch | ≥ 2.2.0 | 开发环境: 2.12.0 + CUDA 12.6 |
| **transformers** | **≥ 4.45, < 5.0** | ⚠️ 必须保持 4.x！5.x 移除了 `load_in_4bit`/`load_in_8bit` |
| accelerate | ≥ 1.0.0 | 模型加载加速（bitsandbytes 依赖） |

### 模型量化

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| bitsandbytes | ≥ 0.45.0 | INT4/INT8 量化，Windows + CUDA 12.x 已支持 |

### Web 可视化

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| streamlit | ≥ 1.30.0 | Web 可视化平台 |
| plotly | ≥ 5.18.0 | 交互式图表 |
| pandas | ≥ 2.0.0 | 数据处理 |
| fastapi | ≥ 0.110.0 | API 后端框架 |
| uvicorn[standard] | ≥ 0.29.0 | ASGI 服务器 |
| python-multipart | ≥ 0.0.12 | 文件上传支持 |

### 工具

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| tqdm | ≥ 4.65.0 | 进度条 |
| psutil | ≥ 5.9.0 | 系统资源监控（CPU/内存） |

### 前端

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Node.js | ≥ 18 | 前端构建 |
| npm / yarn | — | 包管理器 |

### 一键安装

```bash
# Python 依赖
pip install -r requirements.txt

# 前端依赖
cd frontend && npm install && cd ..
```

---

## 🤖 模型下载

本项目使用 **Qwen-1.8B-Chat** 模型。由于模型权重文件较大（~3.5 GB），不包含在 Git 仓库中，请通过以下任一方式下载。

### 方式一：ModelScope 下载（推荐，国内更快）

🔗 **模型主页**：[https://modelscope.cn/models/Qwen/Qwen-1.8B-Chat](https://modelscope.cn/models/Qwen/Qwen-1.8B-Chat)

```bash
# 安装 ModelScope SDK
pip install modelscope

# 下载模型到 models/qwen-1_8b-chat/
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')"
```

### 方式二：Hugging Face 下载

🔗 **模型主页**：[https://huggingface.co/Qwen/Qwen-1.8B-Chat](https://huggingface.co/Qwen/Qwen-1.8B-Chat)

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen-1.8B-Chat --local-dir models/qwen-1_8b-chat
```

### 方式三：百度网盘（备用）

> 📎 网盘链接：https://pan.baidu.com/s/1trocmWlmG3F1lOB6krIz4w
>
> 提取码：t7qk

下载后将所有文件放入 `models/qwen-1_8b-chat/` 目录即可。

### 目录结构检查

下载完成后，`models/qwen-1_8b-chat/` 应包含以下文件：

```
models/qwen-1_8b-chat/
├── model-00001-of-00002.safetensors  (~1.9 GB)
├── model-00002-of-00002.safetensors  (~1.6 GB)
├── model.safetensors.index.json
├── config.json
├── configuration_qwen.py
├── tokenizer_config.json
├── qwen.tiktoken
├── tokenization_qwen.py
├── modeling_qwen.py
├── qwen_generation_utils.py
├── cpp_kernels.py
├── generation_config.json
├── LICENSE
├── NOTICE
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

参见上方 [🤖 模型下载](#-模型下载) 章节。

### 3. 验证模型

```bash
# INT4 量化加载验证（推荐）
python scripts/quantize_model.py --quant int4

# 验证全部三种量化模式
python scripts/quantize_model.py --all --skip-copy
```

### 4. 启动系统

#### 前端 + API 模式（推荐）

```bash
# 终端 1：启动 FastAPI 后端
cd src && uvicorn app:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端开发服务器
cd frontend && npm run dev
```

浏览器打开 `http://localhost:5173` 即可使用。

#### Streamlit 可视化模式

```bash
streamlit run src/web_ui.py
```

#### 分布式模式（3台设备）

**主节点（设备1）**：
```bash
python src/scheduler.py           # 启动调度器 + TCP服务端
streamlit run src/web_ui.py       # 启动可视化界面
```

**从节点1（设备2）**：
```bash
python src/tcp_comm.py --role client1  # 连接主节点，加载中间层
```

**从节点2（设备3）**：
```bash
python src/tcp_comm.py --role client2  # 连接主节点，加载后半层
```

#### 单机模拟模式

```bash
# 修改 config.py: RUN_MODE = "single"
streamlit run src/web_ui.py
```

---

## 📊 量化效果（已验证）

> 测试环境: NVIDIA RTX GPU + CUDA 12.6 + PyTorch 2.12.0 + Qwen-1.8B-Chat (24层 Transformer)  
> 统一 prompt，50 token 生成，3 轮取均值

| 配置 | GPU 显存 | 推理速度 | 备注 |
|------|---------|----------|------|
| FP16 | 3.47 GB | 53.2 tok/s | 基线对照组 |
| FP16 + compile | 3.47 GB | 55.1 tok/s | 算子融合 +3.6% |
| INT8 | 2.30 GB | 9.8 tok/s | 省显存但速度损失大 |
| **INT4** ⭐ | **1.75 GB** | **28.7 tok/s** | **推荐边缘设备：显存减半** |

> INT4 相比 FP16: 显存 **-50%**，速度仍有 29 tok/s  
> torch.compile 仅对 FP16 有效 (+3.6%)，INT4/INT8 下自动跳过（模型模块已内置保护）

---

## 🧪 对照实验组

| 实验组 | 量化 | 算子融合 | KV缓存 | 部署模式 |
|--------|------|----------|--------|----------|
| 基线组 | FP16 | 无 | 传统KV | 单机 |
| 实验组1 | INT4 | 无 | 传统KV | 单机 |
| 实验组2 | INT4 | 融合 | 传统KV | 单机 |
| 实验组3 | INT4 | 融合 | 分页KV | 单机 |
| 实验组4 | INT4 | 融合 | 分页KV | 分布式 |

---

## 📊 核心评判指标

- **显存占用**：量化、分页KV优化效果
- **推理时延 / Token生成速度**：算子融合、整体优化效果
- **CPU负载 / 网络延迟**：分布式通信开销
- **对话通顺度**：量化精度损失评估
- **长时间运行稳定性**：缓存、通信健壮性

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

- [整体架构](docs/整体架构.md)
- [核心技术原理](docs/核心技术原理.md)
- [模块接口说明](docs/模块接口说明.md)
- [运行流程&异常处理](docs/运行流程&异常处理.md)
- [测试与评判标准](docs/测试与评判标准.md)

---

## 📄 许可证

本项目为北京交通大学 2026 年大学生创新创业训练计划项目。

---

© 2026 北京交通大学 · 杨睿涵 · 张禄政 · 王泽远
