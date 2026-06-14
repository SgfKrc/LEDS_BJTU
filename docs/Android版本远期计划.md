# 五、Android 版本 — 远期部署计划

> **状态**: 规划阶段  
> **优先级**: 远期（v0.5.0+）  
> **核心目标**: 将 QLH 边缘推理系统移植到 Android 设备，实现「手机即推理节点」

---

## 5.1 动机与定位

### 5.1.1 为什么做 Android 版本

| 维度 | 说明 |
|------|------|
| **设备普及** | 闲置 Android 手机远多于闲置 PC — 宿舍淘汰机、备用机均可加入集群 |
| **能耗优势** | ARM 架构能效比远高于 x86 — 同功耗下算力密度更高 |
| **始终在线** | 手机 24h 待机 + Wi-Fi 常连接，天然适合做常驻推理节点 |
| **部署零成本** | 无需采购新硬件，旧手机「废物利用」 |
| **边缘特性** | 手机分布在用户身边，天然贴近数据源，符合边缘计算定义 |

### 5.1.2 Android 节点在集群中的角色

```
┌──────────────────────────────────────────────┐
│              PC 主节点 (Master)               │
│   PyTorch + CUDA / llama.cpp + GGUF          │
│   完整推理 + 调度 + Web UI                   │
└──────┬──────────────┬──────────────┬─────────┘
       │ Tailscale    │              │
       ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Android  │  │ Android  │  │  PC 从节点│
│ 从节点 1 │  │ 从节点 2 │  │  从节点 3 │
│ 4-6 层   │  │ 4-6 层   │  │ 6-8 层   │
│ Q4_K_M   │  │ Q4_K_M   │  │ INT4     │
└──────────┘  └──────────┘  └──────────┘
```

- **Android 节点定位**: 轻量级从节点，承担 4-6 层 Transformer 推理
- **通信方式**: Tailscale 虚拟组网（Android 有官方 App）+ TCP 长连接
- **不适用场景**: 主节点（无 Web UI 服务能力）、大规模并发（内存受限）

---

## 5.2 技术架构选型

### 5.2.1 推理引擎 — llama.cpp（GGUF）

| 方案 | 可行性 | 结论 |
|------|--------|------|
| **PyTorch Mobile** | Android 有官方支持，但模型需 TorchScript/ExecuTorch 导出 | 备选 |
| **ONNX Runtime Mobile** | ARM64 + Qualcomm QNN EP 支持，需先导出 ONNX | 备选 |
| **llama.cpp (NDK)** | ✅ 原生支持 ARM NEON，GGUF 直接加载，零依赖 | **首选** |
| **MediaPipe LLM** | Google 官方方案，但仅支持 Gemma 系列 | 不适用 |
| **MLC-LLM** | TVM 编译后端，性能好但复杂度高 | 参考 |

**选择理由 — llama.cpp**:
1. 已有 Windows 端 llama.cpp 引擎代码（`src/llama_engine.py`），逻辑可复用
2. C++ 原生引擎，通过 NDK 交叉编译 → ARM64 二进制，性能极佳
3. GGUF 格式模型与桌面端完全一致，无需重新导出
4. Q4_K_M 量化后 Qwen-1.8B 仅 ~1.16GB — 可装入 4GB+ 手机 RAM
5. 开源社区活跃，持续优化 ARM 性能（NEON / SVE 指令集）

### 5.2.2 应用形态

```
┌────────────────────────────────────────────┐
│           APK 包结构 (≈ 180MB)             │
├────────────────────────────────────────────┤
│  📦 libllama_android.so    ← C++ NDK 编译  │
│  📦 libggml.so             ← GGML 核心库   │
│  📦 Kotlin 业务层           ← JNI 桥接     │
│  📦 Compose UI             ← 现代原生界面   │
│  📦 Tailscale SDK          ← 虚拟组网      │
│  ────────────────────────────────────────  │
│  📂 用户自行放入:                           │
│     models/qwen-1_8b-chat-Q4_K_M.gguf     │
│     (1.16 GB，首次启动引导下载)            │
└────────────────────────────────────────────┘
```

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| **推理引擎** | llama.cpp NDK 编译 | C++ → `.so`，ARM NEON 优化 |
| **JNI 桥接** | Kotlin/JNI | 封装 llama.cpp C API |
| **业务逻辑** | Kotlin + Coroutines | 状态管理、网络通信、任务调度 |
| **UI** | Jetpack Compose | Material 3，自适应手机/平板 |
| **网络** | Tailscale Android SDK | 自动加入组网，保持长连接 |
| **通信** | 纯 Java/Kotlin socket | 复用 TCP 协议，JSON 控制 + 二进制张量 |
| **持久化** | Room (SQLite) | 对话历史 + 节点状态本地存储 |

### 5.2.3 为什么不选 React Native / PWA

| 方案 | 问题 |
|------|------|
| React Native | 需要 Bridge 调用原生 C++ 推理库，性能损耗 + 依赖复杂 |
| PWA (WebView) | 无法访问 TCP socket（受限），无法调用 JNI |
| Flutter | Dart FFI 可调用 C，但生态不如原生成熟 |

**结论**: 原生 Kotlin + JNI 是最直接的路径，性能开销最小，对 llama.cpp C API 的封装最薄。

---

## 5.3 分阶段实施路线

### 阶段 0: 环境验证（~2 天）

**目标**: 在 Android 设备上运行 llama.cpp 命令行推理

**步骤**:
1. Ubuntu/Linux 主机配置 Android NDK (r26+)
2. 交叉编译 llama.cpp 为 ARM64 二进制
   ```bash
   cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
         -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF ..
   make -j$(nproc)
   ```
3. `adb push` 模型文件 + 编译好的 `main` 二进制到手机
4. `adb shell` 运行推理，测试性能和内存占用
5. 验证 Q4_K_M Qwen-1.8B 在骁龙 8xx / 天玑 9xxx 上的 token 生成速度

**验收标准**:
- 骁龙 865 级别 ≥ 5 tok/s（Q4_K_M, 4 线程）
- 峰值 RAM ≤ 2GB（含模型 + KV Cache）
- 推理结果与桌面端一致（确定性采样下）

### 阶段 1: 最小可用 APK（~2 周）

**目标**: 能加载模型 + 接收主节点 TCP 指令 + 执行层推理

**核心模块**:

```
android/
├── app/
│   ├── src/main/
│   │   ├── cpp/
│   │   │   ├── llama_bridge.cpp       # JNI 封装 llama.cpp C API
│   │   │   └── tensor_codec.cpp       # 张量序列化/反序列化（复用协议）
│   │   ├── java/com/qlh/inference/
│   │   │   ├── LlamaEngine.kt         # Kotlin 封装 JNI 接口
│   │   │   ├── TcpClientService.kt    # TCP 长连接 + 心跳（前台 Service）
│   │   │   ├── LayerWorker.kt         # 层范围推理执行器
│   │   │   ├── NodeStateManager.kt    # 节点状态 + 任务队列
│   │   │   ├── TailscaleHelper.kt     # Tailscale 组网集成
│   │   │   └── ui/
│   │   │       ├── MainActivity.kt    # 主界面（状态 + 控制面板）
│   │   │       ├── StatusScreen.kt    # 节点状态仪表盘
│   │   │       └── SetupWizard.kt     # 首次启动引导（模型下载等）
│   │   └── res/
│   └── build.gradle.kts
├── models/                            # 空目录，用户放入 .gguf 文件
├── CMakeLists.txt                     # NDK 编译脚本
└── README.md
```

**关键设计决策**:

1. **前台 Service** — Android 8+ 对后台限制严格，推理服务必须跑在前台 Service 中（通知栏显示「推理引擎运行中」）
2. **Wake Lock** — Wi-Fi Lock + CPU Lock，确保屏幕关闭时不断网、不掉频
3. **温控策略** — 监控电池温度，≥42°C 时降低线程数/暂停推理，防止过热关机
4. **充电检测** — 默认仅在充电时参与推理集群（可在设置中覆盖）

### 阶段 2: 分布式流水线集成（~1 周）

**目标**: Android 节点作为流水线从节点，无缝接入 PC 主节点集群

**需要实现的 TCP 消息**:
- `HEARTBEAT` → 上报设备信息（ARM、RAM、当前电池、温度）
- `NODE_REGISTER` → 注册为 `network_type: tailscale_android`
- `LAYER_FORWARD` → 接收隐藏状态 → 执行层前向 → 返回 `LAYER_RESULT`
- `LAYER_CONFIG` → 接收主节点分发的层范围配置

**调度策略适配**:
- `compute_layer_assignment()` 添加 Android 分数惩罚系数（0.4-0.6），自动减少分配给手机的层数
- 手机节点 `max_new_tokens` 自动限制在 256 以下（温控考虑）

### 阶段 3: 模型管理 + 下载（~3 天）

- 接入 HuggingFace API，从 `RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf` 下载 Q4_K_M
- 支持断点续传（Android DownloadManager）
- SHA256 校验确保文件完整性
- 内置百度网盘备用链接（国内用户加速）

### 阶段 4: 用户体验优化（~1 周）

- 首次启动引导（Tailscale 组网 → 模型下载 → 集群加入，3 步向导）
- 通知栏快速开关（一键暂停/恢复推理）
- 电量感知调度（≥80% → 全速；20-80% → 半速；<20% → 暂停）
- Google Play 上架（通过 llama.cpp MIT 许可 + Qwen Apache 2.0 许可审核）

---

## 5.4 性能预估

### 5.4.1 单机推理（离线模式）

| 芯片 | 等级 | 线程 | Q4_K_M tok/s | 峰值 RAM |
|------|------|------|-------------|----------|
| 骁龙 8 Gen 3 | 旗舰 | 4 | 12-18 | 1.8 GB |
| 骁龙 8+ Gen 1 | 次旗舰 | 4 | 8-12 | 1.8 GB |
| 骁龙 865 | 中端 | 4 | 5-8 | 1.8 GB |
| 天玑 9200 | 旗舰 | 4 | 10-15 | 1.8 GB |
| 骁龙 778G | 中低端 | 2 | 3-4 | 1.9 GB |
| 骁龙 480 | 入门 | 2 | 1-2 | 2.0 GB |

### 5.4.2 分布式流水线（Android + PC 混合）

假设 2 台 Android + 1 台 PC 笔记本组成 3 节点流水线：

| 节点 | 设备 | 分配层数 | 预估单 Token 延迟 |
|------|------|---------|-------------------|
| Master | GTX 1650 笔记本 | Layers 0-7 + Embed | ~80ms |
| Slave 1 | 骁龙 8+ 手机 | Layers 8-15 | ~120ms |
| Slave 2 | 骁龙 865 手机 | Layers 16-23 + LM Head | ~180ms |
| **总计** | | 24 层 | **~430ms + 网络 ~50ms** |

> 对比全 PC 模式（全部层在主节点）: ~200ms/token。  
> 混合模式延迟更高，但释放了 PC 的显存压力，适合并发多请求场景。

---

## 5.5 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| **Google Play 审核拒绝** | 中 | 无法上架 | 先走 APK 侧载（GitHub Release）+ 提供详细使用说明 |
| **温控降频** | 高 | 推理速度骤降 | 温控策略 + 低电量自动暂停 + 散热背夹推荐 |
| **后台被系统杀死** | 高 | 节点离线 | 前台 Service + 通知常驻 + 电池优化白名单引导 |
| **厂商 ROM 兼容性** | 中 | 部分手机无法运行 | 主力适配原生/类原生 ROM，MIUI/ColorOS 提供专用说明 |
| **ARM SVE 指令缺失** | 低 | 老芯片推理极慢 | 最低要求 ARMv8.2-A（2017 年+），自动检测并提示 |
| **llama.cpp GGUF 模型加载失败** | 低 | 模型无法加载 | 内置多个 GGUF 文件名 fallback + 自动校验 |
| **Tailscale 耗电** | 中 | 待机时间缩短 | 仅在充电时启用；提供「按需连接」模式 |

---

## 5.6 文件结构（远期）

```
qlh/
├── android/                          # 新增 Android 工程目录
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── cpp/
│   │   │   │   ├── llama_bridge.cpp
│   │   │   │   ├── llama_bridge.h
│   │   │   │   ├── tensor_codec.cpp
│   │   │   │   └── tensor_codec.h
│   │   │   ├── java/com/qlh/inference/
│   │   │   │   ├── LlamaEngine.kt
│   │   │   │   ├── TcpClientService.kt
│   │   │   │   ├── LayerWorker.kt
│   │   │   │   ├── NodeStateManager.kt
│   │   │   │   ├── BatteryManager.kt
│   │   │   │   ├── TailscaleHelper.kt
│   │   │   │   └── ui/
│   │   │   │       ├── MainActivity.kt
│   │   │   │       ├── StatusScreen.kt
│   │   │   │       ├── SetupWizard.kt
│   │   │   │       └── theme/
│   │   │   └── res/
│   │   └── build.gradle.kts
│   ├── CMakeLists.txt
│   ├── build.gradle.kts
│   └── gradle.properties
├── src/                              # 已有 PC 端代码
│   ├── scheduler.py                  # 修改: Android节点分数策略
│   ├── device_profiler.py            # 修改: Android/ARM 优化提示
│   └── ...
├── docs/
│   ├── Android版本远期计划.md        # 本文档
│   ├── 分布式推理流水线实施计划.md
│   └── ...
└── ...
```

---

## 5.7 时间线总览

```
┌─────────┬──────────┬──────────┬──────────┬─────────┐
│ Q3 2026 │ Q4 2026  │ Q1 2027  │ Q2 2027  │ Q3 2027 │
├─────────┼──────────┼──────────┼──────────┼─────────┤
│ 阶段 0  │ 阶段 1   │ 阶段 2   │ 阶段 3-4 │ 发布    │
│ 环境    │ 最小     │ 分布式   │ 体验优化 │ Google  │
│ 验证    │ 可用APK  │ 集成     │ + 上架   │ Play    │
│ 2天     │ 2周      │ 1周      │ 1.5周    │         │
└─────────┴──────────┴──────────┴──────────┴─────────┘
```

**里程碑**:
- **M1** (阶段 0 完成): `adb shell` 跑通 Qwen-1.8B GGUF 推理
- **M2** (阶段 1 完成): APK 安装后可作为从节点注册到 PC 主节点
- **M3** (阶段 2 完成): Android 节点参与分布式流水线，端到端推理可用
- **M4** (阶段 4 完成): Google Play 上架 + 用户无需 ADB 即可部署

---

## 5.8 与已有系统的关系

| 已实现 | Android 版复用方式 |
|--------|-------------------|
| `tcp_comm.py` 消息协议 | 直接复用 — Kotlin 实现相同的 4 字节长度头 + JSON/二进制体 |
| `scheduler.py` 分层算法 | 不变 — 增加 Android 节点分数衰减系数 |
| `llama_engine.py` | 逻辑映射 — JNI 桥接 llama.cpp C API，等效于 Python 封装 |
| `config.py` | 概念映射 — Kotlin `SharedPreferences` 等效 |
| Tailscale 组网 | 使用官方 Android App + 后台 VPN Service |
| 前端 React UI | 可选 — 开发专用 Android 原生 UI；或通过 PC 主节点的 Web UI 间接管理 |

---

> **编写日期**: 2026-06-14  
> **下次评审**: 阶段 0 启动前（需确认 NDK 编译环境 + 测试机）  
> **关联文档**: [分布式推理流水线实施计划](分布式推理流水线实施计划.md)、[整体架构](整体架构.md)
