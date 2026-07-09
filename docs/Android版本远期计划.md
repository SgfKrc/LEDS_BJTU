# 五、Android 版本 — 远期部署计划

> **状态**: 规划阶段  
> **优先级**: 远期（v0.5.0+）  
> **核心目标**: 将 QLH 边缘推理系统移植到 Android 设备  
> **设计理念**: ⭐ **全有或全无** — 不做层间拆分

---

## 5.1 设计理念：全有或全无

### 5.1.1 为什么不做层间拆分

llama.cpp 的常规 API 面向**完整模型推理**（token in → token out），内部虽然存在 GPU layer offload、tensor split、RPC backend 等机制，但都是 llama.cpp **进程内部的调度逻辑**，不暴露「输入 hidden states → 跑第 N-M 层 → 输出 hidden states」的稳定接口。

要强行实现层间拆分需要：
- Fork llama.cpp 源码，定位每层 hidden state 边界
- 处理 KV cache 切分、position/attention mask 跨层传递
- 保证 llama.cpp 量化张量布局与 PyTorch hidden states 格式兼容
- 跟踪上游 llama.cpp 变更，持续维护 fork

**工程量巨大，维护成本高，且与项目 PC 端的 PyTorch 层间拆分不在同一技术栈，收益有限。**

### 5.1.2 全有或全无

放弃层间拆分，Android 节点只有两种工作模式：

```
┌──────────────────────────────────────────────────────────────┐
│                    Android 节点两种模式                        │
├──────────────────────────┬───────────────────────────────────┤
│  模式 A: 全有 (ALL)       │  模式 B: 全无 (NOTHING)            │
│  Android 本地跑全部层      │  Android 只做展示，主节点全包       │
├──────────────────────────┼───────────────────────────────────┤
│  ┌──────────┐            │  ┌──────────┐                     │
│  │ Android  │            │  │ Android  │  用户输入             │
│  │ llama.cpp│            │  │ 聊天UI   │──────HTTP──────┐     │
│  │ Q4_K_M   │            │  │ 薄客户端  │                │     │
│  │ 1.16GB   │            │  └──────────┘                ▼     │
│  │ 完整推理  │            │                     ┌──────────┐  │
│  └──────────┘            │                     │ PC 主节点 │  │
│      独立运行             │                     │ 调度+推理 │  │
│  无需网络 (离线可用)       │                     └────┬─────┘  │
│                          │                    分发到各PC从节点 │
│                          │                     ← 结果汇总 ←   │
└──────────────────────────┴───────────────────────────────────┘
```

| 维度 | 全有 (ALL) | 全无 (NOTHING) |
|------|-----------|---------------|
| **推理位置** | Android 本地 (llama.cpp) | PC 主节点 + 集群 |
| **模型** | GGUF Q4_K_M (~1.16 GB) | 不需要（无模型） |
| **计算贡献** | 全部 24 层 | 0 层 |
| **网络依赖** | 离线可用 | 必须连接主节点 |
| **APK 大小** | ~180 MB（含 .so） | ~15 MB（纯 UI） |
| **适用场景** | 单机使用、网络不稳定 | 有 PC 集群、追求速度 |
| **手机要求** | 4GB+ RAM，ARMv8.2-A+ | 任意 Android 8+ |

---

## 5.2 技术架构

### 5.2.1 模式 A：全有 — 技术栈

```
┌──────────────────────────────────────────┐
│            Android APK (~180MB)           │
├──────────────────────────────────────────┤
│  📦 libllama_android.so    ← NDK 交叉编译 │
│  📦 libggml.so             ← GGML 核心    │
│  📦 Kotlin 业务层           ← JNI 桥接    │
│  📦 Jetpack Compose UI     ← Material 3  │
│  ─────────────────────────────────────── │
│  📂 用户自行放入 / 应用内下载:             │
│     models/qwen-1_8b-chat-Q4_K_M.gguf    │
│     (~1.16 GB，放外部存储)                │
└──────────────────────────────────────────┘
```

| 层级 | 技术 | 说明 |
|------|------|------|
| 推理引擎 | llama.cpp NDK | C → ARM64 .so，NEON 优化 |
| JNI 桥接 | Kotlin/JNI | 封装 llama.cpp C API |
| 业务逻辑 | Kotlin Coroutines + StateFlow | 推理状态管理、对话历史 |
| UI | Jetpack Compose | Material 3，聊天界面 + 设置 |
| 持久化 | Room (SQLite) | 对话历史、模型路径配置 |
| 可选网络 | OkHttp + Tailscale | 注册到主节点（仅状态上报，不参与推理） |

**关键设计**：
- 前台 Service + 通知栏常驻（"推理引擎运行中"），防止被系统杀死
- Wake Lock（Wi-Fi + CPU），屏幕关闭时不断网不掉频
- 温控策略：电池 ≥42°C 降线程 / 暂停
- 默认仅充电时参与推理（可在设置覆盖）

### 5.2.2 模式 B：全无 — 技术栈

```
┌──────────────────────────────────────────┐
│            Android APK (~15MB)            │
├──────────────────────────────────────────┤
│  📦 Jetpack Compose UI     ← 聊天界面     │
│  📦 OkHttp                 ← HTTP 客户端  │
│  📦 Kotlin Coroutines      ← 异步请求     │
│  📦 Room (SQLite)          ← 本地缓存     │
│  📦 DataStore              ← 偏好设置     │
└──────────────────────────────────────────┘
         │ HTTP POST /api/chat
         ▼
┌──────────────────────────────────────────┐
│           PC 主节点 (已有能力)             │
│  api_server.py  ─→  scheduler.run_pipeline│
│                  ─→  或 mgr.chat() 本地   │
│                  ─→  返回完整响应          │
└──────────────────────────────────────────┘
```

**极简实现**：Android 就是一个带聊天 UI 的 HTTP 客户端，调用主节点已有的 `/api/chat` 接口。主节点收到请求后，按现有逻辑执行：有从节点 → 流水线分发；无从节点 → 本地推理。

**无需任何 PC 端代码改动** — 现有的 `/api/chat` 接口已经支持完整的 `ChatRequest` (message, max_new_tokens, temperature, top_p, session_id)，返回 `ChatResponse` (content, metrics, followups)。

**可选增强**：
- 主节点连接配置（IP:端口 保存到 DataStore）
- 多会话管理（复用 `/api/sessions` 接口）
- 流式展示（如主节点支持 SSE streaming）

---

## 5.3 PC 端需要的改动

虽然模式 B（全无）不需要 PC 端改动，但为了更好地支持 Android 节点管理，建议做以下增强：

### 5.3.1 NodeInfo 增加 `node_type` 字段

```python
# scheduler.py — NodeInfo 新增字段
@dataclass
class NodeInfo:
    # ... 现有字段 ...
    node_type: str = "pc"  # "pc" | "android_full" | "android_thin"
```

| node_type | 含义 | 层分配 | 参与流水线 |
|-----------|------|--------|-----------|
| `pc` | PC 节点（现有） | 正常分配 | ✅ |
| `android_full` | Android 全有模式 | 0 层（自己全跑） | ❌ |
| `android_thin` | Android 全无模式 | 0 层（不跑推理） | ❌ |

### 5.3.2 层分配跳过非 PC 节点

`compute_layer_assignment()` 需要过滤 `node_type != "pc"` 的节点，不给 Android 节点分配层。

### 5.3.3 前端节点列表显示 Android 标识

在管理面板节点列表中，Android 节点显示 🤖 图标和模式标签（"全有"/"全无"），帮助管理员区分设备类型。

### 5.3.4 API 增强（可选）

```python
# api_server.py — 新增端点（可选）
@app.get("/api/cluster/android/config")
async def get_android_config():
    """Android 客户端获取集群连接配置"""
    return {
        "master_host": scheduler.get_lan_ip(),
        "master_port": SERVER_PORT,
        "mode": "thin",  # 或 "full"
    }
```

---

## 5.4 分阶段实施路线

### 阶段 0：环境验证（~2 天）

**目标**：在 Android 设备上跑通 llama.cpp 命令行推理

1. Ubuntu/WSL 配置 Android NDK r26+
2. 交叉编译 llama.cpp → ARM64 二进制
   ```bash
   cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
         -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF ..
   make -j$(nproc)
   ```
3. `adb push` 模型文件 + 编译好的 binary
4. `adb shell` 跑推理，测试性能

**验收标准**：
- 骁龙 865 ≥ 5 tok/s（Q4_K_M, 4 线程）
- 峰值 RAM ≤ 2 GB
- 结果与桌面端一致

### 阶段 1：模式 B 先行 — 全无薄客户端（~1 周）

**目标**：Android APK 可以连接 PC 主节点，发送消息并显示回复

**为什么先做全无**：
- 最简单，不需要 NDK / JNI / 模型加载
- 可以验证 Android ↔ PC 通信链路
- 快速产出可用的 Android 聊天客户端

**核心模块**：

```
android/
├── app/
│   ├── src/main/java/com/qlh/inference/
│   │   ├── MainActivity.kt          # 单 Activity 架构
│   │   ├── ui/
│   │   │   ├── ChatScreen.kt        # 聊天主界面（Compose）
│   │   │   ├── SettingsScreen.kt    # 主节点连接设置
│   │   │   ├── SessionListScreen.kt # 多会话列表
│   │   │   └── theme/Theme.kt       # Material 3 主题
│   │   ├── network/
│   │   │   ├── ApiClient.kt         # OkHttp 封装，调用 /api/chat
│   │   │   ├── ChatRepository.kt    # 消息发送 + 历史管理
│   │   │   └── SseClient.kt         # SSE 流式接收（可选）
│   │   ├── data/
│   │   │   ├── AppDatabase.kt       # Room 数据库
│   │   │   ├── MessageDao.kt        # 消息 DAO
│   │   │   └── SettingsDataStore.kt # 连接配置持久化
│   │   └── service/
│   │       └── TailscaleService.kt  # Tailscale VPN 集成（可选）
│   └── build.gradle.kts
├── build.gradle.kts
└── gradle.properties
```

**关键实现**：
1. `ApiClient.kt` — POST JSON 到 `http://<master_ip>:8000/api/chat`
2. `ChatScreen.kt` — 类微信聊天界面，消息气泡 + 打字动画
3. 连接状态指示器（在线/离线/推理中）
4. 对话历史本地缓存（Room），断网也能看历史

### 阶段 2：模式 A — 全有本地推理（~2 周）

**目标**：Android 本地加载 GGUF 模型，离线完整推理

**核心模块**（在阶段 1 基础上增加）：

```
android/
├── app/src/main/
│   ├── cpp/
│   │   ├── llama_bridge.cpp         # JNI: llama.cpp C API 封装
│   │   ├── llama_bridge.h
│   │   └── CMakeLists.txt
│   ├── java/com/qlh/inference/
│   │   ├── engine/
│   │   │   ├── LlamaEngine.kt       # Kotlin 封装 JNI
│   │   │   ├── InferenceConfig.kt   # 推理参数 (temp, top_p, max_tokens)
│   │   │   └── ModelManager.kt      # 模型加载/卸载/状态
│   │   ├── service/
│   │   │   └── InferenceService.kt  # 前台 Service（推理线程）
│   │   └── ui/
│   │       ├── ModelDownloadScreen.kt  # 模型下载引导
│   │       └── PerformanceMonitor.kt   # tok/s、内存、温度
```

**模型存储策略（重要）**：

Android 系统卸载 APK 时不会提供应用自定义的“是否删除 models 目录”脚本选项。若模型放在应用内部目录（如 `context.filesDir/models`），卸载 APK 时会被系统自动删除，无法实现“默认保留”。

因此正式版应采用以下策略之一：

1. **推荐：SAF 用户选择目录** — 首次下载模型时让用户选择一个外部目录（如 `Download/QLH/models`），应用只保存 URI 权限；卸载 APK 不会删除该目录。
2. **备选：公共 Downloads 目录** — 将模型保存到用户可见的 `Download/QLH/models`，设置页提供“删除本地模型”按钮；卸载 APK 默认保留。
3. **不推荐：内部存储** — 适合测试，卸载 APK 会直接删除模型。

**JNI 桥接核心函数**：

```cpp
// llama_bridge.cpp — 最小必要接口
extern "C" {
    JNIEXPORT jlong JNICALL Java_com_qlh_inference_engine_LlamaEngine_nativeInit(
        JNIEnv*, jobject, jstring model_path, jint n_threads, jint n_ctx);
    
    JNIEXPORT jstring JNICALL Java_com_qlh_inference_engine_LlamaEngine_nativeGenerate(
        JNIEnv*, jobject, jlong engine_ptr, jstring prompt, 
        jint max_tokens, jfloat temperature, jfloat top_p);
    
    JNIEXPORT void JNICALL Java_com_qlh_inference_engine_LlamaEngine_nativeFree(
        JNIEnv*, jobject, jlong engine_ptr);
    
    JNIEXPORT jfloat JNICALL Java_com_qlh_inference_engine_LlamaEngine_nativeGetTokPerSec(
        JNIEnv*, jobject, jlong engine_ptr);
}
```

### 阶段 3：模式切换 + 自动注册（~1 周）

**目标**：两种模式无缝切换，可选注册到主节点

- 设置页面切换「全有/全无」模式
- 全有模式：可选注册到主节点（仅状态上报，不参与层分配）
- 全无模式：自动连接主节点，显示主节点状态
- 主节点管理面板显示 Android 节点（🤖 标识 + 模式标签）

### 阶段 4：体验优化 + 发布（~1 周）

- 首次启动引导（模型下载 → 模式选择 → 连接配置）
- 通知栏快捷控制（暂停/恢复/切换模式）
- 电量感知：≥80% 全速 / 20-80% 半速 / <20% 暂停
- 充电检测：默认仅充电时参与推理
- 温控降频：≥42°C 降 50% 线程
- GitHub Release APK + Google Play（可选）

---

## 5.5 性能预估

### 5.5.1 全有模式（本地推理）

| 芯片 | 等级 | 线程 | Q4_K_M tok/s | 峰值 RAM |
|------|------|------|-------------|----------|
| 骁龙 8 Gen 3 | 旗舰 | 4 | 12-18 | 1.8 GB |
| 骁龙 8+ Gen 1 | 次旗舰 | 4 | 8-12 | 1.8 GB |
| 骁龙 865 | 中端 | 4 | 5-8 | 1.8 GB |
| 天玑 9200 | 旗舰 | 4 | 10-15 | 1.8 GB |
| 骁龙 778G | 中低端 | 2 | 3-4 | 1.9 GB |
| 骁龙 480 | 入门 | 2 | 1-2 | 2.0 GB |

### 5.5.2 全无模式（转发给 PC 集群）

Android 端几乎零开销。延迟取决于 PC 集群推理速度 + 网络 RTT：

| 场景 | 单 token 延迟 | 说明 |
|------|-------------|------|
| PC 主节点单机 | ~50ms + 网络 | 主节点本地推理 |
| PC 3 节点流水线 | ~200ms + 网络 | 分布式层拆分 |
| WiFi 局域网 | +5-15ms | 同路由器 |
| Tailscale 远程 | +30-80ms | 跨子网 |

---

## 5.6 与现有系统的关系

| 已有模块 | Android 全有模式 | Android 全无模式 |
|----------|-----------------|-----------------|
| `tcp_comm.py` 消息协议 | 可选：仅心跳上报 | 不涉及（用 HTTP） |
| `scheduler.py` 分层算法 | 不参与（0 层） | 不参与（0 层） |
| `llama_engine.py` | JNI 等效实现 | 不涉及 |
| `api_server.py` `/api/chat` | 不涉及 | 直接复用 |
| `api_server.py` `/api/sessions` | 不涉及 | 直接复用 |
| Tailscale 组网 | Android 官方 App | Android 官方 App |
| `config.py` | DataStore 等效 | DataStore 等效 |
| 前端 React UI | 不涉及 | Android 原生 UI 替代 |

---

## 5.7 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| Google Play 审核拒绝 | 中 | 无法上架 | APK 侧载（GitHub Release）|
| 温控降频 | 高 | 推理速度骤降 | 温控策略 + 散热背夹推荐 |
| 后台被系统杀死 | 高 | 推理中断 | 前台 Service + 电池优化白名单引导 |
| 厂商 ROM 兼容性 | 中 | 部分手机无法运行 | 主力适配原生/类原生，MIUI/ColorOS 专项说明 |
| ARM SVE 指令缺失 | 低 | 老芯片极慢 | 最低 ARMv8.2-A (2017+)，自动检测提示 |
| GGUF 模型加载失败 | 低 | 模式 A 不可用 | 自动回退到模式 B |
| Tailscale 耗电 | 中 | 待机缩短 | 仅充电时启用；按需连接模式 |
| 网络断连（全无模式） | 中 | 推理不可用 | 自动切换到全有模式（如已下载模型） |

---

## 5.8 时间线

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  阶段 0  │  阶段 1  │  阶段 2  │  阶段 3  │  阶段 4  │
│  环境    │  模式B   │  模式A   │  模式    │  体验    │
│  验证    │  全无    │  全有    │  切换    │  优化    │
│  2天     │  1周     │  2周     │  1周     │  1周     │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

**里程碑**：
- **M0** (阶段 0 完成): `adb shell` 跑通 Qwen-1.8B GGUF 推理
- **M1** (阶段 1 完成): APK 可连接 PC 主节点聊天（全无模式可用）
- **M2** (阶段 2 完成): 离线本地推理可用（全有模式可用）
- **M3** (阶段 3 完成): 双模式无缝切换 + 主节点识别 Android 节点
- **M4** (阶段 4 完成): GitHub Release APK + Google Play 上架

---

> **编写日期**: 2026-07-02  
> **设计理念**: 全有或全无 — 不做层间拆分  
> **关联文档**: [Android SAF模型存储方案](Android SAF模型存储方案.md)、[分布式推理流水线实施计划](分布式推理流水线实施计划.md)、[整体架构](整体架构.md)
