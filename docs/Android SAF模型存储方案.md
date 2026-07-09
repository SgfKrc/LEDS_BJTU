# Android SAF 模型存储方案

> **状态**：设计方案  
> **适用范围**：Android 全有模式（本地 llama.cpp / GGUF 推理）  
> **核心目标**：让 Android 模型文件在卸载 APK 后默认保留，同时允许用户在应用内主动删除  
> **设计原则**：模型文件属于用户资产，不应随 APK 卸载被默认删除

---

## 1. 背景与问题

Android 全有模式需要在手机本地保存 GGUF 模型文件，例如：

```text
qwen-1.8b-Q4_K_M.gguf
```

该文件通常约 1GB+，重新下载成本较高。因此卸载应用时，默认应保留模型文件，只有用户明确选择时才删除。

PC 版可以通过 Inno Setup 卸载脚本实现：

```text
卸载时弹窗：是否同时删除 models 目录？
默认：否
```

但 Android 不同：

| 平台 | 卸载流程控制方 | 能否在卸载时弹应用自定义选项 |
|------|---------------|----------------------------|
| Windows PC | Inno Setup 卸载器 | ✅ 可以 |
| Android | 系统 Package Installer | ❌ 不可以 |

也就是说，Android 应用无法在系统卸载 APK 时弹出自定义确认框。

---

## 2. 当前内部存储方案的问题

当前测试实现中，Android 模型管理器使用：

```kotlin
File(context.filesDir, "models")
```

对应路径类似：

```text
/data/data/com.qlh.inference/files/models/
```

该方案优点：

- 实现简单
- 无需额外权限
- llama.cpp 可直接通过普通文件路径加载

但缺点非常关键：

| 问题 | 说明 |
|------|------|
| 卸载必删 | APK 卸载时，系统会自动删除 `context.filesDir` |
| 无法选择 | 应用无法拦截系统卸载流程并询问是否保留 |
| 用户不可见 | 普通文件管理器通常看不到内部目录 |
| 不适合正式版 | 1GB+ 模型会随卸载直接丢失 |

因此内部存储只适合作为早期测试方案，不适合作为正式的 Android 模型存储方案。

---

## 3. SAF 目标行为

正式版 Android 应采用 SAF（Storage Access Framework）或等价用户可见目录方案，使模型文件独立于 APK 生命周期。

目标行为：

```text
首次进入全有模式
  ↓
用户选择模型目录，例如 Download/QLH/models
  ↓
应用持久化目录访问权限
  ↓
模型下载 / 导入到该目录
  ↓
llama.cpp 从该目录加载 GGUF
  ↓
卸载 APK 时模型默认保留
  ↓
用户可在应用内点击“删除本地模型”主动删除
```

用户体验目标：

| 场景 | 行为 |
|------|------|
| 卸载 APK | 默认保留模型目录 |
| 重新安装 APK | 可重新授权同一目录，复用旧模型 |
| 用户想清理空间 | 应用内提供“删除本地模型”按钮 |
| 模型损坏 | 应用提示重新下载或重新选择目录 |
| 没有目录权限 | 提示重新授权 |

---

## 4. 方案选型

### 4.1 方案 A：SAF 目录 + `/proc/self/fd/<fd>` 路径加载

这是最理想方案。

流程：

```kotlin
val pfd = contentResolver.openFileDescriptor(modelUri, "r")
val fd = pfd.detachFd()
val path = "/proc/self/fd/$fd"
llamaLoadModel(path)
```

架构：

```text
SAF DocumentUri
   ↓ ContentResolver.openFileDescriptor()
ParcelFileDescriptor
   ↓ fd
/proc/self/fd/<fd>
   ↓
llama.cpp native load_model(path)
```

优点：

- 不复制模型
- 不占双份空间
- 卸载 APK 后模型保留
- 用户可见、可备份、可手动替换

风险：

| 风险 | 说明 |
|------|------|
| mmap 兼容性 | llama.cpp 加载 GGUF 时可能依赖 mmap；部分 content provider 的 fd 不一定支持完整 mmap 行为 |
| fd 生命周期 | 必须保证模型加载期间 fd 不被提前关闭 |
| ROM 差异 | 不同厂商文件管理器 / DocumentsProvider 行为可能不同 |
| native 路径假设 | llama.cpp 可能对 `/proc/self/fd` 路径进行 stat/seek，需要实机验证 |

验证重点：

```text
llama.cpp 是否能直接加载 /proc/self/fd/<fd>
```

如果验证通过，优先采用该方案。

---

### 4.2 方案 B：SAF 保留原件 + 内部缓存副本加载

如果方案 A 不稳定，则采用稳妥方案。

流程：

```text
SAF 外部模型目录
  ↓
复制到 context.cacheDir 或 context.filesDir 临时副本
  ↓
llama.cpp 加载普通文件路径
```

优点：

- llama.cpp 不需要适配 fd
- 兼容性最高
- SAF 原始模型仍然保留
- APK 卸载后外部原始模型不丢失

缺点：

| 缺点 | 说明 |
|------|------|
| 占用双份空间 | SAF 原件 + 内部副本，至少 2GB+ |
| 首次启动慢 | 需要复制大文件 |
| 缓存管理复杂 | 需要校验副本是否过期、损坏、空间不足 |

适合场景：

- `/proc/self/fd` 方案在部分机型失败
- 需要优先保证稳定性
- 用户手机空间充足

---

### 4.3 方案 C：公共 Downloads 路径直读

例如：

```text
/storage/emulated/0/Download/QLH/models/qwen-1.8b-Q4_K_M.gguf
```

优点：

- 路径直观
- llama.cpp 可直接读普通路径

缺点：

| 问题 | 说明 |
|------|------|
| Android 10+ Scoped Storage 限制 | targetSdk 较高时，直接读写公共 Downloads 不稳定 |
| 权限复杂 | 可能需要用户授权、SAF 或特殊文件权限 |
| 上架风险 | 申请 MANAGE_EXTERNAL_STORAGE 容易被拒 |

结论：不建议作为主线方案。可以作为调试 / 侧载 APK 的高级模式。

---

## 5. 推荐路线

采用渐进式路线：

```text
阶段 1：SAF 目录选择 + 模型扫描
阶段 2：验证 /proc/self/fd 加载
阶段 3：若失败，回退到内部缓存副本
阶段 4：接入 PC 主节点模型下载
阶段 5：设置页模型管理完善
```

推荐优先级：

| 优先级 | 方案 | 结论 |
|--------|------|------|
| P0 | SAF 目录选择 | 必做 |
| P0 | 保存持久 URI 权限 | 必做 |
| P0 | 扫描 `.gguf` 模型 | 必做 |
| P1 | `/proc/self/fd` 加载 | 优先验证 |
| P1 | 内部缓存副本 fallback | 必须准备 |
| P2 | PC 主节点下载到 SAF | 增强体验 |
| P2 | PC 前端显示 Android 模型状态 | 可选 |

---

## 6. Android 端改造点

### 6.1 权限与目录选择

使用系统目录选择器：

```kotlin
Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
```

用户选择目录后：

```kotlin
contentResolver.takePersistableUriPermission(
    treeUri,
    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
)
```

需要保存：

```kotlin
model_tree_uri = treeUri.toString()
```

保存位置：

```text
SettingsDataStore
```

---

### 6.2 SettingsDataStore 新增字段

建议新增：

```kotlin
val KEY_MODEL_TREE_URI = stringPreferencesKey("model_tree_uri")
val KEY_SELECTED_MODEL_URI = stringPreferencesKey("selected_model_uri")
val KEY_MODEL_STORAGE_MODE = stringPreferencesKey("model_storage_mode")
```

含义：

| Key | 说明 |
|-----|------|
| `model_tree_uri` | 用户选择的模型目录 URI |
| `selected_model_uri` | 当前使用的 GGUF 文件 URI |
| `model_storage_mode` | `saf_fd` / `saf_cache` / `internal_test` |

---

### 6.3 ModelManager 改造

当前：

```kotlin
val modelsDir: File
    get() = File(context.filesDir, "models")
```

应改造成多后端：

```kotlin
sealed class ModelLocation {
    data class InternalFile(val file: File) : ModelLocation()
    data class SafDocument(val uri: Uri) : ModelLocation()
    data class CachedCopy(val sourceUri: Uri, val cacheFile: File) : ModelLocation()
}
```

核心职责：

| 方法 | 职责 |
|------|------|
| `selectModelDirectory(uri)` | 保存 SAF 目录权限 |
| `listModels()` | 扫描目录下 `.gguf` 文件 |
| `selectModel(uri)` | 选择当前模型 |
| `openModelForLlama()` | 返回可供 llama.cpp 加载的路径或 fd token |
| `deleteSelectedModel()` | 用户主动删除模型 |
| `verifyModel()` | SHA256 校验 |

---

### 6.4 llama.cpp 加载接口改造

当前 native 接口：

```kotlin
private external fun nativeLoadModel(path: String, nCtx: Int): Long
```

如果 `/proc/self/fd` 可用，可以保持不变：

```kotlin
nativeLoadModel("/proc/self/fd/$fd", nCtx)
```

但需要保证 fd 生命周期：

```kotlin
class LoadedModelHandle(
    val modelPtr: Long,
    val pfd: ParcelFileDescriptor?
)
```

只要模型仍加载，`ParcelFileDescriptor` 就不能关闭。

如果走缓存副本方案，则仍传普通路径：

```kotlin
nativeLoadModel(cacheFile.absolutePath, nCtx)
```

---

## 7. Android UI 改造点

### 7.1 设置页新增“模型管理”区域

建议在 `SettingsScreen` 增加：

```text
模型管理
├── 存储位置：Download/QLH/models 或 未选择
├── 当前模型：qwen-1.8b-Q4_K_M.gguf
├── 模型大小：1.16 GB
├── 校验状态：已校验 / 未校验 / 校验失败
├── [选择模型目录]
├── [扫描模型]
├── [从主节点下载]
└── [删除本地模型]
```

### 7.2 卸载说明

在模型管理区域显示提示：

```text
模型保存在你选择的外部目录中，卸载 APK 默认不会删除。
如需清理空间，请点击“删除本地模型”。
```

---

## 8. PC 后端适配

SAF 本身是 Android 本地文件授权机制，PC 后端不是必须改。

但如果要支持 Android 从 PC 主节点下载模型，需要复用或增强现有接口：

| 接口 | 当前用途 | Android 用途 |
|------|----------|--------------|
| `GET /api/models/gguf` | 列出 PC 上 GGUF 模型 | Android 显示可下载模型列表 |
| `GET /api/models/download/{filename}` | 下载 GGUF 文件 | Android 下载到 SAF 目录 |

建议补充字段：

```json
{
  "name": "qwen-1.8b-Q4_K_M.gguf",
  "size": 1245540512,
  "sha256": "...",
  "download_url": "/api/models/download/qwen-1.8b-Q4_K_M.gguf"
}
```

后端增强项：

| 增强 | 是否必须 |
|------|----------|
| 返回文件大小 | 建议 |
| 返回 SHA256 | 建议 |
| 支持 Range 请求 | 建议，用于断点续传 |
| 限制路径穿越 | 必须 |
| 下载日志 | 可选 |

---

## 9. PC 前端适配

PC 前端不需要参与 SAF 目录选择。

可选增强：

| 功能 | 说明 |
|------|------|
| Android 节点标识 | 节点列表显示 🤖 Android |
| 模式显示 | 全有 / 全无 |
| 模型状态 | 已就绪 / 未下载 / 校验失败 |
| 下载二维码 | Android 扫码配置主节点地址 |
| 主节点模型列表 | 展示哪些 GGUF 可供 Android 下载 |

这些是体验增强，不阻塞 SAF 主流程。

---

## 10. 数据流设计

### 10.1 用户手动导入模型

```text
用户点击“选择模型目录”
  ↓
ACTION_OPEN_DOCUMENT_TREE
  ↓
保存 treeUri 权限
  ↓
DocumentFile.fromTreeUri()
  ↓
扫描 .gguf
  ↓
用户选择模型
  ↓
保存 selected_model_uri
  ↓
加载模型
```

### 10.2 从 PC 主节点下载模型

```text
Android 设置主节点地址
  ↓
GET /api/models/gguf
  ↓
用户选择 GGUF
  ↓
在 SAF 目录创建文件
  ↓
GET /api/models/download/{filename}
  ↓
写入 SAF OutputStream
  ↓
SHA256 校验
  ↓
保存 selected_model_uri
```

### 10.3 加载模型

优先：

```text
selected_model_uri
  ↓ openFileDescriptor(uri, "r")
  ↓ /proc/self/fd/<fd>
  ↓ llama.cpp load_model
```

失败回退：

```text
selected_model_uri
  ↓ copy to internal cache
  ↓ llama.cpp load_model(cacheFile.absolutePath)
```

---

## 11. 风险与验证清单

| 风险 | 验证方式 | 对策 |
|------|----------|------|
| `/proc/self/fd` 无法 mmap | 在真机加载 1GB GGUF | 回退缓存副本 |
| fd 生命周期提前关闭 | 长时间连续推理 | 将 PFD 绑定到模型句柄生命周期 |
| SAF 权限丢失 | 重启 / 重新安装后测试 | 提示用户重新授权 |
| 下载中断 | 模拟断网 | Range 断点续传 |
| 空间不足 | 低剩余空间设备测试 | 下载前检查可用空间 |
| 模型损坏 | SHA256 mismatch | 删除损坏文件并提示重下 |
| 厂商 ROM DocumentsProvider 异常 | MIUI / ColorOS / HarmonyOS 测试 | 提供内部测试目录 fallback |

---

## 12. 最小可行实现（MVP）

MVP 不追求下载闭环，只验证 SAF 加载：

1. 设置页增加“选择模型目录”按钮
2. 用户手动把 GGUF 放入该目录
3. 应用扫描 `.gguf`
4. 用户选择模型
5. 尝试 `/proc/self/fd` 加载
6. 如果失败，复制到内部缓存再加载

MVP 完成标准：

- 卸载 APK 后，用户选择目录中的 GGUF 仍存在
- 重新安装 APK 后，重新授权同一目录即可复用模型
- 至少一台 Android 真机能完成本地推理
- 失败时有明确提示，不会静默崩溃

---

## 13. 与“全有或全无”理念的关系

SAF 只服务于 Android **全有模式**。

| 模式 | 是否需要 SAF |
|------|-------------|
| 全无模式 | ❌ 不需要，本机不保存模型 |
| 全有模式 | ✅ 需要，本机保存完整 GGUF 模型 |

SAF 不改变调度理念：

- Android 不参与 PC 层间拆分
- Android 要么本地完整推理
- 要么只把请求交给 PC 主节点

---

## 14. 结论

Android 无法像 PC Inno Setup 那样在系统卸载时弹出“是否删除 models”选项。

要实现“默认保留模型，用户主动删除才删除”，正确方向是：

```text
模型不要放在应用内部目录
而是放到用户授权的外部 SAF 目录
```

推荐实现路径：

1. 先做 SAF 目录选择 + 模型扫描
2. 优先验证 `/proc/self/fd/<fd>` 能否直接给 llama.cpp 加载
3. 若不稳定，使用 SAF 原件 + 内部缓存副本 fallback
4. 后续再接入 PC 主节点下载接口和前端状态展示

> **关键验证点**：llama.cpp 是否能稳定加载 `/proc/self/fd/<fd>` 指向的 SAF 模型文件。
