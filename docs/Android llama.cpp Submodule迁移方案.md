# Android llama.cpp Submodule 迁移方案

> 日期：2026-07-14
> 状态：方案已确定，等待空闲维护窗口实施
> 范围：`android/app/src/main/cpp/llama.cpp/` 及 Android Full 原生构建链
> 不包含：本次仅形成文档，不修改 Git 索引、CMake、Gradle、CI 或打包脚本

## 1. 背景

当前主仓库直接跟踪一份完整的 llama.cpp 上游源码树：

- 路径：`android/app/src/main/cpp/llama.cpp/`
- Git 跟踪文件数：2746
- 工作区文件原始大小：约 73.44 MiB
- 上游仓库：`https://github.com/ggml-org/llama.cpp.git`
- 当前记录版本：`47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`
- 当前接入方式：`add_subdirectory(llama.cpp build-llama)`

目录中包含上游的文档、CI、Dockerfile、脚本、示例、测试和工具等完整内容。它们不会进入 APK，但会进入主仓库历史、代码检出和语言统计。

## 2. 决策

将 `android/app/src/main/cpp/llama.cpp/` 从主仓库内置源码迁移为 Git submodule，并继续固定到经过验证的上游 commit。

不采用以下方案：

| 方案 | 不采用原因 |
|---|---|
| CMake `FetchContent` | 首次配置需要联网；Android Gradle 的 flavor、构建类型、ABI 和 CMake 配置哈希可能产生多个 `.cxx` 构建目录，各目录可能分别下载；清理 `.cxx`、更换构建机或 CI runner 后也会重新下载 |
| `.gitignore` + 构建脚本 `git clone` | 版本和源码准备逻辑落入自定义脚本，Android Studio、Gradle 命令行和 CI 容易出现不同入口行为 |
| 继续直接提交完整源码 | 主仓库持续承担无关上游文件和后续升级产生的大量差异 |

选择 submodule 后，CMake 不负责联网。源码只在新克隆或显式初始化 submodule 时下载；执行 `gradlew clean`、重复打包或切换 Full/Release 不会重新下载 submodule。

## 3. 目标状态

迁移后的主仓库只保存：

1. `.gitmodules` 中的 llama.cpp 仓库地址和路径。
2. 路径 `android/app/src/main/cpp/llama.cpp` 对应的 Gitlink（模式 `160000`）。
3. Gitlink 指向的固定 llama.cpp commit。
4. 位于 submodule 外部的版本说明和本项目构建约束。

目标 `.gitmodules` 配置：

```ini
[submodule "android/app/src/main/cpp/llama.cpp"]
    path = android/app/src/main/cpp/llama.cpp
    url = https://github.com/ggml-org/llama.cpp.git
```

不要配置跟踪分支。submodule 的 Gitlink必须固定到明确 commit，避免上游分支更新自动改变 Android 构建结果。

现有 CMake 路径保持不变：

```cmake
add_subdirectory(llama.cpp build-llama)
```

因此迁移不改变 JNI include 路径、链接目标 `llama` 或 APK 内容选择。

## 4. 实施前检查

实施必须在单独维护窗口完成，并满足以下条件：

- Android 相关路径没有未提交修改。
- 当前 Full 和 Lite Release 基线可以构建。
- 能访问上游 Git 仓库，或已准备可信本地镜像。
- 已确认标记的 commit 在上游仓库存在且可检出。
- 已比较当前 vendored 源码与上游 commit，记录所有差异。

当前目录包含本项目文件 `QLH_VENDOR_COMMIT.txt`。它不属于上游源码，迁移前应移动到 submodule 外，例如：

```text
android/app/src/main/cpp/LLAMA_CPP_VERSION.md
```

如果比较发现除版本说明外还有本项目定制代码，不能直接迁移到官方 submodule。应先选择以下一种处理方式：

- 将改动移到 `qlh_llama_jni.cpp` 或外层 CMake，保持上游 submodule 无修改。
- 形成可重复应用、可测试的补丁文件。
- 确实需要长期修改上游时，改用项目维护的 llama.cpp fork，并锁定 fork commit。

禁止让主仓库引用只存在于某位开发者本地、未推送到远端的 submodule commit。

## 5. 建议实施步骤

以下命令是实施清单，不在本文档提交时执行。

### 5.1 校验上游版本

```powershell
git clone https://github.com/ggml-org/llama.cpp.git "$env:TEMP\qlh-llama-verify"
git -C "$env:TEMP\qlh-llama-verify" checkout 47e1de77aa0f06bf73cfd8c5281d95979f89fcbe
```

排除 `QLH_VENDOR_COMMIT.txt` 后，对当前源码树和上游目标 commit 做文件清单及内容比较。比较未通过时暂停迁移并先处理差异。

### 5.2 移出项目版本说明

```powershell
git mv android/app/src/main/cpp/llama.cpp/QLH_VENDOR_COMMIT.txt `
  android/app/src/main/cpp/LLAMA_CPP_VERSION.md
```

版本说明应记录：

- 上游 URL
- 固定 commit
- 更新日期
- Android 验证结果
- 是否存在项目补丁或 fork

Gitlink 是实际版本来源，说明文件用于人工审查，二者必须一致。

### 5.3 用 submodule 替换 vendored 目录

```powershell
git rm -r android/app/src/main/cpp/llama.cpp
git submodule add https://github.com/ggml-org/llama.cpp.git `
  android/app/src/main/cpp/llama.cpp
git -C android/app/src/main/cpp/llama.cpp checkout `
  47e1de77aa0f06bf73cfd8c5281d95979f89fcbe
git add .gitmodules android/app/src/main/cpp/llama.cpp `
  android/app/src/main/cpp/LLAMA_CPP_VERSION.md
```

不要在日常构建脚本中执行 `git submodule update --remote`。该命令会引入未经验证的上游版本。

### 5.4 增加 CMake 前置检查

Full 构建进入 `add_subdirectory()` 前，应检查：

```cmake
set(QLH_LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/llama.cpp")
if(NOT EXISTS "${QLH_LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule is not initialized. Run: "
        "git submodule update --init --recursive")
endif()

add_subdirectory("${QLH_LLAMA_CPP_DIR}" build-llama)
```

该检查必须放在 `QLH_LITE` 的提前 `return()` 之后，保证：

- Full 需要本地 llama.cpp 源码，缺失时立即给出可操作错误。
- Lite 只构建 JNI stub，不应因为未初始化 llama.cpp 而失败。

### 5.5 调整开发和 CI 入口

新克隆推荐：

```bash
git clone --recurse-submodules <qlh-repository-url>
```

已有工作区：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

CI checkout 必须启用 submodule。以 GitHub Actions 为例：

```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive
```

如果使用内部镜像，应通过 Git 配置或 CI checkout 配置替换 URL，不要在 CMake 中加入备用下载逻辑。

## 6. 下载和离线行为

迁移后各操作的预期行为：

| 操作 | 是否下载 llama.cpp |
|---|---:|
| 已初始化工作区重复执行 Full/Lite 打包 | 否 |
| `gradlew clean` 后重新构建 | 否 |
| 删除 `android/app/.cxx` 后重新构建 | 否，submodule 源码不在 `.cxx` 中 |
| 新克隆使用 `--recurse-submodules` | 是，克隆阶段下载一次 |
| 新克隆未初始化 submodule，构建 Lite | 否，应允许成功 |
| 新克隆未初始化 submodule，构建 Full | 否，但应立即报初始化提示并停止 |
| 显式执行 `git submodule update --init --recursive` | 首次或缺对象时下载 |
| 切换到引用另一 llama.cpp commit 的主仓库版本 | 本地没有目标对象时下载 |

“离线可构建”指 submodule、Gradle 依赖、Android SDK/NDK/CMake 均已在本机准备完成。submodule 方案只保证 CMake 不会为 llama.cpp 主动联网，不能替代 Gradle 和 Android 工具链缓存。

## 7. 验收清单

### 7.1 Git 结构

- `git ls-files -s android/app/src/main/cpp/llama.cpp` 显示模式 `160000`。
- `git submodule status` 显示预期 commit，且没有 `-`、`+` 或冲突状态。
- 主仓库不再逐文件跟踪 llama.cpp 上游文件。
- `.gitmodules` 使用 `ggml-org/llama.cpp` 官方 HTTPS 地址。
- submodule 工作区 `git status --short` 为空。

### 7.2 构建行为

- 已初始化 submodule 后，Full Release 构建通过。
- Lite Release 构建通过，且仍只编译 `qlh_llama_jni_stub.cpp`。
- 未初始化 submodule 时，Lite 构建通过。
- 未初始化 submodule 时，Full 构建以明确提示失败。
- 删除 `.cxx` 并断开网络后，在 Android/Gradle 依赖已缓存的环境中 Full 可以重新配置和构建。
- 连续构建 Full 和 Lite 不产生 llama.cpp 下载动作。

### 7.3 功能回归

- Full APK 能加载 `libqlh_llama_jni.so`。
- Full 能加载现有 Qwen GGUF 模型并完成至少一次生成。
- 模型加载、取消、释放和重新加载流程正常。
- Lite 不包含完整 llama.cpp 推理实现，HTTP thin 模式正常。
- Full/Lite Release 签名、包名和版本号保持不变。
- APK native 库及安装包体积与迁移前基线没有无法解释的变化。

### 7.4 仓库效果

- 新迁移提交不再包含 2746 个上游文件的普通 blob 变更。
- GitHub 主仓库语言统计不再计入 submodule 源码。统计刷新可能有延迟，不作为构建验收条件。
- GitHub 自动生成的源码 ZIP 不作为 Android 构建输入；发布源码包如需可直接构建，必须单独包含 submodule 内容。

## 8. 上游升级流程

submodule 迁移后，升级 llama.cpp 必须单独提交并经过 Android 回归：

```powershell
git -C android/app/src/main/cpp/llama.cpp fetch origin
git -C android/app/src/main/cpp/llama.cpp checkout <new-verified-commit>
git add android/app/src/main/cpp/llama.cpp `
  android/app/src/main/cpp/LLAMA_CPP_VERSION.md
```

升级提交应记录：

- 旧 commit 和新 commit
- 上游变更范围
- C API 或 CMake target 变化
- Full/Lite Release 构建结果
- Android 实机模型加载和生成结果
- APK 体积变化

不允许只执行 `git submodule update --remote` 后直接合并。

## 9. 风险与处理

| 风险 | 处理方式 |
|---|---|
| 开发者忘记初始化 | CMake Full 前置检查给出准确命令；README 和 CI 同步更新 |
| CI checkout 未拉取 submodule | checkout 阶段启用 recursive submodules，并在构建前检查状态 |
| 上游 commit 不可达 | 实施前验证官方远端；必要时使用受控 fork 或内部镜像 |
| submodule 内出现本地改动 | 构建前检查 submodule 状态；项目改动放在 JNI 外层或受控 fork |
| 上游升级破坏 C API/CMake target | 固定 commit；升级必须独立评审和实机回归 |
| 源码 ZIP 缺少 submodule | 不把 GitHub 自动源码 ZIP 当作可直接构建发行物 |
| 网络不可用 | 初始化后构建不联网；新环境使用内部镜像或预置完整 Git 工作区 |

## 10. 回滚方案

推荐把迁移做成一个独立提交。如果验收失败，优先直接 revert 该迁移提交，使主仓库恢复迁移前的 vendored 源码和 CMake 状态。

回滚后可执行：

```bash
git submodule deinit -f -- android/app/src/main/cpp/llama.cpp
git revert <submodule-migration-commit>
```

`.git/modules/` 下的本地缓存不影响仓库正确性，不应在自动回滚脚本中递归删除。确需清理时由维护者核对绝对路径后手工处理。

## 11. 实施时机

本方案等待满足以下条件后实施：

- 当前分布式推理和 PC 集显版紧急验证完成。
- 没有 Android Release 打包正在进行。
- 有时间完成一次新克隆验证和至少一台 Android 真机 Full/Lite 回归。
- 能将迁移、测试和必要修正放在同一维护窗口内完成。

在此之前继续使用当前 vendored 源码，不改为 `FetchContent`，也不在打包脚本中增加自动 clone。
