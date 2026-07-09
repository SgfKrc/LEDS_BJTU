# PC 与 Android 端交互体验优化计划

## 背景

当前 PC 端、Android 端和 Android 本地推理链路已经基本打通，但实际试用中暴露出多类体验与产品形态问题：Android 端交互仍不够成熟，控制面板相对 PC 版缺失较多功能；PC/Android 双端缺少复制回答、应用内日志查看等常用能力；同时后续还需要区分普通 Android 版和手机极简版，并为独显 PC 版增加实验用多模型能力。

本文记录已发现的待优化点、影响范围、推荐方案和验收标准，供后续迭代实现。

## 总体目标

1. 改善 Android 聊天输入体验，避免软键盘遮挡输入框或对话内容。
2. 为 PC 和 Android 双端增加一键复制回答功能。
3. 补齐 Android 版缺失的 PC 版用户设置能力，避免控制面板过度阉割。
4. 在普通 Android 版之外，后续额外推出手机用“极简版”，保留当前简化功能，并尽量压缩 APK、运行时和模型存储占用。
5. 独显 PC 版增加多模型实验支持，允许加载更重模型做效果对比。
6. PC 和 Android 均改为静默日志模式，提供应用内查看/清理日志入口。
7. 明确 Android 节点能力边界：不主动参与 llama.cpp 层间拆分；如果接收任务，也只能作为完整推理 Worker，且仅普通版考虑，极简版不参与。
8. 远期研究复杂问题的任务级推理链拆分，将复杂问题拆成多个完整子任务并分发给 Full Inference Worker。
9. 总 README 去除涉及成员真名的信息，满足导师合规要求。
10. PC 独显版引入 Gerrit 风格的主节点转让/备用主节点审查机制，仅 PC 独显版节点可投票，合计 >= +2 通过。
11. Android 普通版与极简版使用独立签名密钥和不同包名，避免相互干扰。

---

## 优化项 1：Android 软键盘遮挡聊天界面

### 现象

Android 端进入聊天页后，用户点击底部输入框唤起软键盘时，当前界面不会自动随键盘抬升，导致：

- 输入框可能被键盘遮挡；
- 最新对话内容不能自动滚动到可见区域；
- 用户需要手动收起键盘或拖动列表，体验不友好。

### 影响范围

- Android 客户端聊天页；
- 主要涉及 `ChatScreen` 输入栏、消息列表和 Scaffold / Column 布局；
- 与全有模式 / 全无模式无关，属于通用 UI 问题。

### 推荐方案

在 Android Compose 聊天页中适配 IME Insets：

1. 根布局或聊天内容容器使用 `imePadding()`，让底部输入栏随软键盘上移。
2. 消息列表使用 `LazyColumn` 并配合 `BringIntoViewRequester` 或滚动控制，在新消息/键盘弹出时保持最后消息可见。
3. 输入区域使用 `navigationBarsPadding()` + `imePadding()`，避免同时被系统导航栏和软键盘遮挡。
4. 检查 Activity 是否使用合适的 `windowSoftInputMode`。如果当前 edge-to-edge 设置导致默认 resize 失效，需要在 Compose 侧显式处理 WindowInsets。

### 可能涉及文件

- `android/app/src/main/java/com/qlh/inference/ui/ChatScreen.kt`
- `android/app/src/main/java/com/qlh/inference/MainActivity.kt`
- `android/app/src/main/AndroidManifest.xml`

### 验收标准

- 点击输入框后，输入框始终位于软键盘上方；
- 键盘弹出时，最后一条消息或正在输入区域不被遮挡；
- 发送消息后，消息列表自动滚动到底部；
- 横屏/三键导航/手势导航下均不出现明显遮挡。

---

## 优化项 2：PC 和 Android 缺少一键复制回答功能

### 现象

当前用户如果想复制 AI 的某一条回答，需要手动选中文本。PC 端和 Android 端都没有明确的“复制回答”按钮，对长回答尤其不方便。

### 影响范围

- PC Web 前端聊天消息气泡；
- Android Compose 聊天消息气泡；
- 不影响后端推理逻辑，只涉及前端/客户端交互。

### 推荐方案

为 assistant 消息增加一键复制入口。

#### PC 端

在 AI 回答气泡中增加复制按钮：

- 鼠标悬停或消息右上角显示“复制”按钮；
- 点击后调用 `navigator.clipboard.writeText(answerText)`；
- 复制成功后短暂显示“已复制”；
- 复制失败时给出提示或回退到旧式复制方案。

#### Android 端

在 AI 回答气泡中增加复制入口：

- 气泡右上角显示复制图标，或长按消息弹出“复制回答”；
- 使用 Android `ClipboardManager` 写入剪贴板；
- 复制成功后通过 Snackbar / Toast 显示“已复制”；
- 仅复制回答正文，不复制 metrics、思考开关状态或内部字段。

### 可能涉及文件

PC 端：

- `frontend/src/components/ChatPanel.jsx`
- 可能涉及 `frontend/src/App.jsx` 中的全局提示 / 状态管理

Android 端：

- `android/app/src/main/java/com/qlh/inference/ui/ChatScreen.kt`

### 验收标准

- PC 端每条 AI 回答旁都有可发现的复制入口；
- Android 端每条 AI 回答可一键复制或长按复制；
- 复制内容与回答文本完全一致；
- 复制成功有明确反馈；
- 用户消息可选是否也支持复制，但最低要求是 assistant 回答支持复制。

---

## 优化项 3：Android 版缺失较多 PC 版用户设置功能

### 现象

Android 版本目前控制面板简化过多，缺少很多 PC 版已有的用户设置能力。对于普通 Android 版而言，过度阉割会导致用户无法调整关键运行参数，也不利于把 Android 端作为完整客户端继续演示。

### 需要补齐的方向

普通 Android 版应尽量对齐 PC 端用户设置，但不强行加入 Android 不具备的分层拆分能力。

建议补齐：

1. **连接设置**：主节点地址、端口、连接测试、连接状态展示。
2. **推理参数**：最大生成 token、temperature、top_p、上下文长度等。
3. **模式切换**：全有模式 / 全无模式状态更清晰，展示当前运行引擎。
4. **模型管理**：SAF 模型目录、已选模型、模型大小、刷新、卸载/释放模型。
5. **日志入口**：查看本地日志、复制日志、清理日志。
6. **基础状态**：设备信息、可用内存、native runtime 是否可用、当前 ABI。

不建议补齐：

- Android 主动参与 PC 版 Transformer 层间拆分分配；
- Android 端复杂节点管理和层覆盖逻辑；
- 重模型实验相关设置。

### 可能涉及文件

- `android/app/src/main/java/com/qlh/inference/ui/SettingsScreen.kt`
- `android/app/src/main/java/com/qlh/inference/MainViewModel.kt`
- `android/app/src/main/java/com/qlh/inference/data/SettingsDataStore.kt`
- `android/app/src/main/java/com/qlh/inference/service/ModelManager.kt`

### 验收标准

- Android 普通版至少覆盖 PC 端常用用户设置；
- 用户不需要回到 PC 才能修改基本推理参数；
- Android 控制面板不再给人“只剩聊天壳”的感觉；
- 不引入 Android 当前无法可靠支持的层间拆分功能。

---

## 优化项 4：Android 普通版与手机极简版分离

### 背景

完成基础交互优化后，Android 应额外推出“手机极简版”。极简版保留当前简化功能，面向普通手机用户，避免复杂控制面板和完整模型目录管理带来的认知负担。同时，极简版应把安装包体积、运行时占用、缓存占用和模型存储占用都作为核心约束，尽量压缩空间占用。

### 产品定位

| 版本 | 定位 | 功能范围 |
|------|------|----------|
| Android 普通版 | 完整客户端 | 全有模式、本地 GGUF、SAF 模型目录、较完整设置、日志查看 |
| Android 极简版 | 手机轻量入口 | 简化 UI、固定/内置模型策略、少量参数、尽量无需管理完整 models 目录、尽量压缩空间占用 |

### 极简版模型策略

极简版将来考虑使用更小/更简化的大模型，只使用 INT4 或等价轻量量化形态。因此：

- 极简版不需要完整 `models/` 目录概念；
- 不要求用户理解 Safetensors / GGUF / 多模型目录；
- 可以只暴露“下载/选择一个推荐模型”或“使用内置推荐模型”的入口；
- 不提供复杂日志查看和清理入口；
- 不提供节点管理、层分配、重模型实验等功能。

### 极简版空间压缩策略

极简版需要把空间占用作为产品约束，而不是只做 UI 简化：

1. **APK 体积压缩**：开启 R8、resource shrink、native strip，只保留必要 ABI，优先 `arm64-v8a`。
2. **依赖裁剪**：不打包普通版才需要的日志查看、完整控制面板、多模型管理、Worker 接收任务等模块。
3. **模型存储压缩**：优先使用单一推荐 INT4 小模型，不要求完整 `models/` 目录结构，不保留多份缓存副本。
4. **运行时缓存控制**：限制 SAF cache / 下载缓存大小，提供自动清理策略，避免长期占用手机存储。
5. **功能入口压缩**：只保留聊天、连接/模式基础设置、必要模型状态提示，隐藏高级参数。
6. **分发物拆分**：极简 APK 与普通 APK 分开构建和分发，不把普通版 native/高级资源带进极简包。

### 技术建议

后续可用 Gradle product flavor 区分：

- `fullRelease`：Android 普通版；
- `liteRelease`：Android 极简版。

不同 flavor 可通过 `BuildConfig` 控制：

- 是否显示高级设置；
- 是否显示日志入口；
- 是否显示完整模型管理；
- 是否允许多模型选择；
- 是否包含 Worker 接收任务能力；
- 是否启用普通版 native/高级资源。

### 验收标准

- 普通版和极简版功能定位清晰；
- 极简版启动路径更短，不暴露复杂控制面板；
- 极简版 APK、缓存、模型存储占用明显低于普通版；
- 极简版不依赖完整 models 目录；
- 极简版不包含 Worker 接收任务能力；
- 普通版仍保留高级设置、日志能力和可选 Worker 能力。

---

## 优化项 5：独显版增加多模型实验支持

### 背景

PC 独显版作为主节点和实验平台，需要支持更重模型，用于测试模型规模、推理效果和性能差异。重模型仅用于实验，不进入集显版、Android 普通版、Android 极简版的功能范围。

### 范围界定

| 版本 | 是否支持重模型实验 | 说明 |
|------|------------------|------|
| PC 独显版 | 是 | 主节点实验平台，可支持多模型/重模型 |
| PC 集显版 | 否 | 保持 GGUF/轻量模型，控制安装包和资源占用 |
| Android 普通版 | 否 | 仅本地轻量 GGUF / 转发 PC |
| Android 极简版 | 否 | 仅推荐小模型/INT4 简化模型 |

### 推荐方案

1. 独显版模型管理支持多个模型配置：
   - 模型名称；
   - 模型路径；
   - 模型类型；
   - 推荐显存；
   - 默认最大上下文；
   - 是否实验模型。
2. 前端设置中增加“实验模型”区域，仅独显版显示。
3. 后端根据配置选择模型加载路径，避免影响集显版默认逻辑。
4. 分发服务器和 README 标注：重模型不随安装包分发，需要用户自行下载。

### 可能涉及文件

- `src/config.py`
- `src/model_module.py`
- `src/model_downloader.py`
- `frontend/src/components/SettingsModal.jsx`
- `frontend/src/components/AdminPanel.jsx`
- `packaging/qlh-cuda.spec`
- `packaging/setup-cuda.iss`

### 验收标准

- 独显版可以选择多个模型配置；
- 集显版/Android 不显示重模型实验入口；
- 未下载重模型时有明确提示，不影响默认模型启动；
- 实验模型的加载失败不会破坏基础推理功能。

---

## 优化项 6：PC / Android 静默日志与应用内日志管理

### 现象与要求

PC 版不应继续显示后台控制台窗口，Android 也不应把日志暴露为用户需要手动查看系统 logcat 的形式。两端都应静默保存日志到本地，并提供应用内一键查看和清理日志功能。

极简版不提供日志查看/清理功能，避免界面复杂化。

### PC 端推荐方案

1. 打包版改为无控制台窗口启动：
   - PyInstaller `console=False` 或使用 windowed bootloader；
   - 后端 stdout/stderr 重定向到本地日志文件。
2. 日志保存到用户可写目录：
   - 安装目录 `logs/`，或 `%LOCALAPPDATA%/QLH-Edge-Inference/logs/`；
   - 按日期滚动，例如 `qlh-YYYY-MM-DD.log`。
3. 前端提供日志入口：
   - 查看最近日志；
   - 复制日志；
   - 清理日志；
   - 打开日志目录。

### Android 端推荐方案

1. 应用内写本地日志文件：
   - `context.filesDir/logs/` 或 `context.cacheDir/logs/`；
   - 按日期或大小滚动。
2. 设置页增加日志管理入口：
   - 查看日志；
   - 复制日志；
   - 分享日志；
   - 清理日志。
3. 保留 `Logcat` 输出用于开发，但用户态问题排查优先依赖应用内日志。

### 极简版限制

- 不显示日志管理入口；
- 仅内部静默记录必要崩溃/错误；
- 如需排查，用户安装普通版或导出系统日志。

### 可能涉及文件

PC 端：

- `packaging/qlh-cpu.spec`
- `packaging/qlh-cuda.spec`
- `packaging/launcher.py`
- `src/api_server.py`
- `frontend/src/components/SettingsModal.jsx`

Android 端：

- `android/app/src/main/java/com/qlh/inference/service/InferenceService.kt`
- `android/app/src/main/java/com/qlh/inference/ui/SettingsScreen.kt`
- 可新增 `android/app/src/main/java/com/qlh/inference/logging/` 相关工具类

### 验收标准

- PC 打包版启动后不再显示后台控制台窗口；
- PC 后端异常能写入本地日志；
- Android 运行时错误能写入应用内日志；
- 普通版提供查看/复制/清理日志；
- 极简版不显示日志入口。

---

## 优化项 7：Android 节点能力边界与任务接收能力研究

### 当前原则

Android 由于 llama.cpp 难以进行 Transformer 层间拆分，当前不参与 PC 集群的主动层分配。也就是说，Android 不作为“某几层 Transformer 的执行节点”加入链式分层流水线。

### 能力边界

Android 如果要接收任务，原则上只能接收**完整推理任务**，不能接收层间拆分任务：

- 可以接收完整 prompt；
- 使用 Android 本地 llama.cpp 完整推理；
- 返回完整回答或流式 token；
- 不接收 hidden states；
- 不执行部分 Transformer 层；
- 不参与 PC 主节点的 layer assignment。

因此，“不参与主动分层分配”不等于“完全不能贡献算力”。Android 普通版后续可以研究作为完整推理 Worker；Android 极简版不考虑接收任务。

### 推荐定位

Android 节点能力按版本区分：

| Android 版本 | 是否接收任务 | 任务类型 | 说明 |
|--------------|--------------|----------|------|
| 普通版 | 可研究 | 仅完整推理任务 | 接收完整 prompt，本机完整 GGUF 推理后返回结果 |
| 极简版 | 不考虑 | 无 | 只作为手机轻量聊天入口，避免耗电、发热和后台占用 |

普通版即使支持 Worker，也必须是用户显式开启，默认不接收后台任务。

### 可能模式

| 模式 | Android 是否适合 | 说明 |
|------|----------------|------|
| 层间拆分节点 | 暂不适合 | llama.cpp 层间拆分困难，需要反向工程/改 native |
| 完整推理 Worker | 仅普通版可研究 | 接收完整请求，本机完整跑 GGUF；极简版不参与 |
| 远程薄客户端 | 已支持 | Android 只负责 UI，PC 集群推理 |
| 本地离线终端 | 已支持/继续完善 | Android 全有模式 |

### 后续研究方向

1. PC 主节点增加“完整任务 Worker”概念，与 layer worker 区分。
2. Android 普通版暴露轻量 HTTP / WebSocket 接口，接收完整推理请求。
3. 调度器根据任务类型决定：
   - 分层流水线任务 -> PC 节点；
   - 独立完整推理任务 -> 可分给 Android 普通版 worker；
   - Android 极简版 -> 永不接收任务；
   - 用户手机前台使用时 -> 不接收后台任务，避免发热耗电。
4. 增加 Android 普通版开关：是否允许作为 Worker 接收任务，默认关闭。

### 验收标准

- Android 不被错误加入 layer assignment；
- 仅 Android 普通版可选作为完整推理 Worker；
- Android 极简版不显示、不启用 Worker 接收任务能力；
- 主节点 UI 清楚区分“分层节点”和“完整任务节点”；
- 用户可关闭 Android Worker 模式，默认关闭，避免手机发热/耗电。

---

## 优化项 8：远期计划 - 复杂问题任务级推理链拆分

### 定位

这是远期计划，不作为当前 Android 普通版/极简版的近期目标，也不替代 PC 端已有的 Transformer 层间拆分流水线。

任务级推理链拆分指的是：把一个复杂问题拆成多个可以独立完整推理的子任务，分发给多个 Full Inference Worker 并行处理，最后由主节点汇总结果。它解决的是复杂任务的并行编排问题，而不是单次 token 生成或单次 model forward 的加速问题。

### 与层拆分的区别

| 维度 | Transformer 层拆分 | 任务级推理链拆分 |
|------|--------------------|------------------|
| 拆分对象 | 模型内部层 | 用户问题 / 推理任务 |
| 节点输入 | hidden states / KV 相关中间状态 | 完整文本 prompt / 子任务描述 |
| 节点输出 | hidden states 或 logits | 文本答案 / 结构化结果 |
| 是否需要完整模型 | 不一定 | 需要完整推理能力 |
| Android 普通版是否适合 | 不适合 | 可作为远期研究 |
| Android 极简版是否适合 | 不适合 | 不考虑 |

### 典型流程

```text
复杂问题
  -> 主节点分析是否可拆分
  -> 生成多个完整子任务
  -> 分发给 PC / Android 普通版 Full Inference Worker
  -> 各 Worker 独立完整推理
  -> 主节点收集结果
  -> 汇总、去重、排序、形成最终回答
```

### 适用场景

适合：

- 多角度分析；
- 多文档总结；
- 多方案比较；
- 代码审查多个文件；
- 搜索、阅读、汇总类任务；
- 需要多个独立观点再综合的复杂问题。

不适合：

- 普通单轮聊天；
- 单个回答的 token 级加速；
- 单次模型 forward 加速；
- PC 层间拆分流水线的直接替代。

### 与 Android Worker 的关系

任务级推理链拆分只考虑 Android 普通版作为 Full Inference Worker，且必须满足：

- Android 普通版只接收完整子任务；
- 不接收 hidden states；
- 不执行部分 Transformer 层；
- 不参与 layer assignment；
- Worker 模式必须用户显式开启，默认关闭；
- Android 极简版不参与任务接收和任务链拆分。

### 可能涉及文件

- `src/graph_orchestrator.py`
- `src/scheduler.py`
- `src/api_server.py`
- `frontend/src/components/AdminPanel.jsx`
- `android/app/src/main/java/com/qlh/inference/service/InferenceService.kt`
- 可新增 PC/Android Worker 协议文档

### 验收标准（远期）

- 主节点能区分“层拆分任务”和“任务级推理链任务”；
- 复杂问题能被拆成多个完整子任务；
- Full Inference Worker 能接收完整子任务并返回文本结果；
- 主节点能汇总多个 Worker 的结果；
- Android 普通版可选参与，极简版不参与；
- 该功能关闭时不影响现有 PC 层拆分推理。

---

## 优化项 9：总 README 去除成员真名

### 要求

导师要求总 README 去除涉及真名的信息。需要避免公开出现成员真实姓名。

### 推荐方案

修改根目录 `README.md`：

- 删除或匿名化团队分工表中的成员真名；
- 删除页脚中的成员真名；
- 可保留项目团队、学校、指导教师等非敏感信息；
- 如需保留分工，用“项目负责人 / 成员 A / 成员 B”或“模型优化组 / 分布式组 / 前端文档组”等角色描述代替。

### 验收标准

- 根 README 不再出现成员真实姓名；
- 项目职责说明仍清晰；
- 不影响项目介绍、使用说明和技术文档索引。

---

## 优化项 10：PC独显版主节点转让与备用主节点审查机制

### 背景

PC 独显版已经支持主节点转让和指定备用主节点，但目前缺少变更审查流程。在生产环境中，主节点角色的转让或备用主节点的指定/变更属于高风险操作——错误的主节点配置可能导致推理请求路由失败、丢失会话信息或破坏分布式流水线。

需要一个类似 Gerrit 的审查机制来保护这类操作。

### Gerrit 风格审查机制设计

#### 审查对象

- 主节点角色转让（当前主节点 → 另一节点）；
- 备用主节点的指定、变更或移除；
- 集群拓扑中的关键配置变更（如主节点/备用节点角色切换）。

#### 投票规则

| 投票角色 | 可投 | 说明 |
|----------|------|------|
| 管理员 | -1 / 0 / +1 | 项目管理方，通过邮箱通知审查请求 |
| 独显版软件投票 | -1 / 0 / +1 | 任何运行 PC 独显版的节点均可投票 |
| 软件版本条件 | 仅独显版节点可以投票 | PC 集显版、Android 各版本不参与此审查 |

#### 通过条件

最终得票 >= **+2** 才可通过并执行变更。

典型通过场景：

| 场景 | 管理员 | 独显版节点 A | 独显版节点 B | 合计 | 结果 |
|------|--------|-------------|-------------|------|------|
| 单管理员赞同 | +1 | 0 | 0 | +1 | 不通过 |
| 管理员 + 一个独显节点赞同 | +1 | +1 | 0 | +2 | 通过 |
| 两个独显节点赞同（无管理员） | 0 | +1 | +1 | +2 | 通过 |
| 管理员 + 独显节点各 -1 | -1 | -1 | 0 | -2 | 阻止 |

#### 审查流程

```text
1. 任一独显版节点发起转让/备用主节点变更请求
2. 系统生成审查票，序列化为 JSON 并持久化
3. 通过邮件通知管理员审查链接
4. 所有 PC 独显版节点可在管理面板看到待审票
5. 管理员和独显节点可以投 +1 / 0 / -1
6. 当合计 >= +2 时，变更通过并自动执行
7. 当合计 <= -2 时，变更被阻止并需要重新发起
8. 审查票有超时时间（如 48 小时），超时自动关闭
9. 审查票关闭后，变更不可执行，需要重新发起
```

#### 邮件通知

- 管理员邮箱在配置文件中设定，可以为空；
- 审查请求通过后端 SMTP 或系统通知发送；
- 如果浏览器前端不可访问，管理员可通过 CLI / API 投票。

### 软件版本限制

| 软件版本 | 能否发起审查 | 能否投票 |
|----------|-------------|---------|
| PC 独显版 | 可以 | 可以 |
| PC 集显版 | 不可以 | 不可以 |
| Android 普通版 | 不可以 | 不可以 |
| Android 极简版 | 不可以 | 不可以 |

### 可能涉及文件

- `src/config.py` — 管理员邮箱、审查票超时参数
- `src/scheduler.py` — 主节点转让、备用节点变更任务
- `src/api_server.py` — 审查 REST API（CRUD 审查票、投票）
- `frontend/src/components/AdminPanel.jsx` — 管理面板审查 UI
- `frontend/src/components/SettingsModal.jsx` — 集群配置变更入口
- 可新增 `src/review.py` — 审查票状态机 + 持久化
- 可新增 `src/mailer.py` — 邮件通知封装

### 验收标准

- 主节点转让需经过审查投票才能执行；
- 审查票创建后持久化，不因节点重启丢失；
- 独显版节点可在管理面板看到待审票并投票；
- 合计 >= +2 时变更自动执行，<= -2 时自动阻止；
- 超时审查票自动关闭，不阻塞后续变更；
- PC 集显版、Android 各版本不显示审查 UI 且无法投票。

---

## 优化项 11：Android 普通版与极简版签名分离

### 背景

当前 Android release APK 使用同一个 keystore（`qlh-release.jks`）签名。后续需要区分普通版和极简版两个 APK，如果它们共用同一个签名密钥：

- 两个 APK 的包名相同则无法共存安装；
- 如果包名不同但签名相同，更新时可能出现签名冲突；
- 从安全和分发角度，不同产品线应该使用独立签名。

### 推荐方案

| APK | 包名 | 签名密钥 | 说明 |
|-----|------|---------|------|
| Android 普通版 | `com.qlh.inference` | `qlh-release.jks` | 当前主线签名 |
| Android 极简版 | `com.qlh.inference.lite` | `qlh-lite-release.jks` | 独立签名，包名加 `.lite` 后缀 |

### 为什么签名应该不同

1. **包名不同 + 签名不同 = 完全独立的两个 App**：用户可以同时安装普通版和极简版，互不干扰。
2. **安全隔离**：如果极简版的签名意外泄露，不会影响普通版的 APK 签名安全。
3. **分发通道独立**：Google Play、F-Droid、直接分发的 APK 各自可以用不同签名策略。
4. **未来可能拆分权限/能力**：极简版本后续可能裁剪更多能力（如网络权限、后台服务），不同签名有助于平台区分。

### 实现步骤

1. **生成极简版 keystore**：
   ```bash
   keytool -genkey -v -keystore qlh-lite-release.jks \
     -keyalg RSA -keysize 2048 -validity 10000 \
     -alias qlh-lite-release
   ```
2. **新增 `keystore-lite.properties`**：存放极简版签名信息（同样不提交 git）。
3. **修改 Gradle flavor 配置**：为 `lite` flavor 指定独立的 `signingConfig`。
4. **Android 源码 `build.gradle.kts`** 中使用两个 `signingConfigs` 分别绑定 `full` / `lite` 构建类型。
5. **AndroidManifest** 中 `lite` flavor 的包名改为 `com.qlh.inference.lite`。

### 可能涉及文件

- `android/keystore-lite.properties` — 新增，不提交 git
- `android/qlh-lite-release.jks` — 新增，不提交 git
- `android/app/build.gradle.kts` — product flavor + signingConfig
- `android/app/src/lite/AndroidManifest.xml` — 极简版独立包名
- `android/.gitignore` — 补充 `*lite*.jks` + `keystore-lite.properties`

### 验收标准

- 普通版 APK：签名使用 `qlh-release.jks`，包名 `com.qlh.inference`
- 极简版 APK：签名使用 `qlh-lite-release.jks`，包名 `com.qlh.inference.lite`
- 同一设备上可同时安装普通版和极简版
- keystore 和 `.properties` 均被 `.gitignore` 排除

---

## 难度分级与分期路线

### 难度分级定义

| 难度 | 含义 | 特征 |
|------|------|------|
| 低 | UI/文档/单端小改 | 不改核心推理链路，风险低，验证路径短 |
| 中 | 双端体验或设置补齐 | 涉及前端/Android 状态管理，但不改调度核心 |
| 高 | 打包、日志、版本分层、多模型 | 涉及构建产物、运行模式、版本差异和回归测试 |
| 研究级 | 新调度范式或跨端 Worker | 需要协议、调度器、端侧能力和长期实验验证 |

### 分期路线

| 阶段 | 时间定位 | 优化项 | 难度 | 说明 |
|------|----------|--------|------|------|
| P0 | 立即/短期 | README 去真名、Android 键盘遮挡、PC/Android 一键复制回答 | 低 | 合规和基础交互优先，改动范围小 |
| P1 | 短期 | Android 普通版补齐常用设置、Android/PC 应用内日志查看与清理 | 中 | 提升可用性和问题排查能力 |
| P2 | 中期 | PC 打包版静默后台窗口、日志静默落盘、Android 普通版/极简版 flavor 分离 | 高 | 涉及打包、运行时、版本差异和分发说明 |
| P2.5 | 中期 | Android 普通版/极简版签名分离（独立 keystore + 不同包名） | 中 | 涉及 Gradle flavor、签名配置、.gitignore 更新 |
| P3 | 中期/长期 | PC 独显版多模型/重模型实验支持、主节点转让/备用主节点审查机制 | 高 | 只进入独显版，涉及调度器 + 审查票状态机 + 邮件通知 |
| P4 | 长期 | Android 普通版完整推理 Worker | 研究级 | Android 只接收完整任务，不接收层间拆分任务，默认关闭 |
| P5 | 远期 | 复杂问题任务级推理链拆分 | 研究级 | 多 Worker 子任务并行 + 主节点汇总，不替代 Transformer 层拆分 |

### 四种软件版本的优化边界

| 软件版本 | 近期优化重点 | 不做/暂不做 |
|----------|--------------|-------------|
| PC 集显版 | 静默日志、复制回答、基础设置稳定、轻量 GGUF/CPU 路线 | 重模型实验、Android Worker |
| PC 独显版 | 多模型实验、重模型效果对比、主节点转让审查、备用主节点管理、主节点能力增强 | Android 极简化策略、Android 签名管理 |
| Android 普通版 | 键盘适配、复制回答、设置补齐、日志管理、本地 GGUF、独立签名 | 层间拆分节点、重模型实验、投票审查 |
| Android 极简版 | 极限压缩体积和存储占用、保留轻量聊天入口 | 日志管理、Worker 接收任务、完整 models 目录、高级设置 |

---

## 建议实现顺序

1. **合规优先**：总 README 去除成员真名。
2. **输入可用性**：修 Android 键盘遮挡问题。
3. **基础交互**：增加 Android / PC 一键复制回答。
4. **Android 普通版补齐**：补 PC 端常用用户设置、日志入口。
5. **日志体验**：PC 改静默窗口 + 双端应用内日志查看/清理。
6. **产品分层**：规划 Android 普通版 / 极简版 flavor，独立签名与包名。
7. **审查机制**：PC 独显版主节点转让/备用主节点审查票 + 投票（P3 级）。
8. **实验能力**：独显 PC 版多模型/重模型实验支持。
9. **完整任务 Worker 研究**：Android 普通版完整推理 Worker 能力，极简版不参与。
10. **远期任务链编排**：复杂问题任务级推理链拆分，多 Worker 子任务并行 + 主节点汇总。

## 回归测试清单

### Android 普通版

- 打开聊天页，点击输入框，确认输入框不会被键盘遮挡；
- 连续发送多条消息，确认列表自动滚动到底部；
- 长回答生成后，点击复制按钮，粘贴到其他 App 验证内容完整；
- 全有模式和全无模式均测试一次；
- 设置页能修改常用推理参数；
- 应用内能查看、复制、清理日志；
- 远期 Worker 模式默认关闭，开启后只接收完整子任务。

### Android 极简版

- 启动路径短，不显示复杂控制面板；
- APK、运行时缓存、模型存储占用尽量压缩；
- 不要求用户管理完整 models 目录；
- 不显示日志管理入口；
- 不显示 Worker 接收任务入口，也不会接收 PC 主节点任务；
- 保留聊天核心功能。
- 使用独立签名密钥和包名，与普通版无冲突可共存安装。

### PC 集显版

- 打包版启动不显示后台控制台窗口；
- 日志静默写入本地；
- 前端能查看/复制/清理日志；
- 不显示重模型实验入口；
- 不显示审查票投票入口。

### PC 独显版

- 支持默认模型正常推理；
- 可配置实验重模型；
- 未下载实验模型时提示明确；
- 重模型加载失败不影响默认模型；
- 日志静默保存并可在应用内查看；
- 主节点转让需审查投票通过才执行；
- 只有 PC 独显版节点可参与投票；
- 合计 >= +2 通过，<= -2 阻止；
- 超时审查票自动关闭。

### README 合规

- 根 README 不再出现成员真实姓名；
- 团队职责以匿名角色或小组方式展示。
