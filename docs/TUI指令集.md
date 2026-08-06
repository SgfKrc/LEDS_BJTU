# TUI 指令集参考

> **状态**：现行
>
> 文档生命周期：**生效中（Active）**
>
> 创建日期：2026-08-05
>
> 适用范围：`src/tui_admin.py` 的 `/` 命令系统（`bjtu` 终端指令集）的完整参考——命令、别名、参数、选项、退出语义与契约测试。命令行为以源码 `COMMANDS` 注册表与 `tests/test_tui_commands.py` 为准；本文档与两者不一致时以源码和测试为准。
>
> 关联文档：[TUI 使用指南](TUI使用指南.md)（启动/参数/排障）· [TUI 适配实施计划](TUI适配实施计划.md)（7 屏契约）· [微服务架构改造计划](微服务架构改造计划.md)（2.6 验收）

---

## 一、通用规则

1. **触发**：任意界面（主菜单、7 个管理屏、`--plain` 纯文本模式）输入 `/` 开头命令后按 `Enter` 执行；命令输入中按 `ESC` 取消。
2. **命令行直调（单命令模式）**：`bjtu <命令>` 直接执行一条命令后退出、不进入交互界面（如 `bjtu shutdown`、`bjtu status`、`bjtu /shutdown`），命令名不带 `/` 也可。注意：**单命令模式不会自动启动后端**——后端未运行时提示"后端未在运行"并以退出码 1 结束；请先用 `bjtu`（交互模式）或 `start_tui.bat` 启动后端。命令必须是**第一个非选项参数**（`bjtu status --port 9000`）；选项在前（`bjtu --port 9000 status`）按交互模式处理（与启动脚本判定一致）。
3. **参数解析**：位置参数 + 选项混用。选项支持 `--key value`、`--key=value`、`--key`（布尔开关）三种写法，选项不占位置参数计数。
4. **校验反馈**：
   - 参数不足 / 过多 → 黄色 `warn` 提示 `参数不足/参数过多。用法: <usage>`；
   - 未知命令、非法取值 → 红色 `err`；
   - 成功 → 绿色 `ok`。
5. **命令不区分大小写**（命令名与多数取值会被 `lower()` 归一化）。
6. **无需进入菜单**：模型切换、量化切换、引擎切换、队列控制、优雅退出等常用操作直接输入命令即可，与菜单动作等价（对应关系见 §四）。

---

## 二、命令总表（27 条）

### 系统

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/help` | `/h` | `/help` | 命令集帮助（分组列出全部命令） |
| `/status` | `/st` | `/status` | 打开系统状态总览屏 |
| `/screen` | `/goto` | `/screen <编号\|名称>` | 跳转管理屏（`1-7` 编号或名称关键字，包含匹配） |
| `/refresh` | `/r` | `/refresh` | 立即刷新当前屏幕（仅交互模式；`--plain` 无"当前屏"概念时提示） |
| `/quit` | `/q` `/exit` | `/quit` | 退出 TUI，**后端保持运行** |
| `/shutdown` | `/halt` | `/shutdown [原因]` | 优雅退出：后端保存/清理资源后退出，TUI 随后退出；后端已停或失败时 TUI 保持不退出 |

### 模型 / 量化 / 引擎

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/model` | — | `/model` | 当前模型详情（标识/名称/量化/引擎/设备/显存/路径） |
| `/models` | — | `/models` | 可用模型列表 + 量化/引擎选项 + 当前值 |
| `/switch` | — | `/switch <模型ID> [--quant 精度] [--engine 引擎] [--compile]` | 切换模型（失败自动回滚）。默认 `--quant int4`、`--engine auto` |
| `/load` | — | `/load [模型ID] [--quant 精度] [--engine 引擎] [--compile]` | 加载模型；缺省模型 ID 用默认 Qwen |
| `/quant` | — | `/quant <int4\|int8\|fp16\|gguf>` | 切换量化精度（重载当前模型）；当前引擎为 `llama_cpp` 时自动转 `auto` 交由后端按文件类型解析 |
| `/engine` | — | `/engine <auto\|llama_cpp\|pytorch\|island>` | 切换推理引擎（重载当前模型）；量化保持当前值，可用 `--quant` 覆盖 |
| `/presets` | — | `/presets` | 预设问题与 Token/显存估算（来自后端 `/presets`） |

### 设备

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/gpu` | — | `/gpu [序号]` | 无参数：列出 GPU（`»` 标记当前）；带序号：切换推理 GPU |
| `/device` | — | `/device <auto\|profile>` | `auto`：按画像自动应用推荐配置；`profile`：查看设备画像 |

### 集群 / 队列

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/nodes` | — | `/nodes` | 节点列表与状态（角色/类型/状态/地址/心跳） |
| `/connect` | — | `/connect <IP> [端口] [--switch]` | 连接主节点（端口默认 `8888`）；本机为主节点时必须加 `--switch` 确认放弃主节点身份 |
| `/dist` | — | `/dist <on\|off\|toggle\|status>` | 分布式推理开关与状态查询 |
| `/queue` | — | `/queue [status\|strategy <fifo\|mlfq>\|pause\|resume\|clear\|cancel <任务ID>]` | 请求队列状态与控制（缺省子命令为 `status`） |

### 日志

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/logs` | — | `/logs [行数] [--remote]` | 打开日志查看屏；行数钳制到 `10-500`；`--remote` 切到后端最近日志（仅交互模式） |
| `/log` | — | `/log <filter <级别>\|token <令牌>>` | 日志级别过滤（`ERROR/WARNING/INFO/DEBUG`，空=全部）；设置/清除日志 Token（空=清除） |

### 设置

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/host` | — | `/host <主机> [端口]` | 切换后端地址（同时清除各屏缓存，下次刷新重新拉取） |
| `/interval` | — | `/interval <秒>` | 自动刷新间隔（钳制 `1-60`） |
| `/timeout` | — | `/timeout <秒>` | HTTP 请求超时（钳制 `1-120`） |
| `/token` | — | `/token <令牌>` | 设置日志访问 Token（留空清除） |

### 会话

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/chat` | — | `/chat <clear>` | 清空对话历史 |
| `/cancel` | — | `/cancel <任务ID>` | 取消生成任务（`POST /chat/generations/{id}/cancel`）；404 时回退取消工作流（`POST /workflows/{id}/cancel`），两者都失败才报错 |

---

## 三、退出语义

| 命令 | TUI 进程 | 后端进程 | 适用场景 |
|------|----------|----------|----------|
| `/quit`（`/q` `/exit`） | 退出 | **保持运行** | 仅关闭管理端，服务继续 |
| `/shutdown`（`/halt`） | 退出 | **优雅退出**（`POST /system/shutdown`，后端保存/清理资源） | 关闭整套服务；后端不可达或拒绝时 TUI 保持不退出并报错 |

> 后端退出后 TUI 若仍运行会显示"后端未启动"提示；`--plain` 模式同样支持两个退出命令。

---

## 四、与菜单操作的对应

| 菜单屏 | 对应命令 |
|--------|----------|
| 1 系统状态总览 | `/status`、`/screen 1` |
| 2 节点管理（发现/连接/注册/注销/删除/转让/备用/容量/邮件/重置） | `/nodes`、`/connect <IP> [端口] [--switch]`（其余为菜单专属动作，无单命令，见下） |
| 3 分布式与分层 | `/dist on\|off\|toggle` |
| 4 请求队列 (MLFQ) | `/queue`（`status/strategy/pause/resume/clear/cancel`） |
| 5 设备画像（GPU 切换/自动配置） | `/gpu [序号]`、`/device auto\|profile` |
| 6 日志查看（本地尾部/后端最近/文件列表/统计） | `/logs [行数] [--remote]`、`/log filter <级别>` |
| 7 设置（后端地址/间隔/超时/Token/连通测试） | `/host <主机> [端口]`、`/interval <秒>`、`/timeout <秒>`、`/token <令牌>` |

> 菜单专属动作（无等价命令）：屏 2 的自动发现、转让日志、注册/注销/删除节点、转让主节点、备用主节点设置、邮件告警测试、重置身份；屏 6 的连通测试（`T`）。这些仍走菜单操作。

---

## 五、契约与测试

- **命令系统单元测试**：`tests/test_tui_commands.py`（52 用例，2026-08-05 复核后全覆盖 27 条命令）——命令解析、参数校验（不足/过多/非法值）、选项解析（`--quant/--engine/--compile/--switch`）、请求构造（body 逐字段）、退出标志（`/quit` vs `/shutdown`）、后端不可达容错。新增/修改命令必须同步该文件。
- **7 屏 × 2 角色走查**：`scripts/tui_walkthrough.py`（`--real` 为真实微服务拓扑模式，master 16 动作 + client 3 屏；默认模式为桩环境全动作走查）。
- **契约来源**：`src/tui_admin.py` `COMMANDS` 注册表（1946 行起）是命令的唯一事实来源；`/help` 与 `bjtu --help` 输出由 `_build_command_help_lines()` 自动生成，与本文档总表一致。

---

## 六、维护说明

新增命令的流程：

1. 在 `src/tui_admin.py` 的 `COMMANDS` 注册表追加条目（name/aliases/usage/summary/handler/min_args/max_args），handler 签名 `(app, args, opts)`；
2. 若属新分组，同步更新 `_build_command_help_lines()` 的 `groups`；
3. 在 `tests/test_tui_commands.py` 补用例（解析、校验、请求构造、边界）；
4. 更新本文档总表与分组表。

---

**维护者**：QLH 开发团队
**下次复核触发**：`src/tui_admin.py` `COMMANDS` 注册表或 `tests/test_tui_commands.py` 变更时
