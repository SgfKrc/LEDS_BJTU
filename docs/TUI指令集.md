# TUI 指令集参考

> **状态**：已归档（Archived，2026-09-17）
>
> 文档生命周期：**已归档（Archived）**——所述实现已移出主仓
>
> 创建日期：2026-08-05
>
> 适用范围（**历史**）：`src/tui_admin.py` 的 `/` 命令系统（35 条）参考——命令、别名、参数、选项、退出语义与契约测试。**该实现已于 2026-09-17 归档到 `_to_delete/`**（连同 `COMMANDS` 注册表与 `tests/test_tui_commands.py`），本文档仅作历史记录保留。
>
> **现行替代**：交互外壳 `src/tui_textual.py`（Textual，9 个功能屏 + 1 个调试兜底屏：聊天/状态/模型/分布式/节点/队列/日志/设备/设置/调试；聊天内命令由 `src/tui_shared.py` 的 `COMMAND_SPECS` 生成（`/help` 与实现同源）：`/help`、`/model`、`/queue`、`/new`、`/resume`、`/rename`、`/sessions`、`/delete-session`、`/reset`、`/route`、`/thinking`、`/cancel`、`/clear`、`/quit`；模型/集群/节点/日志/设备/设置的稳定写操作由屏内按键提供，全部先经确认框）；单命令面 `src/tui_commands.py`（只读：`status`/`models`/`nodes`/`queue`/`device`/`logs`/`help`）。调试屏不计作产品功能覆盖。见 [TUI 使用指南](TUI使用指南.md)。
>

---

## 一、通用规则

1. **触发**：任意界面（主菜单、8 个屏幕、`--plain` 纯文本模式）输入 `/` 开头命令后按 `Enter` 执行；命令输入中按 `ESC` 取消。
2. **命令行直调（单命令模式）**：`bjtu <命令>` 直接执行一条命令后退出、不进入交互界面（如 `bjtu shutdown`、`bjtu status`、`bjtu /shutdown`），命令名不带 `/` 也可。注意：**单命令模式不会自动启动后端**——后端未运行时提示"后端未在运行"并以退出码 1 结束；请先用 `bjtu`（交互模式）或 `start_tui.bat` 启动后端。命令必须是**第一个非选项参数**（`bjtu status --port 9000`）；选项在前（`bjtu --port 9000 status`）按交互模式处理（与启动脚本判定一致）。
3. **参数解析**：位置参数 + 选项混用。选项支持 `--key value`、`--key=value`、`--key`（布尔开关）三种写法，选项不占位置参数计数。
4. **校验反馈**：
   - 参数不足 / 过多 → 黄色 `warn` 提示 `参数不足/参数过多。用法: <usage>`；
   - 未知命令、非法取值 → 红色 `err`；
   - 成功 → 绿色 `ok`。
5. **命令不区分大小写**（命令名与多数取值会被 `lower()` 归一化）。
6. **无需进入菜单**：模型切换、量化切换、引擎切换、队列控制、优雅退出等常用操作直接输入命令即可，与菜单动作等价（对应关系见 §四）。

---

## 二、命令总表（35 条）

### 系统

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/help` | `/h` | `/help` | 命令集帮助（分组列出全部命令） |
| `/status` | `/st` | `/status` | 打开系统状态总览屏 |
| `/screen` | `/goto` | `/screen <编号\|名称>` | 跳转屏幕（`1-8` 编号或名称关键字，包含匹配） |
| `/refresh` | `/r` | `/refresh` | 立即刷新当前屏幕（仅交互模式；`--plain` 无"当前屏"概念时提示） |
| `/quit` | `/q` `/exit` | `/quit` | 退出 TUI，**后端保持运行** |
| `/shutdown` | `/halt` | `/shutdown [原因]` | 优雅退出：后端保存/清理资源后退出，TUI 随后退出；后端已停或失败时 TUI 保持不退出 |

### 模型 / 量化 / 引擎

| 命令 | 别名 | 用法 | 说明 |
|------|------|------|------|
| `/model` | — | `/model` | 当前模型详情（标识/名称/量化/引擎/设备/显存/路径） |
| `/model fleet` | — | `/model fleet` | 模型舰队列表、可用格式/引擎、设备档位与当前模型 |
| `/model select` | — | `/model select <模型ID> [--quant 精度] [--engine 引擎]` | 按模型 ID 选择并切换模型 |
| `/models` | — | `/models` | `/model fleet` 的短命令 |
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
| `/chat` | — | `/chat <clear\|open>` | 打开聊天屏或清空对话历史 |
| `/new` | — | `/new` | 创建并切换新会话 |
| `/sessions` | — | `/sessions` | 列出最近会话 |
| `/resume` | — | `/resume <session_id>` | 恢复历史会话 |
| `/rename` | — | `/rename <标题>` | 重命名当前会话 |
| `/delete-session` | — | `/delete-session` | 删除当前会话 |
| `/route` | — | `/route <auto\|local\|distributed\|required>` | 设置聊天请求路由偏好 |
| `/thinking` | — | `/thinking <on\|off>` | 控制 thinking 内容展示 |
| `/cancel` | — | `/cancel [任务ID]` | 取消聊天生成；带 ID 时兼容取消工作流 |

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
| 2 节点管理（发现/连接/注册/注销/删除/转让/备用/容量/重置） | `/nodes`、`/connect <IP> [端口] [--switch]`（其余为菜单专属动作，无单命令，见下） |
| 3 分布式与分层 | `/dist on\|off\|toggle` |
| 4 请求队列 (MLFQ) | `/queue`（`status/strategy/pause/resume/clear/cancel`） |
| 5 设备画像（GPU 切换/自动配置） | `/gpu [序号]`、`/device auto\|profile` |
| 6 日志查看（本地尾部/后端最近/文件列表/统计） | `/logs [行数] [--remote]`、`/log filter <级别>` |
| 7 设置（后端地址/间隔/超时/Token/连通测试） | `/host <主机> [端口]`、`/interval <秒>`、`/timeout <秒>`、`/token <令牌>` |
| 8 聊天 | `/chat open`、`/new`、`/sessions`、`/resume`、`/rename`、`/delete-session`、`/route`、`/thinking`、`/cancel` |

> 菜单专属动作（无等价命令）：屏 2 的自动发现、转让日志、注册/注销/删除节点、转让主节点、备用主节点设置、重置身份；屏 6 的连通测试（`T`）。这些仍走菜单操作。

---

## 五、契约与测试

- **命令系统单元测试**：`tests/test_tui_commands.py` 与聊天/CLI/启动测试覆盖当前 35 条命令及统一入口；本轮定向回归 **173 passed**。新增/修改命令必须同步源码 `COMMANDS`、测试与本文档。
- **8 屏走查**：既有 7 个管理屏继续沿用 `scripts/tui_walkthrough.py`；聊天屏与统一入口由 `tests/test_tui_chat_screen.py`、`tests/test_qlh_cli.py` 和 backend supervisor 测试覆盖。
- **契约来源**：交互外壳的命令**唯一事实来源**是 `src/tui_shared.py` 的 `COMMAND_SPECS`（`/help` 由它生成，未实现的命令不登记——曾出现过 `/image*` 写着却不能用）。`src/tui_admin.py` 已归档到 `_to_delete/`，故本文档第二节的 35 条总表是**已归档旧 TUI 的历史参考**，不代表现行能力。

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
