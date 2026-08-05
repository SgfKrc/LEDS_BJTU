# TUI 技术 Q&A

> 文档生命周期：**现行（Active）**
>
> 创建日期：2026-08-05
>
> 适用范围：QLH 终端版管理菜单（`src/tui_admin.py`，约 2600 行）的技术栈、架构与实现机制问答——回答"用了什么技术、功能都是怎么实现的"。使用/操作口径见 [TUI使用指南](TUI使用指南.md)，命令语义见 [TUI指令集](TUI指令集.md)，网关注约与验收见 [TUI适配实施计划](TUI适配实施计划.md)；行为以源码为最终依据。
>
> 关联文档：[TUI使用指南](TUI使用指南.md) · [TUI指令集](TUI指令集.md) · [TUI适配实施计划](TUI适配实施计划.md)

---

## Q1：TUI 用了什么技术栈？有第三方依赖吗？

**纯 Python 标准库，零第三方依赖**——这是刻意的设计约束（`tui_admin.py` 文件头注释即声明）。全部 import 只有：

- 通用：`argparse`、`json`、`os`、`shutil`、`sys`、`time`、`unicodedata`、`urllib.*`（`request`/`parse`/`error`）
- Windows 条件导入：`ctypes`（开启 VT 模式）、`msvcrt`（按键）
- Unix 条件导入：`select`、`termios`、`tty`（原始模式与按键）

这样做的好处：任何装了 Python 3.10+ 的机器（Windows/Linux/macOS/Android Termux）都能直接跑，不需要 `pip install`、不需要虚拟环境，天然适配"边缘设备随手管理"的定位。HTTP 客户端用 `urllib.request` 自封装（见 Q9），终端控制全部手写 ANSI/VT 转义（见 Q3/Q4）。

## Q2：TUI 和前端（React）是什么关系？

两者是同一套后端 REST API 的**两种前端**：

| | React 前端 | TUI |
|---|---|---|
| 位置 | `frontend/` | `src/tui_admin.py` |
| 依赖 | Node 生态 | 纯标准库 |
| 适用 | 浏览器、可视化 | SSH/终端、无头服务器、低配边缘设备 |

TUI 通过 `http://host:port/api/*` 与后端 `src/api_server.py`（FastAPI）通信，`--host` 可指向 Tailscale 远程主节点，因此也可以在**没有浏览器**的机器上做完整管理。TUI 本身不含任何推理逻辑，所有状态变更（模型切换、量化、分布式开关、队列控制）都是对后端的 REST 调用，与 React 前端完全等价。

## Q3：跨平台终端控制是怎么实现的？

核心是 `AnsiTerm` 类，分三层：

1. **能力探测**（`__enter__`）：校验 stdin/stdout 是交互终端（`isatty`），不满足直接抛 `TermNotCapable`，上层自动降级纯文本模式（见 Q6）。
2. **进入/退出原始模式**：
   - Windows：`ctypes` 调 `SetConsoleMode` 开 `ENABLE_VIRTUAL_TERMINAL_PROCESSING`（让 cmd/PowerShell 支持 ANSI 转义），无需安装任何包；
   - Unix：保存 `termios` 配置后用 `tty.setcbreak` 关闭行缓冲/回显。
   - 进入时写 `\x1b[?1049h`（备用屏幕）、`\x1b[?25l`（隐藏光标）、清屏；退出时反向恢复，保证 Ctrl+C 或异常后终端不留残影。
3. **渲染原语**：`paint()` 把内部样式（`title/ok/err/warn/dim/sel/head/key/input/cmd`）映射成 SGR 转义码；`size()` 用 `shutil.get_terminal_size` 取窗口尺寸（下限 40×10 防除零）；`write()` 统一输出。

## Q4：界面是怎么画出来的？中文对齐为什么不出乱？

**整屏重绘策略**：每次渲染把整个界面（标题、菜单/屏幕内容、底部消息或命令输入行、快捷键提示）拼成一个字符串，`\x1b[H` 光标归位后一次写入，每行末尾 `\x1b[K` 清行尾。比增量刷新简单可靠，60 行以内界面性能无感。

**CJK 对齐是重点**：终端里中文占 2 列、emoji 占 2 列、零宽字符占 0 列，直接 `len()` 会对不齐。因此：

- `char_width()` / `disp_width()`：按 Unicode East Asian Width 计算每个字符的显示宽度；
- `truncate_display()`：按**显示宽度**截断加 `…`，不会把中文切一半；
- `pad_display()`：按显示宽度填充/居中；
- `make_table()`：CJK 安全的表格对齐，超宽时自动削减最宽的列。

屏幕内容（`lines(width)`）返回 `(样式, 文本)` 列表，`_norm_lines()` 统一规范化（兼容纯字符串），渲染层按样式上色、按宽度换行。

## Q5：按键是怎么读的？方向键/功能键怎么区分？

`AnsiTerm.get_key(timeout)` 分平台实现，统一返回**语义化按键名**（`"UP"/"DOWN"/"ENTER"/"ESC"/"PGUP"/单字符/中文…`）：

- **Windows**：`msvcrt.kbhit()` 轮询 + `getwch()` 读字符；`\x00`/`\xe0` 前缀表示功能键，查表映射方向键/翻页/Home/End；`\x03` 转 `KeyboardInterrupt`。
- **Unix**：`select` 等超时后 `os.read` 批量读；`_parse_input_bytes()` 解析 ESC 转义序列（`\x1b[A`→UP、`\x1b[5~`→PGUP 等，查 `_ESC_MAP`），结尾是 ESC 时额外等 20ms 防止序列被拆断；同时支持 UTF-8 多字节（防止用户误输入中文把解析器搞乱）。

主循环 `run()` 以 0.3 秒超时轮询：无按键继续刷新，EOF（终端关闭）和 Ctrl+C 都安全退出。

## Q6：三种界面模式（ANSI / 纯文本 / 降级）是什么关系？

三层类结构，单一职责：

| 类 | 职责 |
|---|---|
| `BaseApp` | 公共状态（api 客户端、7 个屏幕实例、`exit_requested`/`shutdown_backend` 退出标志）与命令系统（`exec_command`）；不含界面逻辑 |
| `InteractiveApp` | ANSI 全屏交互：主菜单 ↑↓/数字/Enter 导航、`/` 进入命令输入、`q` 优雅退出、`Esc` 仅退出；屏幕内滚动/动作键 |
| `PlainApp` | 纯文本编号菜单：`input()` 逐行交互，适配管道、SSH 无 ANSI、脚本化操作（`--plain` 强制） |

**自动降级链**：`main()` 先尝试 `InteractiveApp`，若终端不支持（非交互终端、TermNotCapable）打印提示后自动切 `PlainApp`——所以 `echo q | python src/tui_admin.py` 也能完整工作（这在脚本化/CI 里很有用）。

交互动作统一抽象为 `BaseUI/TermUI/PlainUI`：屏幕动作处理器（如"连接主节点"）只面向 UI 接口编程，全屏模式弹输入框、纯文本模式走 `input()`。

## Q7：7 个管理屏幕是怎么组织的？

统一走 `Screen` 基类契约，`SCREEN_CLASSES` 注册：

- `refresh(force)`：按 `auto_refresh` + 刷新间隔节流；**所有请求异常都被捕获写入 `self.error`**，屏幕显示"[错误]"而不崩溃——后端挂掉时 TUI 依然可用；
- `fetch()`：拉数据（抽象方法）；
- `lines(width)`：把数据渲染成 `[(样式, 文本)]`；
- `actions()`：返回该屏支持的按键动作 `[(键, 标签, 处理器)]`，交互模式动态渲染成快捷键提示。

7 个屏幕：① 系统状态总览（自动刷新，health/status/当前模型/角色）② 节点管理（12 个动作：连接、注册、注销、转让主节点、备用主节点、最大节点数、自动发现、转让日志、邮件测试、重置身份等）③ 分布式与分层（开关、分层覆盖）④ 请求队列 MLFQ（策略/暂停/恢复/清空/取消）⑤ 设备画像（GPU 列表/选择、自动配置）⑥ 日志查看（本地文件尾部 / 后端 recent / files / stats 四种模式，带日志 Token 鉴权）⑦ 设置（host/port/timeout/interval/token，纯本地状态，不请求后端）。

## Q8：`/` 命令系统是怎么实现的？

**注册表驱动**，`COMMANDS` 是唯一事实来源（27 条命令）：

- 每条命令一个 dict：`name / aliases / usage / summary / handler / min_args / max_args`；
- 解析流程：`exec_command()` 要求以 `/` 开头 → 匹配主名或别名（不区分大小写）→ `_split_cmd_args()` 把参数拆成位置参数 + 选项（`--key value` / `--key=value` / 裸 `--key`=True）→ 按 `min_args`/`max_args` 校验（不足/过多返回 `warn` 提示用法）→ 调用 handler 返回 `(消息, 样式)`；
- 帮助自动生成：`_build_command_help_lines()` 按 7 个分组从注册表生成帮助文本，`/help` 与 `bjtu --help`（argparse epilog）共用，**加新命令不用改帮助代码**；
- 命令不区分大小写，`/switch` 的 `--quant`/`--engine`/`--compile` 选项统一由 `_do_model_change()` 构造请求体。

## Q9：和后端怎么通信？`ApiClient` 做了什么？

`ApiClient` 用 `urllib.request` 手写极简 REST 客户端：

- 请求拼 `http://host:port/api<path>`，JSON body、超时、HTTP 错误、网络错误**全部收敛为 `ApiError`**（带 status），上层只捕获一种异常；
- 日志相关端点带 `X-QLH-Log-Token` 头鉴权（`--log-token` 参数）；
- 命令→端点映射（示例）：`/switch` `/quant` `/engine` → `POST /api/models/switch`；`/load` → `POST /api/models/load`；`/model` → `GET /api/models/current`；`/dist` → `GET|PUT /api/cluster/config/distributed-inference`；`/queue` → `GET /api/cluster/queue` + `POST .../strategy|pause|resume|clear` + `DELETE .../task/{id}`；`/nodes` → `GET /api/cluster/nodes`；`/connect` → `POST /api/cluster/connect`；`/gpu` `/device` → `GET /api/device/profile` + `POST /api/device/select-gpu`；`/logs` → `GET /api/logs/recent|stats`；`/shutdown` → `POST /api/system/shutdown`；`/chat` → `POST /api/chat/clear`；`/cancel` → `POST /api/chat/generations/{id}/cancel`（404 回退工作流取消）。

## Q10：优雅退出是怎么实现的？q / Esc / /quit / /shutdown 有什么区别？

后端提供 `POST /api/system/shutdown`，收到后**保存状态、清理资源再退出进程**（而不是杀进程）。TUI 侧四个出口语义不同：

| 出口 | 行为 |
|---|---|
| `q`（主菜单） | 等同 `/shutdown`：请求后端优雅退出，成功则 TUI 随后端一起结束；**请求失败也退出 TUI**（打印原因，后端保持） |
| `Esc`（主菜单） | 仅退出界面，后端保持运行（常驻模式） |
| `/quit` | 同 Esc，仅退界面 |
| `/shutdown` | 请求优雅退出；失败时 TUI **不退出**，可继续操作 |

实现要点：`cmd_shutdown` 成功时置 `exit_requested` + `shutdown_backend` 两个标志；`main()` 退出语根据标志区分"后端已优雅退出"与"TUI 已退出（后端保持运行）"；q 的失败原因通过 `exit_message` 带给 `main()` 打印，避免全屏模式下消息丢失。

## Q11：`bjtu shutdown` 这种"单命令模式"是怎么实现的？

交互模式之外，TUI 支持**命令行直调**：`bjtu shutdown` / `bjtu status` / `bjtu /shutdown` 执行一条命令后直接退出，不进入界面。链路：

1. `start_tui.bat` / `start_tui.sh` 判断首参：以 `/` 开头或命中内置命令名/别名清单（与 `COMMANDS` 注册表同步，有测试守护）→ **跳过"启动后端"流程**，直接把参数透传给 `tui_admin.py`；
2. `tui_admin.py`：argparse 增加可选位置参数 `命令`；`_is_single_command()` 校验"命令必须是第一个非选项参数"（选项在前按交互模式处理，与脚本判定对齐）；
3. `run_single_command()`：先探测 `/api/health`，**后端未运行则报错退出（退出码 1），绝不自动启动后端**——单命令模式只做"处理"不负责"启动"；后端在线则构造 `BaseApp` 执行命令并打印结果；
4. 退出码语义：命令成功（`ok`）= 0；未知命令/参数错误/后端未运行 = 1，方便脚本判断。

## Q12：中文输出在 Windows 上为什么不会乱码/崩溃？

Windows 上有两个经典坑，都有对应处理：

1. **bat 脚本中文注释会被 cmd 按 GBK 字节扫描而错位**（曾导致 `'src' is not recognized` 这类诡异报错）——`start_tui.bat` 的 `rem` 注释全部使用英文，`echo` 中文保留（原样字节输出 + `chcp 65001` 显示正常）；
2. **Python 输出到管道时默认用 GBK**，`»`、emoji 等字符直接 `UnicodeEncodeError` 崩溃——`_force_utf8_stdout()` 在三条路径（单命令模式、`--plain` 模式、TermNotCapable 降级）统一 `sys.stdout.reconfigure(encoding="utf-8")`；`start_tui.bat` 首行 `chcp 65001` 保证控制台按 UTF-8 显示。

## Q13：一键启动脚本（start_tui.bat / start_tui.sh）做了什么？

`bjtu` 全局命令 → 启动脚本，完成"后端生命周期管理"：

1. 探测 `BACKEND_PORT`（默认 8000，`QLH_BACKEND_PORT` 可覆盖）是否已监听；
2. 未运行则拉起后端：Windows 新开"QLH 后端 API"窗口运行 `python src/api_server.py`（特意不用 `uvicorn -m`，保证 `/api/system/shutdown` 能触发跨平台优雅退出）；Linux/macOS 用 `nohup` 后台运行，PID 写入 `logs/backend_tui.pid`；
3. 轮询 `/api/health` 直至就绪（上限 120 秒，失败给排查提示）；
4. 进入 TUI。退出 TUI 后后端继续运行（常驻服务模式）。

## Q14：TUI 是怎么测试的？质量怎么保证？

`tests/test_tui_commands.py`（73 项）是**契约测试**，不依赖真实后端与终端：

- `FakeApi` 桩：记录每次调用（路径/参数/body）并按路径返回预设响应，可断言"切换模型时请求体包含 `model_id=... quant_type=... engine=...`"这类精确行为；
- `DownApi` 桩：模拟后端不可达（GET/POST 都抛 `ApiError`），覆盖失败路径；
- 测试类按功能划分：命令解析、退出语义（q/ESC//quit//shutdown）、模型命令、设备/集群、设置、屏幕导航、日志、**单命令模式**（退出码语义）、**启动脚本清单与 `COMMANDS` 注册表同步**（防新增命令后脚本忘记更新）、**退出语义矩阵**；
- 端到端验证（手工/脚本）：真实后端 + 管道输入模拟按键，验证 q 优雅退出后 8000 端口释放、后端未运行时不启动等。

## Q15：有哪些已知边界与限制？

- **ANSI 全屏模式需要真正的交互终端**：非终端（管道/重定向/CI）自动降级纯文本，功能等价但交互方式不同；
- **单命令模式不启动后端**：`bjtu shutdown` 等要求后端已运行，未运行报错退出（有意设计，防止"想关后端反而拉起后端"）；
- **远程管理**：`--host` 指向远程节点时，q//shutdown 关闭的是**远程**后端，注意操作对象；
- **TUI 无本地推理能力**：所有操作都是后端 API 的"遥控器"，后端不可达时只能查看错误与退出；
- **日志鉴权**：远程查看日志需 `--log-token`，否则接口 401；
- **性能边界**：整屏重绘策略在极窄/极大窗口下以窗口尺寸为界，表格超宽自动削减列。
