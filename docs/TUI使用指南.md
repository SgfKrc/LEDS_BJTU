# TUI 管理菜单使用指南

> **状态**：现行
>
> **生命周期**：Active（在用）
>
> **更新日期**：2026-08-03
>
> **适用范围**：QLH 终端版管理菜单（`src/tui_admin.py`）与全局 `bjtu` 命令的启动方式、参数与使用说明；实现细节与网关契约见 [TUI 适配实施计划](TUI适配实施计划.md)，功能口径以源码与本文为准

---

## 一、这是什么

`src/tui_admin.py`（纯 Python 标准库，零第三方依赖）是 QLH 的终端管理面，与 Web 管理面板功能对应，覆盖 7 个屏幕：系统总览、节点管理、分布式与分层、请求队列、设备画像、日志、设置。适用于无浏览器环境（SSH、服务器、树莓派等），支持 Windows 10+ / Linux / macOS。

TUI 是**纯 HTTP 客户端**：所有数据来自后端 API（默认 `http://127.0.0.1:8000/api`）。后端未运行时 TUI 无法工作，因此一键启动脚本会先确保后端就绪再进入 TUI。

## 二、全局 `bjtu` 命令（推荐）

把一键启动封装为全局命令 `bjtu`：**在任何目录的终端输入 `bjtu` 即可自动启动后端 + TUI**（无需先进入项目目录）。

### 安装（一次性）

**Windows**：把项目根目录加入 `PATH`：

```bat
:: cmd 或 PowerShell 执行（当前用户，需重新打开终端生效）
setx PATH "%PATH%;G:\C\PYT\qlh"
```

> 或用图形界面：系统属性 → 环境变量 → 用户变量 `Path` → 新建 → 填 QLH 项目根目录（`bjtu.bat` 所在目录）。
> ⚠️ `setx` 会把整个 PATH 写回，若原 PATH 很长有截断风险，优先用图形界面方式。

**Linux / macOS**（推荐符号链接到 PATH 目录）：

```bash
sudo ln -s /path/to/qlh/bjtu.sh /usr/local/bin/bjtu
```

### 用法

```bash
bjtu                                # 启动后端 + TUI（等价 start_tui 一键启动）
bjtu --host 100.x.x.x               # 启动后端后，TUI 管理远程主节点
bjtu --plain                        # 纯文本编号菜单
```

行为与 `start_tui.bat` / `start_tui.sh` 完全相同：探测 `8000` 端口（`QLH_BACKEND_PORT` 可覆盖）→ 未运行则启动后端 → 等待 `/api/health` 就绪 → 进入 TUI；退出 TUI 后后端继续运行（Windows 关闭"QLH 后端 API"窗口停止；Linux/macOS `kill "$(cat logs/backend_tui.pid)"`）。

## 三、一键启动（start_tui.bat / start_tui.sh）

> 一键启动 = 自动检查后端 → 未运行则启动后端 → 等待就绪 → 进入 TUI。

### Windows

双击 `start_tui.bat`，或在 cmd/PowerShell 中执行：

```bat
start_tui.bat
```

启动过程：

1. 检测 `8000` 端口是否已有后端在运行（`QLH_BACKEND_PORT` 可覆盖，见 §四）。
2. 未运行则**新开一个"QLH 后端 API"窗口**启动 `python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000`（若存在 `.venv` 会自动激活）。
3. 轮询 `/api/health` 直至就绪（上限 120 秒）。
4. 在当前窗口进入 TUI，原样透传命令行参数。

退出 TUI 后**后端继续运行**（适合作为常驻服务）。停止后端：关闭"QLH 后端 API"窗口，或在该窗口按 `Ctrl+C`。

### Linux / macOS

```bash
./start_tui.sh              # 需要可执行权限：chmod +x start_tui.sh
```

启动过程与 Windows 相同，区别：

- 后端以 `nohup` **后台运行**，日志写入 `logs/backend_tui.log`，PID 存于 `logs/backend_tui.pid`；
- 退出 TUI 后后端继续运行，停止：

```bash
kill "$(cat logs/backend_tui.pid)"
```

> 提示：若 `logs/backend_tui.log` 里出现启动失败，通常是端口占用、Python 环境或依赖缺失，见 §六排查。

## 四、手动启动（高级/排障用）

不依赖一键脚本，先启动后端，再启动 TUI：

```bash
# 1. 启动后端（项目根目录）
python src/api_server.py            # 或 python -m uvicorn src.api_server:app --port 8000

# 2. 另开终端启动 TUI
python src/tui_admin.py             # 连本机 8000
```

## 五、参数说明

### 一键启动脚本

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `QLH_BACKEND_PORT` | `8000` | 后端端口。改动后 TUI 需用 `--port` 指向同一端口，例如 `QLH_BACKEND_PORT=8100 start_tui.bat --port 8100` |

### `tui_admin.py`（也是 `bjtu` 命令透传的参数）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` | `127.0.0.1` | 后端地址；填 Tailscale IP（如 `100.x.x.x`）可直管远程主节点 |
| `--port` | `8000` | 后端端口 |
| `--plain` | 关 | 强制纯文本编号菜单（管道/CI/老终端自动降级，也可手动指定） |
| `--interval` | `3.0` | 总览/节点/队列屏自动刷新秒数 |
| `--timeout` | `5.0` | HTTP 请求超时秒数 |
| `--log-token` | 空 | 远程访问日志接口所需的 `X-QLH-Log-Token`（本地模式直接读 `logs/` 目录，无需 token） |
| `--no-color` | 关 | 关闭彩色输出 |
| `--version` | — | 打印版本（当前 `1.0.0`） |

典型用法：

```bash
python src/tui_admin.py --host 100.x.x.x           # 管理远程 Tailscale 主节点
python src/tui_admin.py --host 100.x.x.x --log-token xxx   # 远程模式带日志 token
python src/tui_admin.py --plain                    # 纯文本编号菜单
```

## 六、常见问题

| 现象 | 原因与处理 |
|------|-----------|
| 一键启动 120 秒未就绪 | 查看后端窗口日志（Windows）或 `logs/backend_tui.log`（Linux/macOS）。多为端口被占用（改 `QLH_BACKEND_PORT`）、Python 环境缺依赖、`.env`/数据库配置不可达 |
| TUI 显示"后端未启动"提示 | 后端未运行或地址不对：确认 `start_tui.bat` / `start_tui.sh` 已跑完后端启动步骤，或用 `python src/api_server.py` 手动起后端 |
| TUI 报"内部错误" | 多为网关/后端版本与 TUI 契约不一致（字段缺失或类型错误）。契约测试见 `gateway/test/tui-contract.e2e-spec.ts`；排障见 [TUI 适配实施计划](TUI适配实施计划.md) §7 |
| 中文乱码 | Windows：脚本已自动 `chcp 65001`，直接双击即可；若手动启动请先执行 `chcp 65001`。Linux/macOS：确认终端使用 UTF-8 |
| 远程模式日志打不开 | 远程日志需 `--log-token`（未配置 token 时后端也允许放行）；本地模式（TUI 与后端同机）不走 HTTP，直接读 `logs/` 目录 |

## 七、自动化走查与测试

- **契约测试**：`cd gateway && npm run test:tui`（44 用例：38 端点调用点 + 5 项细节 + 错误契约）。
- **7 屏 × 2 角色走查**：`scripts/tui_walkthrough.py --host <网关> --port <端口> --mode master|client`，配套桩 `scripts/dev_stubs.py`（scheduler-svc :8020 + inference-svc :8010，`--client-mode` 模拟从节点身份）与 `src/legacy_control.py`（:8040，`/logs/*`）。
- 2026-08-03 复核：`tui-contract` 44/44、master/client 双角色走查全部 PASS，`tui_admin.py` 自 TUI 适配完成后零改动。

---

**维护者**：QLH 开发团队
**下次复核触发**：`tui_admin.py` 或网关契约发生变更时
