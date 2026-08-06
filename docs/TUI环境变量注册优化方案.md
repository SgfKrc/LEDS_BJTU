# TUI 环境变量注册优化方案

> **状态**：规划（调研完成，尚未实施）
>
> **文档生命周期**：Candidate（L4-TUI 相关，随打包版落地后转 Active）
>
> **调研日期**：2026-08-06
>
> **更新日期**：2026-08-06
>
> **适用范围**：Windows Inno Setup 与 Linux .deb 安装包的环境变量注册能力；`bjtu` 全局命令、`start_tui.bat`/`start_tui.sh` 一键启动、`bjtu chat` T9 聊天页的可选环境变量（`QLH_BACKEND_PORT`、`QLH_CLUSTER_SECRET`、`QLH_MODEL_STORE`、`QLH_SQLITE_PATH` 等）
>
> **关联文档**：[TUI 使用指南](TUI使用指南.md)、[TUI 适配与聊天页实施计划](TUI适配实施计划.md)、[打包说明](../packaging/README.md)

---

## 1. 现状与问题

### 1.1 现状

- `bjtu.bat` / `bjtu.sh` 依赖项目根目录在 `PATH` 中；README 当前引导用户手动 `setx PATH "%PATH%;<项目根>"`（Windows）或 `ln -s <项目根>/bjtu.sh /usr/local/bin/bjtu`（Linux/macOS）。
- 打包版（Inno Setup `.exe` / `.deb`）安装后**不会**注册任何环境变量——用户在新终端里输入 `bjtu` 会提示找不到命令，必须再手动注册。
- 各可选环境变量（端口、集群密钥、模型存储路径等）目前都依赖用户在 `.env` / 系统设置里手工配置。

### 1.2 问题

1. 安装包体验不完整：安装完成后"全局 `bjtu` 命令"这一承诺需要额外手工步骤才能兑现。
2. 手工 `setx` 存在覆盖风险（若用户粘贴错 `%PATH%` 会截断原有路径），且不区分用户/系统级。
3. Linux 用户不一定知道 `profile.d` 与 `/etc/environment` 的区别，容易写进错误文件。
4. 卸载后残留的 PATH 条目与失效的符号链接会长期污染环境。

---

## 2. 目标

1. 打包安装时**可选**地把 QLH 相关环境变量注册进系统变量或用户变量（Windows 注册表 / Linux profile.d）；注册与否由安装向导/参数显式选择，**不是默认强制行为**。
2. 注册完成后，新开终端即可直接使用全局 `bjtu`（含 `bjtu chat`），无需手工 `setx` / `ln -s`。
3. 卸载时只移除本包写入的条目，不触碰用户原有环境变量。
4. 源码模式（非打包）维持现状（README 手动指引），不受影响。

> 说明：将"环境变量写入注册表/环境文件"定位为**可选功能**，是因为修改用户/系统环境变量属于副作用较大的操作（影响所有后续进程），应当由用户在安装时明确知情并勾选，而不是安装器默认替用户决定。

---

## 3. 方案设计

### 3.1 Windows（Inno Setup）

Inno Setup 的 `[Registry]` 段直接支持写注册表；用户级环境变量位于 `HKCU\Environment`，无需管理员权限。

- **注册项**（`[Registry]`）：

  | 值 | 写入目标 | 说明 |
  |---|---|---|
  | `PATH` | `HKCU\Environment` | 追加项目安装目录（合并现有值，去重，长度上限内） |
  | `QLH_BACKEND_PORT` | `HKCU\Environment` | 可选：默认后端端口（如 8000） |
  | `QLH_MODEL_STORE` / `QLH_SQLITE_PATH` | `HKCU\Environment` | 可选：数据目录（默认留空走安装目录） |

- **可选开关**：安装向导新增"注册全局 `bjtu` 命令（写入用户环境变量）"复选框（默认**勾选**但可取消；提供静默参数 `/ENVREG=0|1`）。仅当勾选时才执行 `[Registry]` 写入。
- **卸载清理**：`[UninstallDelete]` 或卸载脚本按"仅删除本安装包添加的 PATH 段 + 本包设置的值"执行；不整体改写 `PATH`。
- **即时生效**：安装完成后广播 `WM_SETTINGCHANGE`（Inno Setup `[Run]` 调用或 Pascal 脚本 `SendMessageTimeout`），提示用户重开终端。
- **系统级（可选）**：若用户选择"所有用户"安装且具备管理员权限，可写 `HKLM\Environment`；否则回落用户级并提示。

> 注意：Windows 用户级环境变量的 `PATH` 是 `REG_EXPAND_SZ`，最大约 32 KB；本包只追加一段短路径，冲突风险低。

### 3.2 Linux（.deb）

- **注册文件**：`postinst` 写入 `/etc/profile.d/qlh.sh`：

  ```sh
  # 由 QLH 安装包生成，卸载时自动移除
  export PATH="$PATH:/opt/qlh-edge-inference/bin"
  export QLH_BACKEND_PORT=8000        # 仅当用户在安装时选择注册环境变量
  ```

- **可选开关**：debconf 提问"是否注册全局 `qlh-launcher` / `bjtu` 命令环境变量？"（`debconf` 默认 yes，可 `DEBIAN_FRONTEND=noninteractive` + 参数关闭）；或读取 `etc/qlh/env-register` 标记文件。
- **卸载清理**：`postrm` 删除 `/etc/profile.d/qlh.sh`（仅本包文件，不动其他文件）。
- **即时生效**：`/etc/profile.d` 对**新登录会话**生效；已打开的终端需重新登录/`source`。

### 3.3 静默与自动化

- Windows：`setup.exe /ENVREG=0` 或 `/ENVREG=1` 覆盖向导默认。
- Linux：`dpkg-preconfigure` / `debconf-set-selections` 预设注册选择。
- 升级安装：检测已有注册则保持，不重复追加（幂等）。

---

## 4. 安全与边界

1. **可选原则**：注册环境变量默认勾选但可关闭；`QLH_CLUSTER_SECRET` 等敏感值**默认不写入**环境变量（写入属明文，仅当用户显式要求且知晓风险时提供高级选项）。
2. **最小写入**：只追加安装目录一段 `PATH`，不合并/改写用户已有条目；卸载只删本包条目。
3. **权限**：用户级（HKCU / `~/.profile`）无需管理员；系统级（HKLM / `/etc/environment`）需要权限且默认不启用。
4. **生效范围**：注册只影响新进程；不修改当前会话、不自动重启服务。
5. **源码模式不变**：非打包运行不受影响，README 手动指引保留。

---

## 5. 验收标准

1. Windows 安装包勾选"注册环境变量"→ 新开终端 `bjtu`、`bjtu chat` 直接可用；不勾选 → 环境完全不被修改。
2. 卸载后 `PATH` 恢复到安装前（无残留 QLH 段），用户其他 PATH 条目不受影响。
3. Linux `.deb` 安装/卸载后 `/etc/profile.d/qlh.sh` 正确创建/删除；新会话 `bjtu` 可用。
4. 重复安装（升级）不产生重复 PATH 段。
5. 静默安装（`/ENVREG=1`、debconf 预设）与交互安装行为一致。

---

## 6. 实施位置（待办，不在本文档范围外另行立计划）

- `packaging/setup.iss` / `packaging/setup-cuda.iss`：`[Registry]` + 向导复选框 + 卸载清理。
- `packaging/linux/postinst` / `postrm`：profile.d 写入/清理 + debconf 提问。
- `packaging/launcher.py`：保持现状（不负责注册，只读取已注册变量）。
- 完成后更新 [打包说明](../packaging/README.md) 与 [TUI 使用指南](TUI使用指南.md) 的安装章节。
