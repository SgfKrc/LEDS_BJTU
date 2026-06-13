; ============================================================
; QLH 边缘推理系统 — Inno Setup 安装脚本
; ============================================================
; 编译命令:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
;   或 双击 setup.iss → Inno Setup Compiler 打开 → Build → Compile
;
; 前置条件:
;   1. 已完成 PyInstaller 打包 → dist/QLH-Edge-Inference/
;   2. 已安装 Inno Setup 6 (默认路径 C:\Program Files (x86)\Inno Setup 6)
;
; 输出: dist/QLH-Edge-Inference-Setup-v0.1.0.exe
; ============================================================

#define MyAppName         "QLH Edge Inference"
#define MyAppNameCN       "轻量化大模型分布式边缘推理系统"
#define MyAppVersion      "0.1.0"
#define MyAppPublisher    "北京交通大学 · 大创项目"
#define MyAppExeName      "QLH-Edge-Inference.exe"
#define MyAppSourceDir    "dist\QLH-Edge-Inference"
#define MyAppOutputDir    "dist"

[Setup]
; 全局唯一标识 — 不要修改（用于升级检测和卸载）
AppId={{F1A3B5C7-8D2E-4F6A-9B1C-3D5E7F8A0B2C}

; 基本信息
AppName={#MyAppName}
AppVerName={#MyAppNameCN} v{#MyAppVersion}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; 图标
SetupIconFile=g:\C\PYT\qlh\packaging\leds.ico

; 输出
OutputDir={#MyAppOutputDir}
OutputBaseFilename=QLH-Edge-Inference-Setup-v{#MyAppVersion}

; 默认安装路径
DefaultDirName={autopf}\QLH-Edge-Inference
DefaultGroupName={#MyAppNameCN}

; 权限 — 最低权限即可，如需写 Program Files 会弹 UAC 提权
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; 压缩
Compression=lzma2/max
SolidCompression=yes

; 界面
WizardStyle=modern
DisableWelcomePage=no

; 卸载信息（控制面板 → 程序和功能）
UninstallDisplayName={#MyAppNameCN}
UninstallDisplayIcon={app}\{#MyAppExeName}

; 仅支持 64 位 Windows 10+
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "chinese"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; 安装界面中文字符串（中文语言激活时使用）
chinese.AppNameCN=轻量化大模型分布式边缘推理系统
chinese.LaunchDesc=立即启动 轻量化大模型分布式边缘推理系统
chinese.InstallDoneMsg=安装完成！%n%n首次启动会自动检测模型文件。%n若缺失模型，程序会弹出下载引导窗口。%n%n启动后浏览器访问: http://localhost:8000%n%n项目: 北京交通大学 · 大学生创新创业训练计划

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; ---- 主程序（PyInstaller 输出） ----
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

; ---- 项目文档 ----
Source: "..\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\整体架构.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\模块接口说明.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\核心技术原理.md"; DestDir: "{app}\docs"; Flags: ignoreversion

; ---- GGUF 转换工具（高级用户） ----
Source: "scripts\convert_to_gguf.py"; DestDir: "{app}\tools"; Flags: ignoreversion

[Dirs]
; 模型和日志目录（用户可写）
Name: "{app}\models"; Permissions: users-modify
Name: "{app}\logs";   Permissions: users-modify

[Icons]
; 开始菜单 → 启动
Name: "{group}\{#MyAppNameCN}"; \
  Filename: "{app}\{#MyAppExeName}"; \
  Comment: "启动 {#MyAppNameCN} (端口 8000)"

; 开始菜单 → 卸载
Name: "{group}\卸载 {#MyAppNameCN}"; \
  Filename: "{uninstallexe}"

; 开始菜单 → 使用说明
Name: "{group}\使用说明"; \
  Filename: "{app}\docs\README.md"

; 桌面快捷方式（勾选 Tasks → desktopicon 时创建）
Name: "{autodesktop}\{#MyAppNameCN}"; \
  Filename: "{app}\{#MyAppExeName}"; \
  Tasks: desktopicon; \
  Comment: "启动 {#MyAppNameCN}"

[Run]
; 安装完成后询问是否立即启动
Filename: "{app}\{#MyAppExeName}"; \
  Description: "{cm:LaunchDesc}"; \
  Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
; 清理运行时产生的文件
Type: files; Name: "{app}\logs\*"
Type: dirifempty; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\models"

[Code]
// 安装完成的提示（中文/英文自适应）
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(CustomMessage('InstallDoneMsg'), mbInformation, MB_OK);
  end;
end;
