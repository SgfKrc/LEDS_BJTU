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
; 输出: dist/QLH-Edge-Inference-Setup-v0.1.7.exe
; ============================================================

#define MyAppName         "QLH Edge Inference"
#define MyAppNameCN       "轻量化大模型分布式边缘推理系统"
#define MyAppVersion      "0.1.7"
#define MyAppPublisher    "北京交通大学 · 大创项目"
#define MyAppExeName      "QLH-Edge-Inference.exe"
#define MyAppSourceDir    "..\dist\QLH-Edge-Inference"
#define MyAppOutputDir    "dist"

[Setup]
; 全局唯一标识 — 不要修改（用于升级检测和卸载）
AppId={{F1A3B5C7-8D2E-4F6A-9B1C-3D5E7F8A0B2C}}

; 基本信息
AppName={#MyAppName}
AppVerName={#MyAppNameCN} v{#MyAppVersion}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; 图标
SetupIconFile=leds.ico

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
; restartreplace: 若文件被旧版进程锁定，标记为重启后替换
; ignoreversion: 不比较版本号，直接覆盖
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs restartreplace

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
; 清理运行时产生的文件（卸载时删除日志，默认保留模型文件）
Type: files; Name: "{app}\logs\*"
Type: dirifempty; Name: "{app}\logs"

[Code]
// ---- 卸载旧版本（通过注册表 AppId 查找已有安装）----
function GetOldUninstallString(var UninstPath: String): Boolean;
begin
  Result := RegQueryStringValue(
    HKEY_LOCAL_MACHINE, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\' +
    '{#emit SetupSetting("AppId")}_is1', 'UninstallString', UninstPath
  ) or RegQueryStringValue(
    HKEY_CURRENT_USER, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\' +
    '{#emit SetupSetting("AppId")}_is1', 'UninstallString', UninstPath
  );
end;

// ---- 安装前检查 ----
function InitializeSetup: Boolean;
var
  UninstPath: String;
  ResultCode: Integer;
begin
  Result := True;

  // 检测是否有已安装的旧版本
  if GetOldUninstallString(UninstPath) then
  begin
    if MsgBox(
      '检测到已安装的旧版本 {#MyAppNameCN}。' + #13#10 +
      '覆盖安装可能导致文件冲突（新旧 DLL 混在一起）。' + #13#10#13#10 +
      '建议：先卸载旧版本，再重新安装。' + #13#10 +
      '模型文件和日志不会被删除。' + #13#10#13#10 +
      '是否自动卸载旧版本后继续？' + #13#10 +
      '（选「否」则直接覆盖安装，不推荐）',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON1
    ) = IDYES then
    begin
      // 静默卸载旧版本
      Exec(RemoveQuotes(ExtractFilePath(UninstPath)),
           '/VERYSILENT /SUPPRESSMSGBOXES',
           '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;

// ---- 卸载时可选删除 models 目录（默认不删除）----
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if DirExists(ExpandConstant('{app}\models')) then
    begin
      if MsgBox(
        '是否同时删除模型目录？' + #13#10#13#10 +
        ExpandConstant('{app}\models') + #13#10#13#10 +
        '注意：模型文件通常较大，重新下载会耗费时间。' + #13#10 +
        '建议默认保留；只有在彻底清理或释放磁盘空间时选择「是」。',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2
      ) = IDYES then
      begin
        DelTree(ExpandConstant('{app}\models'), True, True, True);
        RemoveDir(ExpandConstant('{app}'));
      end;
    end;
  end;
end;

// 安装完成的提示（中文/英文自适应）
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(CustomMessage('InstallDoneMsg'), mbInformation, MB_OK);
  end;
end;
