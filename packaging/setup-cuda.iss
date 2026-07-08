; ============================================================
; QLH 边缘推理系统 — Inno Setup 安装脚本（独显版）
; ============================================================
; 编译命令:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup-cuda.iss
;
; 前置条件:
;   1. 已完成 PyInstaller 打包 → dist/QLH-Edge-Inference-CUDA/
;   2. 已安装 Inno Setup 6
;
; 输出: dist/QLH-Edge-Inference-Setup-v0.1.5-CUDA.exe
; ============================================================

#define MyAppName         "QLH Edge Inference (CUDA)"
#define MyAppNameCN       "轻量化大模型分布式边缘推理系统（独显版）"
#define MyAppVersion      "0.1.5"
#define MyAppPublisher    "北京交通大学 · 大创项目"
#define MyAppExeName      "QLH-Edge-Inference.exe"
#define MyAppSourceDir    "..\dist\QLH-Edge-Inference-CUDA"
#define MyAppOutputDir    "dist"

[Setup]
; 全局唯一标识 — 与集显版不同，允许共存
AppId={{F1A3B5C7-8D2E-4F6A-9B1C-3D5E7F8A0B2D}}

; 基本信息
AppName={#MyAppName}
AppVerName={#MyAppNameCN} v{#MyAppVersion}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; 图标
SetupIconFile=g:\C\PYT\qlh\packaging\leds.ico

; 输出
OutputDir={#MyAppOutputDir}
OutputBaseFilename=QLH-Edge-Inference-Setup-v{#MyAppVersion}-CUDA

; 默认安装路径（与集显版不同目录，避免冲突）
DefaultDirName={autopf}\QLH-Edge-Inference-CUDA
DefaultGroupName={#MyAppNameCN}

; 权限
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; 压缩
Compression=lzma2/max
SolidCompression=yes

; 界面
WizardStyle=modern
DisableWelcomePage=no

; 卸载信息
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
chinese.AppNameCN=轻量化大模型分布式边缘推理系统（独显版）
chinese.LaunchDesc=立即启动 轻量化大模型分布式边缘推理系统
chinese.InstallDoneMsg=安装完成！%n%n本版本支持 NVIDIA GPU 推理（CUDA）+ CPU 推理（llama.cpp）。%n无 GPU 时自动回退 CPU 模式。%n%n首次启动会自动检测模型文件。%n%n启动后浏览器访问: http://localhost:8000%n%n项目: 北京交通大学 · 大学生创新创业训练计划

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; ---- 主程序（PyInstaller 输出） ----
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs restartreplace

; ---- 项目文档 ----
Source: "..\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\整体架构.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\模块接口说明.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\docs\核心技术原理.md"; DestDir: "{app}\docs"; Flags: ignoreversion

; ---- GGUF 转换工具 ----
Source: "scripts\convert_to_gguf.py"; DestDir: "{app}\tools"; Flags: ignoreversion

[Dirs]
Name: "{app}\models"; Permissions: users-modify
Name: "{app}\logs";   Permissions: users-modify

[Icons]
Name: "{group}\{#MyAppNameCN}"; \
  Filename: "{app}\{#MyAppExeName}"; \
  Comment: "启动 {#MyAppNameCN} (端口 8000)"

Name: "{group}\卸载 {#MyAppNameCN}"; \
  Filename: "{uninstallexe}"

Name: "{group}\使用说明"; \
  Filename: "{app}\docs\README.md"

Name: "{autodesktop}\{#MyAppNameCN}"; \
  Filename: "{app}\{#MyAppExeName}"; \
  Tasks: desktopicon; \
  Comment: "启动 {#MyAppNameCN}"

[Run]
Filename: "{app}\{#MyAppExeName}"; \
  Description: "{cm:LaunchDesc}"; \
  Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
Type: files; Name: "{app}\logs\*"
Type: dirifempty; Name: "{app}\logs"

[Code]
// ---- 卸载旧版本 ----
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

function InitializeSetup: Boolean;
var
  UninstPath: String;
  ResultCode: Integer;
begin
  Result := True;
  if GetOldUninstallString(UninstPath) then
  begin
    if MsgBox(
      '检测到已安装的旧版本 {#MyAppNameCN}。' + #13#10 +
      '覆盖安装可能导致文件冲突。' + #13#10#13#10 +
      '建议：先卸载旧版本，再重新安装。' + #13#10 +
      '模型文件和日志不会被删除。' + #13#10#13#10 +
      '是否自动卸载旧版本后继续？',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON1
    ) = IDYES then
    begin
      Exec(RemoveQuotes(ExtractFilePath(UninstPath)),
           '/VERYSILENT /SUPPRESSMSGBOXES',
           '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;

// ---- 卸载时可选删除 models ----
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
        '建议默认保留。',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2
      ) = IDYES then
      begin
        DelTree(ExpandConstant('{app}\models'), True, True, True);
        RemoveDir(ExpandConstant('{app}'));
      end;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(CustomMessage('InstallDoneMsg'), mbInformation, MB_OK);
  end;
end;
