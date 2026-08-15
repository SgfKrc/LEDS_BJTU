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
; 输出: dist/QLH-Edge-Inference-Setup-v0.1.8.2-CUDA.exe
; ============================================================

#define MyAppName         "QLH Edge Inference (CUDA)"
#define MyAppNameCN       "轻量化大模型分布式边缘推理系统（独显版）"
#define MyAppVersion      "0.1.8.3"
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
SetupIconFile=leds.ico

; 输出
OutputDir={#MyAppOutputDir}
OutputBaseFilename=QLH-Edge-Inference-Setup-v{#MyAppVersion}-CUDA

; 默认安装路径（与集显版不同目录，避免冲突）
DefaultDirName={autopf}\QLH-Edge-Inference-CUDA
DefaultGroupName={#MyAppNameCN}

; 权限
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ChangesEnvironment=yes

; 压缩
Compression=lzma2/max
SolidCompression=yes
; 安装包含 7.66GB 冻结 GGUF，超过 4.2GB 单 Setup.exe 上限，必须分片
DiskSpanning=yes
DiskSliceSize=2147483648

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
chinese.EnvRegTask=注册全局 bjtu 命令（写入当前用户环境变量）
chinese.EnvRegWriteError=无法写入当前用户的 PATH；应用已安装，但新终端不会自动找到 bjtu。
english.EnvRegTask=Register the global bjtu command for the current user
english.EnvRegWriteError=Could not update the current user's PATH. The application was installed, but new terminals will not find bjtu automatically.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "envreg"; Description: "{cm:EnvRegTask}"

[Files]
; ---- 主程序（PyInstaller 输出） ----
; build-cuda.bat 已在签名前把入口、版本、文档和工具复制到发布树。
; 安装器只能读取该签名树，禁止再用仓库文件覆盖签名内容。
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs restartreplace

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

[Code]
const
  QlhEnvRegistrationKey = 'Software\QLH\EnvironmentRegistration';
  QlhEnvPathEntryValue = 'PathEntry';
  QlhEnvPathOwnedValue = 'PathOwned';

#include "env-registration.issinc"

function RunDataRetention(CommandName: String): Boolean;
var
  DataRoot, Parameters: String;
  ResultCode: Integer;
  Started: Boolean;
begin
  DataRoot := ExpandConstant('{localappdata}\QLH-Edge-Inference\data');
  Parameters := CommandName + ' --root "' + ExpandConstant('{app}') +
    '" --data-root "' + DataRoot + '" --yes --json';
  ResultCode := -1;
  Started := Exec(
    ExpandConstant('{app}\tools\QLH-Data-Retention.exe'),
    Parameters, '', SW_HIDE, ewWaitUntilTerminated, ResultCode
  );
  Result := Started and (ResultCode = 0);
  if not Result then
    MsgBox('用户数据事务失败，安装或卸载已取消以保护数据。' + #13#10#13#10 +
      '请检查磁盘空间、目录冲突和 .qlh-retention-transaction.json 后重试。' + #13#10 +
      DataRoot, mbError, MB_OK);
end;

procedure RetainUserData;
begin
  if not RunDataRetention('retain') then
    Abort;
end;

procedure ReassociateRetainedData;
begin
  if not RunDataRetention('reassociate') then
    Abort;
end;

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
  if not EnvRegistrationParameterIsValid then
  begin
    MsgBox('ENVREG 只能为 0 或 1。', mbError, MB_OK);
    Result := False;
    exit;
  end;
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

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    RetainUserData
  else if CurUninstallStep = usPostUninstall then
    RemoveRegisteredUserEnvironment;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Verified: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    Verified := Exec(
      ExpandConstant('{app}\tools\QLH-Install-Manifest.exe'),
      'verify --root "' + ExpandConstant('{app}') + '" --level deep' +
      ' --trusted-keys-dir "' + ExpandConstant('{app}\pubkeys') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode
    );
    if (not Verified) or (ResultCode <> 0) then
      RaiseException('UP-N6.0 安装文件 deep 校验失败，安装已中止。');
    ReassociateRetainedData;
    ConfigureRegisteredUserEnvironment;
    MsgBox(CustomMessage('InstallDoneMsg'), mbInformation, MB_OK);
  end;
end;
