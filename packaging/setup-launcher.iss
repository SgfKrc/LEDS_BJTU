; QLH standalone bootstrap launcher installer.
#define MyAppName      "QLH Launcher"
#define MyAppVersion   "0.1.8.2"
#define MyAppPublisher "北京交通大学 · 大创项目"
#define MySourceDir    "..\dist\QLH-Launcher"

[Setup]
AppId={{9B118A4F-76F1-4A1F-B216-313312C9D981}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
SetupIconFile=leds.ico
OutputDir=dist
OutputBaseFilename=QLH-Launcher-Setup-v{#MyAppVersion}
DefaultDirName={localappdata}\Programs\QLH-Launcher
DefaultGroupName=QLH Launcher
PrivilegesRequired=lowest
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\QLH-Launcher.exe

[Languages]
Name: "chinese"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs restartreplace

[Icons]
Name: "{group}\QLH Launcher"; Filename: "{app}\QLH-Launcher.exe"
Name: "{autodesktop}\QLH Launcher"; Filename: "{app}\QLH-Launcher.exe"; Tasks: desktopicon
Name: "{group}\卸载 QLH Launcher"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\QLH-Launcher.exe"; Description: "启动 QLH Launcher"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Verified: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    Verified := Exec(
      ExpandConstant('{app}\tools\QLH-Install-Manifest.exe'),
      'validate --manifest "' + ExpandConstant('{app}\manifest\install-manifest.json') +
      '" --trusted-keys-dir "' + ExpandConstant('{app}\pubkeys') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode
    );
    if (not Verified) or (ResultCode <> 0) then
      RaiseException('UP-N6.0 install manifest verification failed; setup aborted.');
  end;
end;
