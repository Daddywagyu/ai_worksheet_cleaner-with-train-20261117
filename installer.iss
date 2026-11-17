; Inno Setup script - AI Worksheet Cleaner
#define MyAppName "AI Worksheet Cleaner"
#define MyAppVersion "0.1.0"
#define MyAppExeName "AIWorksheetCleaner.exe"

[Setup]
AppId={{3E9A5C71-8B24-4F6D-A0D3-92C7B15E4F08}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\AIWorksheetCleaner
DefaultGroupName={#MyAppName}
OutputDir=installer_output
OutputBaseFilename=AIWorksheetCleaner_Setup_{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "dist\AIWorksheetCleaner\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
