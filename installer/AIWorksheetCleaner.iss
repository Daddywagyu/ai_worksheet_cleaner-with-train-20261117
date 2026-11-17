#define AppName "AI Worksheet Cleaner"
#define AppVersion "0.1.0"
#define AppPublisher "AI Worksheet Cleaner"

[Setup]
AppId={{9D41B75A-6F12-4E50-AE1B-7A7B3D2C81A4}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\AI Worksheet Cleaner
DefaultGroupName={#AppName}
OutputDir=..\installer-output
OutputBaseFilename=AIWorksheetCleaner-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
UninstallDisplayIcon={app}\AIWorksheetCleaner.exe

[Tasks]
Name: "desktopicon"; Description: "สร้าง Shortcut บน Desktop"; GroupDescription: "ตัวเลือกเพิ่มเติม:"; Flags: checkedonce

[Files]
Source: "..\dist\AIWorksheetCleaner\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AI Worksheet Cleaner"; Filename: "{app}\AIWorksheetCleaner.exe"
Name: "{autodesktop}\AI Worksheet Cleaner"; Filename: "{app}\AIWorksheetCleaner.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AIWorksheetCleaner.exe"; Description: "เปิด AI Worksheet Cleaner"; Flags: postinstall nowait skipifsilent
