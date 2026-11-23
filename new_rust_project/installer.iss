; --- Определение основных констант приложения ---
#define MyAppName "TSPiceLaboratory"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "TSPiceLaboratory Team"
#define MyAppURL "https://github.com"
#define MyAppExeName "TSPiceLaboratory.exe"

[Setup]
; Уникальный идентификатор приложения (сгенерируйте новый в IDE Inno Setup: Tools -> Generate GUID)
AppId={{A1B2C3D4-E5F6-7A8B-9C0D-E1F2A3B4C5D6}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableDirPage=no
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Папка для сохранения готового инсталлятора
OutputDir=.\dist_installer
OutputBaseFilename=TSPiceLaboratory_Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Требуется 64-битный режим, так как PyTorch и CUDA строго 64-битные
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Главный исполняемый файл (собранный через PyInstaller или аналоги)
Source: ".\dist\TSPiceLaboratory\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Все сопутствующие бинарники, библиотеки сборки (включая PyQt6, PyTorch, DLL-ки CUDA)
Source: ".\dist\TSPiceLaboratory\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Исходная пустая база данных SQLite, если она поставляется предсозданной (не перезаписываем при обновлении!)
Source: ".\backend\db.sqlite3"; DestDir: "{localappdata}\{#MyAppName}"; Flags: encountersonsystem ignoreversion uninsneveruninstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktops}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Проверка наличия драйверов NVIDIA перед установкой (критично для PyTorch + CUDA 12.8)
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  // Проверяем, существует ли системная утилита nvidia-smi
  if not FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) then
  begin
    if MsgBox('Внимание: На данном компьютере не обнаружена утилита управления NVIDIA CUDA (nvidia-smi.exe).' #13#10 #13#10 +
              'Для работы TSPiceLaboratory требуются вычисления на GPU NVIDIA. Вы хотите продолжить установку?',
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;
