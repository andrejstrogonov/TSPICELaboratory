#define MyAppName "TSpiceLaboratory"
#define MyAppVersion "1.0"
#define MyAppExeName "main.py"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application source modules
Source: "src\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\model.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Transistor characteristics bundled example sweeps for loopback test and verification
Source: "data\examples\*"; DestDir: "{app}\data\examples"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Run local PyQt6 python file windowless using pythonw.exe execution path
Name: "{group}\{#MyAppName}"; Filename: "pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""
Name: "{autodesktop}\{#MyAppName}"; Filename: "pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; Tasks: desktopicon

[Run]
; Auto-install scientific stack via pip silently during wizard execution phase
Filename: "python.exe"; Parameters: "-m pip install -r ""{app}\requirements.txt"""; StatusMsg: "Configuring workspace dependencies (PyQt6, Matplotlib, NumPy, SciPy)..."; Flags: runhidden
Filename: "pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  PythonInstallCmd: String;
  LocalInstallerPath: String;
begin
  Result := '';
  if not Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if MsgBox('Python runtime environment is missing. Would you like to automatically deploy Python 3.11?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      PythonInstallCmd := 'https://python.org';
      LocalInstallerPath := ExpandConstant('{tmp}\python_setup.exe');
      WizardForm.StatusLabel.Caption := 'Fetching Python 3.11 installer...';
      try
        DownloadTemporaryFile(PythonInstallCmd, LocalInstallerPath, '', nil);
        WizardForm.StatusLabel.Caption := 'Installing Python 3.11 silently...';
        if Exec(LocalInstallerPath, '/quiet InstallAllUsers=1 PrependPath=1 TargetDir=' + ExpandConstant('{autopf}\Python311'), '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
        begin
          if ResultCode <> 0 then
            Result := 'Python installation failed with exit code ' + IntToStr(ResultCode) + '. Please install Python manually.';
        end
        else
          Result := 'Failed to launch the Python installer.';
      except
        Result := 'Failed to download the Python installer. Check your internet connection.';
      end;
    end
    else
    begin
      Result := 'Installation cancelled. Python is required to run this application.';
    end;
  end;
end;
