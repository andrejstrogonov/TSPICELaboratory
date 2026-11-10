[Setup]
AppName=TSpiceLaboratory
AppVersion=1.0
DefaultDirName={autopf}\TSpiceLaboratory
DefaultGroupName=TSpiceLaboratory

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

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
    if MsgBox('Python runtime environment is missing. Would you like to automatically deploy Python 3.11 embedded framework?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      PythonInstallCmd := 'https://python.org';
      LocalInstallerPath := ExpandConstant('{tmp}\python_setup.exe');
      StatusMarqueeMode := True;
      WizardForm.StatusLabel.Caption := 'Fetching secure Python binaries from distribution node...';
      try
        DownloadTemporaryFile(PythonInstallCmd, LocalInstallerPath, '', nil);
        WizardForm.StatusLabel.Caption := 'Executing silent installation loop for local hardware architecture...';
        if Exec(LocalInstallerPath, '/quiet InstallAllUsers=1 PrependPath=1 TargetDir={autopf}\Python311', '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
        begin
          if ResultCode <> 0 then
            Result := 'Python platform layout injection aborted with target compilation errors. Deploy manually.';
        end else
          Result := 'Critical core fault executing external runtime subsystem process wrapper.';
      except
        Result := 'Network pipeline transmission failed during target payload extraction loop.';
      end;
    end else
    begin
      Result := 'Deployment routine terminated by user instruction. Python execution layer missing.';
    end;
  end;
end;