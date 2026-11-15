#define MyAppName "TSpiceLaboratory"
#define MyAppVersion "1.0"
#define MyAppExeName "app.py"

#define PythonVersion "3.12.4"
#define PythonInstallerUrl "https://www.python.org/ftp/python/" + PythonVersion + "/python-" + PythonVersion + "-amd64.exe"

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
Source: "src\app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\model.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\cad.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\simulation.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Transistor characteristics bundled example sweeps for loopback test and verification
Source: "data\examples\*"; DestDir: "{app}\data\examples"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Run local PyQt6 python file windowless using the private virtual environment created by [Code];
; ";" switches to the plain pythonw.exe icon without asking Explorer to extract it from the exe
Name: "{group}\{#MyAppName}"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; IconFilename: "{app}\venv\Scripts\pythonw.exe"; IconIndex: 0
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; IconFilename: "{app}\venv\Scripts\pythonw.exe"; IconIndex: 0; Tasks: desktopicon

[Run]
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent; Check: EnvironmentReady

[Code]
const
  VenvPython = '\venv\Scripts\python.exe';
  VenvPythonw = '\venv\Scripts\pythonw.exe';

var
  VenvFailed: Boolean;

function VenvReady: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}' + VenvPython))
    and FileExists(ExpandConstant('{app}' + VenvPythonw));
end;

function EnvironmentReady: Boolean;
begin
  Result := VenvReady;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  PythonInstallCmd: String;
  LocalInstallerPath: String;
begin
  Result := '';
  if not Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if MsgBox('Python runtime environment is missing. Would you like to automatically deploy Python {#PythonVersion}?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      PythonInstallCmd := '{#PythonInstallerUrl}';
      LocalInstallerPath := ExpandConstant('{tmp}\python_setup.exe');
      WizardForm.StatusLabel.Caption := 'Fetching Python {#PythonVersion} installer...';
      try
        DownloadTemporaryFile(PythonInstallCmd, LocalInstallerPath, '', nil);
        WizardForm.StatusLabel.Caption := 'Installing Python {#PythonVersion} silently...';
        if Exec(LocalInstallerPath, '/quiet InstallAllUsers=1 PrependPath=1 TargetDir=' + ExpandConstant('{autopf}\Python312'), '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
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

function InstallDependencies(Status: String): Boolean;
var
  ResultCode: Integer;
  Command: String;
  Arguments: String;
  Output: TExecOutput;
begin
  Command := ExpandConstant('{app}' + VenvPython);
  Arguments := '-m pip install --disable-pip-version-check --no-input -r "' + ExpandConstant('{app}\requirements.txt') + '"';
  WizardForm.StatusLabel.Caption := Status;
  // Exec failures must not be skipped: without the interpreter pip cannot run at all
  if not ExecAndCaptureOutput(Command, Arguments, '', SW_HIDE, ewWaitUntilTerminated, ResultCode, Output) then
  begin
    Result := False;
    Exit;
  end;
  Result := ResultCode = 0;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;
  if VenvReady then
    Exit;

  WizardForm.StatusLabel.Caption := 'Creating the application environment...';
  if not Exec('python', '-m venv "' + ExpandConstant('{app}\venv') + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) or not VenvReady then
  begin
    VenvFailed := True;
    // A previous run may have left an unusable environment; clearing it lets the next run retry
    DelTree(ExpandConstant('{app}\venv'), True, True, True);
    MsgBox('The application environment could not be created in "' + ExpandConstant('{app}\venv') + '".' + #13#10 +
      'Make sure Python 3.12+ is installed and not blocked by antivirus, then run the setup again.',
      mbError, MB_OK);
    Exit;
  end;

  if not InstallDependencies('Configuring workspace dependencies (PyQt6, Matplotlib, NumPy, SciPy)...') then
  begin
    VenvFailed := True;
    MsgBox('Installing the application dependencies failed. Run the setup again with a working internet connection.',
      mbError, MB_OK);
    Exit;
  end;

  // Warm the Matplotlib font cache during setup so the first launch does not rebuild it
  WizardForm.StatusLabel.Caption := 'Preparing plot fonts...';
  Exec(ExpandConstant('{app}' + VenvPython), '-c "from matplotlib.figure import Figure; Figure()"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function GetCustomSetupExitCode: Integer;
begin
  if VenvFailed then
    Result := 1
  else
    Result := 0;
end;
