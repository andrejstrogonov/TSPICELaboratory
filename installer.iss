#define MyAppName "TSpiceLaboratory"
#define MyAppVersion "1.0"
#define MyAppExeName "main.py"

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
Source: "src\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\model.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\cad.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\simulation.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Transistor characteristics bundled example sweeps for loopback test and verification
Source: "data\examples\*"; DestDir: "{app}\data\examples"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Run local PyQt6 python file windowless using the private virtual environment created by [Run]
Name: "{group}\{#MyAppName}"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Build the private virtual environment once; every later run is skipped while venv\Scripts\python.exe exists
Filename: "python.exe"; Parameters: "-m venv ""{app}\venv"""; StatusMsg: "Creating the application environment..."; Flags: runhidden skipifdoesntexist; Check: NeedsEnvironmentSetup
; Auto-install scientific stack via pip silently during wizard execution phase
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-m pip install --disable-pip-version-check --no-input -r ""{app}\requirements.txt"""; StatusMsg: "Configuring workspace dependencies (PyQt6, Matplotlib, NumPy, SciPy)..."; Flags: runhidden; Check: NeedsEnvironmentSetup
; Warm the Matplotlib font cache during setup so the first launch does not rebuild it
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-c ""from matplotlib.figure import Figure; Figure()"""; StatusMsg: "Preparing plot fonts..."; Flags: runhidden; Check: NeedsEnvironmentSetup
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent; Check: EnvironmentReady

[Code]
var
  EnvironmentBuilt: Boolean;
  EnvironmentFailure: String;

function NeedsEnvironmentSetup: Boolean;
begin
  Result := not FileExists(ExpandConstant('{app}\venv\Scripts\python.exe'));
end;

function EnvironmentReady: Boolean;
begin
  Result := EnvironmentBuilt and (EnvironmentFailure = '')
    and FileExists(ExpandConstant('{app}\venv\Scripts\pythonw.exe'));
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

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if NeedsEnvironmentSetup then
      EnvironmentFailure := 'The application environment could not be created. Install Python from python.org and run the setup again.'
    else
      EnvironmentBuilt := True;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  Result := 0;
end;
