@echo off
REM run_demo.bat — sets IDS SDK env vars and launches the demo or a specified script
REM Adjust IDS_ROOT if your SDK is installed elsewhere.
set "IDS_ROOT=C:\Program Files\IDS\ids_peak"

if exist "%IDS_ROOT%\bin" (
  set "PATH=%IDS_ROOT%\bin;%PATH%"
)

if exist "%IDS_ROOT%\GenTL64" (
  set "GENICAM_GENTL64_PATH=%IDS_ROOT%\GenTL64"
) else if exist "%IDS_ROOT%\GenTL" (
  set "GENICAM_GENTL64_PATH=%IDS_ROOT%\GenTL"
)

REM Prefer a local virtualenv if present
if exist ".venv\Scripts\Activate.bat" (
  call ".venv\Scripts\Activate.bat"
  set "PYEXE=python"
) else (
  set "PYEXE=c:\Python\WPy64-313110\python.exe"
)

set SCRIPT=%1%
if "%SCRIPT%"=="" set SCRIPT=beamprofiler_test.py

%PYEXE% %SCRIPT%

exit /B %ERRORLEVEL%
