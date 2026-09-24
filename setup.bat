@echo off
rem Setup (Windows): creates a private Python environment in .venv and installs the
rem robo-advisor with Yahoo Finance support and the test tools. Safe to re-run (updates).
rem You rarely need to call this yourself: ra.bat runs it automatically whenever the
rem environment is missing, incomplete, or older than pyproject.toml's dependency list.
rem
rem   setup.bat            install / update
rem   setup.bat --fresh    delete .venv and reinstall from scratch
setlocal
cd /d "%~dp0"

if /i "%~1"=="--fresh" (
  echo Removing existing .venv ...
  rmdir /s /q .venv 2>nul
)

rem 1. find a Python >= 3.10 (the "py" launcher first, then "python")
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo ERROR: Python 3.10 or newer was not found.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  echo and run setup.bat again.
  exit /b 1
)
for /f "delims=" %%v in ('%PY% --version') do echo Using %%v

rem 2. create the private environment
if not exist ".venv\Scripts\python.exe" (
  echo Creating private Python environment in .venv ...
  %PY% -m venv .venv || exit /b 1
)

rem 3. install the app and its libraries into it
echo Installing robo-advisor and its libraries ^(this can take a few minutes^) ...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install --quiet -e ".[yahoo,dev]" || exit /b 1

rem 4. smoke check, then record which dependency list was installed (ra compares this stamp)
".venv\Scripts\robo-advisor.exe" graph >nul || exit /b 1
".venv\Scripts\python.exe" -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" pyproject.toml > ".venv\.install-stamp" || exit /b 1
echo.
echo Setup complete. The environment is used automatically by ra.bat - no activation needed:
echo   .\ra run --profile examples\client_target.json
echo   .\ra data --provider yahoo
echo   .\ra run --profile examples\client_target.json --provider yahoo --as-of today
endlocal
