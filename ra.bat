@echo off
rem Run the robo-advisor inside the project's private environment (.venv).
rem The environment is installed or updated automatically when it is missing, incomplete, or
rem its dependency list (pyproject.toml) changed since the last install, e.g. after git pull.
rem
rem Usage: .\ra <command> [options]    e.g.  .\ra run --interactive --out clients\jane
rem        .\ra test                   runs the test suite
setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"

set "STATUS=ok"
if not exist "%VENV%\Scripts\robo-advisor.exe" set "STATUS=missing"
if not exist "%VENV%\.install-stamp" set "STATUS=missing"
if "%STATUS%"=="missing" goto :install
"%VENV%\Scripts\python.exe" -c "import hashlib, sys; cur = hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(); sys.exit(0 if cur == open(sys.argv[2]).read().strip() else 1)" "%HERE%pyproject.toml" "%VENV%\.install-stamp" >nul 2>&1
if errorlevel 1 set "STATUS=outdated"
if "%STATUS%"=="ok" goto :run

:install
if "%STATUS%"=="missing" (echo Private environment missing or incomplete - installing the app ...) else (echo Dependencies changed since the last install - updating the environment ...)
call "%HERE%setup.bat" || exit /b 1

:run
if /i "%~1"=="test" goto :test
"%VENV%\Scripts\robo-advisor.exe" %*
exit /b %ERRORLEVEL%

:test
pushd "%HERE%"
"%VENV%\Scripts\python.exe" -m pytest -q
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
