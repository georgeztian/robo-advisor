@echo off
rem Run the robo-advisor inside the project's private environment (.venv), creating it first
rem if needed. Usage: .\ra <command> [options]   e.g.  .\ra run --interactive --provider yahoo
rem                   .\ra test                   runs the test suite
setlocal
set "HERE=%~dp0"
if not exist "%HERE%.venv\Scripts\robo-advisor.exe" (
  echo Private environment not found - running first-time setup ...
  call "%HERE%setup.bat" || exit /b 1
)
if /i "%~1"=="test" goto :test
"%HERE%.venv\Scripts\robo-advisor.exe" %*
exit /b %ERRORLEVEL%

:test
pushd "%HERE%"
"%HERE%.venv\Scripts\python.exe" -m pytest -q
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
