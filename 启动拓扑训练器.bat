@echo off
rem ============================================================================
rem  HuaRongDao topology trainer - launcher
rem ----------------------------------------------------------------------------
rem  THIS FILE MUST STAY PURE ASCII (no byte above 0x7F, no BOM, CRLF endings).
rem
rem  Why (2026-09-17 incident): the old version was UTF-8 *with BOM* and carried
rem  absolute paths containing CJK characters. cmd.exe decodes a .bat's bytes with
rem  the ANSI code page, so those paths became garbage ("D:\XIYue???...") and
rem  `start` reported "The system cannot find the file ...". The console window
rem  closed instantly, which looked like the trainer itself was broken.
rem
rem  `chcp 65001` does NOT fix that: it only changes console *output*, not how cmd
rem  reads this file. So the rule is: keep this file ASCII and let cmd resolve the
rem  non-ASCII path segments itself.
rem
rem  Measured cmd behaviour (do not "simplify" this back):
rem    if exist "..\*\.venv\..."        -> wildcard in a MIDDLE segment: NOT expanded
rem    for %%P in ("..\*\.venv\...")    -> same problem, yields nothing
rem    for /d %%D in ("..\*")           -> directory wildcard: WORKS
rem  Hence the two-step lookup below (first the sibling folder, then .venv inside it).
rem
rem  pythonw.exe is used on purpose: GUI-subsystem interpreter, so no black console
rem  window stays behind the trainer.
rem ============================================================================
setlocal
cd /d "%~dp0"

rem ---- 1) find the sibling work dir, then the venv launcher inside it ----
set "PY="
for /d %%D in ("%~dp0..\*") do (
  if exist "%%~fD\.venv\Scripts\pythonw.exe" set "PY=%%~fD\.venv\Scripts\pythonw.exe"
)

rem ---- 2) find the trainer script: the *.py here containing its class ----
set "APP="
for %%F in ("%~dp0*.py") do (
  if not defined APP (
    findstr /m /c:"class TopologyTrainerApp" "%%~fF" >nul 2>nul && set "APP=%%~fF"
  )
)

if not defined PY (
  echo.
  echo [ERROR] venv Python launcher not found.
  echo         Looked for: ^<workdir^>\..^<any^>\.venv\Scripts\pythonw.exe
  echo         The virtual environment may have been moved or deleted.
  echo.
  pause
  exit /b 1
)
if not defined APP (
  echo.
  echo [ERROR] Trainer script not found next to this .bat file.
  echo         Looked for a .py containing "class TopologyTrainerApp".
  echo.
  pause
  exit /b 1
)

start "TopologyTrainer" "%PY%" "%APP%"
exit /b 0