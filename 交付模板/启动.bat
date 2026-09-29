@echo off
rem ============================================================================
rem  HuaRongDao state-space analyzer - launcher
rem ----------------------------------------------------------------------------
rem  KEEP THIS FILE PURE ASCII (no byte above 0x7F, no BOM, CRLF endings).
rem  Reason: cmd.exe decodes .bat bytes with the ANSI code page. A UTF-8 BOM, or
rem  CJK characters inside this file, would turn paths into garbage and the
rem  window would close instantly -- that exact failure hit the project's other
rem  launcher on 2026-09-17.
rem
rem  The application file name is CJK, so it is NOT typed here. Instead the only
rem  .exe that sits directly in this folder is located by wildcard, and cmd's own
rem  wildcard expansion works on real directory entries, so it is encoding-safe.
rem  (_internal\ is a subfolder, so it is not matched -- verified on 2026-09-19.)
rem  %~dp0 means "this script's own folder", so the whole folder can be copied
rem  anywhere (USB stick, another PC) and still works.
rem ============================================================================
setlocal
cd /d "%~dp0"

set "APP="
for %%F in ("%~dp0*.exe") do if not defined APP if exist "%%~fF" set "APP=%%~fF"

if not defined APP goto :missing
start "" "%APP%"
exit /b 0

:missing
echo.
echo [ERROR] No .exe found next to this launcher.
echo         Keep this file and the application .exe in the same folder.
echo         (Chinese notes: see the .txt file in this folder.)
echo.
pause
exit /b 1