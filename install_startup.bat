@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Attendance System - Register Auto-Start
echo ============================================

cd /d "%~dp0"

set "EXE_PATH=%~dp0dist\AttendanceSystem\AttendanceSystem.exe"

if not exist "!EXE_PATH!" (
    echo ERROR: Built executable not found at:
    echo   !EXE_PATH!
    echo Run build.bat first.
    pause
    exit /b 1
)

echo Registering Task Scheduler entry...
schtasks /create /tn "AttendanceSystem" /tr "\"!EXE_PATH!\"" /sc ONLOGON /rl HIGHEST /f

if %ERRORLEVEL% NEQ 0 (
    echo FAILED to register. Try running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo Auto-start registered. The app will launch on next login.
echo To remove:  schtasks /delete /tn "AttendanceSystem" /f
pause
