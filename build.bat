@echo off
setlocal

echo ============================================
echo  Attendance System - Build
echo ============================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run setup first.
    pause
    exit /b 1
)

echo Cleaning previous build...
if exist "dist\AttendanceSystem" rmdir /s /q "dist\AttendanceSystem"
if exist "build" rmdir /s /q "build"

echo Building with PyInstaller...
.venv\Scripts\pyinstaller attendance.spec --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED.
    pause
    exit /b 1
)

echo Preparing output folders beside AttendanceSystem.exe in dist...
if not exist "dist\AttendanceSystem\config.yaml" copy "config.yaml" "dist\AttendanceSystem\config.yaml"
if not exist "dist\AttendanceSystem\enrollment_faces" mkdir "dist\AttendanceSystem\enrollment_faces"
if not exist "dist\AttendanceSystem\data" mkdir "dist\AttendanceSystem\data"
if not exist "dist\AttendanceSystem\exports" mkdir "dist\AttendanceSystem\exports"
if not exist "dist\AttendanceSystem\logs" mkdir "dist\AttendanceSystem\logs"
if not exist "dist\AttendanceSystem\archive_thumbnails" mkdir "dist\AttendanceSystem\archive_thumbnails"

if exist "enrollment_faces" (
    xcopy /s /e /q /y "enrollment_faces" "dist\AttendanceSystem\enrollment_faces\" >nul 2>&1
)

echo.
echo Build complete: dist\AttendanceSystem\
echo Run:  dist\AttendanceSystem\AttendanceSystem.exe
echo.
echo Client package must also include:
echo   - Tesseract-OCR installed on the client machine
echo   - config.yaml configured for the client's camera and archive paths
echo   - EasyOCR model files downloaded on first run
pause
