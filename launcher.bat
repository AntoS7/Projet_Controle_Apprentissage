@echo off
REM Portable SUMO Project Launcher for Windows
REM This batch file provides easy access to the portable launcher

echo Starting Portable SUMO Project Launcher...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.8+ and add it to PATH.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Run the portable launcher
python portable_launcher.py %*

REM Pause if run without arguments (double-clicked)
if "%~1"=="" (
    echo.
    echo Press any key to exit...
    pause >nul
)
