@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m visual.desktop_player
) else (
  python -m visual.desktop_player
)
if errorlevel 1 pause
