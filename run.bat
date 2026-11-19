@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First follow the setup steps in README.md to create .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launch.py
