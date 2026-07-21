@echo off
REM Launch the research GUI on Windows (RTX 5060 Ti).
REM Assumes a Python environment with the project deps installed
REM (see environment/requirements-app.txt). tkinter ships with the python.org installer.
cd /d "%~dp0\.."
python src\app\gui.py %*
if errorlevel 1 pause
