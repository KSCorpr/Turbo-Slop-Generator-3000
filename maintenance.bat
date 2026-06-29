@echo off
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"
echo Nettoyage et verification apres mise a jour...
echo.
"%PY%" scripts\maintenance.py
echo.
pause
