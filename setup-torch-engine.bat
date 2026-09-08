@echo off
REM ===========================================================================
REM  Branche Test7000 : installe le moteur PyTorch (torch + diffusers).
REM  Plusieurs gigaoctets, une seule fois. Remplace la pile torch du Toolkit
REM  par une plus recente : c'est voulu, voir scripts\setup_torch_engine.py.
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\setup_torch_engine.py %*
echo.
pause
