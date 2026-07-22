@echo off
REM ===========================================================================
REM  Met a jour le moteur avec le BUILD MAISON (CI GitHub Actions du projet),
REM  compile pour les cartes du projet (1080 Ti=61, 2080 Ti=75, 3060=86).
REM
REM  Prerequis : le workflow "Build sd.cpp (Windows CUDA)" doit avoir ete lance
REM  au moins une fois (onglet Actions du depot GitHub > Run workflow), pour que
REM  la release "engine-latest" existe.
REM
REM  Pour revenir au binaire OFFICIEL leejet : utilisez update-engine.bat.
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   Mise a jour du moteur (build MAISON CI, archis 61+75+86)
echo ============================================================
"%PY%" scripts\get_sdcpp.py --source ours --force
echo.
echo Termine. Relancez run.bat.
pause
