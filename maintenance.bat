@echo off
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

if /i "%~1"=="--purge"  goto run
if /i "%~1"=="--help"   goto help
if not "%~1"==""        goto run

echo ============================================================
echo   Maintenance - Turbo Slop Generator 3000
echo ------------------------------------------------------------
echo   Nettoie ce qu'une mise a jour par copier-coller laisse
echo   derriere : code des fonctions retirees, caches Python,
echo   fichiers temporaires. Verifie ensuite que tout compile.
echo.
echo   Les DONNEES (poids, add-ons d'anciennes versions) sont
echo   seulement CHIFFREES ici, pas supprimees. Pour les effacer :
echo       maintenance.bat --purge
echo ============================================================
echo.

:run
"%PY%" scripts\maintenance.py %*
echo.
pause
goto :eof

:help
"%PY%" scripts\maintenance.py --help
echo.
pause
