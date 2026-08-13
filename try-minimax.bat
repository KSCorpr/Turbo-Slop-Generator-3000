@echo off
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

if not "%~1"=="" goto run

echo ============================================================
echo   Essai MiniMax-H3 (video + audio) - Turbo Slop Generator
echo ------------------------------------------------------------
echo   Ceci n'ajoute RIEN a l'application. C'est un essai, pour
echo   savoir si la generation video est jouable sur cette
echo   machine avant d'y consacrer une interface.
echo.
echo   Deux questions, deux seulement :
echo     1. le modele tient-il dans la VRAM de cette carte ?
echo     2. la LoRA Turbo (4 pas) s'applique-t-elle ?
echo.
echo   Etapes, dans l'ordre :
echo     try-minimax.bat --check      verifie tout, ne telecharge rien
echo     try-minimax.bat --download   recupere les poids (~26 Go)
echo     try-minimax.bat --run        lance les deux passes
echo.
echo   Le moteur doit dater du 4 aout 2026 ou apres :
echo   lancez update-engine.bat avant si ce n'est pas fait.
echo.
echo   Les poids vont dans models\ et se suppriment depuis
echo   "Systeme - Gestion et nettoyage" comme le reste.
echo ============================================================
echo.
echo   Lancement de la verification (aucun telechargement)...
echo.

:run
"%PY%" scripts\try_minimax.py %*
echo.
pause
