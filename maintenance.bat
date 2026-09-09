@echo off
REM ===========================================================================
REM  Maintenance : verifie l'installation et nettoie ce qui traine.
REM
REM  Sans argument, le script POSE LA QUESTION :
REM      1  Update   aligne le moteur sd-cli sur le code
REM      2  Clean    efface ce que les fonctions retirees ont laisse
REM      3  Both
REM      0  Verifier seulement (defaut : ne supprime rien)
REM
REM  Le choix 2 MESURE d'abord, affiche le total, et demande confirmation
REM  avant de supprimer quoi que ce soit.
REM
REM  En ligne de commande, les options directes restent disponibles :
REM      maintenance.bat --update-engine
REM      maintenance.bat --purge
REM      maintenance.bat --all
REM
REM  Ne touche JAMAIS a models\custom\, loras\, outputs\, userdata\, python\.
REM  Pour mettre a jour l'APPLICATION : update.bat (ce script ne telecharge
REM  rien).
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\maintenance.py %*
echo.
pause
