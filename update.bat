@echo off
REM ===========================================================================
REM  LE bouton de mise a jour. Il fait les trois choses, dans l'ordre :
REM
REM    1. le CODE de l'application, depuis GitHub ;
REM    2. le MOTEUR sd.cpp (sd-cli), remis au niveau de ce code ;
REM    3. le MENAGE : ce que les fonctions retirees ont laisse derriere elles
REM       — il MONTRE le total et DEMANDE avant de supprimer quoi que ce soit.
REM
REM  Ne touche jamais a vos donnees : models\, loras\, outputs\, userdata\,
REM  tools_repo\, bin\, python\ sont laisses intacts.
REM
REM    update.bat                   met a jour (meme branche que l'install)
REM    update.bat --code-only       le code seulement, sans moteur ni menage
REM    update.bat --check           dit seulement ce qui changerait
REM    update.bat --rollback        annule la derniere mise a jour du code
REM    update.bat --branch main     change de branche VOLONTAIREMENT
REM
REM  La branche suivie est affichee AVANT tout telechargement. Sans --branch,
REM  la mise a jour reste sur la branche d'ou vient cette installation.
REM
REM  FERMEZ l'application avant de lancer ce script (un fichier ouvert ne peut
REM  pas etre remplace sous Windows).
REM ===========================================================================
cd /d "%~dp0"
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\update_app.py %*
echo.
pause
