@echo off
REM ===========================================================================
REM  LE bouton de mise a jour. Il n'y en a plus qu'un, et il fait les QUATRE
REM  etapes, dans l'ordre :
REM
REM    1. le CODE de l'application, depuis GitHub ;
REM    2. le MENAGE : ce que les fonctions retirees ont laisse derriere elles,
REM       puis la verification de l'installation. Il MONTRE le total et
REM       DEMANDE avant de supprimer quoi que ce soit ;
REM    3. le MOTEUR sd.cpp (images et video), remis au niveau du code ;
REM    4. le MOTEUR 3D trellis.cpp — le binaire seul, et seulement s'il est
REM       deja installe (les ~16 Go de modeles 3D ne sont PAS retelecharges).
REM
REM  Ne touche jamais a vos donnees : models\, loras\, outputs\, userdata\,
REM  tools_repo\, bin\, python\ sont laisses intacts.
REM
REM    update.bat                   tout, dans l'ordre
REM    update.bat --clean           le menage seul
REM    update.bat --engine          le moteur sd.cpp seul
REM    update.bat --trellis         le moteur 3D seul
REM    update.bat --code            le code seul
REM    update.bat --check           dit ce que le code changerait, n'ecrit rien
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
