@echo off
REM ===========================================================================
REM  Atelier - Installation portable Windows (cartes RTX)
REM ---------------------------------------------------------------------------
REM  Installe un Python portable dans .\python, les dependances, et le moteur
REM  stable-diffusion.cpp (CUDA) dans .\bin. Les MODELES se telechargent ensuite
REM  a la demande depuis l'onglet Bibliotheque. Rien au niveau systeme.
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "APP_CACHE=%~dp0.cache"
set "APP_TMP=%~dp0tmp"
set "UV_CACHE_DIR=%APP_CACHE%\uv"
set "UV_PYTHON_INSTALL_DIR=%~dp0runtime\uv-python"
set "PIP_CACHE_DIR=%APP_CACHE%\pip"
set "HF_HOME=%APP_CACHE%\huggingface"
set "HF_HUB_CACHE=%APP_CACHE%\huggingface\hub"
set "HUGGINGFACE_HUB_CACHE=%APP_CACHE%\huggingface\hub"
set "HF_DATASETS_CACHE=%APP_CACHE%\huggingface\datasets"
set "TORCH_HOME=%APP_CACHE%\torch"
set "TRITON_CACHE_DIR=%APP_CACHE%\triton"
set "TORCHINDUCTOR_CACHE_DIR=%APP_CACHE%\torchinductor"
set "CUDA_CACHE_PATH=%APP_CACHE%\nv"
set "XDG_CACHE_HOME=%APP_CACHE%\xdg"
set "MPLCONFIGDIR=%APP_CACHE%\matplotlib"
set "NUMBA_CACHE_DIR=%APP_CACHE%\numba"
set "GRADIO_TEMP_DIR=%APP_TMP%\gradio"
set "TEMP=%APP_CACHE%\tmp"
set "TMP=%APP_CACHE%\tmp"
set "TMPDIR=%APP_CACHE%\tmp"
for %%d in ("%APP_CACHE%" "%UV_PYTHON_INSTALL_DIR%" "%GRADIO_TEMP_DIR%" "%TEMP%") do if not exist "%%~d" mkdir "%%~d" >nul 2>&1

set "PYVER=3.11.9"
set "PYDIR=%~dp0python"
set "PY=%PYDIR%\python.exe"
set "PYZIP_URL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

echo ============================================================
echo   Atelier - installation portable
echo ============================================================

if not exist "%PY%" (
    echo [1/4] Telechargement de Python %PYVER% portable...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYZIP_URL%' -OutFile '%TEMP%\pyembed.zip'" || goto :error
    powershell -NoProfile -Command "Expand-Archive -Force '%TEMP%\pyembed.zip' '%PYDIR%'" || goto :error
    powershell -NoProfile -Command "$f=Get-ChildItem '%PYDIR%\python*._pth' | Select-Object -First 1; (Get-Content $f.FullName) -replace '#import site','import site' | Set-Content $f.FullName; Add-Content $f.FullName 'Lib\site-packages'"
    echo [1/4] Installation de pip...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%TEMP%\get-pip.py'" || goto :error
    "%PY%" "%TEMP%\get-pip.py" --no-warn-script-location || goto :error
) else (
    echo [1/4] Python portable deja present.
)

echo [2/4] Installation des dependances (Gradio, huggingface_hub)...
REM Reseau plus robuste : nombreux reessais + timeout long contre les coupures.
set "PIP_NET=--retries 8 --timeout 120 --no-warn-script-location"
"%PY%" -m pip install --upgrade pip %PIP_NET%
"%PY%" -m pip install -r requirements.txt %PIP_NET% || (
    echo Nouvel essai de l'installation des dependances...
    "%PY%" -m pip install -r requirements.txt %PIP_NET% || goto :error
)

REM Auto-reparation : si des outils (Toolkit/Upscale) ont installe transformers/
REM diffusers en version trop recente (incompatible huggingface_hub<1.0 requis
REM par Gradio, ou torch 2.4), on les ramene a une version compatible. On ne les
REM installe PAS s'ils sont absents (install de base legere).
echo [2/4] Verification de compatibilite des outils...
"%PY%" -m pip show transformers >nul 2>&1 && "%PY%" -m pip install "transformers>=4.45,<5" %PIP_NET%
"%PY%" -m pip show diffusers >nul 2>&1 && "%PY%" -m pip install "diffusers>=0.30,<0.32" %PIP_NET%

echo [3/4] Telechargement du moteur stable-diffusion.cpp (CUDA)...
"%PY%" scripts\get_sdcpp.py --variant cuda || goto :error

echo [4/4] Dossiers utilisateur...
for %%d in (models loras outputs tmp userdata .cache runtime) do if not exist "%%d" md "%%d"
"%PY%" -c "import sys; sys.path.insert(0, '.'); from atelier import settings; settings.sync_local_pip_configs()" >nul 2>&1

echo.
echo ============================================================
echo   Installation terminee. Lancez run.bat pour demarrer.
echo   Les modeles se telechargent dans l'onglet Bibliotheque.
echo ============================================================
pause
exit /b 0

:error
echo.
echo *** Erreur pendant l'installation. ***
pause
exit /b 1
