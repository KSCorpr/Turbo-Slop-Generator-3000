@echo off
setlocal
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
set "PY=%~dp0python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" app.py %*
pause
