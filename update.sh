#!/usr/bin/env bash
# Le seul bouton de mise à jour : le code, le ménage, le moteur sd.cpp, puis
# le moteur 3D, puis la vérification des modèles installés. Chaque étape reste
# atteignable seule (--clean, --engine, --trellis, --code, --models).
cd "$(dirname "$0")" || exit 1
APP_ROOT="$(pwd)"
export UV_CACHE_DIR="$APP_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$APP_ROOT/runtime/uv-python"
export PIP_CACHE_DIR="$APP_ROOT/.cache/pip"
export HF_HOME="$APP_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$APP_ROOT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$APP_ROOT/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$APP_ROOT/.cache/huggingface/datasets"
export TORCH_HOME="$APP_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$APP_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$APP_ROOT/.cache/torchinductor"
export CUDA_CACHE_PATH="$APP_ROOT/.cache/nv"
export XDG_CACHE_HOME="$APP_ROOT/.cache/xdg"
export MPLCONFIGDIR="$APP_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$APP_ROOT/.cache/numba"
export GRADIO_TEMP_DIR="$APP_ROOT/tmp/gradio"
export TMP="$APP_ROOT/.cache/tmp"
export TEMP="$APP_ROOT/.cache/tmp"
export TMPDIR="$APP_ROOT/.cache/tmp"
mkdir -p "$APP_ROOT/.cache" "$UV_PYTHON_INSTALL_DIR" "$GRADIO_TEMP_DIR" "$TMPDIR"
PY="./python/bin/python3"
[ -x "$PY" ] || PY="python3"
exec "$PY" scripts/update_app.py "$@"
