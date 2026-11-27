#!/usr/bin/env bash
# ============================================================================
#  Turbo Slop Generator 3000 — installation (Linux et macOS)
#
#  venv local ./venv + dependances + moteur stable-diffusion.cpp.
#  Le moteur telecharge depend de la machine :
#    - Linux  : build CUDA (cartes NVIDIA) ;
#    - macOS  : build Apple Silicon (Metal). Il n'existe pas de build Intel.
#  Les modeles se telechargent a la demande depuis l'onglet Catalogue.
# ============================================================================
set -e
cd "$(dirname "$0")"
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

PY="${PYTHON:-python3}"
VENV="./venv"
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "============================================================"
echo "  Turbo Slop Generator 3000 - installation"
echo "  Systeme : $OS ($ARCH)"
echo "============================================================"

if [ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ]; then
    echo
    echo "ATTENTION : Mac Intel detecte."
    echo "  stable-diffusion.cpp ne publie que des builds Apple Silicon."
    echo "  L'installation va continuer, mais le moteur devra etre compile"
    echo "  a la main, ou l'app tournera sur CPU (tres lent)."
    echo
fi

if [ ! -d "$VENV" ]; then
    echo "[1/4] Creation de l'environnement virtuel..."
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[2/4] Installation des dependances..."
PIP_NET="--retries 8 --timeout 120"
pip install --upgrade pip $PIP_NET
pip install -r requirements.txt $PIP_NET || pip install -r requirements.txt $PIP_NET

# Auto-reparation : si des outils ont installe transformers/diffusers en version
# trop recente (incompatible huggingface_hub<1.0 / torch 2.4), on les ramene a
# une version compatible. On ne les installe PAS s'ils sont absents.
pip show transformers >/dev/null 2>&1 && pip install "transformers>=4.45,<5" $PIP_NET || true
pip show diffusers >/dev/null 2>&1 && pip install "diffusers>=0.30,<0.32" $PIP_NET || true

# Variante du moteur : deduite du systeme par get_sdcpp lui-meme (metal sur
# macOS, cuda ailleurs). On ne la force pas ici pour n'avoir qu'un seul endroit
# ou cette regle est ecrite.
echo "[3/4] Telechargement du moteur stable-diffusion.cpp..."
python scripts/get_sdcpp.py

echo "[4/4] Dossiers utilisateur..."
mkdir -p models loras outputs tmp userdata

echo
echo "============================================================"
echo "  Termine. Lancez ./run.sh pour demarrer."
echo "  Les modeles se telechargent dans l'onglet Catalogue."
if [ "$OS" = "Darwin" ]; then
    echo
    echo "  macOS : le calcul passe par Metal. La memoire est UNIFIEE,"
    echo "  donc la 'VRAM' affichee correspond a la part de RAM que le"
    echo "  GPU peut adresser (~75%)."
fi
echo "============================================================"
