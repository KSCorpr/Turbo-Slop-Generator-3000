#!/usr/bin/env bash
# ============================================================================
#  Atelier - Installation portable Linux (cartes RTX)
#  venv local ./venv + dependances + moteur stable-diffusion.cpp (CUDA).
#  Les modeles se telechargent a la demande depuis l'onglet Bibliotheque.
# ============================================================================
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV="./venv"

echo "============================================================"
echo "  Atelier - installation portable"
echo "============================================================"

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

echo "[3/4] Telechargement du moteur stable-diffusion.cpp (CUDA)..."
python scripts/get_sdcpp.py --variant cuda

echo "[4/4] Dossiers utilisateur..."
mkdir -p models loras outputs tmp userdata

echo
echo "============================================================"
echo "  Termine. Lancez ./run.sh pour demarrer."
echo "  Les modeles se telechargent dans l'onglet Bibliotheque."
echo "============================================================"
