#!/usr/bin/env bash
# Essai MiniMax-H3 (video + audio). N'ajoute rien a l'application : c'est une
# sonde, pour savoir si la generation video est jouable sur cette machine.
#   ./try-minimax.sh --check      verifie tout, ne telecharge rien
#   ./try-minimax.sh --download   recupere les poids (~26 Go)
#   ./try-minimax.sh --run        lance les deux passes
set -e
cd "$(dirname "$0")"
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "============================================================"
    echo "  Essai MiniMax-H3 (video + audio)"
    echo "------------------------------------------------------------"
    echo "  Deux questions, deux seulement :"
    echo "    1. le modele tient-il dans la VRAM de cette carte ?"
    echo "    2. la LoRA Turbo (4 pas) s'applique-t-elle ?"
    echo
    echo "  Etapes :"
    echo "    ./try-minimax.sh --check      verifie, ne telecharge rien"
    echo "    ./try-minimax.sh --download   les poids (~26 Go)"
    echo "    ./try-minimax.sh --run        les deux passes"
    echo
    echo "  Le moteur doit dater du 4 aout 2026 ou apres."
    echo "============================================================"
    echo
fi

python scripts/try_minimax.py "$@"
