#!/usr/bin/env bash
# Nettoyage et verification apres mise a jour par copier-coller.
#   ./maintenance.sh            verifie et nettoie le code
#   ./maintenance.sh --purge    + supprime les donnees des fonctions retirees
set -e
cd "$(dirname "$0")"
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "============================================================"
    echo "  Maintenance - Turbo Slop Generator 3000"
    echo "------------------------------------------------------------"
    echo "  Les DONNEES (poids, add-ons d'anciennes versions) sont"
    echo "  seulement CHIFFREES ici, pas supprimees. Pour les effacer :"
    echo "      ./maintenance.sh --purge"
    echo "============================================================"
    echo
fi

python scripts/maintenance.py "$@"
