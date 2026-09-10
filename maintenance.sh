#!/usr/bin/env bash
# Nettoyage et verification apres mise a jour par copier-coller.
#   ./maintenance.sh                  pose la question : 1 update, 2 clean,
#                                     3 les deux, 0 verifier seulement
#   ./maintenance.sh --update-engine  + aligne le moteur sd-cli sur le code
#   ./maintenance.sh --purge          + supprime les donnees des fonctions retirees
#   ./maintenance.sh --all            tout d'un coup
# À savoir : ./update.sh lance déjà ce script, moteur compris.
set -e
cd "$(dirname "$0")"
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi


python scripts/maintenance.py "$@"
