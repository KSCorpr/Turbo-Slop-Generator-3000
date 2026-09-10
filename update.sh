#!/usr/bin/env bash
# Le bouton de mise à jour : le code, puis le moteur sd.cpp, puis le ménage.
# Voir update.bat pour le détail des options.
cd "$(dirname "$0")" || exit 1
PY="./python/bin/python3"
[ -x "$PY" ] || PY="python3"
exec "$PY" scripts/update_app.py "$@"
