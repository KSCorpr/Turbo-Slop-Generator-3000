#!/usr/bin/env bash
# Le seul bouton de mise à jour : le code, le ménage, le moteur sd.cpp, puis
# le moteur 3D, puis la vérification des modèles installés. Chaque étape reste
# atteignable seule (--clean, --engine, --trellis, --code, --models).
cd "$(dirname "$0")" || exit 1
PY="./python/bin/python3"
[ -x "$PY" ] || PY="python3"
exec "$PY" scripts/update_app.py "$@"
