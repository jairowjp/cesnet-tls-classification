#!/usr/bin/env bash
# =============================================================================
# 04_ejecutar_app_local.sh — Lanza la aplicación en tu equipo, visible solo para ti
# -----------------------------------------------------------------------------
# Origen (27/09/2026): al arrancar la app, Streamlit mostró "Network URL" y "External URL": escuchaba en
# todas las interfaces y cualquier equipo de la red local podía abrirla. Este lanzador la restringe a
# localhost. No se fija en .streamlit/config.toml porque en la nube la dirección la decide la plataforma.
#
# Uso (desde la raíz del repositorio):  bash scripts/utilidades/04_ejecutar_app_local.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."                      # raíz del repositorio, esté donde esté la terminal
PY="${VIRTUAL_ENV:-$(pwd)/../.venv}/bin/python"  # el Python del entorno del proyecto
exec "$PY" -m streamlit run app/principal.py --server.address=localhost --server.port=8501
