#!/usr/bin/env bash
# =============================================================================
# 00_configurar_entorno.sh — Crea el entorno de Python del proyecto desde cero
# -----------------------------------------------------------------------------
# Origen: comandos ejecutados el 26/09/2026 en Kali Linux, reunidos en un script.
# Por qué existe: Kali trae Python 3.14, pero cesnet-datazoo exige pydantic < 2.12,
# que no tiene versión compilada para 3.14 (falla al instalar pydantic-core).
# La solución fue un entorno con Python 3.13 instalado con uv, sin tocar el sistema.
#
# Uso (desde la raíz del repositorio):  bash scripts/utilidades/00_configurar_entorno.sh
# Resultado: ../.venv con Python 3.13 y las versiones exactas de requirements-lock.txt,
#            y el kernel de Jupyter "proyecto_integrador" registrado.
# =============================================================================
set -euo pipefail   # detenerse ante cualquier error, variable no definida o fallo en una tubería

# /tmp es muy pequeño en este equipo (408 MB) y pip lo usa para compilar paquetes grandes
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

command -v uv >/dev/null || { echo "Falta uv: instálalo con 'pipx install uv'"; exit 1; }

uv python install 3.13                       # Python 3.13 independiente del sistema
uv venv --python 3.13 ../.venv               # entorno un nivel arriba del repositorio
# shellcheck disable=SC1091
source ../.venv/bin/activate
uv pip install -r requirements-lock.txt      # versiones exactas: el entorno queda idéntico al original

python -m ipykernel install --user --name proyecto_integrador \
       --display-name "Python 3.13 (proyecto_integrador)"
python -c "import pandas, sklearn, xgboost, torch; print('Entorno listo:', pandas.__version__, torch.__version__)"
