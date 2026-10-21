#!/usr/bin/env bash
# =============================================================================
# 09_correr_semana5.sh — Cómputo de la semana 5, repartido en dos noches
# -----------------------------------------------------------------------------
#   noche1   Parte 0 (baseline y CV agrupada 5 x 3), Workshop 1 (escenarios críticos), análisis de errores
#            y estudio de Optuna (tope de 4,5 h). Unas 6 a 7 horas.
#   noche2   Comparación estadística en los 15 pliegues, reentrenamiento con todo septiembre y cierre
#            (evaluación final única en octubre). Unas 3 horas.
# Cada paso se puede repetir por separado (ver scripts 12 y 13). Si Optuna se interrumpe, al relanzar
# noche1 continúa desde la última prueba guardada.
# Uso:  bash scripts/utilidades/09_correr_semana5.sh noche1|noche2
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${VIRTUAL_ENV:-$(pwd)/../.venv}/bin/python"
paso() { echo; echo "===== $1  ($(date +%H:%M))"; }
inicio=$(date +%s)
case "${1:-}" in
  noche1)
    paso "1/4 Parte 0: baseline y validación cruzada agrupada";   "$PY" scripts/12_semana5_robustez.py parte0
    paso "2/4 Workshop 1: escenarios críticos";                   "$PY" scripts/12_semana5_robustez.py workshop1
    paso "3/4 Análisis de errores y equidad (octubre completo)";  "$PY" scripts/12_semana5_robustez.py errores
    paso "4/4 Workshop 2: estudio de Optuna";                     "$PY" scripts/13_optuna_xgboost.py optimizar
    ;;
  noche2)
    paso "1/3 Comparación en los 15 pliegues (Wilcoxon)";         "$PY" scripts/13_optuna_xgboost.py comparar
    paso "2/3 Reentrenamiento con todo septiembre";               "$PY" scripts/13_optuna_xgboost.py reentrenar
    paso "3/3 Cierre: evaluación final única";                    "$PY" scripts/12_semana5_robustez.py cierre
    ;;
  *) echo "Uso: bash scripts/utilidades/09_correr_semana5.sh noche1|noche2"; exit 1 ;;
esac
echo; echo "Listo en $(( ($(date +%s) - inicio) / 60 )) minutos. Resultados en results/semana5/"
