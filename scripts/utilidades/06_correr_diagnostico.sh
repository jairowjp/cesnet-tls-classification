#!/usr/bin/env bash
# =============================================================================
# 06_correr_diagnostico.sh — Ejecuta todo el diagnóstico de la Actividad de la semana 3
# -----------------------------------------------------------------------------
# Orden: de lo más rápido a lo más lento, así los primeros resultados llegan pronto y, si el tiempo
# se agota, lo ya terminado queda guardado. Cada paso escribe su resultado en results/diagnostico/.
# Se recomienda ejecutarlo dentro de tmux:  tmux new -s diagnostico
#
# Uso: bash scripts/utilidades/06_correr_diagnostico.sh [MUESTRA] [EPOCAS_TRANSFORMER_MEJORADO]
#   MUESTRA: flujos de entrenamiento (submuestra estratificada; 0 = los 640 000). Por defecto 200000.
#   EPOCAS_TRANSFORMER_MEJORADO: límite de épocas del Transformer con la estrategia. Por defecto 25.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
MUESTRA="${1:-200000}"
EPOCAS_TF="${2:-25}"
PY="${VIRTUAL_ENV:-$(pwd)/../.venv}/bin/python"
inicio=$(date +%s)
paso() { echo; echo "===== $1  ($(date +%H:%M))"; }
paso "1/7 Random Forest: curvas C y D, diagnóstico y estrategia 1";  "$PY" scripts/09_diagnostico.py rf --muestra "$MUESTRA"
paso "2/7 CNN 1D base (15 épocas)";                                  "$PY" scripts/09_diagnostico.py red --modelo cnn1d --variante base --muestra "$MUESTRA"
paso "3/7 CNN 1D mejorada (estrategia 2)";                            "$PY" scripts/09_diagnostico.py red --modelo cnn1d --variante mejorada --muestra "$MUESTRA"
paso "4/7 XGBoost base y regularizado (estrategia 3)";               "$PY" scripts/09_diagnostico.py xgb --muestra "$MUESTRA"
paso "5/7 Transformer base (15 épocas)";                              "$PY" scripts/09_diagnostico.py red --modelo transformer --variante base --muestra "$MUESTRA"
paso "6/7 Transformer mejorado (estrategia 2)";                       "$PY" scripts/09_diagnostico.py red --modelo transformer --variante mejorada --muestra "$MUESTRA" --epocas "$EPOCAS_TF"
paso "7/7 Resumen y comparación antes/después";                      "$PY" scripts/09_diagnostico.py resumen
echo; echo "Listo en $(( ($(date +%s) - inicio) / 60 )) minutos. Resultados en results/diagnostico/"
