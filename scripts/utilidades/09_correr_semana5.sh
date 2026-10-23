#!/usr/bin/env bash
# =============================================================================
# 09_correr_semana5.sh — Cómputo de la semana 5, repartido en dos noches
# -----------------------------------------------------------------------------
#   noche1   Parte 0 (baseline y CV agrupada 5 x 3), Workshop 1 (escenarios críticos), análisis de errores
#            y estudio de Optuna (tope de 4,5 h). Unas 6 a 7 horas.
#   noche2   Comparación estadística en los 15 pliegues, reentrenamiento con todo septiembre y cierre
#            (evaluación final única en octubre). Unas 3 horas.
#   prueba3  Prueba rápida de la noche 3 (2 épocas con 5 000 flujos por red, ~5–10 min): verifica que todo
#            funcione en este equipo antes de comprometer horas de cómputo. Sus resultados se sobrescriben.
#   noche3   Comentario de la semana 3: CNN y Transformer con parada temprana real (ReduceLROnPlateau y
#            límite alto de épocas) y la CNN con más capacidad, en la muestra de 200 000 flujos. 2,5 a 4,5 h.
# Cada paso se puede repetir por separado (ver scripts 12 y 13). Si Optuna se interrumpe, al relanzar
# noche1 continúa desde la última prueba guardada.
# Uso:  bash scripts/utilidades/09_correr_semana5.sh noche1|noche2|prueba3|noche3
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
  prueba3)
    paso "1/3 Prueba: pruebas automáticas de las redes";  "$PY" -m pytest -q tests/test_diagnostico.py
    paso "2/3 Prueba: CNN con parada (2 épocas)";        "$PY" scripts/09_diagnostico.py red --modelo cnn1d --variante parada --muestra 5000 --epocas 2
    paso "3/3 Prueba: Transformer con parada (2 épocas)"; "$PY" scripts/09_diagnostico.py red --modelo transformer --variante parada --muestra 5000 --epocas 2
    "$PY" scripts/09_diagnostico.py resumen_parada
    echo; echo "PRUEBA CORRECTA: ya puedes lanzar noche3 (sus resultados reemplazarán los de esta prueba)."
    ;;
  noche3)
    paso "1/4 CNN 1D con parada temprana real";           "$PY" scripts/09_diagnostico.py red --modelo cnn1d --variante parada --muestra 200000
    paso "2/4 Transformer con parada temprana real";      "$PY" scripts/09_diagnostico.py red --modelo transformer --variante parada --muestra 200000
    paso "3/4 CNN 1D con más capacidad";                  "$PY" scripts/09_diagnostico.py red --modelo cnn1d --variante capacidad --muestra 200000
    paso "4/4 Resumen: antes, después, parada y capacidad"; "$PY" scripts/09_diagnostico.py resumen_parada
    ;;
  *) echo "Uso: bash scripts/utilidades/09_correr_semana5.sh noche1|noche2|prueba3|noche3"; exit 1 ;;
esac
echo; echo "Listo en $(( ($(date +%s) - inicio) / 60 )) minutos. Resultados en results/semana5/"
