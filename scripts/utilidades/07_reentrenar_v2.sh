#!/usr/bin/env bash
# =============================================================================
# 07_reentrenar_v2.sh — Reentrena los 4 modelos con la configuración v2 (validada por el diagnóstico)
# -----------------------------------------------------------------------------
#   piloto    Respalda la versión 1 y entrena Random Forest v2 con 200 000 flujos para medir la memoria.
#             Imprime la memoria estimada con los 640 000 flujos. Tarda unos 5 minutos.
#   completo  Reentrena CNN 1D, XGBoost, Random Forest y Transformer; compara con la v1 en VALIDACIÓN
#             (restaura la v1 si la v2 no mejora); regenera la comparativa, prepara la app y ejecuta las pruebas.
#             Tarda unas 5 a 6 horas: ejecútalo dentro de tmux.
# Uso:  bash scripts/utilidades/07_reentrenar_v2.sh piloto|completo
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${VIRTUAL_ENV:-$(pwd)/../.venv}/bin/python"
MODO="${1:-}"
paso() { echo; echo "===== $1  ($(date +%H:%M))"; }

respaldar_v1() {
  if [ -d results/modelos_v1 ]; then echo "Respaldo de la versión 1 ya existente: no se sobrescribe"; return; fi
  cp -r results/modelos results/modelos_v1
  cp -r results/comparativa results/comparativa_v1
  mkdir -p models/v1 && cp models/*.joblib models/*.pt models/v1/ 2>/dev/null || true
  echo "Versión 1 respaldada en results/modelos_v1/, results/comparativa_v1/ y models/v1/"
}

case "$MODO" in
  piloto)
    paso "Respaldo de la versión 1"; respaldar_v1
    paso "Random Forest v2 con 200 000 flujos (prueba de memoria)"
    "$PY" scripts/03_entrenar_clasicos.py --modelo random_forest --limite-filas 200000
    "$PY" - <<'PYEOF'
import json
c = json.load(open("results/modelos/random_forest/metricas.json"))["costo"]
pico = c["memoria_pico_GB"]
estimada = pico * 4  # 4 veces más flujos por árbol; estimación conservadora (la memoria de los datos crece menos)
print(f"\nMemoria máxima con 200 000 flujos: {pico} GB · modelo: {c['tamano_modelo_MB']} MB")
print(f"Estimación con 640 000 flujos: hasta {estimada:.1f} GB")
print("VEREDICTO:", "cabe en la memoria disponible" if estimada <= 9 else "NO cabe con seguridad: hay que reducir el modelo")
PYEOF
    ;;
  completo)
    if [ ! -d results/modelos_v1 ]; then echo "Primero ejecuta el modo piloto (respalda la versión 1)"; exit 1; fi
    inicio=$(date +%s)
    paso "1/7 CNN 1D v2";           "$PY" scripts/05_entrenar_profundos.py --modelo cnn1d
    paso "2/7 XGBoost v2";          "$PY" scripts/03_entrenar_clasicos.py --modelo xgboost
    paso "3/7 Random Forest v2";    "$PY" scripts/03_entrenar_clasicos.py --modelo random_forest
    paso "4/7 Transformer v2";      "$PY" scripts/05_entrenar_profundos.py --modelo transformer
    paso "5/7 Versión 1 frente a versión 2 (decisión por validación)"; "$PY" scripts/11_comparar_versiones.py --restaurar
    paso "6/7 Comparativa y app";   "$PY" scripts/04_comparar_modelos.py && "$PY" scripts/06_preparar_app.py
    paso "7/7 Pruebas";             "$PY" -m pytest -q
    echo; echo "Listo en $(( ($(date +%s) - inicio) / 60 )) minutos."
    ;;
  *) echo "Uso: bash scripts/utilidades/07_reentrenar_v2.sh piloto|completo"; exit 1 ;;
esac
