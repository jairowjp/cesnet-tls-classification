#!/usr/bin/env bash
# =============================================================================
# 02_inspeccionar_dataset.sh — Muestra la estructura y el contenido real del dataset
# -----------------------------------------------------------------------------
# Origen: inspección hecha el 26/09/2026 antes de escribir el EDA, para no suponer
# nombres de columnas ni formatos. Lee directamente del zip, sin descomprimirlo.
# Hallazgos que produjo: 1 203 archivos organizados por semana y día, 45 columnas por
# flujo, la columna CATEGORY ya incluida y 122 días (9,7 GB) entre septiembre y diciembre.
#
# Uso (desde la raíz del repositorio):  bash scripts/utilidades/02_inspeccionar_dataset.sh
# =============================================================================
set -euo pipefail
ZIP="datasets/CESNET-TLS-Year22.zip"
DIA="CESNET-TLS-Year22/WEEK-2022-18/2022-05-08"

echo "== Estructura (primeras entradas y total)"
unzip -l "$ZIP" | head -15
unzip -l "$ZIP" | tail -1

echo "== Mapa de servicios"
head -3 datasets/servicemap.csv; echo "Líneas: $(wc -l < datasets/servicemap.csv)"

echo "== Resumen diario (JSON)"
unzip -p "$ZIP" "$DIA/stats-20220508.json" | head -c 600; echo

echo "== Columnas del CSV de flujos"
unzip -p "$ZIP" "$DIA/flows-20220508.csv.xz" | xz -dc | head -1 | tr ',' '\n' | nl | head -50

echo "== Días y tamaño de septiembre a diciembre de 2022"
unzip -l "$ZIP" | grep -E "flows-2022(09|10|11|12)[0-9]{2}\.csv\.xz" \
  | awk '{n++; s+=$1} END {print n, "días,", s/1e9, "GB comprimidos"}'
