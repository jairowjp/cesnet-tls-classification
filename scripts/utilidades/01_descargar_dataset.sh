#!/usr/bin/env bash
# =============================================================================
# 01_descargar_dataset.sh — Descarga CESNET-TLS-Year22 desde Zenodo y verifica su integridad
# -----------------------------------------------------------------------------
# Origen: descarga realizada el 26/09/2026 (3 h 43 min, 28,4 GB).
# Por qué desde Zenodo: el servidor de la librería DataZoo (liberouter.org) respondía
# con error 503 y tiempo de espera agotado. Zenodo es el registro oficial del mismo
# dataset, publicado por sus autores: doi:10.5281/zenodo.10608607 (CC BY 4.0).
#
# Uso (desde la raíz del repositorio):  bash scripts/utilidades/01_descargar_dataset.sh
# Recomendado ejecutarlo dentro de tmux: la descarga es larga y así no se interrumpe.
# =============================================================================
set -euo pipefail

URL="https://zenodo.org/records/10608607/files"
MD5_OFICIAL="d0dd7c84e2140bba362f6bd23de5cab7"     # publicado en la página de Zenodo
mkdir -p datasets && cd datasets

# -c reanuda si se corta; --tries=0 reintenta sin límite; --read-timeout evita quedarse congelado
wget -c --tries=0 --read-timeout=60 --waitretry=10 \
     -O CESNET-TLS-Year22.zip "$URL/CESNET-TLS-Year22.zip?download=1"
wget -q -O servicemap.csv "$URL/servicemap.csv?download=1"   # mapa servicio → categoría

echo "Verificando integridad (tarda uno o dos minutos)..."
MD5_LOCAL=$(md5sum CESNET-TLS-Year22.zip | cut -d' ' -f1)
if [[ "$MD5_LOCAL" == "$MD5_OFICIAL" ]]; then
  echo "OK: el MD5 coincide con el oficial ($MD5_OFICIAL)"
else
  echo "ERROR: MD5 $MD5_LOCAL distinto del oficial. Vuelve a ejecutar el script para reanudar."; exit 1
fi
