"""
Elimina los archivos Parquet de la muestra que no están registrados en el manifiesto.

Origen del problema (27/09/2026)
--------------------------------
En la primera ejecución de scripts/01_extraer_muestra.py, un día sin flujos detuvo el
proceso. Los procesos paralelos que ya estaban trabajando terminaron y guardaron 18 días
en Parquet, pero el manifiesto no llegó a registrarlos (le falta su total de flujos).
Sin ese total, el EDA calcularía mal la población y la fracción muestreada.

Qué hace
--------
Borra solo esos Parquet "huérfanos"; al volver a ejecutar 01_extraer_muestra.py, se
regeneran con su registro completo. La versión 2 del script de extracción ya evita el
problema, pero esta utilidad queda como herramienta de reparación.

Uso:  python scripts/utilidades/03_limpiar_parquet_huerfanos.py [--aplicar]
      Sin --aplicar solo muestra lo que borraría (modo seguro por defecto).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--muestra", default="data/processed/muestra_2pct")
    ap.add_argument("--aplicar", action="store_true", help="borrar de verdad (sin esto solo se simula)")
    a = ap.parse_args()

    d = Path(a.muestra)
    manifest = json.loads((d / "manifest.json").read_text())
    huerfanos = sorted(p for p in d.glob("*.parquet") if p.stem not in manifest)
    print(f"Parquet sin registro en el manifiesto: {len(huerfanos)}")
    for p in huerfanos:
        print(("  borrado: " if a.aplicar else "  se borraría: ") + p.name)
        if a.aplicar:
            p.unlink()
    if huerfanos and not a.aplicar:
        print("Modo simulación: vuelve a ejecutar con --aplicar para borrarlos.")
    elif huerfanos:
        print("Listo. Ejecuta de nuevo scripts/01_extraer_muestra.py para regenerarlos.")


if __name__ == "__main__":
    main()
