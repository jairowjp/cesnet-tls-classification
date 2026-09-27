"""
Crea la muestra pequeña que se sube a GitHub y que usará la aplicación web.

Por qué hace falta
------------------
La muestra completa (377 MB) no se sube al repositorio. La aplicación y quien revise
el proyecto necesitan, sin embargo, flujos reales para probar el clasificador. Este
script toma hasta N flujos por categoría del mes de PRUEBA (octubre), que ningún
modelo usa para entrenar, y los guarda en data/samples/ junto con una plantilla CSV
que muestra el formato que acepta la aplicación.

Uso:
    python scripts/02_crear_muestra_app.py            # 300 flujos por categoría
    python scripts/02_crear_muestra_app.py --por-clase 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar src/ al ejecutar el script directamente
from src.config import ROOT, load_config  # noqa: E402
from src.data.loader import load_split  # noqa: E402

# Columnas que se excluyen también de la muestra publicada: la aplicación no las necesita
# y DST_ASN delata la categoría (ver EDA, sección 2.10).
NO_PUBLICAR = ["DST_ASN", "DST_PORT", "PROTOCOL"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--por-clase", type=int, default=300, help="flujos por categoría (máximo)")
    a = ap.parse_args()

    seed = load_config()["semilla"]
    test = load_split("prueba").drop(columns=NO_PUBLICAR, errors="ignore")
    # Se barajan los flujos con semilla fija y se toman los primeros N de cada categoría:
    # equivale a un muestreo aleatorio estratificado, reproducible y estable entre versiones de pandas.
    sample = (test.sample(frac=1.0, random_state=seed)
                  .groupby("CATEGORY", observed=True)
                  .head(a.por_clase)
                  .reset_index(drop=True))
    out = ROOT / "data" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out / "muestra_prueba_octubre.parquet", index=False)

    # La plantilla lleva solo columnas de entrada: sin etiqueta ni datos de identificación
    plantilla = sample.drop(columns=["APP", "CATEGORY", "DATE", "TIME_FIRST", "SEQ_HASH"], errors="ignore").head(5)
    plantilla.to_csv(out / "plantilla_clasificador.csv", index=False)

    size_mb = (out / "muestra_prueba_octubre.parquet").stat().st_size / 1e6
    print(f"Muestra: {len(sample):,} flujos de {sample['CATEGORY'].nunique()} categorías ({size_mb:.1f} MB)")
    print(f"Plantilla CSV: {plantilla.shape[1]} columnas de entrada")


if __name__ == "__main__":
    main()
