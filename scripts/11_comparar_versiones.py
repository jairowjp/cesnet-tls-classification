"""
Compara la versión 1 y la versión 2 de cada modelo y decide cuál conservar.

Regla de decisión (protocolo del proyecto): un modelo v2 reemplaza al v1 SOLO si mejora el F1 macro de
VALIDACIÓN. La prueba de octubre se muestra como resultado, pero no interviene en la decisión, para no ajustar
los modelos a los datos de prueba.

Entradas:  results/modelos_v1/<modelo>/metricas.json   (respaldo de la versión 1)
           results/modelos/<modelo>/metricas.json      (versión 2, recién entrenada)
Salidas:   results/comparativa/versiones.csv y versiones.json

Con --restaurar, para cada modelo cuya v2 NO mejore en validación se devuelven a su lugar los resultados
(results/modelos/<modelo>/) y el modelo entrenado (models/<modelo>.*) de la versión 1.

Uso:  python scripts/11_comparar_versiones.py [--restaurar]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1, V2 = ROOT / "results/modelos_v1", ROOT / "results/modelos"
MODELOS = ["random_forest", "xgboost", "cnn1d", "transformer"]
LIMITE_GITHUB_MB = 90  # GitHub rechaza archivos de más de 100 MB: se avisa con margen


def leer(carpeta: Path, modelo: str) -> dict | None:
    ruta = carpeta / modelo / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restaurar", action="store_true", help="devuelve la v1 de los modelos cuya v2 no mejora")
    a = ap.parse_args()
    if not V1.exists():
        sys.exit(
            "No existe results/modelos_v1/. Respalda primero la versión 1 (scripts/utilidades/07_reentrenar_v2.sh)."
        )

    filas = []
    for m in MODELOS:
        v1, v2 = leer(V1, m), leer(V2, m)
        if not v1 or not v2:
            print(f"{m}: falta la versión {'1' if not v1 else '2'}; se omite")
            continue
        if v2.get("modo_prueba"):
            print(f"{m}: la versión 2 es una prueba (submuestra o pocas épocas); no se compara")
            continue
        val1, val2 = v1["validacion"]["f1_macro"], v2["validacion"]["f1_macro"]
        adopta = val2 > val1
        filas.append(
            {
                "modelo": m,
                "f1_validacion_v1": val1,
                "f1_validacion_v2": val2,
                "decision": "v2" if adopta else "v1",
                "f1_prueba_v1": v1["prueba_octubre"]["f1_macro"],
                "f1_prueba_v2": v2["prueba_octubre"]["f1_macro"],
                "mejora_prueba_puntos": 100 * (v2["prueba_octubre"]["f1_macro"] - v1["prueba_octubre"]["f1_macro"]),
                "tamano_MB_v1": v1["costo"]["tamano_modelo_MB"],
                "tamano_MB_v2": v2["costo"]["tamano_modelo_MB"],
                "memoria_pico_GB_v2": v2["costo"].get("memoria_pico_GB"),
            }
        )
        if not adopta and a.restaurar:
            shutil.rmtree(V2 / m)
            shutil.copytree(V1 / m, V2 / m)
            for archivo in (ROOT / "models/v1").glob(f"{m}.*"):
                shutil.copy2(archivo, ROOT / "models" / archivo.name)
            print(f"{m}: la v2 no mejoró en validación ({val2:.4f} ≤ {val1:.4f}); se restauró la v1")

    if not filas:
        sys.exit("No hay modelos para comparar.")
    tabla = pd.DataFrame(filas)
    salida = ROOT / "results/comparativa"
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "versiones.csv", index=False)
    (salida / "versiones.json").write_text(json.dumps(filas, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print("\n================ VERSIÓN 1 FRENTE A VERSIÓN 2 ================")
    print(tabla.round(4).to_string(index=False))
    print("\nDecisión: se conserva la versión con mejor F1 macro de VALIDACIÓN (la prueba no interviene).")
    xgb = tabla[tabla.modelo == "xgboost"]
    if len(xgb) and xgb.iloc[0]["decision"] == "v2" and xgb.iloc[0]["tamano_MB_v2"] > LIMITE_GITHUB_MB:
        print(
            f"\nAVISO: el XGBoost v2 pesa {xgb.iloc[0]['tamano_MB_v2']} MB. La app lo copia a app/models/ y GitHub "
            f"rechaza archivos de más de 100 MB: revísalo antes del commit."
        )


if __name__ == "__main__":
    main()
