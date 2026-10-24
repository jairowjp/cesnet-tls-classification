"""
Exporta los modelos a la carpeta models/ con los nombres que exige el proyecto final y escribe su README.

  models/best_model.pkl   El modelo recomendado (por defecto, XGBoost versión 2; con --version 3, el optimizado
                          con Optuna en la semana 5).
  models/model_v1.pkl     XGBoost versión 1, para comparar y para poder revertir.
  models/README.md        Qué es cada archivo, su desempeño, su tamaño y su huella SHA-256.

Los archivos .pkl son los mismos objetos que guardan los scripts de entrenamiento ({"modelo", "clases",
"variables"}) y se cargan con joblib.load o con src.model.cargar_mejor_modelo, que verifica la huella.

Uso:  python scripts/14_exportar_modelos.py [--version 2|3]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT  # noqa: E402
from src.model import huella  # noqa: E402

MODELOS = ROOT / "models"
ORIGEN = {2: MODELOS / "xgboost.joblib", 3: MODELOS / "xgboost_optuna.joblib", 1: MODELOS / "v1" / "xgboost.joblib"}
LIMITE_GITHUB_MB = 95


def f1_de(version: int) -> tuple[str, str]:
    """F1 macro de validación y de octubre registrados para cada versión (de los resultados del proyecto)."""
    if version == 3:
        r = json.loads((ROOT / "results/semana5/optuna/reentrenamiento.json").read_text())
        c = json.loads((ROOT / "results/semana5/cierre/comparacion.json").read_text())
        return f"{r['validacion_optimizado']['f1_macro']:.3f}", f"{c['ic_optimizado']['valor']:.3f}"
    carpeta = "modelos" if version == 2 else "modelos_v1"
    m = json.loads((ROOT / f"results/{carpeta}/xgboost/metricas.json").read_text())
    return f"{m['validacion']['f1_macro']:.3f}", f"{m['prueba_octubre']['f1_macro']:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, choices=[2, 3], default=2, help="versión que se publica como best_model.pkl")
    a = ap.parse_args()
    filas = []
    for destino, version in (("best_model.pkl", a.version), ("model_v1.pkl", 1)):
        origen = ORIGEN[version]
        if not origen.exists():
            sys.exit(f"No existe {origen.relative_to(ROOT)}. Entrena primero esa versión.")
        mb = origen.stat().st_size / 1e6
        if mb > LIMITE_GITHUB_MB:
            sys.exit(f"{origen.name} pesa {mb:.1f} MB: supera el límite seguro de {LIMITE_GITHUB_MB} MB de GitHub.")
        shutil.copy2(origen, MODELOS / destino)
        val, oct_ = f1_de(version)
        filas.append((destino, version, val, oct_, mb, huella(MODELOS / destino)))
        print(f"{destino:<16} ← XGBoost v{version} · {mb:.1f} MB · F1 validación {val} · octubre {oct_}")
    tabla = "\n".join(
        f"| `{d}` | XGBoost versión {v} | {val} | {o} | {mb:.1f} MB | `{h[:16]}…` |" for d, v, val, o, mb, h in filas
    )
    (MODELOS / "README.md").write_text(
        f"""# Modelos entrenados

| Archivo | Modelo | F1 macro (validación) | F1 macro (octubre) | Tamaño | SHA-256 |
|---|---|---|---|---|---|
{tabla}

- **Validación:** el 20 % de septiembre de 2022. **Octubre:** 872 731 flujos que ningún modelo vio al entrenar.
- Cada archivo contiene un diccionario `{{"modelo", "clases", "variables"}}`: el clasificador de XGBoost, las 23
  categorías en orden y las 60 variables de entrada (ver `src/data_processing.py`).
- Para cargarlo con verificación de integridad:
  ```python
  from src.model import cargar_mejor_modelo
  paquete = cargar_mejor_modelo(sha256="<huella completa de la tabla>")
  ```
  Un archivo de este tipo puede ejecutar código al cargarse: no cargues nunca uno de origen desconocido.
- Los modelos de Random Forest, CNN 1D y Transformer no se publican aquí por su tamaño o porque no se recomiendan;
  sus métricas están en `results/modelos/` y se regeneran con los scripts 03 y 05.
- Generado con `python scripts/14_exportar_modelos.py --version {a.version}`.
""",
        encoding="utf-8",
    )
    print("models/README.md actualizado")


if __name__ == "__main__":
    main()
