"""
Prepara los archivos que necesita la aplicación web.

Qué hace
--------
  1. Calcula la predicción de los cuatro modelos sobre la muestra de octubre de la app
     (data/samples/muestra_prueba_octubre.parquet). La app muestra estas predicciones para
     comparar modelos sin necesitar PyTorch: queda más liviana y con menos superficie de ataque.
  2. Copia el modelo recomendado (XGBoost, 36 MB) a app/models/ y registra su huella SHA-256
     en app/models/manifest.json. La app se niega a cargar el modelo si la huella no coincide
     (OWASP A08: integridad de software y datos).

Se ejecuta en el equipo donde están los modelos entrenados (carpeta models/, que no se sube a GitHub).

Uso:  python scripts/06_preparar_app.py
"""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))  # app/core/

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from core.seguridad import sha256_de  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.data.loader import sequence_features, tabular_features  # noqa: E402

MUESTRA = ROOT / "data/samples/muestra_prueba_octubre.parquet"
SALIDA = ROOT / "app/assets/predicciones_muestra.parquet"
APP_MODELS = ROOT / "app/models"


def predecir_clasico(nombre: str, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Probabilidad de cada categoría para cada flujo (Random Forest o XGBoost)."""
    paquete = joblib.load(ROOT / "models" / f"{nombre}.joblib")  # modelo propio, generado por el script 03
    return paquete["modelo"].predict_proba(tabular_features(df)), list(paquete["clases"])


def predecir_profundo(nombre: str, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Probabilidad de cada categoría (softmax de la salida de la red) para cada flujo."""
    import torch  # solo este script necesita PyTorch; la app no

    from src.models.deep import build_deep_model

    # weights_only=False es necesario porque el archivo guarda también la lista de clases;
    # es seguro aquí porque el archivo lo generó el propio proyecto (scripts/05_entrenar_profundos.py)
    ck = torch.load(ROOT / "models" / f"{nombre}.pt", map_location="cpu", weights_only=False)
    model = build_deep_model(nombre, ck["parametros"], len(ck["clases"]))
    model.load_state_dict(ck["estado"])
    model.eval()
    X, M = sequence_features(df)
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(X), torch.from_numpy(M)), dim=1).numpy()
    return prob, list(ck["clases"])


def main():
    if not MUESTRA.exists():
        sys.exit(f"Falta {MUESTRA.relative_to(ROOT)}. Ejecuta primero: python scripts/02_crear_muestra_app.py")
    df = pd.read_parquet(MUESTRA)
    print(f"Muestra de la app: {len(df):,} flujos de octubre, {df['CATEGORY'].nunique()} categorías")

    # ------------------------------------------------------------ 1. predicciones precalculadas
    # Se guardan la categoría predicha y la probabilidad de cada categoría para los cuatro modelos,
    # así la app muestra la confianza de todos lado a lado sin necesitar PyTorch.
    pred, clases_ref = {}, None
    modelos = [("random_forest", predecir_clasico, "joblib"), ("xgboost", predecir_clasico, "joblib"),
               ("cnn1d", predecir_profundo, "pt"), ("transformer", predecir_profundo, "pt")]
    for nombre, fn, ext in modelos:
        if not (ROOT / "models" / f"{nombre}.{ext}").exists():
            print(f"  {nombre:14s} no encontrado en models/: se omite")
            continue
        prob, clases = fn(nombre, df)
        if clases_ref is None:
            clases_ref = clases
        elif clases != clases_ref:  # todas las columnas prob_* deben referirse a las mismas categorías
            sys.exit(f"{nombre} usa un orden de categorías distinto: vuelve a entrenarlo con el protocolo común")
        pred[f"pred_{nombre}"] = np.asarray(clases)[prob.argmax(axis=1)]
        for k in range(len(clases)):
            pred[f"prob_{nombre}_{k}"] = prob[:, k].astype("float32")
        acierto = (pred[f"pred_{nombre}"] == df["CATEGORY"].astype(str).to_numpy()).mean()
        print(f"  {nombre:14s} acierto sobre la muestra de la app: {acierto:.1%}")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred).to_parquet(SALIDA, index=False)
    (SALIDA.parent / "clases.json").write_text(json.dumps(clases_ref, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------------------ 2. modelo de la app + huella
    origen = ROOT / "models/xgboost.joblib"
    if not origen.exists():
        sys.exit("Falta models/xgboost.joblib. Entrénalo con: python scripts/03_entrenar_clasicos.py --modelo xgboost")
    APP_MODELS.mkdir(parents=True, exist_ok=True)
    destino = APP_MODELS / "xgboost.joblib"
    shutil.copy2(origen, destino)
    paquete = joblib.load(destino)
    manifiesto = {
        destino.name: {
            "sha256": sha256_de(destino),
            "bytes": destino.stat().st_size,
            "clases": list(paquete["clases"]),
            "variables": len(paquete["variables"]),
            "origen": "models/xgboost.joblib, generado por scripts/03_entrenar_clasicos.py",
            "preparado": date.today().isoformat(),
        }
    }
    (APP_MODELS / "manifest.json").write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nPredicciones: {SALIDA.relative_to(ROOT)}")
    print(f"Modelo de la app: {destino.relative_to(ROOT)} ({destino.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA-256: {manifiesto[destino.name]['sha256']}")


if __name__ == "__main__":
    main()
