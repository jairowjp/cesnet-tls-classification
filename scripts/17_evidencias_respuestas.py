"""
Evidencias para el video de respuestas del proyecto final.

  A · Ranking de variables (pregunta 9). Importancia de las 60 variables del modelo publicado con dos métodos:
      la ganancia de XGBoost y el valor SHAP medio absoluto sobre flujos de octubre.
  B · Experimento con el ASN de destino (pregunta 2). Dos XGBoost idénticos, entrenados con la misma submuestra de
      septiembre, con y sin DST_ASN, evaluados en octubre. Muestra qué habría pasado si no se hubiera excluido.
  C · Compresión del modelo optimizado (pregunta 10). Tamaño del modelo de 154 MB con varios métodos de compresión
      frente al límite de 100 MB de GitHub.

No modifica ningún modelo ni resultado existente: todo se guarda en results/evidencias/.
Uso:  python scripts/17_evidencias_respuestas.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from core.explicacion import nombre_legible  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.loader import TABULAR  # noqa: E402
from src.evaluate import metricas  # noqa: E402
from src.model import cargar_mejor_modelo  # noqa: E402
from src.train import entrenador_xgboost  # noqa: E402
from src.utils import configurar_logging, datos_entrenamiento, datos_prueba, guardar  # noqa: E402

CFG = load_config()
SALIDA = ROOT / "results" / "evidencias"
MUESTRA_ENTRENAMIENTO, MUESTRA_PRUEBA, MUESTRA_SHAP = 60_000, 100_000, 3_000
LIMITE_GITHUB_MB = 100


def ranking_variables(log, paquete, X_oct) -> pd.DataFrame:
    """A · Ganancia de XGBoost y SHAP medio absoluto (promedio sobre flujos y categorías) de cada variable."""
    booster = paquete["modelo"].get_booster()
    ganancia = booster.get_score(importance_type="gain")
    gan = np.array([ganancia.get(f"f{i}", 0.0) for i in range(len(TABULAR))])
    rng = np.random.default_rng(CFG["semilla"])
    idx = rng.choice(len(X_oct), min(MUESTRA_SHAP, len(X_oct)), replace=False)
    t0 = time.time()
    contrib = np.asarray(booster.predict(xgb.DMatrix(X_oct[idx]), pred_contribs=True))  # (flujos, clases, 61)
    shap_medio = np.abs(contrib[..., :-1]).mean(axis=tuple(range(contrib.ndim - 1)))
    log.info("SHAP sobre %d flujos de octubre en %.1f min", len(idx), (time.time() - t0) / 60)
    tabla = pd.DataFrame(
        {
            "variable": TABULAR,
            "descripcion": [nombre_legible(v) for v in TABULAR],
            "ganancia_%": 100 * gan / gan.sum(),
            "shap_%": 100 * shap_medio / shap_medio.sum(),
        }
    )
    tabla["puesto_ganancia"] = tabla["ganancia_%"].rank(ascending=False).astype(int)
    tabla["puesto_shap"] = tabla["shap_%"].rank(ascending=False).astype(int)
    tabla = tabla.sort_values("shap_%", ascending=False).reset_index(drop=True)
    guardar(tabla, SALIDA / "importancia_variables.csv")

    fig, ejes = plt.subplots(1, 2, figsize=(15, 6.5))
    for eje, col, titulo in (
        (ejes[0], "shap_%", "Valor SHAP medio (octubre)"),
        (ejes[1], "ganancia_%", "Ganancia de XGBoost"),
    ):
        top = tabla.sort_values(col, ascending=False).head(15)[::-1]
        eje.barh(top["descripcion"], top[col], color="#2B59C3")
        eje.set_title(titulo)
        eje.set_xlabel("% de la importancia total")
        eje.tick_params(axis="y", labelsize=8)
        eje.grid(alpha=0.3, axis="x")
    fig.suptitle("Las 15 variables más importantes del modelo publicado, según dos métodos")
    fig.tight_layout()
    fig.savefig(SALIDA / "importancia_variables.png", dpi=150)
    plt.close(fig)
    comunes = set(tabla.nsmallest(10, "puesto_shap").variable) & set(tabla.nsmallest(10, "puesto_ganancia").variable)
    log.info(
        "Top 5 por SHAP: %s · coinciden %d de las 10 primeras entre ambos métodos",
        list(tabla.head(5).variable),
        len(comunes),
    )
    return tabla


def experimento_asn(log, clases) -> dict:
    """B · Mismo XGBoost con y sin el ASN de destino, entrenado en septiembre y evaluado en octubre."""
    X, y, _, _, df = datos_entrenamiento(MUESTRA_ENTRENAMIENTO)
    if "DST_ASN" not in df.columns:
        log.warning("La muestra no tiene la columna DST_ASN: el experimento B se omite")
        return {"realizado": False, "motivo": "la muestra no contiene DST_ASN"}
    X_te, y_te, df_te = datos_prueba(clases, MUESTRA_PRUEBA)
    asn_tr = df["DST_ASN"].to_numpy(dtype=float)[:, None]
    asn_te = df_te["DST_ASN"].to_numpy(dtype=float)[:, None]
    parametros = CFG["modelos"]["xgboost"]
    resultados = {
        "realizado": True,
        "flujos_entrenamiento": int(len(y)),
        "flujos_prueba": int(len(y_te)),
        "asn_distintos_entrenamiento": int(df["DST_ASN"].nunique()),
    }
    for nombre, Xa, Xb in (("sin_asn", X, X_te), ("con_asn", np.hstack([X, asn_tr]), np.hstack([X_te, asn_te]))):
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=CFG["semilla"])
        t0 = time.time()
        entrenar = entrenador_xgboost(parametros, CFG["semilla"], len(clases))
        predecir = entrenar(Xa[tr], y[tr], Xa[va], y[va])
        pred, proba = predecir(Xb)
        resultados[nombre] = {k: v for k, v in metricas(y_te, pred, proba).items() if k != "peor_categoria"}
        log.info(
            "%s: F1 macro en octubre %.4f (%.1f min)", nombre, resultados[nombre]["f1_macro"], (time.time() - t0) / 60
        )
    # Cuánto pesa el ASN en el modelo que lo usa: se reentrena una vez más para leer su ganancia
    ajustes = {k: v for k, v in parametros.items() if k not in ("early_stopping_rounds", "n_estimators")}
    modelo = xgb.XGBClassifier(**ajustes, n_estimators=200, random_state=CFG["semilla"])
    Xc = np.hstack([X, asn_tr])
    modelo.fit(Xc, y)
    gan = modelo.get_booster().get_score(importance_type="gain")
    total = sum(gan.values())
    resultados["peso_asn_%"] = 100 * gan.get(f"f{Xc.shape[1] - 1}", 0.0) / total
    resultados["diferencia_f1_puntos"] = 100 * (resultados["con_asn"]["f1_macro"] - resultados["sin_asn"]["f1_macro"])
    guardar(resultados, SALIDA / "experimento_asn.json")
    log.info(
        "Con ASN %+.2f puntos de F1 · el ASN concentra el %.1f %% de la ganancia",
        resultados["diferencia_f1_puntos"],
        resultados["peso_asn_%"],
    )
    return resultados


def compresion(log) -> pd.DataFrame | None:
    """C · Tamaño del modelo optimizado con distintos métodos de compresión."""
    ruta = ROOT / "models" / "xgboost_optuna.joblib"
    if not ruta.exists():
        log.warning("No existe models/xgboost_optuna.joblib: la prueba C se omite")
        return None
    paquete = joblib.load(ruta)
    filas = [{"metodo": "Actual (joblib, zlib nivel 3)", "MB": ruta.stat().st_size / 1e6, "segundos": 0.0}]
    with tempfile.TemporaryDirectory() as tmp:
        for etiqueta, compress in (("joblib, zlib nivel 9", ("zlib", 9)), ("joblib, lzma nivel 9", ("lzma", 9))):
            destino = Path(tmp) / "m.joblib"
            t0 = time.time()
            joblib.dump(paquete, destino, compress=compress)
            filas.append({"metodo": etiqueta, "MB": destino.stat().st_size / 1e6, "segundos": time.time() - t0})
        for formato in ("ubj", "json"):
            destino = Path(tmp) / f"m.{formato}"
            t0 = time.time()
            paquete["modelo"].get_booster().save_model(destino)
            filas.append(
                {
                    "metodo": f"Formato nativo de XGBoost (.{formato}, sin comprimir)",
                    "MB": destino.stat().st_size / 1e6,
                    "segundos": time.time() - t0,
                }
            )
    tabla = pd.DataFrame(filas)
    tabla["cabe_en_github"] = tabla["MB"] < LIMITE_GITHUB_MB
    guardar(tabla, SALIDA / "compresion_modelo.csv")
    for _, f in tabla.iterrows():
        log.info("%-48s %7.1f MB · %s", f.metodo, f.MB, "cabe" if f.cabe_en_github else "NO cabe")
    return tabla


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    log = configurar_logging("17_evidencias")
    t0 = time.time()
    paquete = cargar_mejor_modelo()
    clases = list(paquete["clases"])
    X_oct, _, _ = datos_prueba(clases, MUESTRA_PRUEBA)
    resumen = {}
    for nombre, paso in (
        ("A", lambda: ranking_variables(log, paquete, X_oct)),
        ("B", lambda: experimento_asn(log, clases)),
        ("C", lambda: compresion(log)),
    ):
        try:
            paso()
            resumen[nombre] = "completado"
        except Exception as e:  # una parte que falle no impide las demás
            log.exception("Falló la parte %s", nombre)
            resumen[nombre] = f"error: {e}"
    guardar(resumen, SALIDA / "resumen.json")
    log.info("Evidencias terminadas en %.1f min · %s", (time.time() - t0) / 60, json.dumps(resumen, ensure_ascii=False))


if __name__ == "__main__":
    main()
