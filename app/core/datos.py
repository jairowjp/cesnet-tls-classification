"""
Carga de datos y del modelo de la aplicación.

Todo se carga una sola vez y queda en caché:
  * st.cache_data     → tablas (se copian en cada uso, así una página no altera los datos de otra)
  * st.cache_resource → el modelo XGBoost (un único objeto compartido, sin copiar todo el modelo)

La aplicación solo LEE archivos generados por el propio proyecto y versionados en el repositorio.
Si falta alguno, se informa qué script lo genera, sin mostrar trazas internas (OWASP A05).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from core.seguridad import verificar_integridad

ROOT = Path(__file__).resolve().parents[2]  # raíz del repositorio
APP = ROOT / "app"
MODELOS = ["random_forest", "xgboost", "cnn1d", "transformer"]
NOMBRES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "cnn1d": "CNN 1D", "transformer": "Transformer"}


class DatosFaltantes(Exception):
    """Falta un archivo que la aplicación necesita; el mensaje indica cómo generarlo."""


def _exigir(path: Path, como_generarlo: str) -> Path:
    if not path.exists():
        raise DatosFaltantes(f"Falta {path.relative_to(ROOT)}. Genéralo con: {como_generarlo}")
    return path


@st.cache_data(show_spinner=False)
def muestra_octubre() -> pd.DataFrame:
    """Flujos de prueba (octubre de 2022) con las predicciones precalculadas de los 4 modelos."""
    flujos = pd.read_parquet(
        _exigir(ROOT / "data/samples/muestra_prueba_octubre.parquet", "python scripts/02_crear_muestra_app.py")
    )
    pred = pd.read_parquet(_exigir(APP / "assets/predicciones_muestra.parquet", "python scripts/06_preparar_app.py"))
    if len(pred) != len(flujos):
        raise DatosFaltantes(
            "Las predicciones no corresponden a la muestra. Ejecuta: python scripts/06_preparar_app.py"
        )
    df = pd.concat([flujos.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    df["CATEGORY"] = df["CATEGORY"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def clases_precalculadas() -> list[str]:
    """Orden de las categorías en las columnas prob_<modelo>_<k> (el mismo para los cuatro modelos)."""
    return json.loads(
        _exigir(APP / "assets/clases.json", "python scripts/06_preparar_app.py").read_text(encoding="utf-8")
    )


def probabilidades(fila, modelo: str, clases: list[str]):
    """Probabilidades precalculadas de un modelo para un flujo, como serie indexada por categoría."""
    return pd.Series([float(fila[f"prob_{modelo}_{k}"]) for k in range(len(clases))], index=clases)


@st.cache_resource(show_spinner=False)
def modelo_xgboost() -> dict:
    """Modelo recomendado, cargado SOLO si su huella SHA-256 coincide con la registrada (OWASP A08)."""
    archivo = _exigir(APP / "models/xgboost.joblib", "python scripts/06_preparar_app.py")
    entrada = verificar_integridad(archivo, APP / "models/manifest.json")
    paquete = joblib.load(archivo)  # seguro solo porque la línea anterior verificó el archivo
    paquete["sha256"] = entrada["sha256"]
    return paquete


@st.cache_data(show_spinner=False)
def eda() -> dict:
    """Resultados del EDA (results/eda), generados por notebooks/01_eda.ipynb."""
    base = ROOT / "results/eda"
    gen = "jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb"
    return {
        "resumen": json.loads(_exigir(base / "resumen_eda.json", gen).read_text(encoding="utf-8")),
        "clases": pd.read_csv(_exigir(base / "clases_por_particion.csv", gen), index_col=0),
        "firmas": pd.read_csv(_exigir(base / "firma_media_por_categoria.csv", gen), index_col=0),
        "cat_mes": pd.read_csv(_exigir(base / "categorias_por_mes_anio.csv", gen), index_col=0),
        "deriva_js": pd.read_csv(_exigir(base / "deriva_js_anio.csv", gen), index_col=0),
    }


@st.cache_data(show_spinner=False)
def comparativa() -> dict:
    """Métricas, reportes por clase, matrices de confusión y prueba de McNemar de los modelos."""
    comp = ROOT / "results/comparativa"
    gen = "python scripts/04_comparar_modelos.py"
    metricas, reportes, matrices = {}, {}, {}
    for m in MODELOS:
        d = ROOT / "results/modelos" / m
        if (d / "metricas.json").exists():
            metricas[m] = json.loads((d / "metricas.json").read_text(encoding="utf-8"))
            reportes[m] = pd.read_csv(d / "reporte_por_clase.csv", index_col=0)
            matrices[m] = pd.read_csv(d / "matriz_confusion.csv", index_col=0)
    if not metricas:
        raise DatosFaltantes("No hay modelos evaluados en results/modelos/. Entrena al menos uno.")
    return {
        "tabla": pd.read_csv(_exigir(comp / "tabla_comparativa.csv", gen), index_col=0),
        "mcnemar": pd.read_csv(_exigir(comp / "mcnemar.csv", gen)),
        "metricas": metricas,
        "reportes": reportes,
        "matrices": matrices,
    }


@st.cache_data(show_spinner=False)
def diagnostico() -> dict | None:
    """Resultados del diagnóstico de sobreajuste y subajuste (scripts/09_diagnostico.py, semana 3).

    Devuelve None si todavía no se ejecutó el diagnóstico: la página lo informa sin mostrar un error.
    """
    base = ROOT / "results/diagnostico"
    if not (base / "comparacion_antes_despues.csv").exists():
        return None
    historiales = {}
    for modelo in ("xgboost", "cnn1d", "transformer"):
        for momento in ("antes", "despues"):
            ruta = base / modelo / momento / "historial.csv"
            if ruta.exists():
                historiales[(modelo, momento)] = pd.read_csv(ruta)
    rf = base / "random_forest/curvas.json"
    return {
        "comparacion": pd.read_csv(base / "comparacion_antes_despues.csv"),
        "historiales": historiales,
        "rf": json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else None,
    }


def mostrar_error_datos(e: Exception) -> None:
    """Mensaje claro y accionable cuando falta un archivo, sin trazas internas."""
    st.error(str(e) if isinstance(e, DatosFaltantes) else "No se pudieron cargar los datos de esta página.")
    st.stop()
