"""
Utilidades compartidas por los scripts: registro de eventos (logging), carga de datos y guardado de resultados.

Centralizar estas funciones evita repetir código (DRY) y garantiza que todos los scripts de la semana 5
usen exactamente la misma submuestra, con la misma semilla.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.loader import labels, load_months, load_split, tabular_features
from src.evaluate import submuestra_estratificada

RESULTADOS = ROOT / "results" / "semana5"


def configurar_logging(nombre: str) -> logging.Logger:
    """Registro con hora en pantalla y en results/semana5/<nombre>.log, para revisar la ejecución nocturna."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    raiz = logging.getLogger()
    raiz.setLevel(logging.INFO)
    raiz.handlers.clear()
    for manejador in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTADOS / f"{nombre}.log", encoding="utf-8"),
    ):
        manejador.setFormatter(formato)
        raiz.addHandler(manejador)
    for ruidoso in ("optuna", "matplotlib", "PIL"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    return logging.getLogger(nombre)


def datos_entrenamiento(n: int | None = None):
    """Septiembre (o una submuestra estratificada de n flujos): X, y, huellas de secuencia, clases y el DataFrame.

    La submuestra se toma siempre con la semilla del proyecto: parte 0, Optuna y la comparación final usan
    exactamente los mismos flujos y, por tanto, los mismos pliegues.
    """
    cfg = load_config()
    df = load_split("entrenamiento")
    y, clases = labels(df)
    if n:
        df = df.iloc[submuestra_estratificada(y, n, cfg["semilla"])].reset_index(drop=True)
        y, _ = labels(df, clases)
    return tabular_features(df), y, df["SEQ_HASH"].to_numpy(), clases, df


def datos_prueba(clases: list[str], n: int | None = None, meses: list[str] | None = None):
    """Octubre por defecto (o los meses indicados), opcionalmente en submuestra estratificada."""
    cfg = load_config()
    df = load_months(meses) if meses else load_split("prueba")
    y, _ = labels(df, clases)
    if n:
        df = df.iloc[submuestra_estratificada(y, n, cfg["semilla"])].reset_index(drop=True)
        y, _ = labels(df, clases)
    return tabular_features(df), y, df


def cargar_modelo(ruta: Path) -> dict:
    """Carga un modelo guardado por los scripts de entrenamiento ({"modelo", "clases", "variables"})."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Entrena primero el modelo (scripts 03 o 13).")
    return joblib.load(ruta)


def predictor(modelo, lote: int = 200_000):
    """Función X -> (predicciones, probabilidades), por lotes para no saturar la memoria."""

    def predecir(X):
        proba = np.vstack([modelo.predict_proba(X[i : i + lote]) for i in range(0, len(X), lote)])
        return proba.argmax(axis=1), proba

    return predecir


def guardar(obj, ruta: Path) -> None:
    """Guarda un DataFrame (CSV) o un diccionario (JSON con acentos legibles)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(ruta, index=False)
    else:
        ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
