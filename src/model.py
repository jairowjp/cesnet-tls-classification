"""
Modelos del proyecto (punto de entrada que exige la estructura del proyecto final).

Reúne la construcción de los cuatro modelos comparados y la carga del modelo publicado. No duplica código:
delega en src/models/classic.py (Random Forest y XGBoost) y en src/models/deep.py (CNN 1D y Transformer).

Los hiperparámetros viven en configs/experiment.yaml: `modelos` es la versión vigente y `modelos_v1` la
original. La optimización con Optuna de la semana 5 está en src/train.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib

from src.config import ROOT, load_config
from src.models.classic import build_random_forest, build_xgboost, fit_random_forest, fit_xgboost  # noqa: F401

MEJOR_MODELO = ROOT / "models" / "best_model.pkl"


def construir(nombre: str, n_clases: int, parametros: dict | None = None):
    """Construye un modelo sin entrenar con los hiperparámetros de la configuración (o los indicados)."""
    if nombre not in ("random_forest", "xgboost", "cnn1d", "transformer"):
        raise ValueError(f"Modelo desconocido: {nombre}. Opciones: random_forest, xgboost, cnn1d, transformer")
    cfg = load_config()
    params = parametros if parametros is not None else cfg["modelos"][nombre]
    if nombre == "random_forest":
        return build_random_forest(params, cfg["semilla"])
    if nombre == "xgboost":
        return build_xgboost(params, cfg["semilla"], n_clases)
    from src.models.deep import build_deep_model  # cnn1d o transformer: PyTorch solo se importa si hace falta

    return build_deep_model(nombre, params, n_clases)


def huella(ruta: Path) -> str:
    """SHA-256 de un archivo: permite comprobar que el modelo no fue alterado antes de cargarlo."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_mejor_modelo(ruta: Path = MEJOR_MODELO, sha256: str | None = None) -> dict:
    """Carga el modelo publicado ({"modelo", "clases", "variables"}), verificando su huella si se indica.

    Un archivo joblib puede ejecutar código al cargarse, así que nunca se carga uno cuya huella no coincida.
    """
    if sha256 and huella(ruta) != sha256:
        raise ValueError(f"La huella SHA-256 de {ruta.name} no coincide: el archivo pudo ser alterado")
    return joblib.load(ruta)
