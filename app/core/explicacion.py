"""
Abstención y explicación de las predicciones (mitigaciones R1 y R3 del informe ético de la semana 4).

Abstención
----------
Si la confianza del modelo en su respuesta es menor que un umbral, la aplicación no clasifica el flujo: dice
"No clasificado: confianza baja". El modelo está bien calibrado (error de calibración de 0,019), así que su
confianza es una buena señal de cuándo duda. El umbral lo calcula scripts/16_calibrar_abstencion.py con los
datos de VALIDACIÓN y lo guarda en app/assets/umbral_abstencion.json; si ese archivo no existe, se usa 0,5.

Explicación
-----------
Para cada predicción se muestran las variables que más empujaron hacia la categoría elegida. Son valores SHAP
(TreeSHAP) calculados por el propio XGBoost con `pred_contribs=True`, así que no hace falta instalar otra
librería. Un valor positivo acerca el flujo a esa categoría; uno negativo lo aleja.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import TABULAR

UMBRAL_POR_DEFECTO = 0.5
ARCHIVO_UMBRAL = Path(__file__).resolve().parents[1] / "assets" / "umbral_abstencion.json"

_LADO = {"SRC": "del cliente", "DST": "del servidor"}
_ESTADISTICAS = {
    "DURATION": "Duración del flujo",
    "BYTES": "Bytes enviados por el cliente",
    "BYTES_REV": "Bytes enviados por el servidor",
    "PACKETS": "Paquetes enviados por el cliente",
    "PACKETS_REV": "Paquetes enviados por el servidor",
    "PPI_LEN": "Paquetes en la secuencia inicial",
    "PPI_DURATION": "Duración de la secuencia inicial",
    "PPI_ROUNDTRIPS": "Idas y vueltas en la secuencia inicial",
}
_FIN = {"IDLE": "inactividad", "ACTIVE": "tiempo máximo activo", "END": "cierre normal", "OTHER": "otro motivo"}


def nombre_legible(variable: str) -> str:
    """Traduce el nombre técnico de una de las 60 variables a una frase comprensible."""
    if variable in _ESTADISTICAS:
        return _ESTADISTICAS[variable]
    if variable.startswith("FLAG_"):
        bandera = variable.removeprefix("FLAG_").removesuffix("_REV")
        return f"Bandera TCP {bandera} {'del servidor' if variable.endswith('_REV') else 'del cliente'}"
    if variable.startswith("FLOW_ENDREASON_"):
        return f"Fin del flujo por {_FIN.get(variable.rsplit('_', 1)[1], 'otro motivo')}"
    if variable.startswith("PHIST_"):
        _, lado, tipo, k = variable.split("_")
        que = "Paquetes" if tipo == "SIZES" else "Esperas entre paquetes"
        rango = "de tamaño" if tipo == "SIZES" else "de duración"
        return f"{que} {_LADO[lado]} {rango} en el intervalo {int(k) + 1} de 8"
    return variable


@lru_cache(maxsize=1)
def umbral_abstencion() -> dict:
    """Umbral de confianza para abstenerse y de dónde salió (validación o valor por defecto)."""
    if ARCHIVO_UMBRAL.exists():
        return json.loads(ARCHIVO_UMBRAL.read_text(encoding="utf-8"))
    return {"umbral": UMBRAL_POR_DEFECTO, "fuente": "valor por defecto (falta calibrarlo con validación)"}


def decidir(confianza: np.ndarray, umbral: float) -> np.ndarray:
    """True donde el modelo debe abstenerse porque su confianza es menor que el umbral."""
    return np.asarray(confianza) < umbral


def explicar(modelo, X: np.ndarray, clase: int, top: int = 5) -> pd.DataFrame:
    """Las `top` variables que más pesaron en la predicción de `clase` para un flujo (valores SHAP de XGBoost)."""
    import xgboost as xgb

    contrib = modelo.get_booster().predict(xgb.DMatrix(np.asarray(X, dtype=float)[:1]), pred_contribs=True)
    contrib = np.asarray(contrib)
    if contrib.ndim == 3:  # multiclase: (flujos, clases, variables + sesgo)
        valores = contrib[0, clase, :-1]
    else:  # binario: (flujos, variables + sesgo)
        valores = contrib[0, :-1]
    orden = np.argsort(-np.abs(valores))[:top]
    return pd.DataFrame(
        {
            "Variable": [nombre_legible(TABULAR[j]) for j in orden],
            "Aporte": valores[orden],
            "Técnica": [TABULAR[j] for j in orden],
        }
    )
