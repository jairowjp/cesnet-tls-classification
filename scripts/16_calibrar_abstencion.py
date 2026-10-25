"""
Calibra el umbral de abstención de la aplicación con los datos de VALIDACIÓN (mitigación R1 del informe ético).

Si la confianza del modelo en su respuesta es menor que el umbral, la aplicación no clasifica el flujo. Este
script elige ese umbral con el 20 % de validación de septiembre (el mismo corte del script 03), nunca con la
prueba de octubre, para no ajustar una decisión a los datos con los que se evalúa.

Regla, fijada antes de ver los resultados: entre los umbrales de 0,30 a 0,80, el más bajo (el que menos flujos deja
sin clasificar) con el que el acierto en los flujos clasificados llega al menos a 0,94.

Uso:     python scripts/16_calibrar_abstencion.py
Salida:  app/assets/umbral_abstencion.json (lo lee app/core/explicacion.py) y su tabla en results/metrics/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config  # noqa: E402
from src.data_processing import cargar_particion, etiquetas, variables_tabulares  # noqa: E402
from src.model import cargar_mejor_modelo  # noqa: E402

ACIERTO_MINIMO = 0.94
UMBRALES = np.round(np.arange(0.30, 0.81, 0.05), 2)


def tabla_abstencion(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Para cada umbral: qué fracción de flujos se clasifica y con qué acierto."""
    confianza, pred = proba.max(axis=1), proba.argmax(axis=1)
    filas = []
    for u in UMBRALES:
        clasificados = confianza >= u
        filas.append(
            {
                "umbral": float(u),
                "cobertura": float(clasificados.mean()),
                "acierto": float((pred[clasificados] == y[clasificados]).mean()) if clasificados.any() else 0.0,
            }
        )
    return pd.DataFrame(filas)


def elegir(tabla: pd.DataFrame) -> pd.Series:
    """El umbral más bajo que alcanza el acierto mínimo; si ninguno lo alcanza, el de mayor acierto."""
    validos = tabla[tabla.acierto >= ACIERTO_MINIMO]
    return validos.iloc[0] if len(validos) else tabla.loc[tabla.acierto.idxmax()]


def main():
    cfg = load_config()
    paquete = cargar_mejor_modelo()
    df = cargar_particion("entrenamiento")
    y, _ = etiquetas(df, list(paquete["clases"]))
    X = variables_tabulares(df)
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=cfg["datos"]["fraccion_validacion"], stratify=y, random_state=cfg["semilla"]
    )  # el mismo corte del script 03
    proba = paquete["modelo"].predict_proba(X_val)
    tabla = tabla_abstencion(y_val, proba)
    elegido = elegir(tabla)
    salida = {
        "umbral": float(elegido.umbral),
        "cobertura": float(elegido.cobertura),
        "acierto": float(elegido.acierto),
        "acierto_sin_abstencion": float((proba.argmax(axis=1) == y_val).mean()),
        "flujos_validacion": int(len(y_val)),
        "fuente": "validación de septiembre de 2022 (scripts/16_calibrar_abstencion.py)",
        "regla": f"umbral más bajo con acierto de al menos {ACIERTO_MINIMO} en los flujos clasificados",
    }
    (ROOT / "app/assets/umbral_abstencion.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "results/metrics").mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ROOT / "results/metrics/13_abstencion_validacion.csv", index=False)
    print(tabla.round(4).to_string(index=False))
    print(
        f"\nUmbral elegido: {elegido.umbral:.2f} · cobertura {elegido.cobertura:.1%} · acierto {elegido.acierto:.4f} "
        f"(sin abstención: {salida['acierto_sin_abstencion']:.4f})"
    )


if __name__ == "__main__":
    main()
