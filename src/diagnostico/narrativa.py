"""
Análisis en lenguaje claro de las curvas de aprendizaje, calculado a partir de sus métricas.

Lo usan el informe técnico (scripts/10_informe_diagnostico.py) y el notebook overfitting_analysis.ipynb, para
que ambos digan exactamente lo mismo y ninguna afirmación sobre la forma de una curva esté escrita a mano.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.diagnostico.diagnostico import UMBRALES, metricas_curvas


def n(x, k=3) -> str:
    """Número con coma decimal."""
    return f"{float(x):.{k}f}".replace(".", ",")


def analisis_red(R: Path, modelo: str) -> str:
    """Describe las curvas de una red antes y después, a partir de sus métricas."""
    partes = []
    for momento in ("antes", "despues"):
        ruta = R / modelo / momento / "historial.csv"
        if not ruta.exists():
            continue
        m = metricas_curvas(pd.read_csv(ruta))
        tramo = "seguían subiendo" if m["mejora_final"] > UMBRALES["mejora_minima"] else "se habían estabilizado"
        val = "descendía" if m["pendiente_val"] < 0 else "subía"
        partes.append(
            f"{'Antes' if momento == 'antes' else 'Después'} de la estrategia, en {m['pasos']} épocas, el mejor F1 de "
            f"validación ({n(m['mejor_f1_validacion'])}) se alcanzó en la época {m['mejor_paso']}; "
            f"al final, las curvas "
            f"{tramo}, con una brecha de {n(m['brecha_final'])}, y la pérdida de validación {val} en el tramo final."
        )
    return " ".join(partes)


def analisis_xgb(R: Path) -> str:
    ruta = R / "xgboost/antes/historial.csv"
    if not ruta.exists():
        return ""
    m = metricas_curvas(pd.read_csv(ruta))
    ent = "sigue bajando" if m["pendiente_ent"] < -UMBRALES["pendiente_minima"] else "se estabiliza"
    if m["pendiente_val"] > UMBRALES["pendiente_minima"]:
        val = "repunta"
    elif m["pendiente_val"] < -UMBRALES["pendiente_minima"]:
        val = "todavía desciende"
    else:
        val = "se aplana"
    return (
        f"En el tramo final, la pérdida de entrenamiento {ent} mientras la de validación {val}; la brecha final de "
        f"F1 es {n(m['brecha_final'])} y el mejor punto de validación está en la ronda {m['mejor_paso']} de "
        f"{m['pasos']}."
    )


def analisis_rf(rf: dict) -> str:
    val = [sum(v) / len(v) for v in rf["validacion"]]
    tam = rf["tamanos"]
    crece = val[-1] - val[-2] > UMBRALES["mejora_minima"]

    def miles(x):  # separador de miles con espacio, sin tocar las comas decimales del resto del texto
        return f"{int(x):,}".replace(",", " ")

    texto = (
        f"La curva por tamaño muestra que el F1 de validación pasa de {n(val[0])} con {miles(tam[0])} flujos a "
        f"{n(val[-1])} con {miles(tam[-1])}, y que en el último tramo "
        f"{'sigue creciendo: más datos todavía ayudan' if crece else 'ya se estabiliza'}. "
    )
    for param, hp in rf["hiperparametros"].items():
        mejor = hp["valores"][max(range(len(hp["validacion"])), key=lambda i: hp["validacion"][i])]
        texto += (
            f"En la curva de {param}, la mejor validación ({n(max(hp['validacion']))}) se obtiene con "
            f"{'sin límite' if mejor is None else mejor}, y la estrategia eligió "
            f"{'sin límite' if hp['elegido'] is None else hp['elegido']}. "
        )
    return texto
