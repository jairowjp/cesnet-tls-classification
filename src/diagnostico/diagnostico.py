"""
Diagnóstico sistemático (componente 3 de la actividad, 25 %).

Convierte los patrones visuales de la sección 3.1 de la actividad en métricas y reglas explícitas,
para que el diagnóstico sea reproducible y no dependa de "mirar la curva".

Métricas (sección 3.2):
  brecha_final        F1 de entrenamiento − F1 de validación al final. Grande → sobreajuste.
  brecha_relativa     brecha_final / F1 de entrenamiento.
  pendiente_val       pendiente de la pérdida de validación en el último 30 % del entrenamiento,
                      normalizada por su valor medio. Positiva → la validación empeora.
  pendiente_ent       lo mismo para la pérdida de entrenamiento.
  mejora_final        cuánto subió el F1 de validación en el último 30 %. Positiva → aún aprendía.
  mejor_es_ultimo     True si el mejor F1 de validación está en el último 10 % de los pasos.
  f1_entrenamiento    nivel alcanzado: si es bajo, el modelo no logra ni ajustar los datos que ve.

Reglas (umbrales en UMBRALES, documentados en el informe):
  SOBREAJUSTE  brecha_final > brecha_alta, o la pérdida de validación sube mientras la de
               entrenamiento baja (pendiente_val > 0 y pendiente_ent < 0).
  SUBAJUSTE    f1_entrenamiento < f1_bajo con brecha pequeña, o el modelo seguía mejorando al
               terminar (mejor_es_ultimo y mejora_final > mejora_minima).
  ÓPTIMO       ninguna de las anteriores: brecha pequeña y estable, curvas convergidas.
  MIXTO        señales de ambos (por ejemplo, brecha grande y aún mejorando).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UMBRALES = {
    "brecha_alta": 0.05,  # 5 puntos de F1 entre entrenamiento y validación
    "f1_bajo": 0.80,  # por debajo, el modelo no ajusta bien ni siquiera el entrenamiento
    "mejora_minima": 0.005,  # medio punto de F1 en el tramo final = todavía estaba aprendiendo
    "pendiente_minima": 0.002,  # pendiente normalizada que se considera tendencia real (no ruido)
}


def _pendiente(y: np.ndarray) -> float:
    """Pendiente de una recta ajustada, normalizada por el valor medio (cambio relativo por paso)."""
    if len(y) < 3 or np.allclose(y.mean(), 0):
        return 0.0
    return float(np.polyfit(np.arange(len(y)), y, 1)[0] / abs(y.mean()))


def metricas_curvas(hist: pd.DataFrame, tramo: float = 0.3) -> dict:
    """Métricas cuantitativas a partir de un historial por paso (columnas del módulo de tracking)."""
    f1 = hist[["paso", "f1_entrenamiento", "f1_validacion"]].dropna()
    n = len(hist)
    cola = hist.iloc[max(0, int(n * (1 - tramo))) :]
    cola_f1 = f1[f1["paso"] >= cola["paso"].min()]
    mejor_paso = int(f1.loc[f1["f1_validacion"].idxmax(), "paso"])
    ultimo_paso = int(hist["paso"].max())
    f1_ent, f1_val = float(f1["f1_entrenamiento"].iloc[-1]), float(f1["f1_validacion"].iloc[-1])
    return {
        "pasos": ultimo_paso,
        "f1_entrenamiento": round(f1_ent, 4),
        "f1_validacion": round(f1_val, 4),
        "mejor_f1_validacion": round(float(f1["f1_validacion"].max()), 4),
        "mejor_paso": mejor_paso,
        "mejor_es_ultimo": bool(mejor_paso >= ultimo_paso - max(1, int(0.1 * ultimo_paso))),
        "brecha_final": round(f1_ent - f1_val, 4),
        "brecha_relativa": round((f1_ent - f1_val) / max(f1_ent, 1e-9), 4),
        "pendiente_val": round(_pendiente(cola["perdida_validacion"].dropna().to_numpy()), 5),
        "pendiente_ent": round(_pendiente(cola["perdida_entrenamiento"].dropna().to_numpy()), 5),
        "mejora_final": round(
            float(cola_f1["f1_validacion"].iloc[-1] - cola_f1["f1_validacion"].iloc[0]) if len(cola_f1) > 1 else 0.0, 4
        ),
    }


def metricas_tamano(tamanos, puntaje_ent, puntaje_val) -> dict:
    """Métricas a partir de una curva por tamaño del dataset (modelos sin épocas, como Random Forest)."""
    me, mv = puntaje_ent.mean(1), puntaje_val.mean(1)
    return {
        "f1_entrenamiento": round(float(me[-1]), 4),
        "f1_validacion": round(float(mv[-1]), 4),
        "brecha_final": round(float(me[-1] - mv[-1]), 4),
        "brecha_relativa": round(float((me[-1] - mv[-1]) / max(me[-1], 1e-9)), 4),
        "brecha_inicial": round(float(me[0] - mv[0]), 4),
        "mejora_final": round(float(mv[-1] - mv[-2]), 4) if len(mv) > 1 else 0.0,  # ¿más datos aún ayudan?
        "mejor_es_ultimo": bool(np.argmax(mv) == len(mv) - 1),
        "pendiente_val": 0.0,
        "pendiente_ent": 0.0,
    }


def diagnosticar(m: dict, umbrales: dict | None = None) -> tuple[str, list[str]]:
    """Clasifica el comportamiento y devuelve (diagnóstico, razones en lenguaje claro)."""
    u = {**UMBRALES, **(umbrales or {})}
    sobre, sub = [], []
    if m["brecha_final"] > u["brecha_alta"]:
        sobre.append(f"brecha final de {m['brecha_final']:.3f} (mayor que {u['brecha_alta']})")
    if m["pendiente_val"] > u["pendiente_minima"] and m["pendiente_ent"] < -u["pendiente_minima"]:
        sobre.append("la pérdida de validación sube mientras la de entrenamiento baja")
    if m["f1_entrenamiento"] < u["f1_bajo"] and m["brecha_final"] <= u["brecha_alta"]:
        sub.append(f"F1 de entrenamiento bajo ({m['f1_entrenamiento']:.3f} < {u['f1_bajo']}) con brecha pequeña")
    if m["mejor_es_ultimo"] and m["mejora_final"] > u["mejora_minima"]:
        sub.append(f"seguía mejorando al terminar (+{m['mejora_final']:.3f} de F1 de validación en el tramo final)")
    if sobre and sub:
        return "mixto", sobre + sub
    if sobre:
        return "sobreajuste", sobre
    if sub:
        return "subajuste", sub
    return "óptimo", [f"brecha pequeña ({m['brecha_final']:.3f}) y curvas estabilizadas al final del entrenamiento"]
