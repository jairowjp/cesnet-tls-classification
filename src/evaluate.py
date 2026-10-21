"""
Evaluación robusta del modelo (Semana 5 y proyecto final).

Reúne en un solo módulo todo lo que se necesita para evaluar con rigor un clasificador multiclase de
tráfico cifrado:

  * Métricas multiclase, incluida la del dominio: el F1 de la PEOR categoría (el F1 macro puede ocultar
    un servicio mal clasificado; ver el informe de ética de la semana 4).
  * Baseline "dummy" y validación cruzada AGRUPADA: el 30,3 % de las secuencias de paquetes se repite,
    así que una secuencia nunca puede quedar a la vez en entrenamiento y validación (evita la fuga).
  * Intervalos de confianza por bootstrap y pruebas estadísticas pareadas (Wilcoxon, diferencia de F1).
  * Escenarios críticos del Workshop 1, adaptados al tráfico de red, y el relleno de paquetes (TAREA 2).
  * Análisis de errores (pares confundidos, longitud del flujo, calibración, secuencias repetidas) y
    equidad entre categorías con Fairlearn.

Las funciones reciben arreglos de NumPy y no leen archivos: los scripts deciden qué datos usar.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.data.loader import END_REASONS, FLAGS, SKEWED, TABULAR

log = logging.getLogger(__name__)

Perturbacion = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]
Predictor = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]  # X -> (predicciones, probabilidades)

# Variables binarias (banderas TCP y motivo de fin de flujo): no tiene sentido sumarles ruido gaussiano
BINARIAS = [TABULAR.index(c) for c in FLAGS + END_REASONS]
CONTINUAS = [i for i in range(len(TABULAR)) if i not in BINARIAS]


# ======================================================================================== métricas
def _auc_macro(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """AUC-ROC y AUC-PR uno contra el resto, promediadas sobre las clases presentes en y.

    Se calculan clase por clase para no exigir que las 23 categorías aparezcan en cada submuestra.
    """
    roc, pr = [], []
    for c in np.unique(y):
        positivo = y == c
        if 0 < positivo.sum() < len(y):
            roc.append(roc_auc_score(positivo, proba[:, c]))
            pr.append(average_precision_score(positivo, proba[:, c]))
    return float(np.mean(roc)), float(np.mean(pr))


def metricas(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    """Métricas multiclase del proyecto. El F1 de la peor categoría se calcula sobre las clases presentes."""
    presentes = np.unique(y)
    f1_por_clase = f1_score(y, pred, labels=presentes, average=None, zero_division=0)
    res = {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, labels=presentes, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, labels=presentes, average="macro", zero_division=0),
        "precision_macro": precision_score(y, pred, labels=presentes, average="macro", zero_division=0),
        "f1_peor_categoria": float(f1_por_clase.min()),
        "peor_categoria": int(presentes[f1_por_clase.argmin()]),
    }
    if proba is not None:
        res["auc_ovr"], res["auc_pr"] = _auc_macro(y, proba)
    return {k: (float(v) if k != "peor_categoria" else v) for k, v in res.items()}


# ======================================================================== baseline y validación
def submuestra_estratificada(y: np.ndarray, n: int, semilla: int) -> np.ndarray:
    """Índices de una submuestra estratificada por clase (todas las clases conservan su proporción)."""
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(semilla)
    idx = []
    for c in np.unique(y):
        miembros = np.flatnonzero(y == c)
        k = max(1, int(round(n * len(miembros) / len(y))))
        idx.append(rng.choice(miembros, min(k, len(miembros)), replace=False))
    return np.sort(np.concatenate(idx))


def pliegues_agrupados(y: np.ndarray, grupos: np.ndarray, pliegues: int, repeticiones: int, semilla: int):
    """Genera (entrenamiento, validación) estratificados y AGRUPADOS por huella de secuencia.

    Las secuencias idénticas (misma SEQ_HASH) caen siempre en el mismo pliegue: el modelo nunca se
    valida con una secuencia que ya vio al entrenar. Cada repetición usa otra semilla.
    """
    for r in range(repeticiones):
        cv = StratifiedGroupKFold(n_splits=pliegues, shuffle=True, random_state=semilla + r)
        for k, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y, grupos)):
            yield r, k, tr, va


def validacion_cruzada(
    entrenar: Callable, X, y, grupos, pliegues: int, repeticiones: int, semilla: int
) -> pd.DataFrame:
    """Evalúa un modelo con validación cruzada agrupada. `entrenar(Xtr, ytr, Xva, yva)` devuelve un Predictor."""
    filas = []
    for r, k, tr, va in pliegues_agrupados(y, grupos, pliegues, repeticiones, semilla):
        predecir = entrenar(X[tr], y[tr], X[va], y[va])
        pred, proba = predecir(X[va])
        filas.append({"repeticion": r, "pliegue": k, **metricas(y[va], pred, proba)})
        log.info("CV repetición %d pliegue %d: F1 macro %.4f", r + 1, k + 1, filas[-1]["f1_macro"])
    return pd.DataFrame(filas)


# ========================================================== intervalos y pruebas estadísticas
def ic_bootstrap_media(valores, n: int = 2000, semilla: int = 42) -> dict:
    """Media, desviación e IC del 95 % por bootstrap sobre puntajes de pliegues (igual que el notebook)."""
    valores = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = [rng.choice(valores, len(valores)).mean() for _ in range(n)]
    bajo, alto = np.percentile(medias, [2.5, 97.5])
    return {
        "media": float(valores.mean()),
        "desviacion": float(valores.std()),
        "ic95_bajo": float(bajo),
        "ic95_alto": float(alto),
    }


def ic_bootstrap_metrica(y, pred, n: int = 1000, semilla: int = 42, metrica: str = "f1_macro") -> dict:
    """IC del 95 % de una métrica sobre predicciones de prueba, remuestreando flujos."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(semilla)
    valores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        valores.append(metricas(y[i], pred[i])[metrica])
    bajo, alto = np.percentile(valores, [2.5, 97.5])
    return {"valor": metricas(y, pred)[metrica], "ic95_bajo": float(bajo), "ic95_alto": float(alto)}


def ic_diferencia(y, pred_a, pred_b, n: int = 1000, semilla: int = 42) -> dict:
    """IC del 95 % de la DIFERENCIA de F1 macro (B − A), remuestreando los mismos flujos para ambos.

    Si el intervalo no contiene el 0, la diferencia no se explica por el azar del muestreo.
    """
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(semilla)
    dif = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        dif.append(metricas(y[i], pred_b[i])["f1_macro"] - metricas(y[i], pred_a[i])["f1_macro"])
    bajo, alto = np.percentile(dif, [2.5, 97.5])
    observada = metricas(y, pred_b)["f1_macro"] - metricas(y, pred_a)["f1_macro"]
    return {
        "diferencia": float(observada),
        "ic95_bajo": float(bajo),
        "ic95_alto": float(alto),
        "significativa": bool(bajo > 0 or alto < 0),
    }


def wilcoxon_pareado(puntajes_a, puntajes_b) -> dict:
    """Prueba de Wilcoxon de rangos con signo sobre los mismos pliegues (B − A)."""
    a, b = np.asarray(puntajes_a, float), np.asarray(puntajes_b, float)
    if np.allclose(a, b):
        return {"estadistico": 0.0, "p_valor": 1.0, "mediana_diferencia": 0.0, "significativo": False}
    est, p = wilcoxon(b, a)
    return {
        "estadistico": float(est),
        "p_valor": float(p),
        "mediana_diferencia": float(np.median(b - a)),
        "significativo": bool(p < 0.05),
    }


# ================================================================ escenarios críticos (Workshop 1)
def ruido(sigma: float, desviacion: np.ndarray) -> Perturbacion:
    """A · Ruido gaussiano proporcional a la desviación de cada variable CONTINUA (no a las banderas)."""

    def f(X, y, rng):
        X2 = X.copy()
        X2[:, CONTINUAS] += rng.normal(0, 1, (len(X), len(CONTINUAS))) * desviacion[CONTINUAS] * sigma
        return X2, y

    return f


def faltantes(p: float, mediana: np.ndarray) -> Perturbacion:
    """B · Una fracción p de los valores se pierde y se imputa con la mediana de entrenamiento."""

    def f(X, y, rng):
        X2 = X.copy()
        mascara = rng.random(X.shape) < p
        X2[mascara] = np.broadcast_to(mediana, X.shape)[mascara]
        return X2, y

    return f


def dominancia(clase: int, fraccion: float) -> Perturbacion:
    """C · Desbalance extremo: una categoría pasa a ocupar `fraccion` del tráfico de prueba.

    El modelo no cambia, solo el mundo donde se usa: la exactitud se acerca al acierto de esa categoría
    mientras el F1 macro sigue midiendo las demás.
    """

    def f(X, y, rng):
        dom, resto = np.flatnonzero(y == clase), np.flatnonzero(y != clase)
        n_dom = int(round(len(y) * fraccion))
        idx = np.concatenate(
            [
                rng.choice(dom, n_dom, replace=True),
                rng.choice(resto, len(y) - n_dom, replace=len(y) - n_dom > len(resto)),
            ]
        )
        return X[idx], y[idx]

    return f


def variable_caida(columnas: list[int], mediana: np.ndarray) -> Perturbacion:
    """E · Las variables indicadas dejan de llegar y se reciben con un valor constante (la mediana)."""

    def f(X, y, rng):
        X2 = X.copy()
        X2[:, columnas] = mediana[columnas]
        return X2, y

    return f


def relleno(fraccion: float, mtu: int = 1500) -> Perturbacion:
    """TAREA 2 · Relleno de paquetes (padding): una fracción de los flujos rellena cada paquete hasta la MTU.

    Es una técnica real de evasión (y también de privacidad). Efecto sobre las 60 variables tabulares:
      * BYTES y BYTES_REV pasan a ser PACKETS × 1 500 y PACKETS_REV × 1 500.
      * Los histogramas de tamaños (PHIST_SRC_SIZES y PHIST_DST_SIZES) concentran todos los paquetes en el
        último intervalo, el de los paquetes grandes.
    Los tiempos y las banderas no cambian. Las variables de SKEWED están en escala log10(1 + x).
    """
    i = {n: TABULAR.index(n) for n in TABULAR}

    def bruto(col):
        """Deshace la transformación log10(1 + x) que tabular_features aplica a las variables asimétricas."""
        return 10**col - 1

    def f(X, y, rng):
        X2 = X.copy()
        afectados = rng.random(len(X)) < fraccion
        for bytes_col, paq_col, hist in (
            ("BYTES", "PACKETS", "PHIST_SRC_SIZES"),
            ("BYTES_REV", "PACKETS_REV", "PHIST_DST_SIZES"),
        ):
            paquetes = bruto(X2[afectados, i[paq_col]]) if paq_col in SKEWED else X2[afectados, i[paq_col]]
            nuevos_bytes = paquetes * mtu
            X2[afectados, i[bytes_col]] = np.log10(1 + nuevos_bytes) if bytes_col in SKEWED else nuevos_bytes
            cols = [i[f"{hist}_{k}"] for k in range(8)]
            total = X2[np.ix_(afectados, cols)].sum(axis=1)
            X2[np.ix_(afectados, cols[:-1])] = 0
            X2[afectados, cols[-1]] = total
        return X2, y

    return f


def evaluar_perturbado(predecir: Predictor, X, y, perturbar: Perturbacion, repeticiones: int, semilla: int) -> dict:
    """Aplica la perturbación con `repeticiones` semillas distintas y promedia las métricas (notebook)."""
    res = []
    for r in range(repeticiones):
        X2, y2 = perturbar(X, y, np.random.default_rng(semilla + r))
        pred, proba = predecir(X2)
        res.append(metricas(y2, pred, proba))
    return pd.DataFrame(res).drop(columns="peor_categoria").mean().to_dict()


def veredicto(tabla: pd.DataFrame, f1_base: float, umbral: float) -> pd.DataFrame:
    """Agrega la caída de F1 macro respecto de la base y el veredicto OK / FALLA según el umbral fijado."""
    t = tabla.copy()
    t["caida_f1"] = f1_base - t["f1_macro"]
    t["veredicto"] = np.where(t["caida_f1"] <= umbral, "OK", "FALLA")
    return t


# ============================================================================ análisis de errores
def pares_confundidos(y, pred, clases: list[str], top: int = 10) -> pd.DataFrame:
    """Los pares (real → predicha) que más errores concentran, con su peso dentro de la categoría real."""
    y, pred = np.asarray(y), np.asarray(pred)
    err = y != pred
    t = pd.DataFrame({"real": y[err], "predicha": pred[err]}).value_counts().rename("errores").reset_index()
    soporte = pd.Series(y).value_counts()
    t["fraccion_de_la_categoria"] = t["errores"] / t["real"].map(soporte)
    t["fraccion_de_todos_los_errores"] = t["errores"] / err.sum()
    t["real"] = t["real"].map(lambda c: clases[c])
    t["predicha"] = t["predicha"].map(lambda c: clases[c])
    return t.head(top)


def error_por_tramo(y, pred, valores, etiqueta: str, tramos: int = 5) -> pd.DataFrame:
    """Tasa de error y F1 macro por cuantiles de una variable (por ejemplo, la longitud del flujo)."""
    df = pd.DataFrame({"y": y, "pred": pred, "v": valores})
    df["tramo"] = pd.qcut(df["v"], tramos, duplicates="drop")
    filas = []
    for tramo, g in df.groupby("tramo", observed=True):
        filas.append(
            {
                etiqueta: str(tramo),
                "flujos": len(g),
                "tasa_error": float((g.y != g.pred).mean()),
                "f1_macro": metricas(g.y.to_numpy(), g.pred.to_numpy())["f1_macro"],
            }
        )
    return pd.DataFrame(filas)


def calibracion(y, proba, intervalos: int = 10) -> tuple[pd.DataFrame, float]:
    """Diagrama de confiabilidad y error de calibración esperado (ECE) de la confianza máxima.

    Un modelo bien calibrado acierta el 90 % de las veces que dice "90 % de confianza": es la base
    para abstenerse de clasificar cuando la confianza es baja (mitigación R1 del informe de ética).
    """
    confianza, pred = proba.max(axis=1), proba.argmax(axis=1)
    acierto = pred == np.asarray(y)
    bordes = np.linspace(0, 1, intervalos + 1)
    filas, ece = [], 0.0
    for a, b in zip(bordes[:-1], bordes[1:], strict=True):
        m = (confianza > a) & (confianza <= b)
        if m.any():
            filas.append(
                {
                    "intervalo": f"({a:.1f}, {b:.1f}]",
                    "flujos": int(m.sum()),
                    "confianza_media": float(confianza[m].mean()),
                    "acierto": float(acierto[m].mean()),
                }
            )
            ece += m.mean() * abs(confianza[m].mean() - acierto[m].mean())
    return pd.DataFrame(filas), float(ece)


def equidad_por_categoria(y, pred, clases: list[str]) -> tuple[pd.DataFrame, dict]:
    """Métricas por categoría de servicio con Fairlearn (MetricFrame) y sus brechas.

    El sistema no decide sobre personas, así que el "grupo" pertinente es el servicio (informe de ética,
    sección 3.3). La tasa de acierto por grupo equivale al recall de esa categoría (igualdad de oportunidad).
    """
    from fairlearn.metrics import MetricFrame

    nombres = np.asarray(clases)[np.asarray(y)]
    mf = MetricFrame(metrics={"tasa_acierto": accuracy_score}, y_true=y, y_pred=pred, sensitive_features=nombres)
    por_grupo = mf.by_group.sort_values("tasa_acierto")
    brechas = {
        "diferencia_max": float(mf.difference(method="between_groups")["tasa_acierto"]),
        "razon_min": float(mf.ratio(method="between_groups")["tasa_acierto"]),
        "peor_categoria": str(por_grupo.index[0]),
        "tasa_peor": float(por_grupo.iloc[0, 0]),
    }
    return por_grupo, brechas
