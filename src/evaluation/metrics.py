"""
Métricas de desempeño y de costo, iguales para los cuatro modelos.

Por qué un módulo común
-----------------------
Una comparación solo es justa si todos los modelos se miden con el mismo código.
Aquí están las métricas comprometidas en la Presentación del Proyecto (F1 macro,
F1 ponderado, exactitud, latencia) y la que añadió el EDA: el F1 sobre la prueba
sin secuencias repetidas, porque el 30,3 % de los flujos de octubre repite una
secuencia ya vista en septiembre y eso puede inflar el resultado.
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def performance(y_true, y_pred) -> dict:
    """F1 macro (métrica principal), F1 ponderado y exactitud.

    F1 macro promedia todas las clases por igual: con un desbalance de 145 a 1,
    es la única de las tres que no premia ignorar las clases pequeñas.
    """
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def f1_without_repeats(y_true, y_pred, test_hashes, train_hashes) -> dict:
    """F1 macro calculado solo sobre los flujos de prueba cuya secuencia NO aparece en entrenamiento.

    test_hashes y train_hashes son las huellas SEQ_HASH que calcula el script de extracción.
    Si el F1 cae mucho respecto del total, el modelo estaba memorizando secuencias.
    """
    keep = ~np.isin(np.asarray(test_hashes), np.asarray(train_hashes))
    return {
        "f1_macro_sin_repetidos": float(f1_score(np.asarray(y_true)[keep], np.asarray(y_pred)[keep], average="macro")),
        "flujos_evaluados": int(keep.sum()),
        "fraccion_excluida": float(1 - keep.mean()),
    }


def latency_ms_per_flow(predict_fn, X, batch_size: int = 1024, repeats: int = 5) -> float:
    """Latencia media por flujo, en milisegundos, medida en lotes y en CPU.

    Se descarta la primera ejecución (calentamiento: carga de memoria y compilación)
    y se usa la mediana de varias repeticiones, que es robusta a picos del sistema.
    """
    batch = X[:batch_size]
    predict_fn(batch)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(batch)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) / len(batch) * 1000)


def count_parameters(model) -> int:
    """Parámetros entrenables de un modelo de PyTorch (el Transformer debe tener ≤ 2 M)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
