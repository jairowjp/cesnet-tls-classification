"""Pruebas unitarias de las métricas y de la prueba de McNemar.

Se ejecutan con:  pytest -q
Cada prueba usa datos pequeños con resultado conocido, así un cambio accidental en
el cálculo de una métrica se detecta antes de afectar la comparación de modelos.
"""
import numpy as np

from src.evaluation.metrics import f1_without_repeats, latency_ms_per_flow, performance
from src.evaluation.stats import mcnemar_test


def test_performance_prediccion_perfecta():
    y = np.array([0, 1, 2, 1, 0])
    r = performance(y, y)
    assert r["f1_macro"] == 1.0 and r["accuracy"] == 1.0


def test_f1_sin_repetidos_excluye_secuencias_vistas():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])          # acierta en los dos primeros, falla en los dos últimos
    test_h = np.array([10, 20, 30, 40])
    train_h = np.array([30, 40])             # los dos flujos fallados estaban "repetidos"
    r = f1_without_repeats(y_true, y_pred, test_h, train_h)
    assert r["flujos_evaluados"] == 2 and r["f1_macro_sin_repetidos"] == 1.0


def test_latencia_es_positiva():
    X = np.zeros((2048, 4))
    assert latency_ms_per_flow(lambda b: b.sum(axis=1), X) >= 0


def test_mcnemar_modelos_identicos_no_es_significativo():
    y = np.array([0, 1] * 50)
    assert not mcnemar_test(y, y, y)["significativo"]


def test_mcnemar_detecta_diferencia_grande():
    y = np.zeros(200, dtype=int)
    a = y.copy()                              # A acierta todo
    b = np.ones(200, dtype=int)               # B falla todo
    r = mcnemar_test(y, a, b)
    assert r["significativo"] and r["mejor"] == "A"
