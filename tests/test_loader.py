"""Pruebas de la construcción de entradas: formas, exclusiones y máscara de relleno."""
import numpy as np
import pandas as pd

from src.data import loader


def _flujo_sintetico(n=4):
    """Crea n flujos con todas las columnas que espera el cargador.

    Las columnas se reúnen primero en un diccionario y el DataFrame se crea una sola vez:
    agregarlas una por una fragmenta la memoria y pandas emite advertencias de rendimiento.
    """
    rng = np.random.default_rng(0)
    cols = {c: rng.integers(0, 2, n) for c in loader.FLAGS}
    for c in loader.FLOW_STATS + loader.END_REASONS + loader.PHIST:
        cols[c] = rng.integers(1, 100, n)
    for i in range(loader.SEQ):
        activo = i < 5                        # 5 paquetes reales y 25 posiciones de relleno
        cols[f"SIZE_{i}"] = np.full(n, 500 if activo else 0)
        cols[f"DIR_{i}"] = np.full(n, (1 if i % 2 == 0 else -1) if activo else 0)
        cols[f"IPT_{i}"] = np.full(n, 10 if activo else 0)
    cols["CATEGORY"] = pd.Categorical(["Media", "Social", "Media", "Mail"][:n])
    cols["TLS_SNI"] = ["no-debe-usarse.example"] * n  # campo prohibido: no debe aparecer en las entradas
    return pd.DataFrame(cols)


def test_rama_tabular_tiene_60_variables_y_excluye_campos_prohibidos():
    X = loader.tabular_features(_flujo_sintetico())
    assert X.shape == (4, 60)
    assert "TLS_SNI" not in loader.TABULAR and "DST_ASN" not in loader.TABULAR


def test_rama_secuencia_forma_y_mascara_de_relleno():
    X, mask = loader.sequence_features(_flujo_sintetico())
    assert X.shape == (4, 30, 3)
    assert mask[:, :5].sum() == 0 and mask[:, 5:].all()   # relleno marcado solo donde no hay paquete


def test_etiquetas_con_orden_fijo_de_clases():
    y, classes = loader.labels(_flujo_sintetico())
    assert classes == ["Mail", "Media", "Social"] and y.tolist() == [1, 2, 1, 0]
