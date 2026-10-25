"""Pruebas de src/data_processing.py: variables, secuencias, etiquetas y exclusiones de privacidad."""

import numpy as np
import pandas as pd
import pytest

from src import data_processing as dp
from src.data.loader import DIR, IPT, SIZE


@pytest.fixture
def flujos():
    """Tres flujos sintéticos con todas las columnas que usa el proyecto."""
    df = pd.DataFrame({c: [9.0, 99.0, 0.0] for c in dp.TABULAR})
    for i, (s, d, t) in enumerate(zip(SIZE, DIR, IPT, strict=True)):
        df[s] = [1500 if i < 10 else 0, 750, 0]
        df[d] = [1 if i < 10 else 0, -1, 0]  # 0 = posición de relleno
        df[t] = [9.0, 0.0, 0.0]
    df["CATEGORY"] = ["Mail", "Social", "Mail"]
    return df


def test_variables_tabulares_tienen_60_columnas_y_comprimen_las_asimetricas(flujos):
    X = dp.variables_tabulares(flujos)
    assert X.shape == (3, 60) and not np.isnan(X).any()
    i = dp.TABULAR.index("BYTES")  # BYTES está en SKEWED: log10(1 + 9) = 1 y log10(1 + 99) = 2
    assert np.allclose(X[:2, i], [1.0, 2.0])
    j = dp.TABULAR.index("FLAG_ACK")  # las banderas no se transforman
    assert np.allclose(X[:2, j], [9.0, 99.0])


def test_secuencias_y_mascara_de_relleno(flujos):
    X, mascara = dp.secuencias(flujos)
    assert X.shape == (3, dp.SEQ, 3) and mascara.shape == (3, dp.SEQ)
    assert X[0, 0, 0] == pytest.approx(1.0)  # 1 500 bytes normalizados a 1
    assert not mascara[0, :10].any() and mascara[0, 10:].all()
    assert mascara[2].all()  # un flujo sin paquetes queda entero como relleno


def test_etiquetas_con_orden_fijo_y_reutilizable(flujos):
    y, clases = dp.etiquetas(flujos)
    assert clases == ["Mail", "Social"] and y.tolist() == [0, 1, 0]
    y2, _ = dp.etiquetas(flujos.iloc[[1]], clases)  # la misma categoría conserva su número en otra partición
    assert y2.tolist() == [1]
    with pytest.raises(ValueError):
        dp.etiquetas(pd.DataFrame({"CATEGORY": ["Games"]}), clases)


def test_no_se_usan_identificadores_ni_la_variable_que_filtraba_la_respuesta():
    """Privacidad y validez: ni IP, ni SNI, ni JA3, ni el ASN de destino forman parte de las variables."""
    prohibidas = {"SRC_IP", "DST_IP", "TLS_SNI", "TLS_JA3", "DST_ASN", "SRC_ASN"}
    assert not prohibidas & set(dp.TABULAR)  # nombres exactos: PHIST_SRC_IPT (tiempos entre paquetes) sí es válida
