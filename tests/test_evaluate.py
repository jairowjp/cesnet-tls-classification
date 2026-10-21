"""Pruebas de src/evaluate.py y src/train.py (semana 5): métricas, agrupación sin fuga, escenarios y estadística."""

import numpy as np
import pytest

from src import evaluate as ev
from src.data.loader import TABULAR


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    X = rng.normal(0, 1, (400, len(TABULAR))) + y[:, None] * 0.5
    grupos = rng.integers(0, 120, 400)  # huellas repetidas, como SEQ_HASH
    return X, y, grupos


def test_f1_peor_categoria_detecta_la_clase_mal_clasificada():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 0, 1])  # la clase 2 nunca se acierta
    m = ev.metricas(y, pred)
    assert m["f1_peor_categoria"] == 0.0 and m["peor_categoria"] == 2 and m["f1_macro"] > 0


def test_pliegues_agrupados_no_reparten_una_secuencia(datos):
    _, y, grupos = datos
    for _, _, tr, va in ev.pliegues_agrupados(y, grupos, 3, 2, 42):
        assert not set(grupos[tr]) & set(grupos[va])


def test_submuestra_estratificada_conserva_las_proporciones(datos):
    _, y, _ = datos
    i = ev.submuestra_estratificada(y, 200, 42)
    assert abs(len(i) - 200) <= 4 and set(y[i]) == set(y)
    assert np.allclose(np.bincount(y[i]) / len(i), np.bincount(y) / len(y), atol=0.02)


def test_intervalo_de_la_diferencia_detecta_mejora_real():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 3000)
    malo = np.where(rng.random(3000) < 0.6, y, (y + 1) % 3)
    bueno = np.where(rng.random(3000) < 0.9, y, (y + 1) % 3)
    d = ev.ic_diferencia(y, malo, bueno, n=200, semilla=0)
    assert d["significativa"] and d["ic95_bajo"] > 0
    igual = ev.ic_diferencia(y, malo, malo, n=200, semilla=0)
    assert not igual["significativa"]


def test_wilcoxon_necesita_suficientes_pliegues():
    """Con 5 pares el p mínimo posible es 0,0625: por eso la comparación final usa los 15 pliegues (5 x 3)."""
    a = np.linspace(0.80, 0.84, 5)
    assert not ev.wilcoxon_pareado(a, a + 0.01)["significativo"]
    b = np.linspace(0.80, 0.84, 15)
    assert ev.wilcoxon_pareado(b, b + 0.01)["significativo"]


def test_escenarios_conservan_forma_y_no_tocan_las_banderas(datos):
    X, y, _ = datos
    rng = np.random.default_rng(0)
    X2, y2 = ev.ruido(1.0, X.std(axis=0))(X, y, rng)
    assert X2.shape == X.shape and np.array_equal(X2[:, ev.BINARIAS], X[:, ev.BINARIAS])
    X3, _ = ev.faltantes(0.5, np.median(X, axis=0))(X, y, rng)
    assert 0.4 < np.mean(X3 != X) < 0.6
    _, y4 = ev.dominancia(0, 0.9)(X, y, rng)
    assert len(y4) == len(y) and abs(np.mean(y4 == 0) - 0.9) < 0.01


def test_relleno_concentra_los_paquetes_en_el_ultimo_intervalo(datos):
    X, y, _ = datos
    X = np.abs(X)
    X2, _ = ev.relleno(1.0)(X, y, np.random.default_rng(0))
    cols = [TABULAR.index(f"PHIST_SRC_SIZES_{k}") for k in range(8)]
    assert np.allclose(X2[:, cols[:-1]], 0)
    assert np.allclose(X2[:, cols[-1]], X[:, cols].sum(axis=1))
    bytes_ = TABULAR.index("BYTES")
    paquetes = TABULAR.index("PACKETS")
    assert np.allclose(X2[:, bytes_], np.log10(1 + (10 ** X[:, paquetes] - 1) * 1500))


def test_veredicto_aplica_el_umbral_fijado():
    import pandas as pd

    t = ev.veredicto(pd.DataFrame({"f1_macro": [0.90, 0.86, 0.80]}), 0.90, 0.05)
    assert list(t["veredicto"]) == ["OK", "OK", "FALLA"]


def test_optuna_optimiza_y_poda(datos, tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    from src import train as tr

    X, y, grupos = datos
    cfg = {
        "pliegues": 2,
        "rondas_maximas": 10,
        "parada_temprana": 3,
        "espacio": {"max_depth": [2, 4], "learning_rate": [0.05, 0.3]},
    }
    estudio = tr.crear_estudio(tmp_path / "e.db", "prueba", 42)
    estudio.optimize(lambda t: tr.objetivo(t, X, y, grupos, cfg, {"tree_method": "hist"}, 42, 4), n_trials=4)
    assert estudio.best_value > 0.25 and set(estudio.best_params) == {"max_depth", "learning_rate"}
    reanudado = tr.crear_estudio(tmp_path / "e.db", "prueba", 42)  # el estudio se reanuda, no se pierde
    assert len(reanudado.trials) == 4


def test_entrenador_tolera_categorias_ausentes_en_el_entrenamiento(datos):
    """Escenario D o un pliegue sin una categoría pequeña: no debe fallar y la categoría ausente cuenta como error."""
    pytest.importorskip("xgboost")
    from src import train as tr

    X, y, _ = datos
    sin_2 = y != 2
    predecir = tr.entrenador_xgboost({"n_estimators": 10, "tree_method": "hist"}, 42, 4)(
        X[sin_2], y[sin_2], X[sin_2], y[sin_2]
    )
    pred, proba = predecir(X)
    assert proba.shape == (len(X), 4) and np.allclose(proba[:, 2], 0) and not np.any(pred == 2)
