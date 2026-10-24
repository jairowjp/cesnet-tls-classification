"""Pruebas de src/model.py: construcción de los modelos y carga verificada del modelo publicado."""

import joblib
import numpy as np
import pytest

from src import model


def test_construir_xgboost_y_random_forest_con_la_configuracion():
    xgb = model.construir("xgboost", 4)
    rf = model.construir("random_forest", 4)
    assert xgb.get_params()["max_depth"] is not None and rf.get_params()["n_estimators"] > 0


def test_xgboost_entrena_y_predice_probabilidades_por_clase():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    X = rng.normal(0, 1, (200, 60)) + y[:, None]
    m = model.construir("xgboost", 4, {"n_estimators": 20, "max_depth": 3, "tree_method": "hist"})
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (200, 4) and np.allclose(proba.sum(axis=1), 1)


def test_construir_redes_profundas():
    pytest.importorskip("torch")
    assert model.construir("cnn1d", 23) is not None and model.construir("transformer", 23) is not None


def test_modelo_desconocido_da_un_error_claro():
    with pytest.raises(ValueError, match="Modelo desconocido"):
        model.construir("svm", 4)


def test_la_carga_rechaza_un_modelo_alterado(tmp_path):
    """Un archivo joblib puede ejecutar código al cargarse: si la huella no coincide, no se carga."""
    ruta = tmp_path / "best_model.pkl"
    joblib.dump({"modelo": "x", "clases": ["a"], "variables": []}, ruta)
    correcta = model.huella(ruta)
    assert model.cargar_mejor_modelo(ruta, correcta)["clases"] == ["a"]
    with pytest.raises(ValueError, match="no coincide"):
        model.cargar_mejor_modelo(ruta, "0" * 64)
