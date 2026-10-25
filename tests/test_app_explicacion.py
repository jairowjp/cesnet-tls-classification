"""Pruebas de las mitigaciones de la app: abstención, explicación de cada predicción y botón de ejemplo."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def test_decidir_se_abstiene_solo_bajo_el_umbral():
    from core.explicacion import decidir

    assert decidir(np.array([0.3, 0.5, 0.9]), 0.5).tolist() == [True, False, False]


def test_las_60_variables_tienen_nombre_comprensible():
    from core.explicacion import nombre_legible

    from src.data.loader import TABULAR

    assert all(nombre_legible(v) != v for v in TABULAR)


def test_la_explicacion_suma_la_decision_del_modelo():
    """Propiedad de los valores SHAP: las contribuciones más el sesgo reproducen el margen del modelo."""
    xgb = pytest.importorskip("xgboost")
    from core.explicacion import explicar

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = rng.normal(0, 1, (300, 60)) + y[:, None]
    modelo = xgb.XGBClassifier(n_estimators=15, max_depth=3).fit(X, y)
    e = explicar(modelo, X[:1], 2, top=60)
    margen = modelo.get_booster().predict(xgb.DMatrix(X[:1]), output_margin=True)[0, 2]
    contrib = modelo.get_booster().predict(xgb.DMatrix(X[:1]), pred_contribs=True)[0, 2]
    assert len(e) == 60 and np.isclose(contrib.sum(), margen, atol=1e-4)


def test_elegir_umbral_con_la_regla_fijada():
    sys.path.insert(0, str(ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location("s16", ROOT / "scripts/16_calibrar_abstencion.py")
    s16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s16)
    y = np.array([0, 1, 0, 1, 0, 1])
    proba = np.array([[0.9, 0.1], [0.45, 0.55], [0.35, 0.65], [0.2, 0.8], [0.95, 0.05], [0.6, 0.4]])
    tabla = s16.tabla_abstencion(y, proba)
    assert tabla.cobertura.is_monotonic_decreasing and 0 <= tabla.acierto.min() <= 1
    assert s16.elegir(tabla).acierto >= s16.ACIERTO_MINIMO or s16.elegir(tabla).acierto == tabla.acierto.max()


@pytest.mark.skipif(not (ROOT / "app/models/xgboost.joblib").exists(), reason="faltan los artefactos de la app")
def test_boton_probar_con_un_ejemplo_clasifica_20_flujos():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=120)
    at.run()
    at.switch_page("paginas/clasificar.py").run()
    [b for b in at.button if b.label == "Probar con un ejemplo"][0].click().run()
    assert not at.exception and len(at.dataframe[-1].value) == 20
