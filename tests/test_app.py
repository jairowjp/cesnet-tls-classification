"""Pruebas de la aplicación web: el punto de entrada app/app.py y el lanzador de compatibilidad."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ROOT / "app/models/xgboost.joblib").exists(), reason="faltan los artefactos de la app"
)


@pytest.mark.parametrize("entrada", ["app/app.py", "app/principal.py"])
def test_la_app_arranca_sin_errores(entrada):
    """app.py es la app; principal.py es el lanzador que usa Streamlit Cloud. Ambos deben funcionar igual."""
    at = AppTest.from_file(str(ROOT / entrada), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert any("Clasificar tráfico cifrado" in t.value for t in at.title)


def test_la_portada_muestra_el_modelo_recomendado():
    at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=120)
    at.run()
    assert any("XGBoost" in m.value for m in at.markdown)
