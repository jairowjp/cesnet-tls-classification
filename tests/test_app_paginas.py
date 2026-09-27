"""Pruebas de la interfaz: cada página se ejecuta completa con el probador oficial de Streamlit
(AppTest), sin navegador, y se verifica que no produzca errores. También se prueban las
interacciones principales: reproducir otro flujo, alternar métricas y elegir modelos.

Requieren los artefactos de la app (data/samples, app/models, app/assets y results/);
si faltan, las pruebas se omiten en lugar de fallar.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUERIDOS = [
    ROOT / "app/models/xgboost.joblib",
    ROOT / "app/assets/predicciones_muestra.parquet",
    ROOT / "data/samples/muestra_prueba_octubre.parquet",
    ROOT / "results/eda/resumen_eda.json",
    ROOT / "results/comparativa/tabla_comparativa.csv",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUERIDOS), reason="faltan artefactos de la app (ejecuta scripts/06_preparar_app.py)"
)

PAGINAS = [
    "paginas/inicio.py",
    "paginas/datos.py",
    "paginas/modelos.py",
    "paginas/reto.py",
    "paginas/clasificar.py",
    "paginas/conclusiones.py",
    "paginas/encuesta.py",
    "paginas/seguridad.py",
]


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app/principal.py"), default_timeout=120)
    at.run()
    return at


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_se_ejecuta_sin_errores(pagina):
    at = _app()
    at.switch_page(pagina)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]  # tampoco mensajes de error controlados (datos faltantes)
    assert len(at.title) == 1  # cada página tiene un único título principal


def test_inicio_reproduce_otro_flujo():
    at = _app()
    antes = at.session_state["inicio_fila"]
    for _ in range(5):  # con varios intentos, el flujo cambia (elección al azar)
        at.button[0].click().run()
        if at.session_state["inicio_fila"] != antes:
            break
    assert not at.exception and at.session_state["inicio_fila"] != antes


def test_modelos_alterna_excluir_repetidos():
    at = _app()
    at.switch_page("paginas/modelos.py").run()
    at.toggle[0].set_value(True).run()
    assert not at.exception


def test_clasificar_toma_flujo_de_una_categoria():
    at = _app()
    at.switch_page("paginas/clasificar.py").run()
    opciones = at.selectbox[0].options
    at.selectbox[0].set_value(opciones[1]).run()
    assert not at.exception
    assert at.session_state["clasificar_filtro"] == opciones[1]


def test_clasificar_muestra_la_confianza_de_los_cuatro_modelos():
    at = _app()
    at.switch_page("paginas/clasificar.py").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 4  # un gráfico de confianza por modelo


def test_inicio_tiene_portada_institucional():
    """La portada identifica institución, programa, asignatura, autor y docente, y usa el logo servido por la app."""
    at = _app()
    cuerpos = " ".join(h.proto.body for h in at.get("html"))
    for texto in (
        "uees_logo.png",
        "Maestría en Inteligencia Artificial",
        "MIAR0545",
        "Jairo Wladimir Jhayya Perlaza",
        "Gladys María Villegas Rugel",
        "CESNET-TLS-Year22",
    ):
        assert texto in cuerpos, f"falta en la portada: {texto}"
    assert (ROOT / "app/static/uees_logo.png").exists()


def test_reto_completo_de_cinco_rondas():
    """Se juega el reto entero: responder, avanzar y llegar al resultado final."""
    at = _app()
    at.switch_page("paginas/reto.py").run()
    for ronda in range(1, 6):
        assert at.session_state["reto"]["ronda"] == ronda
        at.button(key="reto_opcion_0").click().run()  # responde con la primera opción
        assert not at.exception and at.session_state["reto"]["respuesta"] is not None
        at.button(key="reto_avanzar").click().run()
    assert at.session_state["reto"]["ronda"] == 6  # terminado
    assert 0 <= at.session_state["reto"]["tuyos"] <= 5
    at.button(key="reto_otra_vez").click().run()
    assert at.session_state["reto"]["ronda"] == 1 and not at.exception


def test_encuesta_exige_todas_las_respuestas_y_calcula_el_puntaje():
    at = _app()
    at.switch_page("paginas/encuesta.py").run()
    enviar = next(b for b in at.button if b.label == "Calcular mi puntaje")
    enviar.click().run()
    assert at.error and "Faltan 10" in at.error[0].value  # sin responder: se informa qué falta
    for n in range(1, 11):
        at.radio(key=f"sus_{n}").set_value(5 if n % 2 else 1)
    enviar = next(b for b in at.button if b.label == "Calcular mi puntaje")
    enviar.click().run()
    assert not at.exception and at.session_state["sus_resultado"]["puntaje"] == 100.0
    assert any(c.value.startswith("SUS-") for c in at.code)  # el código se muestra para copiarlo


def test_todas_las_paginas_del_recorrido_invitan_al_siguiente_paso():
    at = _app()
    for pagina in [
        "paginas/inicio.py",
        "paginas/datos.py",
        "paginas/modelos.py",
        "paginas/reto.py",
        "paginas/clasificar.py",
        "paginas/conclusiones.py",
        "paginas/encuesta.py",
    ]:
        at.switch_page(pagina).run()
        textos = " ".join(m.value for m in at.markdown)
        assert "Recorrido: paso" in textos, pagina
