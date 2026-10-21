"""Pruebas del manejo seguro de errores: el usuario nunca ve detalles internos de un error."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app")


def _pagina_que_falla(ruta_app):
    import sys

    sys.path.insert(0, ruta_app)
    from core.errores import ejecutar_pagina

    class Pagina:
        title = "prueba"

        def run(self):
            raise ImportError("cannot import name 'x' from 'core.datos' (/mount/src/proyecto/app/core/datos.py)")

    ejecutar_pagina(Pagina())


def _pagina_que_se_detiene(ruta_app):
    import sys

    import streamlit as st

    sys.path.insert(0, ruta_app)
    from core.errores import ejecutar_pagina

    class Pagina:
        title = "prueba"

        def run(self):
            st.write("antes")
            st.stop()

    ejecutar_pagina(Pagina())
    st.write("después")  # st.stop() debe detener todo el script, también esta línea


def test_un_error_no_expone_rutas_ni_codigo():
    at = AppTest.from_function(_pagina_que_falla, args=(APP,))
    at.run()
    assert not at.exception  # Streamlit no muestra su traza
    assert len(at.error) == 1
    assert "/mount/src" not in at.error[0].value and "core.datos" not in at.error[0].value


def test_st_stop_sigue_funcionando():
    at = AppTest.from_function(_pagina_que_se_detiene, args=(APP,))
    at.run()
    textos = [m.value for m in at.markdown]
    assert "antes" in textos and "después" not in textos and not at.error
