"""Pruebas del estilo de la barra superior y del pie de página.

El estilo del menú depende de identificadores internos de Streamlit (stHeader, stTopNavLink) y del atributo
estándar aria-current="page". Si una actualización de Streamlit los cambia, el menú perdería su diseño sin
ningún error visible: estas pruebas lo detectan antes de publicar.
"""

from pathlib import Path

import streamlit

ROOT = Path(__file__).resolve().parents[1]
IDENTIFICADORES = ("stHeader", "stTopNavLink", "aria-current")


def test_streamlit_conserva_los_identificadores_que_usa_el_menu():
    js = Path(streamlit.__file__).parent / "static/static/js"
    codigo = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in js.glob("index.*.js"))
    faltan = [i for i in IDENTIFICADORES if i not in codigo]
    assert not faltan, (
        f"Streamlit {streamlit.__version__} ya no usa {faltan}: revisa el CSS del menú en app/core/estilo.py"
    )


def test_el_estilo_cubre_barra_menu_activo_y_pie():
    import sys

    sys.path.insert(0, str(ROOT / "app"))
    from core.estilo import CSS

    for selector in ('[data-testid="stHeader"]', '[data-testid="stTopNavLink"]', '[aria-current="page"]', ".pie-banda"):
        assert selector in CSS, f"falta el estilo de {selector}"
    assert CSS.count("{") == CSS.count("}")


def test_isotipo_y_pie_en_todas_las_paginas():
    from streamlit.testing.v1 import AppTest

    assert (ROOT / "app/static/isotipo.png").exists()
    paginas = [
        "inicio",
        "datos",
        "modelos",
        "diagnostico",
        "reto",
        "clasificar",
        "conclusiones",
        "encuesta",
        "seguridad",
    ]
    at = AppTest.from_file(str(ROOT / "app/principal.py"), default_timeout=120)
    at.run()
    for pagina in paginas:
        at.switch_page(f"paginas/{pagina}.py").run()
        assert any("pie-banda" in m.value for m in at.markdown), f"la página {pagina} no muestra el pie de página"
