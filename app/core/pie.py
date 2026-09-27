"""Pie de página común: atribución obligatoria del dataset (CC BY 4.0) y enlace al código."""

import streamlit as st

REPO = "https://github.com/jairowjp/cesnet-tls-classification"


def pie() -> None:
    st.divider()
    st.page_link("paginas/seguridad.py", label="Seguridad y privacidad de este sitio")
    st.markdown(
        f'<p class="nota">Proyecto Integrador en Inteligencia Artificial, UEES, 2026. '
        f"Datos: CESNET-TLS-Year22, © CESNET, licencia CC BY 4.0. "
        f"Código, datos de la muestra y documentación en "
        f'<a href="{REPO}" target="_blank" rel="noopener noreferrer">GitHub</a>.</p>',
        unsafe_allow_html=True,  # texto fijo del código: no contiene datos del usuario
    )
