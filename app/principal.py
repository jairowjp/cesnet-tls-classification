"""
Punto de entrada de la aplicación web.

Ejecución local (desde la raíz del repositorio):
    streamlit run app/principal.py

La configuración de seguridad y del tema está en .streamlit/config.toml.
"""

import sys
from pathlib import Path

# La raíz del repositorio va al path para reutilizar src/ (la misma carga de variables que el entrenamiento).
# app/ ya está en el path porque Streamlit agrega la carpeta del script principal.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from core.estilo import aplicar_estilo  # noqa: E402

st.set_page_config(
    page_title="Clasificar tráfico cifrado | Proyecto Integrador UEES",
    page_icon=":material/swap_vert:",
    layout="wide",
)
aplicar_estilo()

paginas = [
    st.Page("paginas/inicio.py", title="Inicio", default=True),
    st.Page("paginas/datos.py", title="Datos"),
    st.Page("paginas/modelos.py", title="Modelos"),
    st.Page("paginas/reto.py", title="Reto"),
    st.Page("paginas/clasificar.py", title="Clasificar"),
    st.Page("paginas/conclusiones.py", title="Conclusiones"),
    st.Page("paginas/encuesta.py", title="Tu opinión"),
    st.Page("paginas/seguridad.py", title="Seguridad"),
]
# Navegación en la parte superior: más limpia que la barra lateral por defecto de Streamlit
st.navigation(paginas, position="top").run()
