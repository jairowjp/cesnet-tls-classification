"""Pie de página común: créditos, enlaces y atribución obligatoria del dataset (CC BY 4.0)."""
import streamlit as st

REPO = "https://github.com/jairowjp/cesnet-tls-classification"
DATASET = "https://doi.org/10.5281/zenodo.10608607"
INFORME_SEGURIDAD = f"{REPO}/blob/main/docs/seguridad/README.md"


def pie() -> None:
    """Banda oscura al final de cada página. Todo el contenido es fijo del código (sin datos del usuario)."""
    st.markdown(
        f"""
<div class="pie-banda">
  <div>
    <p class="pie-nombre">Tráfico cifrado</p>
    <p>Proyecto Integrador en Inteligencia Artificial (MIAR0545), Maestría en Inteligencia Artificial,
    Universidad de Especialidades Espíritu Santo, 2026.</p>
    <p>Jairo Wladimir Jhayya Perlaza, con la docente Gladys María Villegas Rugel.</p>
  </div>
  <div>
    <p class="pie-titulo">El proyecto</p>
    <p><a href="{REPO}" target="_blank" rel="noopener noreferrer">Código y documentación</a></p>
    <p><a href="{INFORME_SEGURIDAD}" target="_blank" rel="noopener noreferrer">Informe de seguridad</a></p>
    <p><a href="seguridad" target="_self">Seguridad y privacidad de este sitio</a></p>
  </div>
  <div>
    <p class="pie-titulo">Los datos</p>
    <p><a href="{DATASET}" target="_blank" rel="noopener noreferrer">CESNET-TLS-Year22</a>, © CESNET.</p>
    <p>Licencia CC BY 4.0. Solo metadatos de flujos: sin contenido, IP ni dominios.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
