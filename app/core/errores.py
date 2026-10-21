"""
Manejo seguro de errores (OWASP A05: configuración insegura / divulgación de información).

Streamlit Community Cloud muestra la traza de un error (rutas del servidor y líneas de código) aunque
.streamlit/config.toml indique showErrorDetails = "none". Esta capa no depende de la plataforma: si una página
falla, el usuario ve un mensaje claro sin detalles internos y el error completo queda en los registros del
servidor (en Community Cloud: "Manage app"), donde solo lo ve el responsable de la aplicación.

Las señales internas de Streamlit para detener o recargar una página (StopException, RerunException) heredan
de BaseException, no de Exception: no se interceptan, así que st.stop() y el cambio de página siguen funcionando.
"""

import logging

import streamlit as st

registro = logging.getLogger("trafico_cifrado")
MENSAJE = (
    "Esta página no pudo cargarse. Recárgala en unos segundos; si el problema continúa, el error quedó "
    "registrado para revisarlo."
)


def ejecutar_pagina(pagina) -> None:
    """Ejecuta la página elegida en la navegación sin exponer detalles internos si falla."""
    try:
        pagina.run()
    except Exception:  # cualquier error de la página; las señales de control de Streamlit no llegan aquí
        registro.exception("Error al ejecutar la página %s", getattr(pagina, "title", "desconocida"))
        st.error(MENSAJE)
