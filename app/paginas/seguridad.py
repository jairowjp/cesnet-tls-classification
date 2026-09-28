"""
Seguridad y privacidad: el sitio informa, en lenguaje claro, cómo trata los datos y cómo se protege.

Cada afirmación remite a un archivo del repositorio donde se puede comprobar: el control y la
prueba automática que lo verifica. La transparencia es parte de la seguridad.
"""

import pandas as pd
import streamlit as st
from core.pie import REPO, pie

ARCHIVO = f"{REPO}/blob/main"

st.title("Seguridad y privacidad")
st.markdown(
    "Este sitio se diseñó con la seguridad incluida desde el principio, siguiendo la guía OWASP Top 10, que "
    "reúne los riesgos más comunes de las aplicaciones web. Aquí se explica qué datos usa, qué hace con lo que "
    "subes y cómo se protege. Cada punto se puede verificar en el código publicado."
)

st.header("Qué datos usa", divider="gray")
st.markdown("""
- **Solo metadatos de conexiones:** tamaños, direcciones y tiempos de los primeros paquetes de cada flujo,
  registrados en 2022 en la red académica nacional de la República Checa (dataset CESNET-TLS-Year22).
- **Nada se descifró:** el dataset nunca contuvo el contenido de las comunicaciones.
- **Sin datos que identifiquen personas ni servidores:** se excluyeron las direcciones IP, los nombres de
  dominio (SNI), las huellas de los clientes (JA3) y el sistema autónomo de destino.
""")

st.header("Qué pasa con lo que haces aquí", divider="gray")
st.markdown("""
- **Los archivos CSV que subes se procesan en memoria** para clasificarlos y no se guardan ni se envían a nadie.
- **La encuesta no se almacena en el sitio:** al terminar recibes un código corto con tus respuestas y tu
  perfil, y tú decides si enviarlo. No se pide tu nombre ni ningún dato personal.
- **Sin estadísticas de uso ni rastreo:** el sitio no envía datos de tu visita a terceros. La tipografía
  y el logo se sirven desde la propia aplicación, sin consultar servidores externos.
- **Solo una cookie técnica:** la que usa la protección contra falsificación de peticiones (XSRF).
""")

st.header("Cómo se protege", divider="gray")
controles = pd.DataFrame(
    [
        (
            "Que un archivo malicioso dañe la aplicación",
            "Se revisan el tamaño (2 MB), las filas (1 000), las columnas, los tipos y los rangos "
            "de cada archivo antes de usarlo",
            "app/core/seguridad.py",
        ),
        (
            "Que se inyecte código en la página (XSS)",
            "Los textos se insertan como texto, nunca como código, y los datos se escapan antes de llegar al navegador",
            "app/core/firma.py",
        ),
        (
            "Que una hoja de cálculo ejecute fórmulas al abrir una descarga",
            "Todo texto que empiece con = + - @ se neutraliza",
            "app/core/seguridad.py",
        ),
        (
            "Que alguien reemplace el modelo por uno alterado",
            "El modelo solo se carga si su huella digital SHA-256 coincide con la registrada",
            "app/models/manifest.json",
        ),
        (
            "Que un error revele detalles internos",
            "Los errores muestran un mensaje general, sin trazas del código",
            ".streamlit/config.toml",
        ),
        (
            "Que un ataque tome control del servidor",
            "La aplicación corre con un usuario sin privilegios de administrador",
            "Dockerfile",
        ),
    ],
    columns=["Riesgo", "Cómo se evita", "Dónde verificarlo"],
)
st.dataframe(controles, hide_index=True)
st.markdown(
    f"Cada control tiene pruebas automáticas que intentan romperlo, por ejemplo con archivos maliciosos o un "
    f"modelo alterado: [tests/test_app_seguridad.py]({ARCHIVO}/tests/test_app_seguridad.py). "
    f"El detalle completo está en [SECURITY.md]({ARCHIVO}/SECURITY.md)."
)

st.header("Límites que conviene conocer", divider="gray")
st.markdown(f"""
- **Cabeceras de seguridad del navegador:** la plataforma Streamlit no permite configurar todas, por ejemplo
  una política de contenido (CSP) estricta. En la publicación, parte de ellas depende de la plataforma de alojamiento.
- **Revisión externa:** la aplicación se escaneó con OWASP ZAP (de forma activa en una copia local y de forma
  pasiva en la versión publicada) y sus dependencias se auditaron con pip-audit. No se encontraron vulnerabilidades
  explotables; las alertas restantes dependen de la plataforma de alojamiento y están documentadas en el
  [informe de seguridad]({ARCHIVO}/docs/seguridad/README.md).
""")

st.header("Reportar un problema", divider="gray")
st.markdown(
    f"Si encuentras una vulnerabilidad, repórtala en la sección de [issues del repositorio]({REPO}/issues), "
    f"sin incluir datos personales."
)
pie()
