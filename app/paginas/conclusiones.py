"""
Conclusiones: la respuesta a la pregunta del proyecto, sus matices y sus límites.

Las cifras se calculan a partir de los resultados versionados; el texto interpreta, no repite a mano.
"""

import streamlit as st
from core.datos import NOMBRES, comparativa, frase_significancia, mostrar_error_datos
from core.estilo import decimal
from core.guia import siguiente_paso
from core.pie import REPO, pie

try:
    C = comparativa()
except Exception as e:
    mostrar_error_datos(e)

t, M = C["tabla"], C["metricas"]
mejor = t["f1_macro"].idxmax()
perdida_rep = {m: 100 * (t.loc[m, "f1_macro"] - t.loc[m, "f1_macro_sin_repetidos"]) for m in t.index}
menos_memoriza = min(perdida_rep, key=perdida_rep.get)
mas_liviano = t["tamano_MB"].idxmin()

# ¿Las redes convergieron? Se deduce del historial de entrenamiento guardado en sus métricas
sin_converger = []
for m in ("cnn1d", "transformer"):
    hist = M.get(m, {}).get("entrenamiento", {}).get("historial", [])
    if hist:
        mejor_epoca = max(hist, key=lambda h: h["f1_macro_validacion"])["epoca"]
        if mejor_epoca == len(hist):
            sin_converger.append((NOMBRES[m], len(hist)))

st.title("Conclusiones", anchor=False)

st.header("Qué se recomienda", divider="gray", anchor=False)
st.markdown(
    f"Para clasificar tráfico cifrado usando solo metadatos de flujo, el modelo recomendado es **{NOMBRES[mejor]}**. "
    f"Alcanza un F1 macro de **{decimal(t.loc[mejor, 'f1_macro'])}** en octubre, se mantiene en "
    f"**{decimal(t.loc[mejor, 'f1_macro_sin_repetidos'])}** cuando se excluyen las secuencias ya vistas, clasifica un "
    f"flujo en **{decimal(t.loc[mejor, 'latencia_ms'])} ms** y {frase_significancia(C['mcnemar'], mejor)}."
)
st.markdown(
    "El resultado coincide con la revisión crítica publicada en 2025 sobre este campo: cuando se evalúa con rigor, "
    "un modelo clásico bien ajustado iguala o supera a arquitecturas profundas mucho más costosas de entrenar."
)

st.header("Lo que matiza el resultado", divider="gray", anchor=False)
st.markdown(f"""
- **{NOMBRES[menos_memoriza]} es el que menos memoriza:** pierde solo {decimal(perdida_rep[menos_memoriza], 1)} puntos
  de F1 al excluir las secuencias repetidas, frente a {decimal(perdida_rep[mejor], 1)} de {NOMBRES[mejor]}.
- **{NOMBRES[mas_liviano]} es el más liviano:** {t.loc[mas_liviano, "tamano_MB"]:g} MB frente a
  {t.loc[mejor, "tamano_MB"]:g} MB de {NOMBRES[mejor]}, útil para equipos de red con poca memoria.
- **Todos pierden acierto con el tiempo:** entre octubre y diciembre el tráfico cambia y los modelos se desactualizan.
  En producción habría que reentrenarlos de forma periódica.
""")

st.header("Limitaciones", divider="gray", anchor=False)
limitacion_epocas = ""
if sin_converger:
    detalle = " y ".join(f"{n} ({e} épocas)" for n, e in sin_converger)
    limitacion_epocas = (
        f"- **No todas las redes terminaron de aprender:** {detalle} lograron su mejor validación en la "
        f"última época entrenada. Con más entrenamiento podrían acortar la distancia.\n"
    )
st.markdown(
    limitacion_epocas
    + """
- **Las redes solo vieron la secuencia de paquetes:** por diseño no recibieron las estadísticas del flujo que sí usan
  los modelos de árboles. Un modelo híbrido podría combinar ambas fuentes.
- **Una sola red de origen:** los datos provienen de una red académica checa de 2022. Los resultados deben validarse
  con tráfico de otras redes, por ejemplo, ecuatorianas, antes de generalizarlos.
- **Las categorías pequeñas tienen precisión baja:** los pesos por clase hacen que el modelo las sospeche de más.
  Ajustar esos pesos o los umbrales de decisión es el siguiente paso.
"""
)

st.header("Datos, código y créditos", divider="gray", anchor=False)
st.markdown(f"""
- **Dataset:** CESNET-TLS-Year22 de K. Hynek, J. Luxemburk, J. Pešek, T. Čejka y P. Šiška, publicado en
  *Scientific Data* (2024). © CESNET, licencia CC BY 4.0, doi:10.5281/zenodo.10608607.
- **Código y documentación:** [repositorio del proyecto en GitHub]({REPO}), con el EDA, los scripts de entrenamiento,
  la bitácora y los entregables del curso.
- **Tipografía:** Archivo, de Omnibus-Type, bajo SIL Open Font License, servida desde la propia aplicación.
- **Autor:** Jairo Wladimir Jhayya Perlaza, Maestría en Inteligencia Artificial,
  Universidad de Especialidades Espíritu Santo.
""")
siguiente_paso("paginas/conclusiones.py")
pie()
