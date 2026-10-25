"""
Acerca del modelo: qué hace, cómo se evaluó, sus limitaciones y para qué no debe usarse.

Es la ficha del modelo en versión resumida (la completa está en docs/ficha_del_modelo.md del repositorio).
Las cifras se leen de los resultados del proyecto, así que nunca quedan desactualizadas.
"""

import streamlit as st
from core.datos import comparativa, mostrar_error_datos, peor_categoria
from core.estilo import decimal
from core.explicacion import umbral_abstencion
from core.pie import pie

try:
    comp = comparativa()
except Exception as e:
    mostrar_error_datos(e)

t = comp["tabla"].loc["xgboost"]
peor, f1_peor = peor_categoria(comp["reportes"]["xgboost"])
umbral = umbral_abstencion()

st.title("Acerca del modelo", anchor=False)
st.markdown(
    "Esta página resume qué hace el modelo que clasifica en la aplicación, qué tan bien funciona, dónde falla y "
    "para qué no debe usarse. Es la versión breve de su ficha técnica."
)

st.header("Qué hace", divider="gray", anchor=False)
st.markdown(
    "Recibe los metadatos de una conexión TLS cifrada (tamaños, direcciones y tiempos de sus paquetes, y "
    "estadísticas del flujo) y estima a cuál de **23 categorías de servicio** pertenece, por ejemplo streaming, "
    "banca en línea o mensajería. **Nunca descifra ni lee el contenido** de la comunicación."
)

st.header("El modelo", divider="gray", anchor=False)
st.markdown(
    "Es un **XGBoost**, un conjunto de árboles de decisión, entrenado con el tráfico de septiembre de 2022 de la red "
    "académica checa CESNET (dataset CESNET-TLS-Year22, licencia CC BY 4.0). Usa 60 variables por flujo y fue el "
    "mejor de los cuatro modelos comparados, con diferencias estadísticamente significativas."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("F1 macro en octubre", decimal(t["f1_macro"], 3))
c2.metric(f"F1 de la peor categoría ({peor})", decimal(f1_peor, 3))
c3.metric("F1 en diciembre", decimal(t["f1_macro_2022-12"], 3))
c4.metric("Tiempo por flujo", f"{decimal(t['latencia_ms'], 3)} ms")
st.markdown(
    '<p class="nota">Octubre de 2022 son flujos que el modelo nunca vio al entrenar. El F1 macro da el mismo peso a '
    "las 23 categorías, y la peor categoría muestra cuánto puede alejarse un servicio del promedio.</p>",
    unsafe_allow_html=True,  # texto fijo del código
)

st.header("Cómo decide y cuándo se abstiene", divider="gray", anchor=False)
st.markdown(
    f"Junto a cada clasificación, la aplicación muestra **las variables que más pesaron en la decisión** (valores SHAP "
    f"calculados por el propio modelo). Además, si la confianza del modelo es menor que "
    f"**{decimal(umbral['umbral'] * 100, 0)} %**, no da una respuesta: indica que el flujo queda sin clasificar. "
    "El modelo está bien calibrado, así que su confianza es una señal fiable de cuándo duda."
)
if "cobertura" in umbral:
    st.markdown(
        "Con ese umbral, en los datos de validación el modelo clasifica el "
        f"**{decimal(umbral['cobertura'] * 100, 1)} %** "
        f"de los flujos y acierta en el **{decimal(umbral['acierto'] * 100, 1)} %** de ellos."
    )

st.header("Limitaciones", divider="gray", anchor=False)
st.markdown(
    "- **Datos de otro país y otra época.** Aprendió del tráfico checo de 2022. Servicios propios "
    "de otros países, como "
    "bancos o aplicaciones locales, no están representados y probablemente se clasificarían peor.\n"
    "- **Deriva.** Su desempeño baja con los meses, unos 4,7 puntos de octubre a diciembre. Debe reentrenarse "
    "periódicamente.\n"
    "- **Servicios parecidos.** Confunde sobre todo servicios que comparten infraestructura, como Media, Social y "
    "Advertising.\n"
    "- **Datos incompletos o distintos.** Pierde bastante precisión si faltan datos del flujo o si el tráfico cambia "
    "mucho de composición.\n"
    "- **Relleno de paquetes.** Si los paquetes se rellenan hasta un tamaño fijo, el modelo deja de reconocer los "
    "servicios. Es una técnica legítima de privacidad y también una forma de evadirlo."
)

st.header("Usos previstos y usos prohibidos", divider="gray", anchor=False)
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "**Usos previstos**\n\n"
        "- Investigación y docencia sobre clasificación de tráfico cifrado.\n"
        "- Estadísticas **agregadas** de una red para planificar su capacidad o diagnosticar problemas, con "
        "supervisión humana."
    )
with c2:
    st.markdown(
        "**Usos prohibidos**\n\n"
        "- Vigilar o perfilar a personas a partir de su tráfico.\n"
        "- Priorizar, degradar o bloquear servicios con fines comerciales, lo que además contraviene la neutralidad "
        "de red.\n"
        "- Tomar decisiones automáticas sobre personas sin revisión humana."
    )
st.markdown(
    '<p class="nota">El análisis completo de riesgos y mitigaciones está en el informe de impacto social y '
    "responsabilidad del proyecto (semana 4), en la carpeta docs/ del repositorio.</p>",
    unsafe_allow_html=True,  # texto fijo del código
)

st.header("Autoría", divider="gray", anchor=False)
st.markdown(
    "Proyecto Integrador en Inteligencia Artificial, Maestría en Inteligencia Artificial, Universidad de "
    "Especialidades Espíritu Santo. Autor: Jairo Wladimir Jhayya Perlaza. Docente: Gladys María Villegas Rugel. "
    "Código abierto con licencia MIT."
)
pie()
