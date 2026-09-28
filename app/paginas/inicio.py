"""
Inicio: el momento central de la aplicación.

Se reproduce, paquete a paquete, un flujo real de octubre de 2022 que ningún modelo vio al
entrenar, y al final XGBoost revela su categoría. Se elige al azar y se informa si el modelo
acertó o se equivocó: mostrar también los errores es parte de una evaluación honesta.
"""

import numpy as np
import streamlit as st
from core.datos import NOMBRES, comparativa, modelo_xgboost, mostrar_error_datos, muestra_octubre
from core.estilo import decimal
from core.firma import mostrar_firma
from core.guia import resultado_modelo, siguiente_paso, sugerencia
from core.pie import pie

from src.data.loader import DIR, IPT, SIZE, tabular_features

try:
    df = muestra_octubre()
    paquete = modelo_xgboost()
    comp = comparativa()
except Exception as e:
    mostrar_error_datos(e)

# Flujo actual: se guarda en la sesión para que no cambie con cada interacción
if "inicio_fila" not in st.session_state:
    st.session_state.inicio_fila = int(np.random.default_rng().integers(len(df)))

# ------------------------------------------------------------------ portada institucional
# Contenido fijo del código (sin datos del usuario). El logo se sirve desde app/static (static serving).
st.html(
    """
    <header class="portada-cabecera">
      <img src="app/static/uees_logo.png" alt="Universidad de Especialidades Espíritu Santo">
      <div class="portada-programa">
        <strong>Maestría en Inteligencia Artificial</strong><br>
        Proyecto Integrador en Inteligencia Artificial (MIAR0545)
      </div>
    </header>
    """
)
st.title("Clasificar tráfico cifrado sin descifrarlo", anchor=False)
st.html(
    """
    <p class="portada-oficial">Análisis comparativo de modelos de aprendizaje automático y arquitecturas Transformer
    para la clasificación de tráfico de red cifrado sobre el dataset CESNET-TLS-Year22</p>
    <dl class="portada-datos">
      <div><dt>Autor</dt><dd>Jairo Wladimir Jhayya Perlaza</dd></div>
      <div><dt>Docente</dt><dd>Gladys María Villegas Rugel</dd></div>
      <div><dt>Lugar y año</dt><dd>Guayaquil, 2026</dd></div>
    </dl>
    """
)
st.markdown(
    "Casi todo el tráfico de Internet viaja cifrado: nadie en el camino puede leer su contenido. "
    "Aun así, cada conexión deja una forma: cuántos paquetes envía, de qué tamaño, en qué dirección "
    "y con qué ritmo. Esta aplicación muestra que esa forma basta para reconocer qué tipo de servicio hay detrás."
)

fila = df.iloc[st.session_state.inicio_fila]
X = tabular_features(df.iloc[[st.session_state.inicio_fila]])
prob = paquete["modelo"].predict_proba(X)[0]
pred = paquete["clases"][int(prob.argmax())]
real = fila["CATEGORY"]
n_paquetes = int(fila["PPI_LEN"])
acierto = pred == real

mostrar_firma(
    fila[SIZE].to_numpy(),
    fila[DIR].to_numpy(),
    fila[IPT].to_numpy(),
    veredicto={
        "titulo": f"XGBoost dice: {pred} ({prob.max() * 100:.0f} %)",
        "detalle": (
            f"Categoría real: {real}. {'Acertó' if acierto else 'Se equivocó'} a partir de "
            f"{n_paquetes} paquetes, sin leer su contenido."
        ),
    },
    animar=True,
    clave=str(st.session_state.inicio_fila),
    alto=270,
)

# Los otros tres modelos opinan sobre el mismo flujo: la pregunta del proyecto es comparativa.
# Sus predicciones se calcularon con scripts/06_preparar_app.py (la app no necesita cargar PyTorch).
otros = [m for m in ("random_forest", "cnn1d", "transformer") if f"pred_{m}" in df.columns]
if otros:
    st.markdown("**Los otros modelos, sobre el mismo flujo**")
    for col, m in zip(st.columns(len(otros)), otros, strict=True):
        with col:
            confianza = max(float(fila[c]) for c in df.columns if c.startswith(f"prob_{m}_"))
            resultado_modelo(NOMBRES[m], fila[f"pred_{m}"], real, confianza)

sugerencia(
    "Pulsa «Reproducir otro flujo» varias veces: cada servicio deja una forma distinta. Algunos son ráfagas "
    "cortas de pocos paquetes; otros, largas conversaciones en las que el servidor envía mucho más de lo que recibe."
)
col_boton, col_leyenda = st.columns([1, 3], vertical_alignment="center")
with col_boton:
    if st.button("Reproducir otro flujo", type="primary"):
        st.session_state.inicio_fila = int(np.random.default_rng().integers(len(df)))
        st.rerun()
with col_leyenda:
    st.markdown(
        '<p class="nota">Cada barra es un paquete. Las <span class="sube">azules</span> van del cliente al servidor '
        'y las <span class="baja">ámbar</span> vuelven del servidor. La altura indica el tamaño del paquete '
        "y la separación, el tiempo que pasó desde el anterior.</p>",
        unsafe_allow_html=True,  # texto fijo del código
    )

with st.container(border=True):
    st.markdown("**¿Por dónde empezar?** Recorre el proyecto en siete pasos o pon a prueba tu intuición directamente.")
    a, b = st.columns(2)
    a.page_link("paginas/datos.py", label="Empezar el recorrido por los datos")
    b.page_link("paginas/reto.py", label="Ir directo al reto contra los modelos")

st.header("Cómo se construyó", divider="gray", anchor=False)
anio = comp["metricas"].get("xgboost", {})
pasos = st.columns(4)
contenido = [
    (
        "1. Un año de tráfico real",
        "CESNET-TLS-Year22 registró 507,7 millones de conexiones cifradas durante 2022 "
        "en una red académica en producción.",
    ),
    (
        "2. Meses separados",
        "Los modelos aprendieron con septiembre y se evaluaron con octubre, "
        "como ocurriría en una red real: siempre se predice el futuro.",
    ),
    (
        "3. Cuatro modelos",
        "Dos basados en árboles de decisión, que leen estadísticas del flujo, "
        "y dos redes neuronales, que leen la secuencia de paquetes.",
    ),
    ("4. Comparación estadística", "Una diferencia solo cuenta si la prueba de McNemar descarta que sea azar."),
]
for col, (titulo, texto) in zip(pasos, contenido, strict=True):
    with col:
        st.markdown(f"**{titulo}**")
        st.markdown(f'<p class="nota">{texto}</p>', unsafe_allow_html=True)  # textos fijos del código

if anio:
    f1 = anio["prueba_octubre"]["f1_macro"]
    # Cuántas veces más liviana es la red más pequeña frente a XGBoost (calculado, no escrito a mano)
    redes = [
        comp["metricas"][m]["costo"]["tamano_modelo_MB"] for m in ("cnn1d", "transformer") if m in comp["metricas"]
    ]
    veces = anio["costo"]["tamano_modelo_MB"] / min(redes) if redes else None
    st.header("Qué se encontró", divider="gray", anchor=False)
    st.markdown(
        f"El mejor modelo es **{NOMBRES['xgboost']}**: acierta con un F1 macro de **{decimal(f1)}** en octubre, "
        f"clasifica un flujo en **{decimal(anio['costo']['latencia_ms_por_flujo'])} ms** y supera a los demás con "
        f"diferencias estadísticamente significativas. "
        + (
            f"Las redes neuronales, en cambio, son hasta {veces:.0f} veces más livianas y memorizan menos. "
            if veces
            else ""
        )
        + "La comparación completa está en la página de modelos."
    )
    st.page_link("paginas/modelos.py", label="Ver la comparación de modelos")
siguiente_paso("paginas/inicio.py")
pie()
