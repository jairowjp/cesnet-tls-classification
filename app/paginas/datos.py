"""
Datos: el EDA de forma interactiva.

Permite elegir dos categorías y compararlas en tres aspectos: la forma típica de sus conexiones
(firma media), cuánto tráfico representan (desbalance) y cómo cambia su peso durante el año (deriva).
Todas las cifras provienen de results/eda, generado por notebooks/01_eda.ipynb.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.datos import eda, mostrar_error_datos
from core.estilo import CONFIG_GRAFICO, NEUTRO, TINTA, TINTA_SUAVE, decimal, miles
from core.firma import mostrar_firma
from core.guia import siguiente_paso, sugerencia
from core.pie import pie

try:
    E = eda()
except Exception as e:
    mostrar_error_datos(e)

R = E["resumen"]
clases = E["clases"]
categorias = list(clases.index)

st.title("Los datos")
st.markdown(
    f"El dataset CESNET-TLS-Year22 registró **{miles(R['anio']['flujos_totales'])}** conexiones TLS durante 2022 en la "
    f"red académica nacional de la República Checa. Para este proyecto se tomó una muestra aleatoria del 2 % de "
    f"septiembre a diciembre: **{miles(R['muestra']['filas'])}** flujos de "
    f"**{len(categorias)}** categorías de servicio."
)

st.header("La forma de cada servicio", divider="gray")
st.markdown("Elige dos categorías para comparar cómo empieza una conexión típica de cada una.")
c1, c2 = st.columns(2)
por_defecto_a = categorias.index("Media") if "Media" in categorias else 0
por_defecto_b = (
    categorias.index("Software updates") if "Software updates" in categorias else min(1, len(categorias) - 1)
)
cat_a = c1.selectbox("Primera categoría", categorias, index=por_defecto_a)
cat_b = c2.selectbox("Segunda categoría", categorias, index=por_defecto_b)

sugerencia(
    "Compara Media con Music: las dos transmiten contenido y sus firmas se parecen. Después compara Media con "
    "Notifications, cuyas conexiones son cortas y casi no se parecen a ninguna otra."
)
for col, cat in [(c1, cat_a), (c2, cat_b)]:
    firma = E["firmas"].loc[cat].to_numpy(dtype=float)  # tamaño medio × dirección en cada posición
    with col:
        mostrar_firma(
            np.abs(firma),
            np.sign(firma),
            np.full(len(firma), 20.0),
            veredicto={
                "titulo": cat,
                "detalle": f"Firma media de {miles(clases.loc[cat, 'entrenamiento'])} flujos de septiembre",
            },
            animar=False,
            alto=250,
            clave=cat,
        )
st.markdown(
    '<p class="nota">La firma media promedia el tamaño y la dirección de cada posición. Las diferencias entre '
    "categorías, sobre todo a partir del tercer paquete, son las que aprovechan los modelos.</p>",
    unsafe_allow_html=True,
)

st.header("Cuánto tráfico hay de cada tipo", divider="gray")
entr = clases["entrenamiento"].sort_values()
# Cobalto y ámbar se reservan para la dirección de los paquetes; las categorías usan tinta
colores = [TINTA if c == cat_a else (TINTA_SUAVE if c == cat_b else NEUTRO) for c in entr.index]
fig = go.Figure(
    go.Bar(
        x=entr.values,
        y=entr.index,
        orientation="h",
        marker_color=colores,
        hovertemplate="<b>%{y}</b><br>%{x:,} flujos en entrenamiento<extra></extra>",
    )
)
# hovermode="y": el recuadro aparece al pasar por cualquier punto de la fila, no solo sobre la barra
fig.update_layout(
    height=620, hovermode="y", xaxis=dict(type="log", title="Flujos en entrenamiento (escala logarítmica)")
)
st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)
mayor, menor = entr.index[-1], entr.index[0]
st.markdown(
    f"La categoría más frecuente, {mayor}, tiene **{entr.iloc[-1] / entr.iloc[0]:.0f} veces** más flujos que la menos "
    f"frecuente, {menor}. Por eso los modelos se entrenan con pesos por clase y se evalúan con F1 macro, que da el "
    f"mismo peso a todas las categorías."
)

st.header("Cómo cambia el tráfico durante el año", divider="gray")
mes = E["cat_mes"].T * 100  # meses en filas, categorías en columnas, en porcentaje
fig = go.Figure()
for cat, color, trazo in [(cat_a, TINTA, "solid"), (cat_b, TINTA_SUAVE, "dash")]:
    fig.add_trace(
        go.Scatter(
            x=mes.index,
            y=mes[cat],
            mode="lines+markers",
            name=cat,
            line=dict(color=color, width=2.5, dash=trazo),
            hovertemplate="%{y:.2f} % de los flujos<extra>" + cat + "</extra>",
        )
    )
for m in ["2022-09", "2022-10"]:
    if m in mes.index:
        fig.add_vline(x=m, line=dict(color=TINTA_SUAVE, dash="dot", width=1))
# "x unified": un solo recuadro con las dos categorías de ese mes, para compararlas
fig.update_layout(height=340, hovermode="x unified", yaxis=dict(title="% de los flujos del mes"))
st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)
js = E["deriva_js"].iloc[:, 0]
st.markdown(
    f"Las líneas punteadas marcan septiembre (entrenamiento) y octubre (prueba). Frente a septiembre, el mes más "
    f"distinto es **{js.idxmax()}** (distancia de Jensen-Shannon de {decimal(js.max())}), y octubre es de los más "
    f"parecidos ({decimal(js.get('2022-10', float('nan')))}). Los modelos se prueban también con noviembre "
    f"y diciembre para medir "
    f"cuánto pierden cuando el tráfico cambia."
)

st.header("Calidad y precauciones", divider="gray")
Q, S, F = R["calidad"], R["repetidos"], R["fuga"]
tabla = pd.DataFrame(
    {
        "Comprobación": [
            "Datos completos",
            "Valores imposibles",
            "Secuencias de prueba ya vistas en entrenamiento",
            "Campo con fuga de información (ASN de destino)",
        ],
        "Resultado": [
            f"{decimal(Q['completitud_%'], 1)} %",
            "Ninguno",
            f"{decimal(S['prueba_con_secuencia_vista_en_entrenamiento_%'], 1)} %",
            f"NMI {decimal(F['DST_ASN']['NMI_con_categoria'])}: se excluye",
        ],
    }
)
st.dataframe(tabla, hide_index=True)
siguiente_paso("paginas/datos.py")
pie()
