"""
Modelos: la comparación completa, para explorar en lugar de solo leer.

  * Desempeño frente a costo, con la opción de excluir las secuencias repetidas.
  * La prueba de McNemar explicada en lenguaje claro para cualquier par de modelos.
  * El F1 de cada categoría, la matriz de confusión y la pérdida por deriva.
Todas las cifras vienen de results/modelos y results/comparativa.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.datos import NOMBRES, comparativa, mostrar_error_datos
from core.estilo import COBALTO, CONFIG_GRAFICO, FONDO, LINEA, NEUTRO, TINTA, TINTA_SUAVE, decimal, miles
from core.guia import siguiente_paso, sugerencia
from core.pie import pie

try:
    C = comparativa()
except Exception as e:
    mostrar_error_datos(e)

tabla, mc, M = C["tabla"], C["mcnemar"], C["metricas"]
modelos = [m for m in NOMBRES if m in tabla.index]
nombre = lambda m: NOMBRES.get(m, m)  # noqa: E731

st.title("Los modelos", anchor=False)
st.markdown(
    "Se compararon cuatro modelos con el mismo protocolo: mismos datos, mismos meses y mismas métricas. "
    "Dos leen estadísticas del flujo (Random Forest y XGBoost) y dos leen la secuencia de paquetes "
    "(CNN 1D y Transformer)."
)

# ------------------------------------------------------------------ desempeño frente a costo
st.header("Acierto frente a costo", divider="gray", anchor=False)
sin_rep = st.toggle(
    "Evaluar solo con flujos cuya secuencia no apareció en entrenamiento",
    help="El 30 % de los flujos de octubre repite exactamente una secuencia de septiembre. "
    "Excluirlos muestra cuánto del acierto viene de memorizar.",
)
sugerencia(
    "Activa el interruptor de arriba y observa qué modelo pierde más acierto: el que más cae es el que más "
    "dependía de memorizar secuencias ya vistas."
)
col_f1 = "f1_macro_sin_repetidos" if sin_rep else "f1_macro"
fig = go.Figure()
for m in modelos:
    fila = tabla.loc[m]
    recomendado = m == tabla["f1_macro"].idxmax()
    fig.add_trace(
        go.Scatter(
            x=[fila["latencia_ms"]],
            y=[fila[col_f1]],
            mode="markers+text",
            name=nombre(m),
            text=[f"{nombre(m)}  {decimal(fila[col_f1])}"],
            textposition="middle right",
            marker=dict(
                size=12 + 26 * np.sqrt(fila["tamano_MB"] / tabla["tamano_MB"].max()),
                color=COBALTO if recomendado else NEUTRO,
                line=dict(color=TINTA, width=1),
            ),
            hovertemplate=(
                f"<b>{nombre(m)}</b><br>F1 macro: %{{y:.3f}}<br>Latencia: %{{x:.4f}} ms"
                f"<br>Tamaño: {fila['tamano_MB']:g} MB<extra></extra>"
            ),
        )
    )
for valor, texto in [(0.70, "mínimo comprometido"), (0.85, "excelencia")]:
    fig.add_hline(
        y=valor,
        line=dict(color=TINTA_SUAVE, dash="dot", width=1),
        annotation_text=texto,
        annotation_position="bottom left",
        annotation_font=dict(color=TINTA_SUAVE, size=12),
    )
fig.update_layout(
    height=430,
    showlegend=False,
    xaxis=dict(type="log", title="Latencia por flujo en CPU (ms, escala logarítmica)"),
    yaxis=dict(
        title="F1 macro en octubre de 2022",
        range=[min(0.65, tabla[col_f1].min() - 0.05), max(0.95, tabla[col_f1].max() + 0.03)],
    ),
)
st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)
st.markdown(
    '<p class="nota">El tamaño de cada punto es proporcional al tamaño del modelo en disco. '
    "Arriba a la izquierda está lo deseable: acertar mucho y responder rápido.</p>",
    unsafe_allow_html=True,
)

vista = pd.DataFrame(
    {
        "Modelo": [nombre(m) for m in modelos],
        "F1 macro": [tabla.loc[m, "f1_macro"] for m in modelos],
        "F1 sin repetidos": [tabla.loc[m, "f1_macro_sin_repetidos"] for m in modelos],
        "F1 diciembre": [tabla.loc[m, "f1_macro_2022-12"] for m in modelos],
        "Latencia (ms)": [tabla.loc[m, "latencia_ms"] for m in modelos],
        "Tamaño (MB)": [tabla.loc[m, "tamano_MB"] for m in modelos],
        "Entrenamiento (min)": [tabla.loc[m, "entrenamiento_min"] for m in modelos],
    }
).sort_values("F1 macro", ascending=False)
st.dataframe(
    vista,
    hide_index=True,
    column_config={
        "F1 macro": st.column_config.NumberColumn(format="%.3f"),
        "F1 sin repetidos": st.column_config.NumberColumn(format="%.3f"),
        "F1 diciembre": st.column_config.NumberColumn(format="%.3f"),
        "Latencia (ms)": st.column_config.NumberColumn(format="%.4f"),
        "Tamaño (MB)": st.column_config.NumberColumn(format="%.2f"),
    },
)

# ------------------------------------------------------------------ McNemar
st.header("¿La diferencia es real o es azar?", divider="gray", anchor=False)
st.markdown(
    "La prueba de McNemar compara dos modelos solo en los flujos donde **no están de acuerdo**. "
    "Elige un par para ver qué dice."
)
sugerencia("Elige Random Forest y XGBoost: los dos leen las mismas estadísticas, pero uno acierta mucho más.")
c1, c2 = st.columns(2)
a = c1.selectbox("Modelo A", modelos, index=modelos.index("xgboost") if "xgboost" in modelos else 0, format_func=nombre)
b = c2.selectbox("Modelo B", [m for m in modelos if m != a], format_func=nombre)
par = mc[((mc.modelo_A == a) & (mc.modelo_B == b)) | ((mc.modelo_A == b) & (mc.modelo_B == a))]
if not par.empty:
    p = par.iloc[0]
    gana_a = p["discrepancias_A_acierta"] if p["modelo_A"] == a else p["discrepancias_B_acierta"]
    gana_b = p["discrepancias_B_acierta"] if p["modelo_A"] == a else p["discrepancias_A_acierta"]
    fig = go.Figure(
        go.Bar(
            x=[gana_a, gana_b],
            y=[f"Solo acierta {nombre(a)}", f"Solo acierta {nombre(b)}"],
            orientation="h",
            marker_color=[TINTA, NEUTRO],
            text=[miles(gana_a), miles(gana_b)],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,} flujos<extra></extra>",
        )
    )
    fig.update_layout(height=170, hovermode="y", xaxis=dict(visible=False), margin=dict(l=10, r=60, t=10, b=10))
    st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)
    mejor = a if gana_a > gana_b else b
    if p["significativo"]:
        st.markdown(
            f"Cuando discrepan, **{nombre(mejor)}** acierta **{max(gana_a, gana_b) / max(min(gana_a, gana_b), 1):.1f} "
            f"veces más**. Con esos volúmenes, la probabilidad de que la diferencia sea azar es prácticamente "
            f"nula (p ajustado por Holm < 0,05): **la diferencia es real**."
        )
    else:
        st.markdown(
            f"Las discrepancias están equilibradas y la prueba no descarta el azar (p ajustado = "
            f"{decimal(p['p_holm'])}): **los dos modelos son estadísticamente equivalentes**, "
            "así que conviene el más simple."
        )

# ------------------------------------------------------------------ F1 por categoría
st.header("Qué tan bien reconoce cada categoría", divider="gray", anchor=False)
elegidos = st.pills(
    "Modelos",
    modelos,
    selection_mode="multi",
    default=[m for m in ("xgboost", "transformer") if m in modelos],
    format_func=nombre,
)
if elegidos:
    base = C["reportes"][elegidos[0]]
    cats = [c for c in base.index if c not in ("accuracy", "macro avg", "weighted avg")]
    orden = base.loc[cats, "f1-score"].sort_values().index
    tonos = [TINTA, TINTA_SUAVE, NEUTRO, LINEA]
    fig = go.Figure()
    for i, m in enumerate(elegidos):
        r = C["reportes"][m].loc[orden]
        fig.add_trace(
            go.Bar(
                x=r["f1-score"],
                y=orden,
                orientation="h",
                name=nombre(m),
                marker_color=tonos[i % 4],
                customdata=np.stack([r["precision"], r["recall"]], axis=1),
                hovertemplate="F1 %{x:.3f}   precisión %{customdata[0]:.3f}   "
                "recall %{customdata[1]:.3f}<extra>" + nombre(m) + "</extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        hovermode="y unified",  # un solo recuadro con todos los modelos para esa categoría
        height=int(max(520, 18 * len(orden) * len(elegidos) + 80)),
        xaxis=dict(title="F1", range=[0, 1]),
    )
    st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)
    st.markdown(
        '<p class="nota">Pasa el cursor por una categoría para ver el F1, la precisión y el recall de cada modelo. '
        "En las categorías "
        "pequeñas el recall suele ser alto y la precisión baja: los pesos por clase hacen que el modelo "
        'las "sospeche" con facilidad.</p>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ matriz de confusión
st.header("Con qué se confunde cada categoría", divider="gray", anchor=False)
m_conf = st.selectbox(
    "Modelo",
    modelos,
    index=modelos.index("xgboost") if "xgboost" in modelos else 0,
    format_func=nombre,
    key="modelo_confusion",
)
cm = C["matrices"][m_conf]
cmn = cm.div(cm.sum(axis=1).clip(lower=1), axis=0)
fig = go.Figure(
    go.Heatmap(
        z=cmn.values,
        x=cmn.columns,
        y=cmn.index,
        zmin=0,
        zmax=1,
        colorscale=[[0, FONDO], [0.15, "#C9D6F0"], [1, COBALTO]],
        colorbar=dict(title="Proporción"),
        hovertemplate="Real: %{y}<br>Predicha: %{x}<br>%{z:.1%} de la fila<extra></extra>",
    )
)
fig.update_layout(
    height=640,
    xaxis=dict(title="Categoría predicha", tickangle=-45),
    yaxis=dict(title="Categoría real", autorange="reversed"),
)
st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)

# ------------------------------------------------------------------ deriva
st.header("Cuánto pierden cuando el tráfico cambia", divider="gray", anchor=False)
fig = go.Figure()
meses = ["Octubre", "Noviembre", "Diciembre"]
for i, m in enumerate(modelos):
    y = [tabla.loc[m, "f1_macro"], tabla.loc[m, "f1_macro_2022-11"], tabla.loc[m, "f1_macro_2022-12"]]
    fig.add_trace(
        go.Scatter(
            x=meses,
            y=y,
            mode="lines+markers",
            name=nombre(m),
            line=dict(
                color=[TINTA, TINTA_SUAVE, NEUTRO, "#7C8C9C"][i % 4],
                width=2.5,
                dash=["solid", "solid", "dash", "dot"][i % 4],
            ),
        )
    )
fig.update_layout(height=360, hovermode="x unified", yaxis=dict(title="F1 macro"))
st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None)
caidas = {m: 100 * (tabla.loc[m, "f1_macro"] - tabla.loc[m, "f1_macro_2022-12"]) for m in modelos}
st.markdown(
    f"Todos pierden acierto a medida que se alejan del mes de entrenamiento. El que menos pierde es "
    f"**{nombre(min(caidas, key=caidas.get))}** ({decimal(min(caidas.values()), 1)} puntos entre octubre y "
    f"diciembre) y el que más, **{nombre(max(caidas, key=caidas.get))}** ({decimal(max(caidas.values()), 1)} puntos)."
)
siguiente_paso("paginas/modelos.py")
pie()
