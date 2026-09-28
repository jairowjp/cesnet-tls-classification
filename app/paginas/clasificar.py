"""
Clasificar: el modelo recomendado (XGBoost) trabajando en vivo.

Dos formas de probarlo:
  1. Flujos reales de octubre de 2022, que ningún modelo usó para entrenar.
  2. Un CSV propio con el formato de la plantilla. Pasa por core/seguridad.py → validar_csv
     antes de tocar el modelo: tamaño, filas, columnas, tipos y rangos (OWASP A03 y A04).
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.datos import (
    NOMBRES,
    ROOT,
    clases_precalculadas,
    comparativa,
    modelo_xgboost,
    mostrar_error_datos,
    muestra_octubre,
    probabilidades,
)
from core.estilo import CONFIG_GRAFICO, NEUTRO, TINTA, decimal, miles
from core.firma import mostrar_firma
from core.guia import siguiente_paso, sugerencia
from core.pie import pie
from core.seguridad import MAX_BYTES, MAX_FILAS, neutralizar_formulas, validar_csv

from src.data.loader import DIR, IPT, SIZE, tabular_features

try:
    df = muestra_octubre()
    paquete = modelo_xgboost()
    clases_pre = clases_precalculadas()
except Exception as e:
    mostrar_error_datos(e)
try:
    comp = comparativa()  # opcional: solo aporta el orden de los modelos y su latencia en la evaluación
except Exception:
    comp = None

modelo, clases = paquete["modelo"], list(paquete["clases"])

st.title("Clasificar un flujo", anchor=False)
st.markdown(
    "El modelo recomendado, **XGBoost**, clasifica aquí en vivo. Puedes usar flujos reales de octubre de 2022 "
    "o subir los tuyos con el formato de la plantilla."
)

tab_real, tab_csv = st.tabs(["Flujos de octubre de 2022", "Tu propio archivo"])

# ================================================================== flujos reales
with tab_real:
    categorias = ["Cualquiera"] + sorted(df["CATEGORY"].unique())
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    filtro = c1.selectbox(
        "Categoría real del flujo",
        categorias,
        help="Elige una categoría para ver cómo le va al modelo con ella, o deja «Cualquiera».",
    )
    if (
        c2.button("Tomar un flujo", type="primary")
        or "clasificar_fila" not in st.session_state
        or st.session_state.get("clasificar_filtro") != filtro
    ):
        candidatos = df.index if filtro == "Cualquiera" else df.index[df["CATEGORY"] == filtro]
        st.session_state.clasificar_fila = int(np.random.default_rng().choice(candidatos))
        st.session_state.clasificar_filtro = filtro

    sugerencia(
        "Elige una categoría pequeña, como Virtual assistant o Remote desktop, y pulsa «Tomar un flujo» varias "
        "veces: ahí es donde los modelos dudan más y se nota la diferencia entre ellos."
    )
    i = st.session_state.clasificar_fila
    fila = df.loc[i]
    t0 = time.perf_counter()
    prob = modelo.predict_proba(tabular_features(df.loc[[i]]))[0]
    ms = (time.perf_counter() - t0) * 1000
    pred, real = clases[int(prob.argmax())], fila["CATEGORY"]

    mostrar_firma(
        fila[SIZE].to_numpy(),
        fila[DIR].to_numpy(),
        fila[IPT].to_numpy(),
        veredicto={
            "titulo": f"XGBoost dice: {pred}",
            "detalle": f"Categoría real: {real}. {'Acertó' if pred == real else 'Se equivocó'}.",
        },
        animar=True,
        clave=str(i),
        alto=270,
    )

    st.subheader("Qué dice cada modelo", anchor=False)
    st.markdown(
        '<p class="nota">Cada gráfico muestra las cinco categorías que cada modelo considera más probables. '
        "XGBoost clasifica en vivo; los otros tres usan sus probabilidades calculadas y guardadas con los modelos "
        "entrenados, para que la aplicación no necesite cargar las redes neuronales.</p>",
        unsafe_allow_html=True,  # texto fijo del código
    )
    # Orden de los modelos: de mayor a menor F1 macro en la evaluación (si está disponible)
    orden = [
        m for m in ("xgboost", "random_forest", "transformer", "cnn1d") if m == "xgboost" or f"pred_{m}" in df.columns
    ]
    if comp is not None:
        orden = sorted(orden, key=lambda m: -comp["tabla"]["f1_macro"].get(m, 0))
    for fila_modelos in (orden[:2], orden[2:]):
        for col, m in zip(st.columns(2), fila_modelos, strict=False):
            p = pd.Series(prob, index=clases) if m == "xgboost" else probabilidades(fila, m, clases_pre)
            dice = p.idxmax()
            top = p.sort_values(ascending=False).head(5)
            etiquetas = [f"{c} (real)" if c == real else c for c in top.index]
            with col:
                st.markdown(f"**{NOMBRES[m]}**  \n{'Acierta' if dice == real else 'Se equivoca'}: {dice}")
                fig = go.Figure(
                    go.Bar(
                        x=top.values[::-1],
                        y=etiquetas[::-1],
                        orientation="h",
                        marker_color=[TINTA if c == dice else NEUTRO for c in top.index][::-1],
                        text=[f"{v * 100:.1f} %".replace(".", ",") for v in top.values[::-1]],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.1%}<extra></extra>",
                    )
                )
                fig.update_layout(
                    height=230,
                    hovermode="y",
                    xaxis=dict(visible=False, range=[0, 1.2]),
                    margin=dict(l=10, r=10, t=5, b=5),
                )
                st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None, key=f"confianza_{m}")
                if comp is not None and m in comp["tabla"].index:
                    nota = (
                        f"En la evaluación: {decimal(comp['tabla'].loc[m, 'latencia_ms'], 3)} ms por flujo, en lotes."
                    )
                    if m == "xgboost":
                        nota = f"En vivo: {decimal(ms, 1)} ms para este flujo. " + nota
                    st.markdown(f'<p class="nota">{nota}</p>', unsafe_allow_html=True)  # cifras calculadas por la app

# ================================================================== CSV propio
with tab_csv:
    st.markdown(
        f"Sube un CSV con **una fila por flujo** y las mismas columnas de la plantilla: estadísticas del flujo, "
        f"histogramas y la secuencia de sus primeros 30 paquetes. Máximo **{MAX_FILAS} filas** y "
        f"**{MAX_BYTES // (1024 * 1024)} MB**."
    )
    plantilla = ROOT / "data/samples/plantilla_clasificador.csv"
    if plantilla.exists():
        st.download_button(
            "Descargar la plantilla", plantilla.read_bytes(), file_name="plantilla_clasificador.csv", mime="text/csv"
        )
    archivo = st.file_uploader("Archivo CSV", type=["csv"], accept_multiple_files=False)
    if archivo is not None:
        resultado = validar_csv(archivo.getvalue())
        if not resultado.valido:
            # Los mensajes son textos fijos de validar_csv: nunca repiten el contenido del archivo
            st.error("El archivo no se puede clasificar:\n\n" + "\n".join(f"- {e}" for e in resultado.errores))
        else:
            datos = resultado.datos
            prob = modelo.predict_proba(tabular_features(datos))
            salida = pd.DataFrame(
                {
                    "Flujo": np.arange(1, len(datos) + 1),
                    "Categoría predicha": [clases[k] for k in prob.argmax(axis=1)],
                    "Confianza": prob.max(axis=1),
                }
            )
            st.success(f"{miles(len(datos))} flujos clasificados.")
            st.dataframe(
                salida,
                hide_index=True,
                column_config={
                    "Confianza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
                },
            )
            primero = datos.iloc[0]
            mostrar_firma(
                primero[SIZE].to_numpy(),
                primero[DIR].to_numpy(),
                primero[IPT].to_numpy(),
                veredicto={
                    "titulo": f"Flujo 1: {salida.iloc[0]['Categoría predicha']}",
                    "detalle": "Firma del primer flujo del archivo.",
                },
                animar=False,
                alto=250,
                clave="csv",
            )
            st.download_button(
                "Descargar resultados",
                neutralizar_formulas(salida).to_csv(index=False).encode("utf-8"),
                file_name="clasificacion.csv",
                mime="text/csv",
            )
siguiente_paso("paginas/clasificar.py")
pie()
