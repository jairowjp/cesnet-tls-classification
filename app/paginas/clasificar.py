"""
Clasificar: el modelo recomendado (XGBoost) trabajando en vivo.

Dos formas de probarlo (con abstención cuando el modelo duda y la explicación de cada decisión):
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
from core.explicacion import decidir, explicar, umbral_abstencion
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
UMBRAL = umbral_abstencion()["umbral"]


def mostrar_explicacion(X_flujo, clase: int) -> None:
    """¿Por qué esta categoría? Las cinco variables que más pesaron, con valores SHAP del propio XGBoost."""
    e = explicar(modelo, X_flujo, clase)
    fig = go.Figure(
        go.Bar(
            x=e["Aporte"][::-1],
            y=e["Variable"][::-1],
            orientation="h",
            marker_color=[TINTA if v > 0 else NEUTRO for v in e["Aporte"][::-1]],
            hovertemplate="<b>%{y}</b><br>Aporte: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=5, b=5), xaxis=dict(title="Aporte a la decisión"))
    st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None, key=f"explicacion_{clase}")
    st.markdown(
        '<p class="nota">Las barras oscuras empujan el flujo hacia esta categoría y las claras lo alejan. Son valores '
        "SHAP calculados por el propio modelo: sumados, reproducen exactamente su decisión.</p>",
        unsafe_allow_html=True,  # texto fijo del código
    )


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
    with st.spinner("Clasificando el flujo…"):
        X_flujo = tabular_features(df.loc[[i]])
        t0 = time.perf_counter()
        prob = modelo.predict_proba(X_flujo)[0]
        ms = (time.perf_counter() - t0) * 1000
    pred, real = clases[int(prob.argmax())], fila["CATEGORY"]
    se_abstiene = bool(decidir(prob.max(), UMBRAL))
    if se_abstiene:
        dos = pd.Series(prob, index=clases).nlargest(2)
        veredicto = {
            "titulo": f"XGBoost se abstiene: confianza baja ({decimal(prob.max() * 100, 0)} %)",
            "detalle": f"Duda entre {dos.index[0]} y {dos.index[1]}. Categoría real: {real}.",
        }
    else:
        veredicto = {
            "titulo": f"XGBoost dice: {pred}",
            "detalle": f"Categoría real: {real}. {'Acertó' if pred == real else 'Se equivocó'}.",
        }

    mostrar_firma(
        fila[SIZE].to_numpy(),
        fila[DIR].to_numpy(),
        fila[IPT].to_numpy(),
        veredicto=veredicto,
        animar=True,
        clave=str(i),
        alto=270,
    )

    st.subheader(f"¿Por qué {pred}?", anchor=False)
    mostrar_explicacion(X_flujo, int(prob.argmax()))
    if se_abstiene:
        st.info(
            f"La confianza del modelo es menor que {decimal(UMBRAL * 100, 0)} %, así que la aplicación no da una "
            "respuesta: prefiere reconocer la duda antes que arriesgar un error. La explicación muestra hacia dónde se "
            "inclinaba."
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
    ejemplo = ROOT / "data/samples/ejemplo_clasificador.csv"
    if st.button("Probar con un ejemplo", help="Carga 20 flujos reales de octubre con el formato de la plantilla"):
        st.session_state.csv_ejemplo = True
    archivo = st.file_uploader("Archivo CSV", type=["csv"], accept_multiple_files=False)
    if archivo is not None:
        st.session_state.csv_ejemplo = False  # un archivo propio reemplaza al ejemplo
    contenido = archivo.getvalue() if archivo is not None else None
    if contenido is None and st.session_state.get("csv_ejemplo") and ejemplo.exists():
        contenido = ejemplo.read_bytes()
        st.caption("Clasificando el archivo de ejemplo: 20 flujos reales de octubre de 2022.")
    if contenido is not None:
        resultado = validar_csv(contenido)  # el ejemplo pasa por la misma validación que un archivo subido
        if not resultado.valido:
            # Los mensajes son textos fijos de validar_csv: nunca repiten el contenido del archivo
            st.error("El archivo no se puede clasificar:\n\n" + "\n".join(f"- {e}" for e in resultado.errores))
        else:
            datos = resultado.datos
            with st.spinner(f"Clasificando {miles(len(datos))} flujos…"):
                prob = modelo.predict_proba(tabular_features(datos))
            dudosos = decidir(prob.max(axis=1), UMBRAL)
            salida = pd.DataFrame(
                {
                    "Flujo": np.arange(1, len(datos) + 1),
                    "Categoría predicha": [
                        "No clasificado: confianza baja" if d else clases[k]
                        for k, d in zip(prob.argmax(axis=1), dudosos, strict=True)
                    ],
                    "Más probable": [clases[k] for k in prob.argmax(axis=1)],
                    "Confianza": prob.max(axis=1),
                }
            )
            st.success(
                f"{miles(len(datos) - int(dudosos.sum()))} de {miles(len(datos))} flujos clasificados. "
                f"En {miles(int(dudosos.sum()))} el modelo se abstuvo por tener una confianza menor que "
                f"{decimal(UMBRAL * 100, 0)} %."
            )
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
