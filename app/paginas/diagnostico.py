"""
Diagnóstico: ¿aprendieron bien los modelos? (Actividad de la semana 3).

Muestra, de forma interactiva, el diagnóstico de sobreajuste y subajuste generado por
scripts/09_diagnostico.py: curvas de entrenamiento frente a validación por época o ronda, curvas de
Random Forest por tamaño del dataset y por hiperparámetro, y la comparación antes/después de las
estrategias de mejora. Todas las cifras se leen de results/diagnostico/.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.datos import diagnostico
from core.estilo import CONFIG_GRAFICO, NEUTRO, TINTA, TINTA_SUAVE, decimal
from core.guia import siguiente_paso, sugerencia
from core.pie import pie

NOMBRE = {"xgboost": "XGBoost", "cnn1d": "CNN 1D", "transformer": "Transformer"}
ETIQUETA = {"sobreajuste": "Sobreajuste", "subajuste": "Subajuste", "mixto": "Mixto", "óptimo": "Óptimo"}

st.title("¿Aprendieron bien los modelos?", anchor=False)
st.markdown(
    "Un modelo puede fallar de dos maneras opuestas. Si **memoriza** los ejemplos de entrenamiento, acierta mucho con "
    "ellos pero poco con datos nuevos: es el **sobreajuste**. Si **no alcanza a aprender** los patrones, "
    "falla en ambos: "
    "es el **subajuste**. Para saber cuál de los dos ocurre se comparan, paso a paso, el desempeño en los datos de "
    "entrenamiento y en los de validación, que el modelo no usa para aprender."
)

D = diagnostico()
if D is None:
    st.info("El diagnóstico todavía no se ha ejecutado. Se genera con `python scripts/09_diagnostico.py todo`.")
    siguiente_paso("paginas/diagnostico.py")
    pie()
    st.stop()

comp = D["comparacion"]


def fila(modelo: str, momento: str) -> pd.Series | None:
    sel = comp[(comp["modelo"] == modelo) & (comp["momento"] == momento)]
    return sel.iloc[0] if len(sel) else None


# ------------------------------------------------------------------ resumen del diagnóstico
st.header("Qué se encontró", divider="gray", anchor=False)
filas = []
for modelo in dict.fromkeys(comp["modelo"]):
    a, d = fila(modelo, "Antes"), fila(modelo, "Después")
    if a is None or d is None:
        continue
    filas.append(
        {
            "Modelo": modelo,
            "Diagnóstico inicial": ETIQUETA.get(str(a.get("diagnostico")), str(a.get("diagnostico"))),
            "F1 en datos nuevos, antes": a["f1_prueba"],
            "F1 en datos nuevos, después": d["f1_prueba"],
            "Mejora (puntos)": (d["f1_prueba"] - a["f1_prueba"]) * 100,
            "Brecha antes": a["brecha"],
            "Brecha después": d["brecha"],
        }
    )
tabla = pd.DataFrame(filas)
st.dataframe(
    tabla,
    hide_index=True,
    column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in tabla.columns if c.startswith(("F1", "Brecha"))
    }
    | {"Mejora (puntos)": st.column_config.NumberColumn(format="%+.1f")},
)
if len(tabla):
    mejor = tabla.loc[tabla["Mejora (puntos)"].idxmax()]
    subajuste = (tabla["Diagnóstico inicial"] == "Subajuste").sum()
    st.markdown(
        f"El problema dominante no era el sobreajuste sino el **subajuste**: {subajuste} de {len(tabla)} modelos no "
        f"terminaban de aprender. La mayor mejora la logró **{mejor['Modelo']}**: "
        f"{decimal(mejor['F1 en datos nuevos, antes'])} → {decimal(mejor['F1 en datos nuevos, después'])} de F1 macro "
        f"en octubre, datos que ningún modelo vio al entrenar."
    )
st.markdown(
    '<p class="nota">La brecha es el F1 de entrenamiento menos el de validación. '
    "Una brecha grande indica memorización; "
    "una brecha pequeña con un F1 bajo indica que el modelo no aprendió lo suficiente.</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ curvas por época o ronda
st.header("Las curvas de aprendizaje", divider="gray", anchor=False)
disponibles = [m for m in NOMBRE if (m, "antes") in D["historiales"]]
if disponibles:
    c1, c2, c3 = st.columns(3)
    modelo = c1.selectbox("Modelo", disponibles, format_func=NOMBRE.get)
    momento = (
        c2.segmented_control(
            "Momento",
            ["antes", "despues"],
            default="antes",
            format_func=lambda m: "Antes" if m == "antes" else "Después",
        )
        or "antes"
    )
    metrica = (
        c3.segmented_control(
            "Métrica", ["f1", "perdida"], default="f1", format_func=lambda m: "F1 macro" if m == "f1" else "Pérdida"
        )
        or "f1"
    )
    sugerencia(
        "Elige la CNN 1D o el Transformer y compara antes con después: en ambos casos las dos curvas siguen subiendo "
        "juntas al final, la señal de que el modelo aún estaba aprendiendo."
    )
    h = D["historiales"].get((modelo, momento))
    if h is None:
        st.info("No hay resultados para esa combinación.")
    else:
        ent, val = f"{metrica}_entrenamiento", f"{metrica}_validacion"
        datos = h[["paso", ent, val]].dropna()
        unidad = "Ronda" if modelo == "xgboost" else "Época"
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=datos["paso"],
                y=datos[ent],
                name="Entrenamiento",
                line=dict(color=TINTA, width=2.5),
                hovertemplate=f"{unidad} %{{x}}<br>Entrenamiento: %{{y:.4f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=datos["paso"],
                y=datos[val],
                name="Validación",
                line=dict(color=TINTA_SUAVE, width=2.5, dash="dash"),
                hovertemplate=f"{unidad} %{{x}}<br>Validación: %{{y:.4f}}<extra></extra>",
            )
        )
        mejor_i = datos[val].idxmin() if metrica == "perdida" else datos[val].idxmax()
        fig.add_annotation(
            x=datos.loc[mejor_i, "paso"],
            y=datos.loc[mejor_i, val],
            text="Mejor validación",
            showarrow=True,
            arrowhead=2,
            ax=-60,
            ay=-40 if metrica == "f1" else 40,
        )
        fig.update_layout(
            height=380,
            hovermode="x unified",
            xaxis=dict(title=unidad),
            yaxis=dict(
                title="F1 macro" if metrica == "f1" else "Pérdida",
                type="log" if (metrica == "perdida" and modelo == "xgboost") else "linear",
            ),
        )
        st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None, key=f"curva_{modelo}_{momento}_{metrica}")
        f = fila(NOMBRE[modelo], "Antes" if momento == "antes" else "Después")
        if f is not None and isinstance(f.get("diagnostico"), str):
            st.markdown(
                f"Diagnóstico de esta corrida: **{ETIQUETA.get(f['diagnostico'], f['diagnostico'])}**, con una "
                f"brecha final de {decimal(f['brecha'])}."
            )

# ------------------------------------------------------------------ Random Forest: tamaño e hiperparámetros
if D["rf"]:
    st.header("Random Forest: más datos y más libertad", divider="gray", anchor=False)
    rf = D["rf"]
    c1, c2 = st.columns(2)
    with c1:
        tam = rf["tamanos"]
        ent = [sum(v) / len(v) for v in rf["entrenamiento"]]
        val = [sum(v) / len(v) for v in rf["validacion"]]
        fig = go.Figure(
            [
                go.Scatter(x=tam, y=ent, name="Entrenamiento", line=dict(color=TINTA, width=2.5)),
                go.Scatter(
                    x=tam, y=val, name="Validación cruzada", line=dict(color=TINTA_SUAVE, width=2.5, dash="dash")
                ),
            ]
        )
        fig.update_layout(
            height=320,
            title="Según la cantidad de datos",
            hovermode="x unified",
            xaxis=dict(title="Flujos de entrenamiento"),
            yaxis=dict(title="F1 macro"),
        )
        st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None, key="rf_tamano")
    with c2:
        param = st.selectbox("Hiperparámetro", list(rf["hiperparametros"]), key="rf_param")
        hp = rf["hiperparametros"][param]
        etiquetas = ["sin límite" if v is None else str(v) for v in hp["valores"]]
        fig = go.Figure(
            [
                go.Scatter(x=etiquetas, y=hp["entrenamiento"], name="Entrenamiento", line=dict(color=TINTA, width=2.5)),
                go.Scatter(
                    x=etiquetas,
                    y=hp["validacion"],
                    name="Validación cruzada",
                    line=dict(color=TINTA_SUAVE, width=2.5, dash="dash"),
                ),
            ]
        )
        elegido = "sin límite" if hp["elegido"] is None else str(hp["elegido"])
        fig.add_vline(x=etiquetas.index(elegido), line=dict(color=NEUTRO, dash="dot"))
        fig.update_layout(
            height=320,
            title=f"Según {param}",
            hovermode="x unified",
            xaxis=dict(title=param, type="category"),
            yaxis=dict(title="F1 macro"),
        )
        st.plotly_chart(fig, config=CONFIG_GRAFICO, theme=None, key="rf_param_graf")
    st.markdown(
        "Con la configuración original, el diagnóstico de Random Forest fue "
        f"**{ETIQUETA.get(rf['diagnostico'], rf['diagnostico'])}**. "
        "La línea punteada vertical marca el valor elegido por la estrategia de mejora."
    )

# ------------------------------------------------------------------ lección
st.header("La lección", divider="gray", anchor=False)
st.markdown(
    "Una brecha pequeña entre entrenamiento y validación **no siempre es buena señal**: puede significar que el modelo "
    "es igual de deficiente en ambos conjuntos. Por eso el diagnóstico combina la brecha con el nivel de desempeño "
    "y con "
    "la tendencia de las curvas. El detalle completo está en el informe técnico de la actividad y en el notebook "
    "`overfitting_analysis.ipynb` del repositorio."
)
siguiente_paso("paginas/diagnostico.py")
pie()
