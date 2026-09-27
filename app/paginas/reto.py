"""
Reto: "Tú contra los modelos".

El usuario ve la firma de un flujo real de octubre SIN su categoría y elige entre cuatro opciones.
Luego descubre la respuesta y lo que dijeron los cuatro modelos. Cinco rondas y un marcador.

Por qué: es la forma más directa de entender el proyecto. A simple vista, una persona no puede
"leer" una conexión cifrada; el modelo, entrenado con cientos de miles de ejemplos, sí.

Las opciones incluyen la categoría real y las dos que XGBoost consideró más probables después de
ella: así el reto es difícil pero justo, porque las alternativas son las que realmente se confunden.
"""

import numpy as np
import streamlit as st
from core.datos import NOMBRES, clases_precalculadas, mostrar_error_datos, muestra_octubre, probabilidades
from core.firma import mostrar_firma
from core.guia import resultado_modelo, siguiente_paso
from core.pie import pie

from src.data.loader import DIR, IPT, SIZE

RONDAS = 5
MODELOS = ("xgboost", "random_forest", "transformer", "cnn1d")

try:
    df = muestra_octubre()
    clases = clases_precalculadas()
except Exception as e:
    mostrar_error_datos(e)


def nueva_ronda() -> None:
    """Elige un flujo al azar y arma sus cuatro opciones."""
    rng = np.random.default_rng()
    i = int(rng.integers(len(df)))
    real = df.loc[i, "CATEGORY"]
    alternativas = [
        c for c in probabilidades(df.loc[i], "xgboost", clases).sort_values(ascending=False).index if c != real
    ][:2]
    resto = [c for c in clases if c != real and c not in alternativas]
    alternativas.append(str(rng.choice(resto)))
    opciones = [real, *alternativas]
    rng.shuffle(opciones)
    st.session_state.reto.update(fila=i, opciones=opciones, respuesta=None)


def reiniciar() -> None:
    st.session_state.reto = {"ronda": 1, "tuyos": 0, "modelo": 0, "fila": None, "opciones": [], "respuesta": None}
    nueva_ronda()


if "reto" not in st.session_state:
    reiniciar()
R = st.session_state.reto

st.title("Tú contra los modelos")
st.markdown(
    "Esta es la firma de una conexión real de octubre de 2022, sin su categoría. Solo ves lo mismo que el modelo: "
    "tamaños, direcciones y tiempos de sus paquetes. ¿A qué tipo de servicio pertenece?"
)

terminado = R["ronda"] > RONDAS
if not terminado:
    st.progress((R["ronda"] - 1) / RONDAS, text=f"Ronda {R['ronda']} de {RONDAS}")
st.markdown(
    f'<p class="marcador">Tus aciertos: <strong>{R["tuyos"]}</strong>'
    f'<span class="separa">XGBoost: <strong>{R["modelo"]}</strong></span></p>',
    unsafe_allow_html=True,  # solo números calculados por la app
)

if terminado:
    # ------------------------------------------------------------------ resultado final
    st.header("Resultado", divider="gray")
    tuyos, modelo = R["tuyos"], R["modelo"]
    if tuyos > modelo:
        mensaje = "Le ganaste al modelo. Pocas personas lo logran: tienes buen ojo para el tráfico."
    elif tuyos == modelo:
        mensaje = "Empate. Tu intuición estuvo a la altura del mejor modelo en estas rondas."
    else:
        mensaje = (
            "El modelo ganó. Es lo esperable: aprendió de cientos de miles de conexiones, "
            "y una persona no ve esos patrones a simple vista."
        )
    st.markdown(f"Acertaste **{tuyos} de {RONDAS}** y XGBoost **{modelo} de {RONDAS}**. {mensaje}")
    st.button("Jugar otra vez", type="primary", on_click=reiniciar, key="reto_otra_vez")
else:
    fila = df.loc[R["fila"]]
    real = fila["CATEGORY"]
    respondida = R["respuesta"] is not None
    mostrar_firma(
        fila[SIZE].to_numpy(),
        fila[DIR].to_numpy(),
        fila[IPT].to_numpy(),
        veredicto={"titulo": f"Era {real}", "detalle": f"Flujo de {int(fila['PPI_LEN'])} paquetes."}
        if respondida
        else None,
        animar=not respondida,
        clave=f"reto-{R['ronda']}-{respondida}",
    )

    if not respondida:
        st.markdown("**Elige una categoría**")
        columnas = st.columns(2)
        for k, opcion in enumerate(R["opciones"]):
            if columnas[k % 2].button(opcion, key=f"reto_opcion_{k}", width="stretch"):
                R["respuesta"] = opcion
                R["tuyos"] += int(opcion == real)
                R["modelo"] += int(fila["pred_xgboost"] == real)
                st.rerun()
    else:
        if R["respuesta"] == real:
            st.success(f"Acertaste: es {real}.")
        else:
            st.error(f"Elegiste {R['respuesta']}, pero es {real}.")
        st.markdown("**Lo que dijeron los modelos**")
        presentes = [m for m in MODELOS if f"pred_{m}" in df.columns]
        for col, m in zip(st.columns(len(presentes)), presentes, strict=True):
            with col:
                resultado_modelo(NOMBRES[m], fila[f"pred_{m}"], real, probabilidades(fila, m, clases).max())
        ultima = R["ronda"] == RONDAS

        def avanzar() -> None:
            R["ronda"] += 1
            if R["ronda"] <= RONDAS:
                nueva_ronda()

        st.button(
            "Ver el resultado" if ultima else "Siguiente ronda", type="primary", on_click=avanzar, key="reto_avanzar"
        )

siguiente_paso("paginas/reto.py")
pie()
