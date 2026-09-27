"""
Tu opinión: encuesta de usabilidad SUS (Brooke, 1996), integrada en el sitio.

Alimenta el hito H6 del proyecto: la prueba de usabilidad con al menos cinco evaluadores y un
puntaje SUS de 68 o más. El sitio NO guarda las respuestas (el almacenamiento de la plataforma gratuita no es
permanente y guardar datos de terceros agregaría riesgos). Al terminar, el evaluador recibe un código
corto que copia y envía al autor por mensaje o correo; el autor los reúne con scripts/07_consolidar_sus.py.
El código no contiene nombre ni alias: solo el perfil, las respuestas y las tareas.
"""

import streamlit as st
from core.guia import siguiente_paso
from core.pie import pie
from core.sus import PERFILES, PREGUNTAS, RESULTADOS_TAREA, TAREAS, codificar, interpretar, puntaje_sus

ESCALA = {1: "1  Totalmente en desacuerdo", 2: "2", 3: "3", 4: "4", 5: "5  Totalmente de acuerdo"}

st.title("Tu opinión")
st.markdown(
    "Tu evaluación ayuda a medir qué tan fácil es usar este sitio. Son 10 afirmaciones de la escala SUS, "
    "un estándar internacional de usabilidad, y 4 preguntas sobre las tareas que intentaste. Toma unos 3 minutos."
)
st.markdown(
    '<p class="nota">El sitio no guarda tus respuestas. Al final recibes un código corto: lo copias y se lo envías '
    "al autor por mensaje o correo. No se pide tu nombre.</p>",
    unsafe_allow_html=True,
)

with st.form("encuesta"):
    perfil = st.selectbox("Tu perfil", PERFILES)

    st.subheader("Qué tan de acuerdo estás")
    respuestas = []
    for n, texto in enumerate(PREGUNTAS, start=1):
        respuestas.append(
            st.radio(f"{n}. {texto}", list(ESCALA), format_func=ESCALA.get, index=None, horizontal=True, key=f"sus_{n}")
        )

    st.subheader("Qué tareas lograste")
    tareas = {
        t: st.radio(t, RESULTADOS_TAREA, index=None, horizontal=True, key=f"tarea_{k}")
        for k, t in enumerate(TAREAS, start=1)
    }
    enviado = st.form_submit_button("Calcular mi puntaje", type="primary")

if enviado:
    faltan = [n for n, r in enumerate(respuestas, start=1) if r is None]
    if faltan:
        st.error(f"Faltan {len(faltan)} afirmaciones por responder: la {', la '.join(map(str, faltan))}.")
    else:
        st.session_state.sus_resultado = {
            "puntaje": puntaje_sus(respuestas),
            "codigo": codificar(perfil, respuestas, {t: (v or "") for t, v in tareas.items()}),
        }

if "sus_resultado" in st.session_state:
    resultado = st.session_state.sus_resultado
    puntaje = resultado["puntaje"]
    st.header(f"Tu puntaje: {puntaje:.1f} de 100".replace(".", ","), divider="gray")
    st.markdown(
        f"Según la escala de referencia, tu evaluación indica una usabilidad **{interpretar(puntaje)}**. "
        "El proyecto se propuso alcanzar al menos 68, el promedio de referencia de la escala SUS."
    )
    st.subheader("Tu código de respuesta")
    st.markdown("Cópialo con el botón de la derecha y envíaselo al autor por WhatsApp o correo.")
    st.code(resultado["codigo"], language=None)  # st.code incluye un botón para copiar
    st.markdown(
        '<p class="nota">El código resume tus respuestas y tu perfil, sin datos personales. Incluye caracteres '
        "de control: si se copia mal, el autor lo notará al registrarlo. ¡Gracias por tu tiempo!</p>",
        unsafe_allow_html=True,
    )

siguiente_paso("paginas/encuesta.py")
pie()
