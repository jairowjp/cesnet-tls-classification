"""
Guía de participación: el sitio propone la siguiente acción en cada momento.

  * sugerencia(): una idea concreta para probar con los controles de la página actual.
  * siguiente_paso(): al final de cada página, invita al siguiente paso del recorrido.

El recorrido es una secuencia real (cada paso se apoya en el anterior), por eso se numera.
Todos los textos son fijos del código: nunca incluyen datos del usuario.
"""

import streamlit as st

# Orden del recorrido: (archivo de la página, invitación para llegar a ella)
RECORRIDO = [
    ("paginas/inicio.py", ""),
    ("paginas/datos.py", "Conoce los datos con los que aprendieron los modelos."),
    ("paginas/modelos.py", "Ya viste los datos. Ahora descubre qué modelo los clasifica mejor."),
    ("paginas/reto.py", "¿Podrías hacerlo tú? Compite contra los modelos en cinco rondas."),
    ("paginas/clasificar.py", "Explora cualquier flujo de octubre o sube los tuyos."),
    ("paginas/conclusiones.py", "Mira qué modelo se recomienda y cuáles son los límites del estudio."),
    ("paginas/encuesta.py", "Ayuda a evaluar el sitio: son 10 preguntas breves."),
]
ETIQUETAS = {
    "paginas/datos.py": "Ir a los datos",
    "paginas/modelos.py": "Ir a los modelos",
    "paginas/reto.py": "Empezar el reto",
    "paginas/clasificar.py": "Ir a clasificar",
    "paginas/conclusiones.py": "Ver las conclusiones",
    "paginas/encuesta.py": "Responder la encuesta",
}


def sugerencia(texto: str) -> None:
    """Recuadro con una idea para probar en la página. `texto` debe ser fijo del código."""
    st.markdown(f'<div class="sugerencia"><strong>Prueba esto.</strong> {texto}</div>', unsafe_allow_html=True)


def siguiente_paso(pagina_actual: str) -> None:
    """Invitación al siguiente paso del recorrido, con su posición (paso n de 7)."""
    rutas = [r for r, _ in RECORRIDO]
    if pagina_actual not in rutas:
        return
    i = rutas.index(pagina_actual)
    st.markdown(f'<p class="nota recorrido">Recorrido: paso {i + 1} de {len(RECORRIDO)}</p>', unsafe_allow_html=True)
    if i + 1 < len(RECORRIDO):
        destino, invitacion = RECORRIDO[i + 1]
        with st.container(border=True):
            st.markdown(f"**{invitacion}**")
            st.page_link(destino, label=ETIQUETAS[destino])
    else:
        with st.container(border=True):
            st.markdown("**Terminaste el recorrido. Gracias por participar.**")
            st.page_link("paginas/inicio.py", label="Volver al inicio")


def porcentaje(valor: float) -> str:
    """Porcentaje en español con espacio de no separación: «100 %» nunca se parte entre dos líneas."""
    return f"{valor * 100:.0f}\u00a0%"


def resultado_modelo(nombre: str, dice: str, real: str, confianza: float) -> None:
    """Resultado de un modelo en líneas separadas: nombre, veredicto, categoría predicha y confianza.

    Los textos provienen del propio proyecto (nombres de modelos y categorías del dataset).
    """
    veredicto = "Acierta" if dice == real else "Se equivoca"
    st.markdown(f"**{nombre}**  \n{veredicto}  \n{dice}  \nConfianza: {porcentaje(confianza)}")
