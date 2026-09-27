"""
Sistema de diseño de la aplicación.

Idea central: "ver el tráfico sin leerlo". Los dos colores de acento no son decoración:
codifican la DIRECCIÓN de cada paquete en todos los gráficos de la aplicación.

  COBALTO → paquetes del cliente al servidor (la petición)
  ÁMBAR   → paquetes del servidor al cliente (la respuesta)

La pareja azul-naranja se distingue también con los tipos de daltonismo más comunes.
La tipografía es Archivo (licencia SIL OFL), servida desde la propia app (app/static/fonts),
sin pedir nada a servidores externos. Se usa su eje de ancho: expandida en los títulos.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Paleta -------------------------------------------------------------------------------
FONDO = "#F4F6F8"  # gris azulado claro: aspecto de sala de control técnica
SUPERFICIE = "#FFFFFF"
TINTA = "#1B2530"  # texto principal (azul muy oscuro, no negro puro)
TINTA_SUAVE = "#5B6B7A"  # texto secundario
LINEA = "#D5DCE3"  # rejillas y bordes
COBALTO = "#2B59C3"  # cliente → servidor
AMBAR = "#D98A00"  # servidor → cliente
ERROR = "#B3261E"  # solo para errores reales, nunca como decoración
NEUTRO = "#A9B6C3"  # categorías no destacadas en los gráficos
GRANATE = "#791632"  # color institucional de la UEES (tomado del logo): solo en la portada

FUENTE = "Archivo, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

CSS = f"""
<style>
  /* Contenedor principal: ancho cómodo de lectura, contenido alineado a la izquierda */
  [data-testid="stMainBlockContainer"] {{ max-width: 1120px; padding-top: 6rem; padding-bottom: 4rem; }}

  /* Títulos: el eje de ancho de Archivo hace que el tipo sea parte del diseño */
  h1 {{ font-stretch: 125%; font-weight: 700; font-size: clamp(2.1rem, 4.2vw, 3.3rem) !important;
        line-height: 1.04 !important; letter-spacing: -0.015em; color: {TINTA}; }}
  h2 {{ font-stretch: 112%; font-weight: 650; letter-spacing: -0.01em; color: {TINTA}; }}
  h3 {{ font-weight: 600; color: {TINTA}; }}

  /* Texto: líneas de menos de 80 caracteres, cifras tabulares para comparar números */
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, .portada-oficial, .nota {{
      max-width: 72ch; line-height: 1.6; text-align: justify; text-justify: inter-word; }}
  [data-testid="stDataFrame"], .cifra {{ font-variant-numeric: tabular-nums; }}

  /* En columnas angostas la justificación abre huecos entre palabras: ahí el texto va alineado a la izquierda */
  [data-testid="stColumn"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stColumn"] [data-testid="stMarkdownContainer"] li,
  [data-testid="stColumn"] .nota {{ text-align: left; }}

  /* Textos de apoyo */
  .nota {{ color: {TINTA_SUAVE}; font-size: 0.92rem; }}
  .sube {{ color: {COBALTO}; font-weight: 600; }}
  .baja {{ color: {AMBAR}; font-weight: 600; }}

  /* Foco visible para quien navega con teclado (accesibilidad) */
  a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, input:focus-visible {{
      outline: 2px solid {COBALTO} !important; outline-offset: 2px; }}

  /* Portada institucional (página de inicio): identidad UEES con el granate del logo */
  .portada-cabecera {{ display: flex; justify-content: space-between; align-items: center; gap: 1.5rem;
                       border-bottom: 3px solid {GRANATE}; padding-bottom: 0.9rem; margin-bottom: 1.6rem; }}
  .portada-cabecera img {{ height: 92px; width: auto; display: block; }}
  .portada-programa {{ text-align: right; color: {TINTA_SUAVE}; line-height: 1.45; font-size: 0.95rem; }}
  .portada-programa strong {{ color: {GRANATE}; font-weight: 650; }}
  .portada-oficial {{ color: {TINTA_SUAVE}; font-size: 1.06rem; line-height: 1.5; max-width: 68ch;
                     margin: 0 0 1.4rem 0; }}
  .portada-datos {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem 2rem;
                    margin: 0 0 2.2rem 0 !important; padding: 0 !important; }}
  .portada-datos div {{ border-left: 2px solid {LINEA}; padding-left: 0.8rem; margin: 0; }}
  .portada-datos dt {{ color: {TINTA_SUAVE}; font-size: 0.85rem; margin: 0; }}
  .portada-datos dd {{ color: {TINTA}; font-weight: 600; margin: 0.15rem 0 0 0 !important; }}
  @media (max-width: 720px) {{
    .portada-cabecera {{ flex-direction: column; align-items: flex-start; }}
    .portada-programa {{ text-align: left; }}
    .portada-datos {{ grid-template-columns: 1fr; }}
  }}

  /* Participación guiada: sugerencias, marcador del reto y posición en el recorrido */
  .sugerencia {{ background: #E8ECF0; border-left: 3px solid {TINTA}; padding: 0.7rem 1rem; margin: 0.6rem 0 1.2rem 0;
                 max-width: 72ch; line-height: 1.55; text-align: justify; }}
  .marcador {{ font-size: 1.1rem; font-variant-numeric: tabular-nums; }}
  .marcador .separa {{ margin-left: 2rem; }}
  .recorrido {{ margin-top: 2.5rem; margin-bottom: 0.3rem; }}

  @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def aplicar_estilo() -> None:
    """Inyecta el CSS propio. Es contenido fijo del código, nunca datos del usuario."""
    st.html(CSS)


def _plantilla() -> go.layout.Template:
    """Plantilla común de Plotly: misma tipografía, colores y rejilla en todos los gráficos."""
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FUENTE, color=TINTA, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[COBALTO, AMBAR, TINTA_SUAVE, NEUTRO],
        # automargin: Plotly amplía el margen lo necesario para que las etiquetas largas
        # (nombres de categorías como "Analytics & Telemetry") se lean completas
        xaxis=dict(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA, ticks="", automargin=True),
        yaxis=dict(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA, ticks="", automargin=True, ticksuffix="  "),
        margin=dict(l=10, r=10, t=30, b=10),
        # Recuadro al pasar el mouse: alto contraste (tinta oscura con texto blanco) y letra legible
        hoverlabel=dict(
            bgcolor=TINTA, bordercolor=TINTA, align="left", font=dict(family=FUENTE, color="#FFFFFF", size=14)
        ),
        # Formato numérico del español en todos los gráficos: coma decimal y espacio para los miles
        separators=", ",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return t


pio.templates["proyecto"] = _plantilla()
pio.templates.default = "proyecto"

# Configuración de la barra de herramientas de Plotly: sin logotipo ni botones de edición
CONFIG_GRAFICO = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# --- Formato de números en español ----------------------------------------------------------
def miles(n) -> str:
    """Entero con espacio como separador de miles (norma del español técnico): 3 133 138."""
    return f"{int(round(float(n))):,}".replace(",", " ")


def decimal(x, k: int = 3) -> str:
    """Número con coma decimal: 0,886."""
    return f"{float(x):.{k}f}".replace(".", ",")
