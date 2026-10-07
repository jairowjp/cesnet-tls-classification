"""
Genera el isotipo de la aplicación (app/static/isotipo.png): una mini firma de paquetes junto al nombre.

La firma repite el lenguaje visual de la app: barras cobalto hacia arriba (cliente → servidor) y ámbar
hacia abajo (servidor → cliente). El texto usa Archivo, la tipografía de la app (app/static/fonts).
Se usa en la barra de navegación con st.logo. Uso:  python scripts/utilidades/08_generar_isotipo.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FUENTE = ROOT / "app/static/fonts/Archivo-Variable.ttf"
SALIDA = ROOT / "app/static/isotipo.png"
COBALTO, AMBAR, TINTA, LINEA = "#2B59C3", "#D98A00", "#1B2530", "#D5DCE3"

font_manager.fontManager.addfont(str(FUENTE))
archivo = font_manager.FontProperties(fname=str(FUENTE), weight="bold")

fig = plt.figure(figsize=(6.4, 1.0), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 64)
ax.set_ylim(-5, 5)
ax.axis("off")
ax.plot([0, 9.6], [0, 0], color=LINEA, lw=1.2)
# (posición, altura): positivo sube (cliente → servidor), negativo baja (servidor → cliente)
for x, h in [(0.3, 2.6), (2.2, -3.6), (4.1, 4.2), (6.0, -2.0), (7.9, 2.0)]:
    ax.add_patch(plt.Rectangle((x, 0 if h > 0 else h), 1.25, abs(h), color=COBALTO if h > 0 else AMBAR, lw=0))
ax.text(12.0, -0.15, "Tráfico cifrado", fontproperties=archivo, fontsize=27, color=TINTA, va="center", ha="left")
fig.savefig(SALIDA, transparent=True, dpi=200)
# Recorte al contenido visible: sin espacio transparente sobrante, el logo se ve centrado y a buen tamaño
from PIL import Image  # noqa: E402

imagen = Image.open(SALIDA)
x0, y0, x1, y1 = imagen.getbbox()
imagen.crop((max(0, x0 - 6), max(0, y0 - 6), x1 + 6, y1 + 6)).save(SALIDA, optimize=True)
print(f"Isotipo generado: {SALIDA.relative_to(ROOT)}")
