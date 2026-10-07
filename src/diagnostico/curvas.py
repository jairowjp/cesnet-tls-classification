"""
Curvas de aprendizaje (componente 2 de la actividad, 30 %).

Todas las figuras cumplen las especificaciones de la actividad (sección 2.3):
  * 300 DPI;                          * títulos descriptivos y ejes con unidades;
  * leyenda;                          * colores contrastantes para entrenamiento y validación;
  * cuadrícula;                       * anotaciones en los puntos críticos.

Colores: azul cobalto (entrenamiento) y ámbar (validación). La pareja azul-naranja se distingue
con los tipos de daltonismo más comunes, y se refuerza con estilos de línea distintos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figuras sin ventana gráfica
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ENTRENAMIENTO, VALIDACION, TINTA, GRIS = "#2B59C3", "#D98A00", "#1B2530", "#8A97A6"
DPI = 300

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _guardar(fig, archivo: Path) -> Path:
    archivo = Path(archivo)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(archivo, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return archivo


def curva_entrenamiento(
    hist: pd.DataFrame,
    columna_ent: str,
    columna_val: str,
    titulo: str,
    eje_y: str,
    archivo: Path,
    menor_es_mejor: bool,
    eje_x: str = "Época",
    escala_log: bool = False,
) -> Path:
    """Curva de entrenamiento frente a validación por época o ronda (curvas A y B de la actividad).

    Anota el mejor punto de validación y la brecha final, y sombrea la brecha entre ambas curvas.
    La brecha se expresa siempre de modo que un valor POSITIVO indique sobreajuste:
    validación − entrenamiento para la pérdida y entrenamiento − validación para el F1.
    """
    datos = hist[["paso", columna_ent, columna_val]].dropna()
    x, ent, val = datos["paso"].to_numpy(), datos[columna_ent].to_numpy(), datos[columna_val].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    marcas = len(x) <= 50
    ax.plot(x, ent, color=ENTRENAMIENTO, lw=2, marker="o" if marcas else None, ms=3.5, label="Entrenamiento")
    ax.plot(x, val, color=VALIDACION, lw=2, ls="--", marker="s" if marcas else None, ms=3.5, label="Validación")
    ax.fill_between(x, ent, val, color=GRIS, alpha=0.18, label="Brecha")
    if escala_log:
        ax.set_yscale("log")
    i = int(np.argmin(val) if menor_es_mejor else np.argmax(val))
    ax.scatter([x[i]], [val[i]], s=90, facecolors="none", edgecolors=TINTA, lw=1.6, zorder=5)
    ax.annotate(
        f"Mejor validación\n{eje_x.lower()} {int(x[i])}: {val[i]:.3f}",
        (x[i], val[i]),
        xytext=(0.55, 0.55 if menor_es_mejor else 0.30),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=TINTA, lw=1),
        fontsize=9,
    )
    brecha = (val[-1] - ent[-1]) if menor_es_mejor else (ent[-1] - val[-1])
    operacion = "validación − entrenamiento" if menor_es_mejor else "entrenamiento − validación"
    ax.text(
        0.98,
        0.97 if menor_es_mejor else 0.04,
        f"Brecha final ({operacion}): {brecha:+.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top" if menor_es_mejor else "bottom",
        fontsize=8.5,
        color=TINTA,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRIS, lw=0.6),
    )
    ax.set(title=titulo, xlabel=eje_x, ylabel=eje_y + (" (escala logarítmica)" if escala_log else ""))
    # Rincón libre según la forma típica: la pérdida baja desde arriba y el F1 sube desde abajo
    ax.legend(loc="lower left" if menor_es_mejor else "upper left")
    return _guardar(fig, archivo)


def curva_por_tamano(tamanos, puntaje_ent, puntaje_val, titulo: str, archivo: Path) -> Path:
    """Curva de aprendizaje por tamaño del conjunto de entrenamiento (curva C).

    puntaje_ent y puntaje_val: matrices (tamaños × pliegues) de learning_curve. Se dibuja la media
    y una banda de ± 1 desviación estándar entre pliegues.
    """
    me, de = puntaje_ent.mean(1), puntaje_ent.std(1)
    mv, dv = puntaje_val.mean(1), puntaje_val.std(1)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(tamanos, me, color=ENTRENAMIENTO, lw=2, marker="o", label="Entrenamiento")
    ax.fill_between(tamanos, me - de, me + de, color=ENTRENAMIENTO, alpha=0.15)
    ax.plot(tamanos, mv, color=VALIDACION, lw=2, ls="--", marker="s", label="Validación cruzada")
    ax.fill_between(tamanos, mv - dv, mv + dv, color=VALIDACION, alpha=0.15)
    ax.annotate(
        f"Brecha con todos los datos: {me[-1] - mv[-1]:.3f}",
        (tamanos[-1], (me[-1] + mv[-1]) / 2),
        xytext=(-190, 0),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRIS, lw=0.6),
    )
    ax.annotate(f"{mv[-1]:.3f}", (tamanos[-1], mv[-1]), xytext=(4, -12), textcoords="offset points", fontsize=8.5)
    ax.set(title=titulo, xlabel="Flujos de entrenamiento", ylabel="F1 macro")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f} mil"))
    ax.legend(loc="lower right")
    return _guardar(fig, archivo)


def curva_por_hiperparametro(
    valores, puntaje_ent, puntaje_val, parametro: str, titulo: str, archivo: Path, elegido=None
) -> Path:
    """Curva de validación por hiperparámetro (curva D).

    Marca el valor con mejor validación y, si se indica, el elegido por la estrategia de mejora.
    Los valores se dibujan como categorías, así un valor "sin límite" (None) también tiene lugar.
    """
    etiquetas = ["sin límite" if v is None else str(v) for v in valores]
    pos = np.arange(len(valores))
    me, mv = puntaje_ent.mean(1), puntaje_val.mean(1)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(pos, me, color=ENTRENAMIENTO, lw=2, marker="o", label="Entrenamiento")
    ax.plot(pos, mv, color=VALIDACION, lw=2, ls="--", marker="s", label="Validación cruzada")
    ax.fill_between(pos, mv - puntaje_val.std(1), mv + puntaje_val.std(1), color=VALIDACION, alpha=0.15)
    i = int(np.argmax(mv))
    ax.scatter([pos[i]], [mv[i]], s=90, facecolors="none", edgecolors=TINTA, lw=1.6, zorder=5)
    ax.annotate(
        f"Mejor validación: {mv[i]:.3f}",
        (pos[i], mv[i]),
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=TINTA, lw=1),
    )
    if elegido is not None and elegido in valores:
        j = valores.index(elegido)
        ax.axvline(pos[j], color=GRIS, ls=":", lw=1.4)
        derecha = j < len(valores) / 2  # el texto va hacia el lado con más espacio
        ax.annotate(
            "Elegido (menor brecha\nsin perder validación)",
            (pos[j], ax.get_ylim()[1]),
            xytext=(6 if derecha else -6, -26),
            textcoords="offset points",
            ha="left" if derecha else "right",
            fontsize=8.5,
            color=TINTA,
        )
    ax.set_xticks(pos, etiquetas)
    ax.set(title=titulo, xlabel=parametro, ylabel="F1 macro")
    ax.legend(loc="best")
    return _guardar(fig, archivo)


def comparacion_antes_despues(tabla: pd.DataFrame, titulo: str, archivo: Path) -> Path:
    """Barras agrupadas antes/después por modelo: F1 de validación y brecha de generalización.

    tabla: columnas modelo, momento ("Antes"/"Después"), f1_validacion, brecha.
    """
    modelos = list(dict.fromkeys(tabla["modelo"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, nombre in zip(
        axes, ("f1_validacion", "brecha"), ("F1 macro de validación", "Brecha (ent. − val.)"), strict=True
    ):
        pos = np.arange(len(modelos))
        for k, (momento, color, trama) in enumerate((("Antes", GRIS, None), ("Después", ENTRENAMIENTO, None))):
            vals = [tabla[(tabla.modelo == m) & (tabla.momento == momento)][col].mean() for m in modelos]
            barras = ax.bar(pos + (k - 0.5) * 0.38, vals, 0.38, color=color, hatch=trama, label=momento)
            ax.bar_label(barras, fmt="%.3f", fontsize=8, padding=2)
        if col == "f1_validacion":  # el eje empieza cerca del mínimo para que se vean diferencias de 1 o 2 puntos
            ax.set_ylim(max(0.0, tabla[col].min() - 0.08), min(1.0, tabla[col].max() + 0.04))
        ax.axhline(0, color=TINTA, lw=0.8) if col == "brecha" else None
        ax.set_xticks(pos, modelos)
        ax.set(title=nombre, ylabel=nombre)
        ax.legend(loc="best")
    fig.suptitle(titulo, fontweight="bold")
    return _guardar(fig, archivo)
