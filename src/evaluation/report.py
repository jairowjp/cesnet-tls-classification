"""
Salidas comunes de evaluación para todos los modelos.

Por qué un módulo compartido
----------------------------
Los modelos clásicos y los profundos se entrenan con scripts distintos, pero sus resultados
deben ser idénticos en formato para que scripts/04_comparar_modelos.py los compare sin
excepciones. Aquí vive la matriz de confusión que ambos generan.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figuras sin ventana gráfica (funciona igual en Kali, Colab o un servidor)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    """Matriz de confusión normalizada por fila: cada fila suma 1 y su diagonal es el recall de esa clase."""
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Categoría predicha")
    ax.set_ylabel("Categoría real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Proporción de la fila")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
