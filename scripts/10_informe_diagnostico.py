"""
Genera el informe técnico de la Actividad de la semana 3 (diagnostic_report.pdf) en formato UEES.

Todo se calcula a partir de results/diagnostico/: las tablas, las cifras y también las frases del
análisis de curvas. Ninguna afirmación sobre la forma de una curva está escrita a mano: si la curva
de validación de Random Forest sube con más datos, el informe lo dice; si se aplana, dice eso.

Uso:  python scripts/10_informe_diagnostico.py
Salida: docs/semana3/diagnostic_report.pdf
Requisitos: reportlab (uv pip install reportlab) y los resultados de scripts/09_diagnostico.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diagnostico.diagnostico import UMBRALES, metricas_curvas  # noqa: E402
from src.diagnostico.narrativa import analisis_red, analisis_rf, analisis_xgb  # noqa: E402

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    sys.exit("Falta reportlab. Instálalo con: uv pip install reportlab")

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/diagnostico"
SALIDA = ROOT / "docs/semana3/diagnostic_report.pdf"
LOGO = ROOT / "app/static/uees_logo.png"

# Tipografía con todos los caracteres del español y de los nombres de autores (Č, −): DejaVu, incluida en matplotlib
FUENTES = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
for nombre, archivo in (
    ("DV", "DejaVuSans"),
    ("DVB", "DejaVuSans-Bold"),
    ("DVI", "DejaVuSans-Oblique"),
    ("DVBI", "DejaVuSans-BoldOblique"),
):
    pdfmetrics.registerFont(TTFont(nombre, str(FUENTES / f"{archivo}.ttf")))
for negrita, cursiva, nombre in ((0, 0, "DV"), (1, 0, "DVB"), (0, 1, "DVI"), (1, 1, "DVBI")):
    addMapping("DV", negrita, cursiva, nombre)

GRANATE, TINTA, LINEA = colors.HexColor("#791632"), colors.HexColor("#1B2530"), colors.HexColor("#D5DCE3")
_base = getSampleStyleSheet()["Normal"]
P = ParagraphStyle(
    "p", parent=_base, fontName="DV", fontSize=10, leading=14, alignment=TA_JUSTIFY, textColor=TINTA, spaceAfter=6
)
H1 = ParagraphStyle("h1", parent=P, fontName="DVB", fontSize=13, textColor=GRANATE, spaceBefore=10, alignment=0)
C = ParagraphStyle("c", parent=P, alignment=TA_CENTER)
LEY = ParagraphStyle("ley", parent=C, fontName="DVI", fontSize=8.5, leading=11)
CEL = ParagraphStyle("cel", parent=P, fontSize=8.3, leading=10.5, alignment=0, spaceAfter=0)
REF = ParagraphStyle("ref", parent=P, fontSize=9, leading=12, leftIndent=12, firstLineIndent=-12)
NOMBRES = {"xgboost": "XGBoost", "cnn1d": "CNN 1D", "transformer": "Transformer"}


def n(x, k=3) -> str:
    """Número con coma decimal."""
    return f"{float(x):.{k}f}".replace(".", ",")


def puntos(x) -> str:
    return f"{float(x) * 100:+.1f}".replace(".", ",")


def tabla(filas, anchos):
    datos = [[Paragraph(f"<font color='white'><b>{c}</b></font>", CEL) for c in filas[0]]]
    datos += [[Paragraph(str(c), CEL) for c in f] for f in filas[1:]]
    t = Table(datos, colWidths=anchos, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GRANATE),
                ("GRID", (0, 0), (-1, -1), 0.4, LINEA),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F8")]),
            ]
        )
    )
    return t


def figuras(rutas_y_leyendas, ancho=8.1 * cm):
    """Figuras de dos en dos, con su leyenda; omite las que no existan."""
    celdas = []
    for ruta, leyenda in rutas_y_leyendas:
        if ruta.exists():
            celdas.append([Image(str(ruta), width=ancho, height=ancho * 0.59), Paragraph(leyenda, LEY)])
    filas = [celdas[i : i + 2] for i in range(0, len(celdas), 2)]
    salida = []
    for f in filas:
        t = Table([[c[0] for c in f], [c[1] for c in f]], colWidths=[ancho + 0.3 * cm] * len(f))
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        salida += [KeepTogether(t), Spacer(1, 6)]
    return salida


def pie(c, d):
    c.saveState()
    c.setFillColor(GRANATE)
    c.rect(0, A4[1] - 0.5 * cm, A4[0], 0.5 * cm, fill=1, stroke=0)
    c.setFont("DV", 8)
    c.setFillColor(TINTA)
    c.drawString(2 * cm, 1.2 * cm, "UEES · Maestría en Inteligencia Artificial · Proyecto Integrador (MIAR0545)")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {d.page}")
    c.restoreState()


# ------------------------------------------------------------------------------------- informe
def main():
    comp_ruta = R / "comparacion_antes_despues.csv"
    if not comp_ruta.exists():
        sys.exit("Faltan los resultados. Ejecuta primero: python scripts/09_diagnostico.py todo")
    comp = pd.read_csv(comp_ruta)
    rf = (
        json.loads((R / "random_forest/curvas.json").read_text())
        if (R / "random_forest/curvas.json").exists()
        else None
    )
    par = {
        m: (
            comp[(comp.modelo == m) & (comp.momento == "Antes")].iloc[0],
            comp[(comp.modelo == m) & (comp.momento == "Después")].iloc[0],
        )
        for m in dict.fromkeys(comp.modelo)
        if len(comp[comp.modelo == m]) == 2
    }
    diag_ini = {
        m: (str(a["diagnostico"]) if pd.notna(a["diagnostico"]) else "sin diagnóstico") for m, (a, _) in par.items()
    }
    n_sub = sum(d == "subajuste" for d in diag_ini.values())
    optimos = [m for m, d in diag_ini.items() if d == "óptimo"]
    mejoras = {m: d["f1_prueba"] - a["f1_prueba"] for m, (a, d) in par.items()}

    e = [Spacer(1, 1.5 * cm)]
    if LOGO.exists():
        e.append(Image(str(LOGO), width=7 * cm, height=3.23 * cm))
    e += [
        Spacer(1, 1 * cm),
        Paragraph("<b>MAESTRÍA EN INTELIGENCIA ARTIFICIAL</b>", C),
        Paragraph("Proyecto Integrador en Inteligencia Artificial (MIAR0545)", C),
        Spacer(1, 1.5 * cm),
        Paragraph("<font size=18 color='#791632'><b>Reporte técnico</b></font>", C),
        Spacer(1, 0.4 * cm),
        Paragraph(
            "<font size=14><b>Diagnóstico de sobreajuste y subajuste mediante curvas de aprendizaje</b></font>", C
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Clasificación de tráfico de red cifrado sobre el dataset CESNET-TLS-Year22", C),
        Spacer(1, 3 * cm),
        Paragraph("<b>Autor:</b> Jairo Wladimir Jhayya Perlaza", C),
        Paragraph("<b>Docente:</b> Gladys María Villegas Rugel", C),
        Paragraph("<b>Actividad:</b> Semana 3. Diagnóstico de Overfitting/Underfitting", C),
        Spacer(1, 1.5 * cm),
        Paragraph("Guayaquil, octubre de 2026", C),
        PageBreak(),
    ]

    # 1. Resumen ejecutivo
    lista_mejoras = "; ".join(
        f"{m}: {n(a['f1_prueba'])} → {n(d['f1_prueba'])} ({puntos(mejoras[m])} puntos)" for m, (a, d) in par.items()
    )
    e += [
        Paragraph("1. Resumen ejecutivo", H1),
        Paragraph(
            "Se diagnosticó el comportamiento de los cuatro modelos del proyecto (Random Forest, XGBoost, una red "
            "convolucional 1D y un Transformer ligero), que clasifican tráfico TLS cifrado en 23 categorías de "
            "servicio a partir de metadatos de flujo. Se implementó un sistema de tracking que registra la pérdida "
            "y el F1 macro de entrenamiento y validación en cada época o ronda, con MLflow como registro de "
            "experimentos; se generaron curvas de aprendizaje por época, por tamaño del dataset y por "
            "hiperparámetro, y el diagnóstico se automatizó con métricas cuantitativas y reglas explícitas.",
            P,
        ),
        Paragraph(
            f"<b>Hallazgo principal:</b> el problema dominante fue el <b>subajuste</b> ({n_sub} de {len(par)} "
            f"modelos). Diagnósticos iniciales: "
            f"{'; '.join(f'{m}: {d}' for m, d in diag_ini.items())}. "
            f"{'Ningún modelo presentó un comportamiento óptimo en su configuración inicial, ' if not optimos else ''}"
            f"lo que indica que todos tenían margen de mejora.",
            P,
        ),
        Paragraph(
            f"<b>Resultados en datos nunca vistos</b> (prueba de octubre de 2022, F1 macro): {lista_mejoras}.", P
        ),
        Paragraph(
            "<b>Lección metodológica:</b> una brecha pequeña entre entrenamiento y validación no es un objetivo en sí "
            "misma, porque puede indicar un modelo igualmente deficiente en ambos conjuntos. Por eso el "
            "diagnóstico combina la brecha con el nivel de desempeño y la tendencia de las curvas.",
            P,
        ),
        PageBreak(),
    ]

    # 2. Metodología
    e += [
        Paragraph("2. Metodología", H1),
        Paragraph(
            "<b>Datos.</b> Muestra del 2 % de CESNET-TLS-Year22: entrenamiento y validación con septiembre de 2022 "
            "(división 80/20 estratificada) y prueba con octubre. Para acotar el cómputo, el entrenamiento usó una "
            "submuestra estratificada de 200 000 flujos; la validación y la prueba se usaron completas. Antes y "
            "después se comparan con la misma muestra.",
            P,
        ),
        Paragraph(
            "<b>Métrica principal.</b> F1 macro, porque el desbalance entre clases es de 145:1 y la exactitud "
            "resultaría engañosa. Es el <i>score</i> de la curva B de la actividad.",
            P,
        ),
        Paragraph(
            "<b>Tracking.</b> Módulo <i>src/diagnostico/tracking.py</i>: (a) <i>RegistroMetricas</i>, que guarda cada "
            "paso en CSV/JSON y en MLflow (SQLite local); (b) un <i>callback</i> personalizado de XGBoost "
            "(<i>TrainingCallback</i>) que registra la pérdida en cada ronda y el F1 cada 10 rondas; (c) un bucle "
            "de PyTorch con registro manual por época; y (d) <i>learning_curve</i> y <i>validation_curve</i> de "
            "scikit-learn para Random Forest, con validación cruzada estratificada de 3 pliegues.",
            P,
        ),
        Paragraph(
            "<b>Sobre TensorFlow/Keras.</b> El proyecto no usa Keras: sus redes están implementadas en PyTorch. El "
            "requisito de <i>callbacks</i> personalizados se cubrió con el mismo patrón en las dos librerías que sí "
            "usa: el <i>callback</i> de XGBoost y el registro por época del bucle de PyTorch.",
            P,
        ),
        Paragraph(
            "<b>Protocolo.</b> El diagnóstico usó solo entrenamiento y validación; la prueba de octubre se usó una "
            "única vez, en la comparación final, para evitar el sobreajuste al conjunto de validación.",
            P,
        ),
        tabla(
            [
                ["Métrica", "Definición", "Señal (umbral)"],
                [
                    "Brecha final",
                    "F1 de entrenamiento − F1 de validación",
                    f"> {n(UMBRALES['brecha_alta'], 2)}: sobreajuste",
                ],
                [
                    "Pendientes de pérdida",
                    "Pendiente normalizada en el último 30 % del entrenamiento",
                    "Validación sube y entrenamiento baja: sobreajuste",
                ],
                [
                    "Nivel de F1 de entrenamiento",
                    "F1 alcanzado en entrenamiento",
                    f"< {n(UMBRALES['f1_bajo'], 2)} con brecha pequeña: subajuste",
                ],
                [
                    "Mejora final",
                    "Cambio del F1 de validación en el tramo final, con el mejor punto al final",
                    f"> {n(UMBRALES['mejora_minima'])}: aún aprendía",
                ],
            ],
            [3.6 * cm, 7.2 * cm, 5.8 * cm],
        ),
        Spacer(1, 6),
        Paragraph(
            "Con señales de ambos tipos, el diagnóstico es <i>mixto</i>; sin ninguna, <i>óptimo</i>. Las reglas se "
            "verificaron con pruebas automáticas sobre curvas de forma conocida.",
            P,
        ),
    ]

    # 3. Resultados del diagnóstico
    filas = [["Modelo", "Diagnóstico", "F1 ent.", "F1 val.", "Brecha"]]
    for m, (a, _) in par.items():
        filas.append([m, diag_ini[m].capitalize(), n(a["f1_entrenamiento"]), n(a["f1_validacion"]), n(a["brecha"])])
    e += [Paragraph("3. Resultados del diagnóstico", H1), tabla(filas, [3.4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])]
    if not optimos:
        e.append(
            Paragraph(
                "Ningún modelo resultó <i>óptimo</i> en su configuración inicial. No es una omisión del análisis: "
                "es el resultado, y justifica las estrategias de mejora de la sección 5.",
                P,
            )
        )

    # 4. Análisis de curvas
    e += [
        Paragraph("4. Análisis de las curvas de aprendizaje", H1),
        Paragraph("<i>Las frases de esta sección se generan a partir de las métricas de cada curva.</i>", P),
    ]
    for red in ("cnn1d", "transformer"):
        if (R / red / "antes/historial.csv").exists():
            e.append(Paragraph(f"<b>{NOMBRES[red]}.</b> {analisis_red(R, red)}", P))
    if (R / "xgboost/antes/historial.csv").exists():
        e.append(Paragraph(f"<b>XGBoost.</b> {analisis_xgb(R)}", P))
    if rf:
        e.append(Paragraph(f"<b>Random Forest.</b> {analisis_rf(rf)}", P))
    e += figuras(
        [
            (R / "xgboost/antes/A_perdida.png", "Figura 1. XGBoost (antes): pérdida por ronda (curva A)."),
            (R / "xgboost/antes/B_f1.png", "Figura 2. XGBoost (antes): F1 macro por ronda (curva B)."),
            (R / "cnn1d/antes/B_f1.png", "Figura 3. CNN 1D antes de la estrategia (curva B)."),
            (R / "cnn1d/despues/B_f1.png", "Figura 4. CNN 1D después de la estrategia (curva B)."),
            (R / "transformer/antes/B_f1.png", "Figura 5. Transformer antes de la estrategia (curva B)."),
            (R / "transformer/despues/B_f1.png", "Figura 6. Transformer después de la estrategia (curva B)."),
            (R / "random_forest/C_curva_por_tamano.png", "Figura 7. Random Forest: curva por tamaño del dataset (C)."),
            (
                R / "random_forest/D_curva_min_samples_leaf.png",
                "Figura 8. Random Forest: curva por min_samples_leaf (D).",
            ),
        ]
    )

    # 5. Estrategias
    estrategias = {
        "Random Forest": "Ajuste de complejidad guiado por las curvas D",
        "XGBoost": "Regularización: submuestreo de filas y columnas 0,8, L2 = 5, min_child_weight = 5",
        "CNN 1D": "Más épocas, tasa de aprendizaje OneCycle y parada temprana",
        "Transformer": "Más épocas, tasa de aprendizaje OneCycle y parada temprana",
    }
    filas = [["Modelo", "Estrategia", "F1 prueba antes", "F1 prueba después", "Brecha antes → después"]]
    for m, (a, d) in par.items():
        filas.append(
            [m, estrategias.get(m, ""), n(a["f1_prueba"]), n(d["f1_prueba"]), f"{n(a['brecha'])} → {n(d['brecha'])}"]
        )
    e += [
        PageBreak(),
        Paragraph("5. Estrategias implementadas y resultados", H1),
        tabla(filas, [2.6 * cm, 6.4 * cm, 2.2 * cm, 2.2 * cm, 3.2 * cm]),
        Spacer(1, 8),
    ]
    e += figuras(
        [(R / "comparacion_antes_despues.png", "Figura 9. Comparación antes/después (validación de septiembre).")],
        ancho=16 * cm,
    )
    if "Random Forest" in par:
        a, d = par["Random Forest"]
        mejor_con_mas_brecha = d["brecha"] > a["brecha"] and d["f1_prueba"] > a["f1_prueba"]
        e.append(
            Paragraph(
                f"En Random Forest la brecha pasó de {n(a['brecha'])} a {n(d['brecha'])} y el F1 de prueba de "
                f"{n(a['f1_prueba'])} a {n(d['f1_prueba'])}: "
                f"{'una brecha mayor convivió con una mejor generalización. ' if mejor_con_mas_brecha else ''}"
                "El ajuste de complejidad se eligió con las curvas de validación, no con la prueba.",
                P,
            )
        )

    # 6. Conclusiones
    e += [
        Paragraph("6. Conclusiones y recomendaciones futuras", H1),
        Paragraph(
            f"1. El diagnóstico sistemático cambió la lectura del proyecto: el problema dominante fue el subajuste "
            f"({n_sub} de {len(par)} modelos), no el sobreajuste.",
            P,
        ),
        Paragraph("2. La brecha de generalización debe interpretarse junto con el nivel de desempeño.", P),
    ]
    redes_sub = [
        NOMBRES[m]
        for m in ("cnn1d", "transformer")
        if (R / m / "despues/historial.csv").exists()
        and metricas_curvas(pd.read_csv(R / m / "despues/historial.csv"))["mejor_es_ultimo"]
    ]
    if redes_sub:
        e.append(
            Paragraph(
                f"3. {' y '.join(redes_sub)} todavía mejoraban al final del entrenamiento mejorado: se recomienda "
                "entrenar hasta que actúe la parada temprana y explorar mayor capacidad.",
                P,
            )
        )
    e += [
        Paragraph(
            "4. Repetir el análisis con los 640 000 flujos de entrenamiento y actualizar la comparación del proyecto "
            "y su aplicación web, donde este diagnóstico ya se presenta de forma interactiva.",
            P,
        ),
        Paragraph("7. Referencias técnicas", H1),
    ]
    for r in [
        "Hynek, K., Luxemburk, J., Pešek, J., Čejka, T., y Šiška, P. (2024). CESNET-TLS-Year22: A year-spanning TLS "
        "network traffic dataset from backbone lines. <i>Scientific Data, 11</i>, 1156.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning "
        "Research, 12</i>, 2825-2830.",
        "Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
        "<i>Proceedings of KDD '16</i>, 785-794.",
        "Smith, L. N., y Topin, N. (2019). Super-convergence: Very fast training of neural networks "
        "using large learning rates. <i>Proc. SPIE 11006</i>.",
        "Prechelt, L. (1998). Early stopping: But when? En <i>Neural Networks: Tricks of the Trade</i> "
        "(pp. 55-69). Springer.",
        "Goodfellow, I., Bengio, Y., y Courville, A. (2016). <i>Deep Learning</i>, caps. 5 y 7. MIT Press.",
        "Zaharia, M. et al. (2018). Accelerating the machine learning lifecycle with MLflow. <i>IEEE Data Engineering "
        "Bulletin, 41</i>(4), 39-45.",
    ]:
        e.append(Paragraph(r, REF))
    e.append(
        Paragraph(
            "Código, figuras, notebook y resultados: https://github.com/jairowjp/cesnet-tls-classification · "
            "Aplicación web: https://trafico-cifrado-uees.streamlit.app (página Diagnóstico)",
            P,
        )
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(SALIDA),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Diagnóstico de sobreajuste y subajuste",
        author="Jairo Wladimir Jhayya Perlaza",
    )
    doc.build(e, onFirstPage=pie, onLaterPages=pie)
    print(f"Informe generado: {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
