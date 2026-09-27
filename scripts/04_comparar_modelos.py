"""
Compara todos los modelos entrenados con el mismo protocolo y decide cuál recomendar.

Qué hace
--------
  1. Reúne las métricas de cada modelo (results/modelos/<modelo>/metricas.json) en una tabla.
  2. Aplica la prueba de McNemar a cada par de modelos sobre las mismas predicciones de octubre.
  3. Corrige los valores p con el método de Holm. Con varios pares, comparar cada uno a α = 0,05
     infla la probabilidad de declarar una diferencia falsa; Holm controla ese error en conjunto.
  4. Grafica F1 macro frente a latencia: la relación entre desempeño y costo que pide el proyecto.

Regla de recomendación del proyecto
-----------------------------------
Un modelo más complejo solo se recomienda si supera al más simple con diferencia significativa
(p corregido < 0,05). Si no, se recomienda el más simple y barato.

Uso:  python scripts/04_comparar_modelos.py
Salida: results/comparativa/ (tabla_comparativa.csv, mcnemar.csv, f1_vs_latencia.png)
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar src/ al ejecutar el script

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.evaluation.stats import mcnemar_test  # noqa: E402

# Orden de complejidad: de más simple a más complejo (decide a quién favorece la regla en un empate)
COMPLEJIDAD = ["random_forest", "xgboost", "cnn1d", "lstm", "transformer"]


def holm(pvalues: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni: ordena los p de menor a mayor y los multiplica por (m - i),
    manteniendo la monotonía. Es menos conservadora que Bonferroni y controla el error familiar."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def main():
    alpha = load_config()["umbrales"]["alfa_mcnemar"]
    base = ROOT / "results" / "modelos"
    out = ROOT / "results" / "comparativa"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. tabla de métricas
    modelos = sorted((p.parent.name for p in base.glob("*/metricas.json")),
                     key=lambda m: COMPLEJIDAD.index(m) if m in COMPLEJIDAD else 99)
    if len(modelos) < 2:
        sys.exit("Se necesitan al menos dos modelos entrenados en results/modelos/")
    filas, preds = [], {}
    for m in modelos:
        r = json.loads((base / m / "metricas.json").read_text())
        if r.get("modo_prueba"):
            print(f"AVISO: {m} se entrenó en modo prueba; sus resultados no son oficiales")
        filas.append({
            "modelo": m,
            "f1_macro": r["prueba_octubre"]["f1_macro"],
            "f1_macro_sin_repetidos": r["prueba_sin_repetidos"]["f1_macro_sin_repetidos"],
            "f1_ponderado": r["prueba_octubre"]["f1_weighted"],
            "exactitud": r["prueba_octubre"]["accuracy"],
            **{f"f1_macro_{mes}": v for mes, v in r["deriva_f1_macro"].items()},
            "latencia_ms": r["costo"]["latencia_ms_por_flujo"],
            "tamano_MB": r["costo"]["tamano_modelo_MB"],
            "entrenamiento_min": round(r["entrenamiento"]["segundos"] / 60, 1),
        })
        p = np.load(base / m / "predicciones_prueba.npz")
        preds[m] = (p["y_true"], p["y_pred"])
    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla.round(4).to_csv(out / "tabla_comparativa.csv")

    # ------------------------------------------------------------ 2 y 3. McNemar por pares + Holm
    y_ref = preds[modelos[0]][0]
    for m in modelos[1:]:
        if not np.array_equal(preds[m][0], y_ref):
            sys.exit(f"Las etiquetas de prueba de {m} no coinciden: los modelos no se evaluaron sobre los mismos datos")
    pares = []
    for a, b in itertools.combinations(modelos, 2):
        r = mcnemar_test(y_ref, preds[a][1], preds[b][1], alpha)
        pares.append({"modelo_A": a, "modelo_B": b, "discrepancias_A_acierta": r["tabla"][0][1],
                      "discrepancias_B_acierta": r["tabla"][1][0], "estadistico": r["estadistico"],
                      "p_valor": r["p_valor"], "mejor": {"A": a, "B": b}.get(r["mejor"], "empate")})
    mc = pd.DataFrame(pares)
    mc["p_holm"] = holm(mc["p_valor"].tolist())
    mc["significativo"] = mc["p_holm"] < alpha
    mc.to_csv(out / "mcnemar.csv", index=False)

    # ------------------------------------------------------------ 4. gráfica desempeño frente a costo
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(tabla["latencia_ms"], tabla["f1_macro"], s=tabla["tamano_MB"].clip(20, 400), color="#4a4a4a", alpha=0.8)
    for m, row in tabla.iterrows():
        ax.annotate(f"{m}\n{row['tamano_MB']:.0f} MB", (row["latencia_ms"], row["f1_macro"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    umbrales = load_config()["umbrales"]
    ax.axhline(umbrales["f1_macro_minimo"], ls="--", lw=0.8, color="grey")
    ax.axhline(umbrales["f1_macro_excelencia"], ls=":", lw=0.8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Latencia por flujo en CPU (ms, escala log)")
    ax.set_ylabel("F1 macro en octubre de 2022")
    ax.set_title("Desempeño frente a costo (tamaño del punto = tamaño del modelo)")
    fig.tight_layout()
    fig.savefig(out / "f1_vs_latencia.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ resumen en pantalla
    pd.set_option("display.width", 160)
    print("\n================ TABLA COMPARATIVA ================")
    print(tabla.round(4).to_string())
    print("\n================ McNEMAR (corrección de Holm) ================")
    columnas = ["modelo_A", "modelo_B", "discrepancias_A_acierta", "discrepancias_B_acierta",
                "p_holm", "significativo", "mejor"]
    print(mc[columnas].to_string(index=False))
    # Regla del proyecto: se recomienda el modelo MÁS SIMPLE entre los que no difieren
    # significativamente del de mayor F1 macro (empate estadístico = gana el más barato).
    mejor = tabla["f1_macro"].idxmax()

    def significativo(a: str, b: str) -> bool:
        fila = mc[((mc.modelo_A == a) & (mc.modelo_B == b)) | ((mc.modelo_A == b) & (mc.modelo_B == a))]
        return bool(fila["significativo"].iloc[0])

    empatados = [mejor] + [m for m in modelos if m != mejor and not significativo(m, mejor)]
    recomendado = min(empatados, key=lambda m: COMPLEJIDAD.index(m) if m in COMPLEJIDAD else 99)
    print("\n================ RECOMENDACIÓN ================")
    print(f"Mayor F1 macro: {mejor} ({tabla.loc[mejor, 'f1_macro']:.4f})")
    if len(empatados) > 1:
        print(f"Sin diferencia significativa con {mejor}: {', '.join(m for m in empatados if m != mejor)}")
    else:
        print(f"{mejor} supera con significancia a todos los demás modelos (p de Holm < {alpha})")
    print(f"Modelo recomendado: {recomendado} (el más simple entre los estadísticamente equivalentes al mejor)")
    print(f"Resultados en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
