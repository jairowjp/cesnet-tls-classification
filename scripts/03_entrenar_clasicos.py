"""
Entrena y evalúa un modelo clásico (Random Forest o XGBoost) con el protocolo del proyecto.

Protocolo (idéntico para todos los modelos, ver docs/semana2/02_analisis_comparativo_algoritmos.pdf)
------------------------------------------------------------------------------------------------
  1. Entrenamiento con septiembre de 2022; 20 % separado para validación (estratificado, semilla fija).
  2. Prueba con octubre de 2022: F1 macro (principal), F1 ponderado, exactitud, reporte por clase
     y matriz de confusión.
  3. F1 macro sobre la prueba SIN secuencias repetidas (el 30,3 % de octubre repite una secuencia
     de septiembre, ver EDA sección 2.11).
  4. Deriva: F1 macro en noviembre y diciembre, para medir cuánto envejece el modelo.
  5. Costo: tiempo de entrenamiento, latencia por flujo en CPU y tamaño del modelo en disco.
  6. Comparación contra los umbrales comprometidos (F1 macro ≥ 0,70; latencia ≤ 10 ms).

Salidas
-------
  results/modelos/<modelo>/metricas.json          todas las métricas y la verificación de umbrales
  results/modelos/<modelo>/reporte_por_clase.csv   precisión, recall y F1 de las 23 categorías
  results/modelos/<modelo>/matriz_confusion.png    matriz normalizada por fila
  results/modelos/<modelo>/predicciones_prueba.npz predicciones de octubre (para la prueba de McNemar)
  models/<modelo>.joblib                           modelo entrenado (no se sube a GitHub)

Uso:
    python scripts/03_entrenar_clasicos.py --modelo random_forest
    python scripts/03_entrenar_clasicos.py --modelo xgboost
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar src/ al ejecutar el script

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.data.loader import TABULAR, labels, load_months, load_split, tabular_features  # noqa: E402
from src.evaluation.metrics import f1_without_repeats, latency_ms_per_flow, performance  # noqa: E402
from src.evaluation.report import plot_confusion  # noqa: E402
from src.models.classic import MODELS, top_features  # noqa: E402


def memoria_pico_gb() -> float:
    """Memoria RAM máxima usada por el proceso hasta ahora, en GB (para dimensionar el equipo necesario)."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=list(MODELS), default="random_forest")
    ap.add_argument(
        "--limite-filas",
        type=int,
        default=0,
        help="usar solo N flujos de entrenamiento (prueba rápida del pipeline; 0 = todos)",
    )
    a = ap.parse_args()

    cfg = load_config()
    seed, umbrales = cfg["semilla"], cfg["umbrales"]
    params = cfg["modelos"][a.modelo]
    out = ROOT / "results" / "modelos" / a.modelo
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 1. datos
    t0 = time.time()
    train_df = load_split("entrenamiento")
    if a.limite_filas:
        # Prueba rápida: submuestra aleatoria con semilla fija. Los resultados NO son los oficiales.
        train_df = train_df.sample(n=min(a.limite_filas, len(train_df)), random_state=seed)
        print(f"MODO PRUEBA: {len(train_df):,} flujos de entrenamiento", flush=True)
    y_all, classes = labels(train_df)  # el orden de clases lo fija entrenamiento
    X_all = tabular_features(train_df)
    train_hashes = train_df["SEQ_HASH"].to_numpy()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=cfg["datos"]["fraccion_validacion"], stratify=y_all, random_state=seed
    )
    del train_df
    print(
        f"Entrenamiento: {len(y_tr):,} · validación: {len(y_val):,} · {len(classes)} clases "
        f"(carga en {time.time() - t0:.0f} s)",
        flush=True,
    )

    # ------------------------------------------------------------------ 2. entrenamiento
    build, fit = MODELS[a.modelo]
    extra = {"n_classes": len(classes)} if a.modelo == "xgboost" else {}
    model = build(params, seed, **extra)
    t0 = time.time()
    fit(model, X_tr, y_tr, X_val, y_val)
    train_s = time.time() - t0
    val_perf = performance(y_val, model.predict(X_val))
    print(f"Entrenado en {train_s / 60:.1f} min · F1 macro en validación: {val_perf['f1_macro']:.4f}", flush=True)

    # ------------------------------------------------------------------ 3. prueba (octubre)
    test_df = load_split("prueba")
    y_te, _ = labels(test_df, classes)
    X_te = tabular_features(test_df)
    pred = model.predict(X_te)
    test_perf = performance(y_te, pred)
    sin_rep = f1_without_repeats(y_te, pred, test_df["SEQ_HASH"].to_numpy(), train_hashes)
    np.savez_compressed(out / "predicciones_prueba.npz", y_true=y_te, y_pred=pred)

    rep = pd.DataFrame(
        classification_report(
            y_te, pred, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0
        )
    ).T
    rep.to_csv(out / "reporte_por_clase.csv")
    cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "matriz_confusion.csv")
    plot_confusion(
        cm,
        classes,
        out / "matriz_confusion.png",
        f"{a.modelo.replace('_', ' ').title()} · octubre 2022 · F1 macro {test_perf['f1_macro']:.3f}",
    )

    # ------------------------------------------------------------------ 4. deriva (noviembre y diciembre)
    deriva = {}
    for mes in cfg["datos"]["meses"]["deriva"]:
        d = load_months([mes])
        y_d, _ = labels(d, classes)
        deriva[mes] = performance(y_d, model.predict(tabular_features(d)))["f1_macro"]
        del d

    # ------------------------------------------------------------------ 5. costo
    latency = latency_ms_per_flow(model.predict, X_te)
    model_path = ROOT / "models" / f"{a.modelo}.joblib"
    joblib.dump({"modelo": model, "clases": classes, "variables": TABULAR}, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6

    # ------------------------------------------------------------------ 6. resultados y umbrales
    res = {
        "modelo": a.modelo,
        "modo_prueba": bool(a.limite_filas),
        "parametros": params,
        "entrenamiento": {"flujos": int(len(y_tr)), "segundos": round(train_s, 1)},
        "validacion": val_perf,
        "prueba_octubre": test_perf,
        "prueba_sin_repetidos": sin_rep,
        "deriva_f1_macro": deriva,
        "caida_f1_octubre_a_diciembre_puntos": round(100 * (test_perf["f1_macro"] - deriva.get("2022-12", np.nan)), 2),
        "costo": {
            "memoria_pico_GB": memoria_pico_gb(),
            "latencia_ms_por_flujo": round(latency, 5),
            "tamano_modelo_MB": round(size_mb, 1),
        },
        "top_variables": top_features(model, TABULAR),
        "umbrales": {
            "f1_macro_minimo_cumplido": test_perf["f1_macro"] >= umbrales["f1_macro_minimo"],
            "f1_macro_excelencia_cumplido": test_perf["f1_macro"] >= umbrales["f1_macro_excelencia"],
            "latencia_cumplida": latency <= umbrales["latencia_ms_maxima"],
        },
    }
    (out / "metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("\n================ RESULTADOS ================")
    print(
        f"F1 macro octubre ............ {test_perf['f1_macro']:.4f}  (mínimo {umbrales['f1_macro_minimo']}, "
        f"excelencia {umbrales['f1_macro_excelencia']})"
    )
    print(
        f"F1 macro sin repetidos ...... {sin_rep['f1_macro_sin_repetidos']:.4f}  "
        f"({sin_rep['fraccion_excluida']:.1%} de la prueba excluida)"
    )
    print(f"F1 ponderado / exactitud .... {test_perf['f1_weighted']:.4f} / {test_perf['accuracy']:.4f}")
    print("Deriva (F1 macro) ........... " + " · ".join(f"{m}: {v:.4f}" for m, v in deriva.items()))
    print(f"Latencia por flujo .......... {latency:.4f} ms (máximo {umbrales['latencia_ms_maxima']} ms)")
    print(f"Tamaño del modelo ........... {size_mb:.1f} MB")
    print(f"Umbrales .................... {res['umbrales']}")
    print(f"Resultados en ............... {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
