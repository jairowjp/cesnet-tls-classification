"""
Semana 5 · Workshop 2: optimización y sensibilidad de hiperparámetros de XGBoost con Optuna.

Subcomandos (en este orden):
  optimizar   Estudio de Optuna (TPE, poda por mediana) con validación cruzada agrupada SOLO sobre
              septiembre. Genera las figuras del workshop: historial, importancia (fANOVA y reducción de
              impureza), slice, dependencia parcial con un modelo sustituto, contorno y dependencia parcial 2D.
  comparar    La mejor configuración frente a la v2 en los MISMOS 15 pliegues de la parte 0 (5 x 3),
              con la prueba de Wilcoxon pareada (con 5 pliegues sería imposible lograr p < 0,05).
  reentrenar  La mejor configuración con TODO septiembre (mismo protocolo que el script 03) y su
              comparación con la v2 en validación. La prueba de octubre se usa solo en el cierre (script 12).

Uso:  python scripts/13_optuna_xgboost.py optimizar|comparar|reentrenar
Salidas: results/semana5/optuna/ y models/xgboost_optuna.joblib
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src import train as tr  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data.loader import TABULAR  # noqa: E402
from src.utils import RESULTADOS, configurar_logging, datos_entrenamiento, guardar  # noqa: E402

CFG = load_config()
S5, SEMILLA = CFG["semana5"], CFG["semilla"]
SALIDA = RESULTADOS / "optuna"
BASE_DATOS = SALIDA / "estudio.db"
NOMBRE = "xgboost_semana5"
V2 = CFG["modelos"]["xgboost"]
FIJOS = {
    k: v
    for k, v in V2.items()
    if k not in S5["optuna"]["espacio"] and k not in ("n_estimators", "early_stopping_rounds")
}


def configuracion_final(mejores: dict) -> dict:
    """La mejor configuración con el mismo presupuesto de rondas y parada temprana que la v2 (comparación justa)."""
    return {
        **FIJOS,
        **mejores,
        "n_estimators": V2["n_estimators"],
        "early_stopping_rounds": V2["early_stopping_rounds"],
    }


def optimizar(log):
    X, y, g, clases, _ = datos_entrenamiento(S5["muestra_entrenamiento"])
    cfg_o = S5["optuna"]
    estudio = tr.crear_estudio(BASE_DATOS, NOMBRE, SEMILLA)
    hechas = len([t for t in estudio.trials if t.state.is_finished()])
    faltan = max(0, cfg_o["pruebas"] - hechas)
    log.info(
        "Optuna · %d flujos · %d pruebas previas · faltan %d · tope %.1f h",
        len(y),
        hechas,
        faltan,
        cfg_o["horas_maximas"],
    )
    if faltan:
        estudio.optimize(
            lambda t: tr.objetivo(t, X, y, g, cfg_o, FIJOS, SEMILLA, len(clases)),
            n_trials=faltan,
            timeout=cfg_o["horas_maximas"] * 3600,
            callbacks=[
                lambda e, t: log.info(
                    "Prueba %d · %s · F1 %s · mejor %.4f",
                    t.number,
                    t.state.name,
                    f"{t.value:.4f}" if t.value else "—",
                    e.best_value,
                )
            ],
        )
    analizar(estudio, log)


def analizar(estudio: optuna.Study, log) -> None:
    """Todas las figuras y tablas del Workshop 2 a partir de las pruebas del estudio."""
    from optuna.visualization.matplotlib import plot_contour, plot_optimization_history, plot_slice

    SALIDA.mkdir(parents=True, exist_ok=True)
    completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    tabla = estudio.trials_dataframe()
    guardar(tabla, SALIDA / "pruebas.csv")
    imp = tr.importancias(estudio, SEMILLA)
    estados = tabla["state"].value_counts().to_dict()
    guardar(
        {
            "mejor_f1_cv": estudio.best_value,
            "mejores_parametros": estudio.best_params,
            "rondas_medias_mejor": estudio.best_trial.user_attrs.get("rondas_medias"),
            "pruebas": estados,
            "importancias": imp,
            "rango_f1_completas": [float(min(t.value for t in completas)), float(max(t.value for t in completas))],
        },
        SALIDA / "resumen.json",
    )
    ax = plot_optimization_history(estudio)
    ax.set_title("Historial de la optimización")
    ax.set_xlabel("Número de prueba")
    ax.set_ylabel("F1 macro (validación cruzada)")
    for linea in ax.get_legend().get_texts():
        linea.set_text(
            {"Objective Value": "F1 de cada prueba", "Best Value": "Mejor F1 hasta esa prueba"}.get(
                linea.get_text(), linea.get_text()
            )
        )
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(SALIDA / "historial.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # Importancia con dos métodos, lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, (metodo, valores) in zip(axes, imp.items(), strict=True):
        s = pd.Series(valores).sort_values()
        ax.barh(s.index, s.values)
        ax.set_title(f"Importancia ({'fANOVA' if metodo == 'fanova' else 'reducción de impureza'})")
    fig.tight_layout()
    fig.savefig(SALIDA / "importancias.png", dpi=150)
    plt.close(fig)
    axs = plot_slice(estudio)
    f = np.ravel(axs)[0].figure
    f.set_size_inches(20, 4)
    f.savefig(SALIDA / "slice.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # Dependencia parcial con un modelo sustituto (Random Forest que predice el F1 a partir de los hiperparámetros)
    df_t = tabla[tabla["state"] == "COMPLETE"]
    cols = [c for c in df_t.columns if c.startswith("params_")]
    Xp = df_t[cols].astype(float).rename(columns=lambda c: c.removeprefix("params_"))
    cols = list(Xp.columns)
    plt.style.use("default")  # Optuna activa un estilo con fondo gris; se vuelve al estilo del proyecto
    sustituto = RandomForestRegressor(n_estimators=300, random_state=SEMILLA).fit(Xp, df_t["value"])
    fig, axs = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.6), sharey=True)
    PartialDependenceDisplay.from_estimator(sustituto, Xp, features=cols, ax=axs)
    np.ravel(axs)[0].set_ylabel("F1 macro estimado (dependencia parcial)")
    fig.suptitle("Efecto de cada hiperparámetro sobre el F1 macro (modelo sustituto)")
    fig.tight_layout()
    fig.savefig(SALIDA / "dependencia_parcial.png", dpi=150)
    plt.close(fig)
    # Interacción del par más importante: contorno de Optuna y dependencia parcial 2D
    par = list(imp["fanova"])[:2]
    ax = plot_contour(estudio, params=par)
    ax.set_title(f"F1 macro según {par[0]} y {par[1]} (pruebas de Optuna)")
    ax.figure.axes[-1].set_ylabel("F1 macro (validación cruzada)")  # barra de color, en español
    ax.figure.set_size_inches(7, 5)
    ax.figure.savefig(SALIDA / "contorno.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 5))
    PartialDependenceDisplay.from_estimator(sustituto, Xp, features=[(par[0], par[1])], ax=ax)
    ax.set_title(f"Interacción entre {par[0]}\ny {par[1]} (dependencia parcial 2D)", fontsize=11)
    fig.tight_layout()
    fig.savefig(SALIDA / "dependencia_parcial_2d.png", dpi=150)
    plt.close(fig)
    log.info(
        "Mejor F1 (CV) %.4f · %s · importancias fANOVA %s",
        estudio.best_value,
        estudio.best_params,
        {k: round(v, 3) for k, v in imp["fanova"].items()},
    )


def comparar(log):
    X, y, g, clases, _ = datos_entrenamiento(S5["muestra_entrenamiento"])
    previos = RESULTADOS / "parte0" / "pliegues.csv"
    if not previos.exists():
        raise FileNotFoundError(
            "Falta la parte 0 (scripts/12_semana5_robustez.py parte0): ahí están los pliegues de la v2"
        )
    v2 = pd.read_csv(previos).query("modelo == 'xgboost_v2'").reset_index(drop=True)
    mejores = optuna.load_study(study_name=NOMBRE, storage=f"sqlite:///{BASE_DATOS}").best_params
    cv = S5["validacion_cruzada"]
    log.info(
        "Comparación · mejor configuración en los mismos %d pliegues de la parte 0", cv["pliegues"] * cv["repeticiones"]
    )
    opt = ev.validacion_cruzada(
        tr.entrenador_xgboost(configuracion_final(mejores), SEMILLA, len(clases)),
        X,
        y,
        g,
        cv["pliegues"],
        cv["repeticiones"],
        SEMILLA,
    )
    opt.insert(0, "modelo", "xgboost_optimizado")
    guardar(opt, SALIDA / "pliegues_optimizado.csv")
    res = {
        m: {
            "v2": ev.ic_bootstrap_media(v2[m], S5["bootstrap"], SEMILLA),
            "optimizado": ev.ic_bootstrap_media(opt[m], S5["bootstrap"], SEMILLA),
            "wilcoxon": ev.wilcoxon_pareado(v2[m], opt[m]),
        }
        for m in ("f1_macro", "f1_peor_categoria")
    }
    guardar(res, SALIDA / "comparacion_cv.json")
    w = res["f1_macro"]["wilcoxon"]
    log.info(
        "Wilcoxon (15 pliegues) · mediana de la diferencia %+.4f · p = %.4g · %s",
        w["mediana_diferencia"],
        w["p_valor"],
        "significativo" if w["significativo"] else "no significativo",
    )


def reentrenar(log):
    X, y, _, clases, _ = datos_entrenamiento()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=CFG["datos"]["fraccion_validacion"], stratify=y, random_state=SEMILLA
    )  # mismo protocolo que el script 03
    mejores = optuna.load_study(study_name=NOMBRE, storage=f"sqlite:///{BASE_DATOS}").best_params
    params = configuracion_final(mejores)
    log.info("Reentrenamiento con %d flujos · %s", len(y_tr), params)
    t0 = time.time()
    modelo = tr.build_xgboost(params, SEMILLA, len(clases))
    tr.fit_xgboost(modelo, X_tr, y_tr, X_val, y_val)  # un solo entrenamiento, con parada temprana en validación
    minutos = (time.time() - t0) / 60
    proba = modelo.predict_proba(X_val)
    val = ev.metricas(y_val, proba.argmax(axis=1), proba)
    v2_val = json.loads((ROOT / "results/modelos/xgboost/metricas.json").read_text())["validacion"]["f1_macro"]
    rondas = [int(getattr(modelo, "best_iteration", params["n_estimators"]))]
    joblib.dump(
        {"modelo": modelo, "clases": clases, "variables": TABULAR},
        ROOT / "models" / "xgboost_optuna.joblib",
        compress=3,
    )
    guardar(
        {
            "parametros": params,
            "rondas_usadas": rondas[0],
            "minutos_entrenamiento": minutos,
            "validacion_optimizado": val,
            "f1_macro_validacion_v2": v2_val,
            "mejora_validacion_puntos": 100 * (val["f1_macro"] - v2_val),
        },
        SALIDA / "reentrenamiento.json",
    )
    log.info(
        "Validación · optimizado %.4f · v2 %.4f · diferencia %+.2f puntos · %.1f min",
        val["f1_macro"],
        v2_val,
        100 * (val["f1_macro"] - v2_val),
        minutos,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=["optimizar", "comparar", "reentrenar"])
    paso = ap.parse_args().paso
    log = configurar_logging(f"13_{paso}")
    t0 = time.time()
    try:
        {"optimizar": optimizar, "comparar": comparar, "reentrenar": reentrenar}[paso](log)
    except Exception:
        log.exception("Falló el paso %s", paso)
        raise
    log.info("Paso %s terminado en %.1f min", paso, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
