"""
Semana 5 · Parte 0, Workshop 1, análisis de errores y cierre (adaptación del notebook de la docente).

Subcomandos (en este orden):
  parte0     Baseline dummy frente a XGBoost: validación cruzada AGRUPADA 5 x 3 con IC 95 % por bootstrap,
             y tabla de métricas en una submuestra de octubre.
  workshop1  Escenarios críticos sobre el modelo de producción (v2): ruido, faltantes, dominancia de una
             categoría, cambio de distribución, variables caídas y relleno de paquetes (TAREA 2), con el
             umbral de aceptación fijado ANTES en configs/experiment.yaml.
  errores    Análisis de errores del modelo de producción sobre todo octubre: pares confundidos, error según
             la longitud del flujo, calibración, secuencias repetidas frente a nuevas y equidad por categoría.
  cierre     Evaluación final única: v2 frente al modelo optimizado (requiere el script 13), con IC de la
             diferencia, McNemar y prueba de estrés de ambos.

Uso:  python scripts/12_semana5_robustez.py parte0|workshop1|errores|cierre
Salidas: results/semana5/<subcomando>/
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data.loader import TABULAR  # noqa: E402
from src.evaluation.stats import mcnemar_test  # noqa: E402
from src.train import entrenador_xgboost  # noqa: E402
from src.utils import (  # noqa: E402  # noqa: E402
    RESULTADOS,
    cargar_modelo,
    configurar_logging,
    datos_entrenamiento,
    datos_prueba,
    guardar,
    predictor,
)

CFG = load_config()
S5, SEMILLA = CFG["semana5"], CFG["semilla"]
MODELO_V2 = ROOT / "models" / "xgboost.joblib"
MODELO_OPT = ROOT / "models" / "xgboost_optuna.joblib"


def entrenador_dummy(Xtr, ytr, Xva=None, yva=None):
    """Baseline: predice al azar respetando la proporción de clases. Si un modelo no lo supera, no aporta nada."""
    m = DummyClassifier(strategy="stratified", random_state=SEMILLA).fit(Xtr, ytr)
    return lambda X: (m.predict(X), m.predict_proba(X))


def parte0(log):
    X, y, g, clases, _ = datos_entrenamiento(S5["muestra_entrenamiento"])
    cv = S5["validacion_cruzada"]
    log.info("Parte 0 · %d flujos de septiembre · CV agrupada %d x %d", len(y), cv["pliegues"], cv["repeticiones"])
    folds = {}
    for nombre, entrenar in (
        ("dummy", entrenador_dummy),
        ("xgboost_v2", entrenador_xgboost(CFG["modelos"]["xgboost"], SEMILLA, len(clases))),
    ):
        t0 = time.time()
        folds[nombre] = ev.validacion_cruzada(entrenar, X, y, g, cv["pliegues"], cv["repeticiones"], SEMILLA)
        folds[nombre].insert(0, "modelo", nombre)
        log.info("%s listo en %.1f min", nombre, (time.time() - t0) / 60)
    tabla_cv = pd.concat(folds.values())
    guardar(tabla_cv, RESULTADOS / "parte0" / "pliegues.csv")
    resumen = {
        n: {m: ev.ic_bootstrap_media(f[m], S5["bootstrap"], SEMILLA) for m in ("f1_macro", "f1_peor_categoria")}
        for n, f in folds.items()
    }
    guardar(resumen, RESULTADOS / "parte0" / "intervalos.json")
    # Tabla dummy vs. modelo de producción en una submuestra de octubre (como la tabla de la parte 0 del notebook)
    Xte, yte, _ = datos_prueba(clases, S5["muestra_estres"])
    produccion = cargar_modelo(MODELO_V2)["modelo"]
    filas = {
        "dummy": ev.metricas(yte, *entrenador_dummy(X, y)(Xte)),
        "xgboost_v2 (producción)": ev.metricas(yte, *predictor(produccion)(Xte)),
    }
    guardar(pd.DataFrame(filas).T.reset_index(names="modelo"), RESULTADOS / "parte0" / "tabla_octubre.csv")
    for n, r in resumen.items():
        f1 = r["f1_macro"]
        log.info(
            "%-11s F1 macro (CV) = %.4f ± %.4f  IC95 %% [%.4f, %.4f]",
            n,
            f1["media"],
            f1["desviacion"],
            f1["ic95_bajo"],
            f1["ic95_alto"],
        )


def workshop1(log):
    X, y, _, clases, df_tr = datos_entrenamiento(S5["muestra_entrenamiento"])
    Xte, yte, _ = datos_prueba(clases, S5["muestra_estres"])
    modelo = cargar_modelo(MODELO_V2)["modelo"]
    predecir, rep, esc = predictor(modelo), S5["repeticiones"], S5["escenarios"]
    mediana, desviacion = np.median(X, axis=0), X.std(axis=0)
    base = ev.metricas(yte, *predecir(Xte))
    filas = [
        {"escenario": "Base", "nivel": "sin perturbar", **{k: v for k, v in base.items() if k != "peor_categoria"}}
    ]

    def agregar(escenario, nivel, perturbar, n=rep):
        filas.append(
            {"escenario": escenario, "nivel": nivel, **ev.evaluar_perturbado(predecir, Xte, yte, perturbar, n, SEMILLA)}
        )
        log.info("%-24s %-22s F1 macro %.4f", escenario, nivel, filas[-1]["f1_macro"])

    for s in esc["ruido"]:
        agregar("A · Ruido", f"σ={s}", ev.ruido(s, desviacion))
    for p in esc["faltantes"]:
        agregar("B · Faltantes", f"{int(p * 100)} %", ev.faltantes(p, mediana))
    mayoritaria = int(np.bincount(yte).argmax())
    for f in esc["dominancia"]:
        agregar("C · Dominancia", f"{clases[mayoritaria]} {int(f * 100)} %", ev.dominancia(mayoritaria, f))
    top3 = list(np.argsort(modelo.feature_importances_)[::-1][:3])
    for c in top3:
        agregar("E · Variable caída", TABULAR[c], ev.variable_caida([c], mediana), n=1)
    agregar("E · Variable caída", "las 3 juntas", ev.variable_caida(top3, mediana), n=1)
    for p in esc["relleno"]:
        agregar("F · Relleno (TAREA 2)", f"{int(p * 100)} % de los flujos", ev.relleno(p))

    # D · Cambio de distribución: entrenar con flujos cortos y probar con largos (como tumores pequeños/grandes)
    longitud = df_tr["PACKETS"].to_numpy() + df_tr["PACKETS_REV"].to_numpy()
    corto = longitud <= np.median(longitud)
    idx_c = np.flatnonzero(corto)
    rng = np.random.default_rng(SEMILLA)
    rng.shuffle(idx_c)
    corte = int(len(idx_c) * 0.7)
    entrenar = entrenador_xgboost(CFG["modelos"]["xgboost"], SEMILLA, len(clases))
    tr_i, val_i = idx_c[:corte], idx_c[corte:]
    mod_corto = entrenar(X[tr_i], y[tr_i], X[val_i], y[val_i])
    ref = ev.metricas(y[val_i], *mod_corto(X[val_i]))
    largo = ~corto
    shift = ev.metricas(y[largo], *mod_corto(X[largo]))
    for nivel, m in (("flujos cortos (referencia)", ref), ("flujos largos (población distinta)", shift)):
        filas.append(
            {
                "escenario": "D · Cambio de distribución",
                "nivel": nivel,
                **{k: v for k, v in m.items() if k != "peor_categoria"},
            }
        )
    # D' · Deriva temporal real: el modelo de producción en diciembre (mismo tamaño de submuestra)
    Xdic, ydic, _ = datos_prueba(clases, S5["muestra_estres"], meses=[CFG["datos"]["meses"]["deriva"][-1]])
    dic = ev.metricas(ydic, *predecir(Xdic))
    filas.append(
        {
            "escenario": "D · Cambio de distribución",
            "nivel": "deriva real: diciembre",
            **{k: v for k, v in dic.items() if k != "peor_categoria"},
        }
    )

    tabla = ev.veredicto(pd.DataFrame(filas), base["f1_macro"], S5["umbral_caida_f1"])
    # El escenario D compara un MISMO modelo (entrenado con flujos cortos) en su población y en otra distinta,
    # como en el notebook: la caída se mide respecto de su propia referencia, no del modelo de producción.
    es_ref = tabla.nivel.str.startswith("flujos cortos")
    es_shift = tabla.nivel.str.startswith("flujos largos")
    tabla.loc[es_shift, "caida_f1"] = ref["f1_macro"] - shift["f1_macro"]
    tabla.loc[es_shift, "veredicto"] = "OK" if ref["f1_macro"] - shift["f1_macro"] <= S5["umbral_caida_f1"] else "FALLA"
    tabla.loc[es_ref, ["caida_f1", "veredicto"]] = [0.0, "referencia"]
    guardar(tabla, RESULTADOS / "workshop1" / "resumen_escenarios.csv")
    guardar(
        {
            "umbral_caida_f1": S5["umbral_caida_f1"],
            "f1_base": base["f1_macro"],
            "categoria_dominante": clases[mayoritaria],
            "variables_top3": [TABULAR[c] for c in top3],
            "flujos_cortos_entrenamiento": int(corto.sum()),
        },
        RESULTADOS / "workshop1" / "configuracion.json",
    )
    figura_escenarios(tabla, RESULTADOS / "workshop1" / "escenarios.png")
    log.info(
        "Escenarios que FALLAN el umbral: %s",
        tabla.loc[tabla.veredicto == "FALLA", ["escenario", "nivel"]].values.tolist(),
    )


def figura_escenarios(t: pd.DataFrame, ruta: Path) -> None:
    """Tres paneles como el notebook: F1 macro, F1 de la peor categoría y exactitud por nivel de perturbación."""
    paneles = [
        ("A · Ruido", "Ruido (σ × desviación)"),
        ("B · Faltantes", "Datos faltantes"),
        ("C · Dominancia", "Una categoría domina el tráfico"),
        ("F · Relleno (TAREA 2)", "Relleno de paquetes"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for ax, (esc, titulo) in zip(axes, paneles, strict=True):
        sub = t[t.escenario == esc]
        for col, marca, nombre in (
            ("f1_macro", "o-", "F1 macro"),
            ("f1_peor_categoria", "s-", "F1 peor categoría"),
            ("accuracy", "^--", "exactitud"),
        ):
            ax.plot(sub["nivel"], sub[col], marca, label=nombre)
        ax.set_title(titulo)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def errores(log):
    _, _, g_tr, clases, _ = datos_entrenamiento()
    Xte, yte, df_te = datos_prueba(clases)
    pred, proba = predictor(cargar_modelo(MODELO_V2)["modelo"])(Xte)
    salida = RESULTADOS / "errores"
    guardar(ev.pares_confundidos(yte, pred, clases, top=15), salida / "pares_confundidos.csv")
    longitud = df_te["PACKETS"].to_numpy() + df_te["PACKETS_REV"].to_numpy()
    guardar(ev.error_por_tramo(yte, pred, longitud, "paquetes_del_flujo"), salida / "error_por_longitud.csv")
    confiabilidad, ece = ev.calibracion(yte, proba)
    guardar(confiabilidad, salida / "calibracion.csv")
    repetida = np.isin(df_te["SEQ_HASH"].to_numpy(), g_tr)
    rep = pd.DataFrame(
        [
            {"secuencias": nombre, "flujos": int(m.sum()), **ev.metricas(yte[m], pred[m])}
            for nombre, m in (("repetidas (vistas en septiembre)", repetida), ("nuevas", ~repetida))
        ]
    )
    guardar(rep, salida / "repetidas_vs_nuevas.csv")
    por_grupo, brechas = ev.equidad_por_categoria(yte, pred, clases)
    guardar(por_grupo.reset_index(names="categoria"), salida / "equidad_por_categoria.csv")
    guardar({"ece": ece, **brechas, "f1_macro_octubre": ev.metricas(yte, pred)["f1_macro"]}, salida / "resumen.json")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="calibración perfecta")
    ax.plot(confiabilidad["confianza_media"], confiabilidad["acierto"], "o-", label=f"XGBoost v2 (ECE = {ece:.3f})")
    ax.set_xlabel("Confianza del modelo")
    ax.set_ylabel("Acierto real")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida / "calibracion.png", dpi=150)
    plt.close(fig)
    log.info(
        "Errores · ECE %.4f · peor categoría %s (acierto %.3f) · %d flujos",
        ece,
        brechas["peor_categoria"],
        brechas["tasa_peor"],
        len(yte),
    )


def cierre(log):
    _, _, _, clases, _ = datos_entrenamiento(S5["muestra_entrenamiento"])
    Xte, yte, _ = datos_prueba(clases)
    v2, opt = cargar_modelo(MODELO_V2)["modelo"], cargar_modelo(MODELO_OPT)["modelo"]
    pred_v2, proba_v2 = predictor(v2)(Xte)
    pred_op, proba_op = predictor(opt)(Xte)
    final = pd.DataFrame(
        {"xgboost_v2": ev.metricas(yte, pred_v2, proba_v2), "xgboost_optimizado": ev.metricas(yte, pred_op, proba_op)}
    ).T.reset_index(names="modelo")
    guardar(final, RESULTADOS / "cierre" / "evaluacion_final.csv")
    comparacion = {
        "diferencia_f1": ev.ic_diferencia(yte, pred_v2, pred_op, 1000, SEMILLA),
        "mcnemar": mcnemar_test(yte, pred_v2, pred_op),
        "ic_v2": ev.ic_bootstrap_metrica(yte, pred_v2, 1000, SEMILLA),
        "ic_optimizado": ev.ic_bootstrap_metrica(yte, pred_op, 1000, SEMILLA),
    }
    guardar(comparacion, RESULTADOS / "cierre" / "comparacion.json")
    # Prueba de estrés de ambos modelos (como el final del notebook), en la submuestra de octubre
    X, _, _, _, _ = datos_entrenamiento(S5["muestra_entrenamiento"])
    Xs, ys, _ = datos_prueba(clases, S5["muestra_estres"])
    mediana, desviacion = np.median(X, axis=0), X.std(axis=0)
    mayoritaria = int(np.bincount(ys).argmax())
    filas = []
    for nombre, m in (("xgboost_v2", v2), ("xgboost_optimizado", opt)):
        for esc, f in (
            ("sin perturbar", lambda Z, w, r: (Z, w)),
            ("ruido σ=1.0", ev.ruido(1.0, desviacion)),
            ("faltantes 30 %", ev.faltantes(0.3, mediana)),
            ("dominancia 95 %", ev.dominancia(mayoritaria, 0.95)),
            ("relleno 100 %", ev.relleno(1.0)),
        ):
            filas.append(
                {
                    "modelo": nombre,
                    "escenario": esc,
                    **ev.evaluar_perturbado(predictor(m), Xs, ys, f, S5["repeticiones"], SEMILLA),
                }
            )
    guardar(pd.DataFrame(filas), RESULTADOS / "cierre" / "estres_comparado.csv")
    d = comparacion["diferencia_f1"]
    log.info(
        "Cierre · F1 v2 %.4f · optimizado %.4f · diferencia %+.4f IC95 %% [%+.4f, %+.4f] · %s",
        final.f1_macro[0],
        final.f1_macro[1],
        d["diferencia"],
        d["ic95_bajo"],
        d["ic95_alto"],
        "SIGNIFICATIVA" if d["significativa"] else "no significativa",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=["parte0", "workshop1", "errores", "cierre"])
    paso = ap.parse_args().paso
    log = configurar_logging(f"12_{paso}")
    t0 = time.time()
    try:
        {"parte0": parte0, "workshop1": workshop1, "errores": errores, "cierre": cierre}[paso](log)
    except Exception:
        log.exception("Falló el paso %s", paso)
        raise
    log.info("Paso %s terminado en %.1f min", paso, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
