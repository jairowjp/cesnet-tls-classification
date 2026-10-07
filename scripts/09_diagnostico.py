"""
Diagnóstico de sobreajuste y subajuste (Actividad de la semana 3).

Subcomandos (cada uno se puede repetir por separado):
  rf           Random Forest: curva por tamaño del dataset (C), curvas por hiperparámetro (D),
               diagnóstico y estrategia 1: reducir la complejidad del modelo.
  xgb          XGBoost con callback de registro por ronda (curvas A y B), antes y con la
               estrategia 3: regularización (submuestreo de filas y columnas, L2).
  red          CNN 1D o Transformer con registro por época (curvas A y B). Variante "base" (el
               entrenamiento del proyecto) o "mejorada" (estrategia 2: más épocas, OneCycle y
               parada temprana).
  resumen      Reúne los diagnósticos y la comparación antes/después (tabla y figura).
  todo         Ejecuta todo en orden.

Protocolo (evita el "sobreajuste a la validación" que advierte la clase):
  * el diagnóstico usa entrenamiento y VALIDACIÓN (septiembre, división 80/20 estratificada);
  * la prueba (octubre) se usa solo al final, para la comparación antes/después.

--muestra N toma una submuestra estratificada de N flujos del entrenamiento para reducir el tiempo
de cómputo (la validación y la prueba se usan completas). Antes y después usan la misma muestra.

Uso:
    python scripts/09_diagnostico.py todo --muestra 200000
    python scripts/09_diagnostico.py red --modelo transformer --variante base --muestra 200000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar src/ al ejecutar el script

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split, validation_curve  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.data.loader import labels, load_split, sequence_features, tabular_features  # noqa: E402
from src.diagnostico import curvas  # noqa: E402
from src.diagnostico.diagnostico import UMBRALES, diagnosticar, metricas_curvas, metricas_tamano  # noqa: E402
from src.diagnostico.tracking import CallbackRegistroXGB, RegistroMetricas, entrenar_red  # noqa: E402

SALIDA = ROOT / "results/diagnostico"
EXPERIMENTO = "diagnostico-overfitting"
CFG = load_config()
SEMILLA = CFG["semilla"]


# --------------------------------------------------------------------------------------------- datos
def preparar_datos(muestra: int, rama: str):
    """Entrenamiento y validación (septiembre, 80/20 estratificado) y prueba (octubre).

    rama: "tabular" (60 variables) o "secuencia" (30 × 3 con máscara).
    """
    t0 = time.time()
    sep = load_split("entrenamiento")
    y_all, clases = labels(sep)
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_all)), test_size=CFG["datos"]["fraccion_validacion"], stratify=y_all, random_state=SEMILLA
    )
    if muestra and muestra < len(idx_tr):
        idx_tr, _ = train_test_split(idx_tr, train_size=muestra, stratify=y_all[idx_tr], random_state=SEMILLA)
    oct_ = load_split("prueba")
    y_te, _ = labels(oct_, clases)

    def x(df, idx=None):
        parte = df if idx is None else df.iloc[idx]
        return tabular_features(parte) if rama == "tabular" else sequence_features(parte)

    datos = {
        "clases": clases,
        "y_tr": y_all[idx_tr],
        "y_val": y_all[idx_val],
        "y_te": y_te,
        "X_tr": x(sep, idx_tr),
        "X_val": x(sep, idx_val),
        "X_te": x(oct_),
    }
    print(
        f"Datos ({rama}): entrenamiento {len(idx_tr):,} · validación {len(idx_val):,} · prueba {len(y_te):,} "
        f"· {len(clases)} clases ({time.time() - t0:.0f} s)",
        flush=True,
    )
    return datos


def tamano_mb(modelo) -> float:
    """Tamaño del modelo serializado, en MB (costo de despliegue)."""
    buf = io.BytesIO()
    joblib.dump(modelo, buf, compress=3)
    return round(buf.tell() / 1e6, 2)


def guardar_json(ruta: Path, datos: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=1, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------------- Random Forest
def elegir_parametro(valores, p_ent, p_val, tolerancia=0.01):
    """Estrategia de reducción de complejidad: entre los valores cuya validación queda a menos de
    `tolerancia` de la mejor, elige el de MENOR brecha (principio de parsimonia)."""
    mv, brecha = p_val.mean(1), p_ent.mean(1) - p_val.mean(1)
    candidatos = [i for i in range(len(valores)) if mv[i] >= mv.max() - tolerancia]
    return valores[min(candidatos, key=lambda i: brecha[i])]


def ejecutar_rf(muestra: int) -> None:
    from sklearn.ensemble import RandomForestClassifier

    base = {k: v for k, v in CFG["modelos"]["random_forest"].items()}
    d = preparar_datos(muestra, "tabular")
    carpeta = SALIDA / "random_forest"
    # Las curvas con validación cruzada entrenan muchos bosques: se usan hasta 100 000 flujos
    n_cv = min(100_000, len(d["y_tr"]))
    idx = (
        np.arange(len(d["y_tr"]))
        if n_cv == len(d["y_tr"])
        else train_test_split(np.arange(len(d["y_tr"])), train_size=n_cv, stratify=d["y_tr"], random_state=SEMILLA)[0]
    )
    X, y = d["X_tr"][idx], d["y_tr"][idx]
    cv = StratifiedKFold(3, shuffle=True, random_state=SEMILLA)
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEMILLA, **base)

    with RegistroMetricas(EXPERIMENTO, "random_forest-curvas", carpeta, {**base, "flujos_cv": n_cv}) as reg:
        print("Curva por tamaño del dataset (C)…", flush=True)
        tam, p_ent, p_val = learning_curve(
            rf,
            X,
            y,
            train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            shuffle=True,
            random_state=SEMILLA,
        )
        for t, e, v in zip(tam, p_ent.mean(1), p_val.mean(1), strict=True):
            reg.registrar(int(t), f1_entrenamiento=float(e), f1_validacion=float(v))
        curvas.curva_por_tamano(
            tam,
            p_ent,
            p_val,
            "Random Forest: F1 macro según el tamaño del entrenamiento",
            carpeta / "C_curva_por_tamano.png",
        )
        hoja = [1, 2, 5, 10, 20, 50, 100]
        prof = [5, 10, 15, 20, 25, 30, None]
        res_param = {}
        for nombre, valores in (("min_samples_leaf", hoja), ("max_depth", prof)):
            print(f"Curva por hiperparámetro (D): {nombre}…", flush=True)
            e, v = validation_curve(
                rf, X, y, param_name=nombre, param_range=valores, cv=cv, scoring="f1_macro", n_jobs=1
            )
            res_param[nombre] = {
                "valores": valores,
                "entrenamiento": e.mean(1).tolist(),
                "validacion": v.mean(1).tolist(),
                "elegido": elegir_parametro(valores, e, v),
            }
            curvas.curva_por_hiperparametro(
                valores,
                e,
                v,
                nombre,
                f"Random Forest: F1 macro según {nombre}",
                carpeta / f"D_curva_{nombre}.png",
                elegido=res_param[nombre]["elegido"],
            )
        m = metricas_tamano(tam, p_ent, p_val)
        diag, razones = diagnosticar(m)
        reg.cerrar({**m, "diagnostico": diag})
    guardar_json(
        carpeta / "curvas.json",
        {
            "tamanos": tam.tolist(),
            "entrenamiento": p_ent.tolist(),
            "validacion": p_val.tolist(),
            "hiperparametros": res_param,
            "metricas": m,
            "diagnostico": diag,
            "razones": razones,
        },
    )
    print(f"Diagnóstico Random Forest: {diag.upper()} · {'; '.join(razones)}", flush=True)

    # Estrategia 1: reducir la complejidad con los valores elegidos por las curvas D
    despues = {
        **base,
        "min_samples_leaf": res_param["min_samples_leaf"]["elegido"],
        "max_depth": res_param["max_depth"]["elegido"],
    }
    filas = []
    for momento, params in (("antes", base), ("despues", despues)):
        with RegistroMetricas(EXPERIMENTO, f"random_forest-{momento}", carpeta / momento, params) as reg:
            t0 = time.time()
            modelo = RandomForestClassifier(class_weight="balanced", random_state=SEMILLA, **params).fit(
                d["X_tr"], d["y_tr"]
            )
            seg = time.time() - t0
            r = {
                "modelo": "Random Forest",
                "momento": "Antes" if momento == "antes" else "Después",
                "f1_entrenamiento": f1_score(d["y_tr"], modelo.predict(d["X_tr"]), average="macro"),
                "f1_validacion": f1_score(d["y_val"], modelo.predict(d["X_val"]), average="macro"),
                "f1_prueba": f1_score(d["y_te"], modelo.predict(d["X_te"]), average="macro"),
                "tamano_MB": tamano_mb(modelo),
                "segundos": round(seg, 1),
                "parametros": params,
            }
            r["brecha"] = r["f1_entrenamiento"] - r["f1_validacion"]
            # Random Forest no tiene épocas: su diagnóstico usa el nivel de F1 y la brecha final
            r["diagnostico"], r["razones"] = diagnosticar(
                {
                    "f1_entrenamiento": r["f1_entrenamiento"],
                    "brecha_final": r["brecha"],
                    "mejor_es_ultimo": False,
                    "mejora_final": 0.0,
                    "pendiente_val": 0.0,
                    "pendiente_ent": 0.0,
                }
            )
            reg.registrar(1, **{k: v for k, v in r.items() if isinstance(v, float)})
            reg.cerrar({k: v for k, v in r.items() if k not in ("parametros", "razones")})
            filas.append(r)
            print(
                f"  {r['momento']:8s} F1 ent. {r['f1_entrenamiento']:.4f} · val. {r['f1_validacion']:.4f} · "
                f"prueba {r['f1_prueba']:.4f} · brecha {r['brecha']:.4f} · {r['tamano_MB']} MB",
                flush=True,
            )
    guardar_json(carpeta / "antes_despues.json", {"estrategia": "Reducción de complejidad", "filas": filas})


# --------------------------------------------------------------------------------------- XGBoost
def ejecutar_xgb(muestra: int) -> None:
    from xgboost import XGBClassifier

    d = preparar_datos(muestra, "tabular")
    carpeta = SALIDA / "xgboost"
    base = {k: v for k, v in CFG["modelos"]["xgboost"].items()}
    regularizado = {**base, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, "min_child_weight": 5}
    # El callback mide en una submuestra fija del entrenamiento (igual tamaño que la validación como máximo)
    n_seg = min(50_000, len(d["y_tr"]))
    i_seg = np.random.default_rng(SEMILLA).choice(len(d["y_tr"]), n_seg, replace=False)
    filas = []
    for momento, params in (("antes", base), ("despues", regularizado)):
        with RegistroMetricas(EXPERIMENTO, f"xgboost-{momento}", carpeta / momento, params) as reg:
            cb = CallbackRegistroXGB(reg, d["X_tr"][i_seg], d["y_tr"][i_seg], d["X_val"], d["y_val"], cada=10)
            modelo = XGBClassifier(
                objective="multi:softprob",
                num_class=len(d["clases"]),
                eval_metric="mlogloss",
                random_state=SEMILLA,
                callbacks=[cb],
                **params,
            )
            t0 = time.time()
            modelo.fit(
                d["X_tr"],
                d["y_tr"],
                sample_weight=compute_sample_weight("balanced", d["y_tr"]),
                eval_set=[(d["X_tr"][i_seg], d["y_tr"][i_seg]), (d["X_val"], d["y_val"])],
                verbose=False,
            )
            seg = time.time() - t0
            hist = reg.historial()
            m = metricas_curvas(hist)
            diag, razones = diagnosticar(m)
            r = {
                "modelo": "XGBoost",
                "momento": "Antes" if momento == "antes" else "Después",
                **m,
                "f1_prueba": f1_score(d["y_te"], modelo.predict(d["X_te"]), average="macro"),
                "brecha": m["brecha_final"],
                "tamano_MB": round(len(modelo.get_booster().save_raw("ubj")) / 1e6, 2),
                "segundos": round(seg, 1),
                "rondas": int(modelo.best_iteration) + 1,
                "diagnostico": diag,
                "razones": razones,
                "parametros": params,
            }
            reg.cerrar({k: v for k, v in r.items() if k not in ("parametros", "razones")})
            for col, eje, menor in (("perdida", "Pérdida logarítmica multiclase", True), ("f1", "F1 macro", False)):
                curvas.curva_entrenamiento(
                    hist,
                    f"{col}_entrenamiento",
                    f"{col}_validacion",
                    f"XGBoost ({r['momento'].lower()}): {eje} por ronda",
                    eje,
                    carpeta / momento / f"{'A' if col == 'perdida' else 'B'}_{col}.png",
                    menor_es_mejor=menor,
                    eje_x="Ronda",
                    escala_log=menor,  # la pérdida de XGBoost cae varios órdenes de magnitud
                )
            filas.append(r)
            print(
                f"XGBoost {r['momento']}: {diag.upper()} · F1 val. {m['f1_validacion']:.4f} · prueba "
                f"{r['f1_prueba']:.4f} · brecha {m['brecha_final']:.4f} · {r['rondas']} rondas",
                flush=True,
            )
    guardar_json(carpeta / "antes_despues.json", {"estrategia": "Regularización", "filas": filas})


# ------------------------------------------------------------------------------- redes profundas
def ejecutar_red(nombre: str, variante: str, muestra: int, epocas: int | None = None) -> None:
    import torch

    from src.models.deep import build_deep_model

    d = preparar_datos(muestra, "secuencia")
    (X_tr, M_tr), (X_val, M_val), (X_te, M_te) = d["X_tr"], d["X_val"], d["X_te"]
    params = CFG["modelos"][nombre]
    if variante == "base":
        conf = {"epocas": params["epocas"], "programador": "constante", "paciencia": None}
    else:  # estrategia 2 contra el subajuste: más épocas, tasa de aprendizaje OneCycle y parada temprana
        conf = {"epocas": 40, "programador": "onecycle", "paciencia": 6}
    if epocas:  # límite opcional de épocas (pruebas o falta de tiempo); queda registrado en la configuración
        conf["epocas"] = epocas
    carpeta = SALIDA / nombre / ("antes" if variante == "base" else "despues")
    torch.manual_seed(SEMILLA)
    modelo = build_deep_model(nombre, params, len(d["clases"]))
    pesos = compute_class_weight("balanced", classes=np.arange(len(d["clases"])), y=d["y_tr"])
    with RegistroMetricas(EXPERIMENTO, f"{nombre}-{variante}", carpeta, {**params, **conf, "muestra": muestra}) as reg:
        t0 = time.time()
        modelo, epocas = entrenar_red(
            modelo,
            X_tr,
            M_tr,
            d["y_tr"],
            X_val,
            M_val,
            d["y_val"],
            reg,
            epocas=conf["epocas"],
            lr=params["lr"],
            pesos_clase=pesos,
            lote=CFG["entrenamiento_profundo"]["lote"],
            programador=conf["programador"],
            paciencia=conf["paciencia"],
            semilla=SEMILLA,
        )
        seg = time.time() - t0
        hist = reg.historial()
        m = metricas_curvas(hist)
        diag, razones = diagnosticar(m)
        modelo.eval()
        with torch.no_grad():
            pred = np.concatenate(
                [
                    modelo(torch.from_numpy(X_te[i : i + 8192]), torch.from_numpy(M_te[i : i + 8192])).argmax(1).numpy()
                    for i in range(0, len(X_te), 8192)
                ]
            )
        r = {
            "modelo": {"cnn1d": "CNN 1D", "transformer": "Transformer"}[nombre],
            "momento": "Antes" if variante == "base" else "Después",
            **m,
            "f1_prueba": f1_score(d["y_te"], pred, average="macro"),
            "brecha": m["brecha_final"],
            "epocas_ejecutadas": epocas,
            "segundos": round(seg, 1),
            "diagnostico": diag,
            "razones": razones,
            "configuracion": conf,
        }
        reg.cerrar({k: v for k, v in r.items() if k not in ("configuracion", "razones")})
    for col, eje, menor in (("perdida", "Pérdida de entropía cruzada ponderada", True), ("f1", "F1 macro", False)):
        curvas.curva_entrenamiento(
            hist,
            f"{col}_entrenamiento",
            f"{col}_validacion",
            f"{r['modelo']} ({r['momento'].lower()}): {eje} por época",
            eje,
            carpeta / f"{'A' if col == 'perdida' else 'B'}_{col}.png",
            menor_es_mejor=menor,
        )
    guardar_json(carpeta / "resultado.json", r)
    print(
        f"{r['modelo']} {r['momento']}: {diag.upper()} · {'; '.join(razones)} · prueba {r['f1_prueba']:.4f}", flush=True
    )


# ----------------------------------------------------------------------------------------- resumen
def ejecutar_resumen() -> None:
    filas = []
    for archivo in ("random_forest/antes_despues.json", "xgboost/antes_despues.json"):
        if (SALIDA / archivo).exists():
            filas += json.loads((SALIDA / archivo).read_text())["filas"]
    for red in ("cnn1d", "transformer"):
        for momento in ("antes", "despues"):
            if (SALIDA / red / momento / "resultado.json").exists():
                filas.append(json.loads((SALIDA / red / momento / "resultado.json").read_text()))
    if not filas:
        sys.exit("No hay resultados en results/diagnostico/. Ejecuta primero los subcomandos.")
    tabla = pd.DataFrame(filas)
    cols = [
        c
        for c in (
            "modelo",
            "momento",
            "diagnostico",
            "f1_entrenamiento",
            "f1_validacion",
            "f1_prueba",
            "brecha",
            "mejor_paso",
            "pasos",
            "rondas",
            "epocas_ejecutadas",
            "tamano_MB",
            "segundos",
        )
        if c in tabla
    ]
    tabla[cols].to_csv(SALIDA / "comparacion_antes_despues.csv", index=False)
    rf_diag = SALIDA / "random_forest/curvas.json"
    diagnosticos = {"umbrales": UMBRALES}
    if rf_diag.exists():
        c = json.loads(rf_diag.read_text())
        diagnosticos["Random Forest (curva por tamaño)"] = {
            "diagnostico": c["diagnostico"],
            "razones": c["razones"],
            **c["metricas"],
        }
    for f in filas:
        if f.get("diagnostico"):
            diagnosticos[f"{f['modelo']} ({f['momento']})"] = {
                "diagnostico": f["diagnostico"],
                "razones": f.get("razones", []),
            }
    guardar_json(SALIDA / "diagnosticos.json", diagnosticos)
    curvas.comparacion_antes_despues(
        tabla,
        "Antes y después de las estrategias de mejora (validación de septiembre)",
        SALIDA / "comparacion_antes_despues.png",
    )
    pd.set_option("display.width", 180)
    print(tabla[cols].round(4).to_string(index=False))
    print(f"\nResultados en {SALIDA.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="comando", required=True)
    for nombre in ("rf", "xgb", "todo"):
        p = sub.add_parser(nombre)
        p.add_argument(
            "--muestra", type=int, default=0, help="submuestra estratificada del entrenamiento (0 = completo)"
        )
    p = sub.add_parser("red")
    p.add_argument("--modelo", choices=["cnn1d", "transformer"], required=True)
    p.add_argument("--variante", choices=["base", "mejorada"], default="base")
    p.add_argument("--epocas", type=int, default=None, help="límite de épocas (por defecto: 15 base, 40 mejorada)")
    p.add_argument("--muestra", type=int, default=0)
    sub.add_parser("resumen")
    a = ap.parse_args()
    if a.comando == "rf":
        ejecutar_rf(a.muestra)
    elif a.comando == "xgb":
        ejecutar_xgb(a.muestra)
    elif a.comando == "red":
        ejecutar_red(a.modelo, a.variante, a.muestra, a.epocas)
    elif a.comando == "resumen":
        ejecutar_resumen()
    else:  # todo: el orden va de lo más rápido a lo más lento, así los primeros resultados llegan pronto
        ejecutar_rf(a.muestra)
        for red, variante in (
            ("cnn1d", "base"),
            ("cnn1d", "mejorada"),
            ("transformer", "base"),
            ("transformer", "mejorada"),
        ):
            ejecutar_red(red, variante, a.muestra)
        ejecutar_xgb(a.muestra)
        ejecutar_resumen()


if __name__ == "__main__":
    main()
