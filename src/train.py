"""
Entrenamiento y optimización de hiperparámetros de XGBoost (Semana 5, Workshop 2).

La búsqueda usa Optuna (optimización bayesiana con TPE) y, como pide el workshop, se evalúa SOLO con
datos de entrenamiento mediante validación cruzada agrupada por huella de secuencia: la prueba de
octubre no interviene en ninguna decisión.

Decisiones de diseño
--------------------
  * El modelo se construye y entrena con src/models/classic.py (build_xgboost y fit_xgboost), así la
    búsqueda usa exactamente los mismos pesos por clase y la misma parada temprana que la versión 2.
  * Poda (MedianPruner): después de cada pliegue se compara el F1 parcial con la mediana de las pruebas
    anteriores; las configuraciones claramente peores se abandonan para ahorrar cómputo.
  * El estudio se guarda en SQLite: si el equipo se apaga, se reanuda donde quedó, y cualquier persona
    puede reabrirlo para repetir el análisis de sensibilidad (reproducibilidad).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna

from src.evaluate import metricas, pliegues_agrupados
from src.models.classic import build_xgboost, fit_xgboost

log = logging.getLogger(__name__)
ESCALA_LOG = {"learning_rate", "reg_lambda"}  # parámetros que se exploran en escala logarítmica
ENTEROS = {"max_depth", "min_child_weight"}


def sugerir(trial: optuna.Trial, espacio: dict) -> dict:
    """Traduce el espacio de búsqueda de configs/experiment.yaml a sugerencias de Optuna."""
    params = {}
    for nombre, (bajo, alto) in espacio.items():
        if nombre in ENTEROS:
            params[nombre] = trial.suggest_int(nombre, int(bajo), int(alto))
        else:
            params[nombre] = trial.suggest_float(nombre, float(bajo), float(alto), log=nombre in ESCALA_LOG)
    return params


def entrenador_xgboost(params: dict, semilla: int, n_clases: int, registro: list | None = None):
    """Devuelve una función entrenar(Xtr, ytr, Xva, yva) -> predictor, compatible con evaluate.validacion_cruzada.

    Si en el entrenamiento faltan categorías (un pliegue sin flujos de un servicio muy pequeño, o el escenario
    D, que entrena solo con flujos cortos), XGBoost aprende las presentes y sus probabilidades se reubican en
    las n_clases columnas originales: las categorías no vistas cuentan como errores, que es lo correcto.
    Si se pasa `registro`, guarda en él la ronda óptima de cada entrenamiento (parada temprana).
    """

    def entrenar(Xtr, ytr, Xva, yva):
        vistas = np.unique(ytr)
        codigo = np.full(n_clases, -1)
        codigo[vistas] = np.arange(len(vistas))
        en_val = np.isin(yva, vistas)  # la parada temprana solo puede vigilar clases vistas
        modelo = build_xgboost(params, semilla, len(vistas))
        fit_xgboost(modelo, Xtr, codigo[ytr], Xva[en_val], codigo[yva[en_val]])
        if registro is not None:
            registro.append(int(getattr(modelo, "best_iteration", modelo.n_estimators)))

        def predecir(X):
            proba = np.zeros((len(X), n_clases))
            proba[:, vistas] = modelo.predict_proba(X)
            return proba.argmax(axis=1), proba

        return predecir

    return entrenar


def objetivo(trial: optuna.Trial, X, y, grupos, cfg_optuna: dict, fijos: dict, semilla: int, n_clases: int) -> float:
    """F1 macro medio en validación cruzada agrupada para la configuración sugerida por Optuna."""
    params = {
        **fijos,
        **sugerir(trial, cfg_optuna["espacio"]),
        "n_estimators": cfg_optuna["rondas_maximas"],
        "early_stopping_rounds": cfg_optuna["parada_temprana"],
    }
    rondas, puntajes = [], []
    entrenar = entrenador_xgboost(params, semilla, n_clases, rondas)
    for _, k, tr, va in pliegues_agrupados(y, grupos, cfg_optuna["pliegues"], 1, semilla):
        pred, proba = entrenar(X[tr], y[tr], X[va], y[va])(X[va])
        m = metricas(y[va], pred, proba)
        puntajes.append(m["f1_macro"])
        trial.set_user_attr(f"f1_peor_pliegue_{k}", m["f1_peor_categoria"])
        trial.report(float(np.mean(puntajes)), step=k)
        if trial.should_prune():
            log.info("Prueba %d podada tras el pliegue %d (F1 parcial %.4f)", trial.number, k + 1, np.mean(puntajes))
            raise optuna.TrialPruned()
    trial.set_user_attr("rondas_medias", float(np.mean(rondas)))
    trial.set_user_attr("f1_desviacion", float(np.std(puntajes)))
    return float(np.mean(puntajes))


def crear_estudio(ruta_db: Path, nombre: str, semilla: int) -> optuna.Study:
    """Crea o reanuda el estudio (TPE con semilla fija y poda por mediana), guardado en SQLite."""
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=nombre,
        storage=f"sqlite:///{ruta_db}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )


def importancias(estudio: optuna.Study, semilla: int) -> dict:
    """Importancia de cada hiperparámetro con dos métodos distintos: fANOVA y reducción de impureza.

    Si ambos coinciden en el orden, la conclusión sobre qué hiperparámetros importan es robusta.
    """
    from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator

    return {
        "fanova": optuna.importance.get_param_importances(estudio, evaluator=FanovaImportanceEvaluator(seed=semilla)),
        "impureza": optuna.importance.get_param_importances(
            estudio, evaluator=MeanDecreaseImpurityImportanceEvaluator(seed=semilla)
        ),
    }
