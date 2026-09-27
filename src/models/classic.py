"""
Modelos clásicos de la línea base: Random Forest y XGBoost.

Por qué son la línea base
-------------------------
El Análisis comparativo (docs/semana2) les dio el mayor puntaje (4,50): entrenan en CPU en
minutos, predicen en microsegundos y permiten explicar qué variables pesan. Todo modelo
profundo del proyecto se medirá contra ellos, y solo se recomendará si los supera con
significancia estadística (prueba de McNemar, p < 0,05).

Manejo del desbalance (145 a 1, EDA sección 2.6)
------------------------------------------------
Ambos modelos reciben pesos inversos a la frecuencia de cada clase: n / (k · n_clase).
Es la misma fórmula de los pesos calculados en el EDA (results/eda/pesos_por_clase.json).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def build_random_forest(params: dict, seed: int) -> RandomForestClassifier:
    """Random Forest con pesos por clase.

    class_weight="balanced" aplica exactamente n / (k · n_clase), la fórmula del EDA.
    """
    return RandomForestClassifier(class_weight="balanced", random_state=seed, **params)


def fit_random_forest(model, X_train, y_train, X_val=None, y_val=None):
    """Entrena Random Forest. No usa validación durante el ajuste: no tiene parada temprana."""
    return model.fit(X_train, y_train)


def build_xgboost(params: dict, seed: int, n_classes: int) -> XGBClassifier:
    """XGBoost multiclase con parada temprana.

    objective="multi:softprob" devuelve una probabilidad por clase, que la aplicación
    mostrará en el clasificador en vivo. eval_metric="mlogloss" es la pérdida que se vigila
    en validación para detener el entrenamiento cuando deja de mejorar (evita sobreajuste).
    """
    return XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss",
                         random_state=seed, **params)


def fit_xgboost(model, X_train, y_train, X_val, y_val):
    """Entrena XGBoost con pesos por muestra y validación para la parada temprana."""
    w = compute_sample_weight("balanced", y_train)   # misma fórmula de pesos que Random Forest
    model.fit(X_train, y_train, sample_weight=w, eval_set=[(X_val, y_val)], verbose=False)
    return model


# Registro de modelos disponibles: el script de entrenamiento elige por nombre.
MODELS = {
    "random_forest": (build_random_forest, fit_random_forest),
    "xgboost": (build_xgboost, fit_xgboost),
}


def top_features(model, names: list[str], k: int = 15) -> list[tuple[str, float]]:
    """Las k variables más importantes según el propio modelo (importancia por impureza o ganancia).

    Sirve para la interpretabilidad que piden los administradores de red: qué rasgos del
    flujo usa el modelo para decidir.
    """
    imp = np.asarray(model.feature_importances_)
    order = np.argsort(imp)[::-1][:k]
    return [(names[i], float(imp[i])) for i in order]
