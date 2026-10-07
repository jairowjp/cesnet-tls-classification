"""Pruebas del diagnóstico: cada regla se verifica con curvas de comportamiento conocido."""

import numpy as np
import pandas as pd

from src.diagnostico.diagnostico import diagnosticar, metricas_curvas, metricas_tamano
from src.diagnostico.tracking import RegistroMetricas


def _historial(f1_ent, f1_val, perdida_ent, perdida_val):
    n = len(f1_ent)
    return pd.DataFrame(
        {
            "paso": np.arange(1, n + 1),
            "f1_entrenamiento": f1_ent,
            "f1_validacion": f1_val,
            "perdida_entrenamiento": perdida_ent,
            "perdida_validacion": perdida_val,
        }
    )


def test_detecta_sobreajuste():
    """Entrenamiento casi perfecto, validación estancada y su pérdida subiendo: patrón clásico de sobreajuste."""
    t = np.linspace(0, 1, 30)
    h = _historial(0.70 + 0.29 * t, np.full(30, 0.80), 1.0 - 0.9 * t, 0.6 + 0.4 * t**2)
    diag, razones = diagnosticar(metricas_curvas(h))
    assert diag == "sobreajuste" and any("brecha" in r for r in razones)


def test_detecta_subajuste_por_no_converger():
    """Ambas curvas siguen subiendo juntas al terminar: el modelo aún aprendía."""
    t = np.linspace(0, 1, 15)
    h = _historial(0.40 + 0.40 * t, 0.39 + 0.40 * t, 1.5 - t, 1.5 - t)
    assert diagnosticar(metricas_curvas(h))[0] == "subajuste"


def test_detecta_subajuste_por_rendimiento_bajo():
    """Curvas planas y juntas, pero con F1 bajo: el modelo no alcanza a ajustar ni el entrenamiento."""
    h = _historial(np.full(20, 0.55), np.full(20, 0.54), np.full(20, 1.2), np.full(20, 1.21))
    assert diagnosticar(metricas_curvas(h))[0] == "subajuste"


def test_detecta_comportamiento_optimo():
    """Curvas que convergen con brecha pequeña y estable en un nivel alto."""
    t = np.linspace(0, 1, 30)
    meseta = 0.90 - 0.30 * np.exp(-8 * t)
    h = _historial(meseta + 0.01, meseta, 0.3 + np.exp(-8 * t), 0.32 + np.exp(-8 * t))
    assert diagnosticar(metricas_curvas(h))[0] == "óptimo"


def test_curva_por_tamano_con_brecha_grande_es_sobreajuste():
    tam = np.array([1000, 5000, 10000])
    ent = np.array([[1.0, 1.0], [0.99, 0.99], [0.99, 0.98]])
    val = np.array([[0.70, 0.71], [0.78, 0.77], [0.81, 0.80]])
    assert diagnosticar(metricas_tamano(tam, ent, val))[0] in ("sobreajuste", "mixto")


def test_registro_guarda_historial_sin_mlflow(tmp_path):
    with RegistroMetricas("prueba", "corrida", tmp_path, {"a": 1}, usar_mlflow=False) as reg:
        reg.registrar(1, perdida_entrenamiento=1.0, perdida_validacion=1.1)
        reg.registrar(2, perdida_entrenamiento=0.8, perdida_validacion=0.9)
        reg.cerrar({"diagnostico": "óptimo"})
    hist = pd.read_csv(tmp_path / "historial.csv")
    assert list(hist["paso"]) == [1, 2] and (tmp_path / "resumen.json").exists()


def test_narrativa_respeta_coma_decimal_y_miles():
    """El análisis en lenguaje claro usa coma decimal y no confunde los miles con los decimales."""
    from src.diagnostico.narrativa import analisis_rf

    rf = {"tamanos": [9600, 96000], "entrenamiento": [[0.99], [0.99]], "validacion": [[0.80], [0.85]],
          "hiperparametros": {"max_depth": {"valores": [5, None], "entrenamiento": [0.8, 0.99],
                                            "validacion": [0.78, 0.85], "elegido": None}}}
    texto = analisis_rf(rf)
    assert "0,800" in texto and "96 000" in texto and "sigue creciendo" in texto and "sin límite" in texto
