"""
Prueba de McNemar para comparar dos clasificadores sobre los mismos flujos.

Por qué McNemar
---------------
Dos modelos pueden diferir en F1 por azar. McNemar mira solo los flujos en los que
los modelos discrepan (uno acierta y el otro no) y dice si esa diferencia es
estadísticamente significativa. Es la prueba recomendada por Dietterich (1998)
para comparar clasificadores entrenados una vez sobre el mismo conjunto de prueba.

Regla del proyecto: la red profunda solo se recomienda si supera a la línea base
con p < 0,05; si no, se recomienda la línea base, por ser más barata.
"""
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true, pred_a, pred_b, alpha: float = 0.05) -> dict:
    """Compara el modelo A con el modelo B sobre las mismas predicciones de prueba.

    Tabla de contingencia:
                      B acierta   B falla
        A acierta        n11        n10
        A falla          n01        n00
    Solo n10 y n01 (las discrepancias) determinan el resultado.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = [[int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    discrepancias = table[0][1] + table[1][0]
    if discrepancias == 0:
        # Los modelos aciertan y fallan exactamente en los mismos flujos: no hay diferencia que probar.
        # La fórmula dividiría entre cero, así que se informa p = 1 (ninguna evidencia de diferencia).
        return {"estadistico": 0.0, "p_valor": 1.0, "significativo": False, "tabla": table, "mejor": "empate"}
    res = mcnemar(table, exact=False, correction=True)
    return {"estadistico": float(res.statistic), "p_valor": float(res.pvalue),
            "significativo": bool(res.pvalue < alpha), "tabla": table,
            "mejor": "A" if table[0][1] > table[1][0] else "B"}
