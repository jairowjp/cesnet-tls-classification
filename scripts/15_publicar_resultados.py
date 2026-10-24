"""
Reúne los resultados clave del proyecto en las tres carpetas que exige la estructura del proyecto final.

  results/figures/   Las figuras principales, numeradas en el orden en que se cuenta el proyecto.
  results/metrics/   Las tablas y métricas finales (CSV y JSON).
  results/reports/   Los informes entregados en cada semana (PDF).

Las carpetas detalladas (eda/, modelos/, comparativa/, diagnostico/, semana5/, …) se conservan, porque la
aplicación web, los notebooks y los scripts dependen de ellas. Este script solo COPIA: no recalcula nada, y se
puede volver a ejecutar cuando cambie algún resultado.

Uso:  python scripts/15_publicar_resultados.py
"""

import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
DOCS = R.parent / "docs"

FIGURAS = {
    "01_eda_flujos_por_dia.png": "eda/figuras/01_flujos_por_dia.png",
    "02_eda_categorias_por_mes.png": "eda/figuras/02_categorias_por_mes_anio.png",
    "03_eda_deriva_temporal.png": "eda/figuras/03_deriva_js_anio.png",
    "04_eda_desbalance_de_clases.png": "eda/figuras/04_clases_entrenamiento.png",
    "05_eda_firma_por_categoria.png": "eda/figuras/08_firma_secuencia_por_categoria.png",
    "06_modelos_f1_frente_a_latencia.png": "comparativa/f1_vs_latencia.png",
    "07_xgboost_matriz_de_confusion.png": "modelos/xgboost/matriz_confusion.png",
    "08_diagnostico_antes_despues.png": "diagnostico/comparacion_antes_despues.png",
    "09_diagnostico_redes_parada_y_capacidad.png": "diagnostico/comparacion_parada.png",
    "10_optimizacion_importancia_hiperparametros.png": "semana5/optuna/importancias.png",
    "11_optimizacion_dependencia_parcial.png": "semana5/optuna/dependencia_parcial.png",
    "12_optimizacion_interaccion.png": "semana5/optuna/contorno.png",
    "13_evaluacion_escenarios_criticos.png": "semana5/workshop1/escenarios.png",
    "14_evaluacion_calibracion.png": "semana5/errores/calibracion.png",
}
METRICAS = {
    "01_comparacion_modelos_v2.csv": "comparativa/tabla_comparativa.csv",
    "02_mcnemar_entre_modelos.csv": "comparativa/mcnemar.csv",
    "03_version1_frente_a_version2.csv": "comparativa/versiones.csv",
    "04_baseline_e_intervalos.json": "semana5/parte0/intervalos.json",
    "05_escenarios_criticos.csv": "semana5/workshop1/resumen_escenarios.csv",
    "06_optuna_resumen.json": "semana5/optuna/resumen.json",
    "07_optuna_comparacion_15_pliegues.json": "semana5/optuna/comparacion_cv.json",
    "08_evaluacion_final_octubre.csv": "semana5/cierre/evaluacion_final.csv",
    "09_evaluacion_final_pruebas_estadisticas.json": "semana5/cierre/comparacion.json",
    "10_equidad_por_categoria.csv": "semana5/errores/equidad_por_categoria.csv",
    "11_pares_confundidos.csv": "semana5/errores/pares_confundidos.csv",
    "12_redes_parada_y_capacidad.csv": "diagnostico/comparacion_parada.csv",
}
INFORMES = {
    "01_presentacion_del_proyecto.pdf": "semana1/02_presentacion_del_proyecto.pdf",
    "02_ficha_decision_tecnica.pdf": "semana2/01_ficha_decision_tecnica.pdf",
    "03_analisis_comparativo_algoritmos.pdf": "semana2/02_analisis_comparativo_algoritmos.pdf",
    "04_eda_cesnet_tls_year22.pdf": "semana2/03_eda_cesnet_tls_year22.pdf",
    "05_diagnostico_sobreajuste_subajuste.pdf": "semana3/diagnostic_report.pdf",
    "06_impacto_social_y_responsabilidad.pdf": "semana4/Workshop_Etica_Impacto_Social_Jairo_Jhayya.pdf",
    "07_evaluacion_y_optimizacion.pdf": "semana5/Informe_Semana5_Evaluacion_Optimizacion_Jairo_Jhayya.pdf",
}


def copiar(destino: Path, mapa: dict, base: Path) -> list[str]:
    destino.mkdir(parents=True, exist_ok=True)
    faltan = []
    for nombre, origen in mapa.items():
        if (base / origen).exists():
            shutil.copy2(base / origen, destino / nombre)
        else:
            faltan.append(origen)
    return faltan


def main():
    faltan = (
        copiar(R / "figures", FIGURAS, R) + copiar(R / "metrics", METRICAS, R) + copiar(R / "reports", INFORMES, DOCS)
    )
    for carpeta, mapa in (("figures", FIGURAS), ("metrics", METRICAS), ("reports", INFORMES)):
        print(f"results/{carpeta}/: {len(mapa) - sum(1 for o in mapa.values() if o in faltan)} de {len(mapa)} archivos")
    if faltan:
        print("No se encontraron (revisa que existan):", *faltan, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
