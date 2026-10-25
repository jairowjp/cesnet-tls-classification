"""
Procesamiento de datos del proyecto (punto de entrada que exige la estructura del proyecto final).

Este módulo reúne, con nombres claros, todo lo que transforma los flujos de CESNET-TLS-Year22 en datos listos
para los modelos. No duplica código: delega en src/data/loader.py, donde está la implementación probada desde
el Sprint 1.

Qué hace cada función
---------------------
cargar_mes(es)        Lee la muestra del 2 % de los meses indicados (archivos parquet por día).
cargar_particion      Septiembre (entrenamiento), octubre (prueba) o noviembre y diciembre (deriva).
variables_tabulares   Las 60 variables del flujo que usan Random Forest y XGBoost (estadísticas, banderas TCP,
                      motivo de fin e histogramas), con log10(1 + x) en las variables muy asimétricas.
secuencias            Los primeros 30 paquetes (tamaño, dirección y tiempo) y su máscara, para la CNN y el
                      Transformer.
etiquetas             Las 23 categorías de servicio codificadas como enteros, en un orden fijo.

Exclusiones deliberadas: las direcciones IP, el nombre de servidor (SNI), la huella JA3 y el ASN de destino no
se usan nunca. Las tres primeras identifican a personas o servicios concretos; el ASN de destino filtraba la
respuesta (información mutua normalizada de 0,430 con la categoría, ver el EDA).
"""
from src.data.loader import (  # noqa: F401  (se reexportan para usarlos desde este módulo)
    SEQ,
    SKEWED,
    TABULAR,
    labels,
    load_months,
    load_split,
    sequence_features,
    tabular_features,
)

cargar_meses = load_months
cargar_particion = load_split
variables_tabulares = tabular_features
secuencias = sequence_features
etiquetas = labels

__all__ = ["SEQ", "SKEWED", "TABULAR", "cargar_meses", "cargar_particion", "etiquetas", "secuencias",
           "variables_tabulares"]
