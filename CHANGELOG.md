# Registro de cambios

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Las fechas son las reales de cada entrega.

## [Sin publicar] · Sprint 1
### Añadido
- `src/models/classic.py`: Random Forest y XGBoost con pesos por clase (misma fórmula del EDA).
- `scripts/utilidades/`: configuración del entorno, descarga y verificación del dataset, inspección y reparación de la muestra.
- `scripts/README.md` (índice y origen de cada script) y `docs/bitacora.md` (registro cronológico).
- `scripts/03_entrenar_clasicos.py`: entrenamiento y evaluación con el protocolo del proyecto: F1 macro,
  F1 sin secuencias repetidas, deriva en noviembre y diciembre, latencia, tamaño del modelo y verificación
  de umbrales. Incluye un modo de prueba rápida (`--limite-filas`).
### Cambiado
- Random Forest acotado (150 árboles, hojas de ≥ 10 flujos, 50 % de datos por árbol) para que quepa en la
  memoria del equipo: con 23 clases, sin límites superaría los 16 GB de RAM.
- Prueba de McNemar: devuelve p = 1 cuando los modelos no discrepan (antes podía declarar diferencia falsa).
- Pruebas de carga sin advertencias de rendimiento.

## [0.2.0] – 2026-09-27 · Semana 2
### Añadido
- Descarga del dataset completo desde Zenodo y verificación de integridad con MD5.
- `scripts/01_extraer_muestra.py`: muestra reproducible del 2 % de septiembre a diciembre de 2022, reanudable y
  tolerante a días sin flujos.
- `notebooks/01_eda.ipynb`: EDA completo con 11 secciones; resultados en `results/eda/`.
- Ficha de Decisión Técnica, Análisis comparativo de algoritmos y documento del EDA en `docs/semana2/`.
- Módulos `src/` de carga de datos y evaluación, pruebas unitarias y espacio de trabajo de VS Code.
### Cambiado
- Número de categorías corregido de 24 a **23**, según el mapa oficial de servicios y los datos.
- Acceso a los datos: de la librería DataZoo (servidor caído) a los archivos oficiales de Zenodo.
- Métrica añadida: F1 macro sobre la prueba sin secuencias repetidas (30,3 % de la prueba repite secuencias).

## [0.1.0] – 2026-09-22 · Semana 1
### Añadido
- Workshop de Metodología SMART y Presentación del Proyecto en `docs/semana1/`.
