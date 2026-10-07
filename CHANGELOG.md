# Registro de cambios

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Las fechas son las reales de cada entrega.

## [0.4.0] – 2026-10-07 · Aplicación web, seguridad, diagnóstico y modelos v2

### Versión 2 de los modelos
- `README.md`: estado real del proyecto y resultados de la versión 1 y la 2.
#### Cambiado
- `configs/experiment.yaml`: configuración v2 validada por el diagnóstico de la semana 3 (Random Forest con hojas de 1
  flujo y profundidad 20; XGBoost con hasta 1 000 rondas y regularización; redes con hasta 50 y 40 épocas, tasa OneCycle
  y paciencia 6). La v1 se conserva como `modelos_v1`, punto de partida reproducible del diagnóstico.
- `scripts/05_entrenar_profundos.py`: programador OneCycle y paciencia por modelo. Los scripts 03 y 05 registran la
  memoria máxima usada.
#### Añadido
- `scripts/11_comparar_versiones.py` y `scripts/utilidades/07_reentrenar_v2.sh`: reentrenamiento con respaldo de la v1,
  prueba piloto de memoria y decisión por F1 de validación (la prueba no interviene).
#### Corregido
- `src/models/deep.py`: las claves de entrenamiento (programador, paciencia) ya no llegan al constructor de la red.

### Semana 3: diagnóstico de sobreajuste y subajuste
#### Añadido
- `src/diagnostico/`: tracking con MLflow (SQLite local), callback personalizado de XGBoost y bucle de PyTorch con
  registro por época; curvas A-D a 300 DPI; diagnóstico cuantitativo con reglas y umbrales; análisis en lenguaje claro
  calculado desde las métricas (`narrativa.py`), compartido por el informe y el notebook.
- `scripts/09_diagnostico.py` y `06_correr_diagnostico.sh`: diagnóstico de los 4 modelos y 3 estrategias de mejora.
  Resultado: el problema dominante era el subajuste; F1 macro de prueba +9,6 (Random Forest), +13,4 (CNN 1D) y
  +11,2 puntos (Transformer); en XGBoost la regularización redujo la brecha de 0,105 a 0,090.
- `scripts/10_informe_diagnostico.py`: informe técnico en formato UEES generado desde los resultados.
- `notebooks/overfitting_analysis.ipynb`: las 8 secciones de la actividad.
- Página **Diagnóstico** en la aplicación web, con curvas interactivas; el recorrido guiado pasa a ocho pasos.
- 8 pruebas nuevas (reglas de diagnóstico, registro, narrativa y página web).

### Sprint 3: aplicación web
#### Añadido
- Aplicación Streamlit de cinco páginas (`app/`) con sistema de diseño propio: la firma del flujo como elemento
  central, cobalto y ámbar reservados para la dirección de los paquetes y tipografía Archivo servida desde la app.
- Núcleo de seguridad (`app/core/seguridad.py`): verificación SHA-256 del modelo antes de cargarlo, validación
  estricta de CSV (tamaño, filas, columnas, tipos y rangos) y neutralización de fórmulas en las descargas.
- Configuración segura de Streamlit (`.streamlit/config.toml`), `Dockerfile` con usuario sin privilegios y
  `requirements-app.txt` sin PyTorch.
- `scripts/06_preparar_app.py` y 22 pruebas nuevas: 14 de seguridad (incluidos intentos de inyección) y 8 de interfaz.
- Participación guiada: recorrido de siete pasos con invitación al siguiente en cada página, sugerencias en contexto
  y portada institucional con el logo y el granate de la UEES.
- Página **Reto**: el usuario adivina la categoría de flujos reales y compite contra los modelos en cinco rondas.
- Página **Tu opinión**: encuesta SUS con puntaje inmediato y un código de respuesta corto que el evaluador copia y
  envía (el sitio no guarda nada ni pide datos personales). El código lleva caracteres de control que detectan el
  100 % de los errores de copia de un carácter. `scripts/07_consolidar_sus.py` los verifica y recalcula cada puntaje.
- Página **Seguridad**: qué datos se usan, qué pasa con lo que sube el usuario y cómo verificar cada control.
- Clasificar y la portada muestran la predicción y la confianza de los cuatro modelos, no solo de XGBoost.
- Publicación en Streamlit Community Cloud (gratuito): dependencias propias de la app en `app/requirements.txt`,
  sin PyTorch. Hugging Face dejó de ofrecer Docker en su plan gratuito; el `Dockerfile` queda como plan B.
- `scripts/utilidades/04_ejecutar_app_local.sh`: la app se lanza en local solo en `localhost`.
- Informe de seguridad (`docs/seguridad/README.md`): pip-audit, ZAP activo en local, ZAP pasivo en la URL pública
  y `scripts/utilidades/05_resumen_zap.py`, que clasifica cada alerta. Sin alertas de riesgo alto ni
  vulnerabilidades en el código del proyecto.
- Auditoría de dependencias con pip-audit sin vulnerabilidades conocidas (`docs/seguridad/01_pip_audit.txt`) y
  escaneo activo con OWASP ZAP sin vulnerabilidades explotables (`docs/seguridad/02_zap_local.json`).

## [0.3.0] – 2026-09-27 · Sprint 1: modelado completo
### Añadido
- `src/models/classic.py`: Random Forest y XGBoost con pesos por clase (misma fórmula del EDA).
- `scripts/utilidades/`: configuración del entorno, descarga y verificación del dataset, inspección y reparación de la muestra.
- `scripts/README.md` (índice y origen de cada script) y `docs/bitacora.md` (registro cronológico).
- `scripts/03_entrenar_clasicos.py`: entrenamiento y evaluación con el protocolo del proyecto: F1 macro,
  F1 sin secuencias repetidas, deriva en noviembre y diciembre, latencia, tamaño del modelo y verificación
  de umbrales. Incluye un modo de prueba rápida (`--limite-filas`).
- `scripts/04_comparar_modelos.py`: tabla comparativa, McNemar por pares con corrección de Holm, gráfica F1 frente a
  latencia y recomendación del modelo más simple entre los estadísticamente equivalentes al mejor.
- `src/models/deep.py`: CNN 1D, Transformer encoder ligero (69 703 parámetros, con máscara de relleno) y LSTM
  opcional; los tres ignoran el contenido del relleno (verificado con pruebas).
- `scripts/05_entrenar_profundos.py`: entrenamiento con parada temprana, mismo protocolo y formato de resultados que
  los modelos clásicos; latencia medida en CPU para comparar en igualdad.
- `notebooks/02_entrenar_profundos_colab.ipynb`: entrenamiento con GPU gratuita en Google Colab.
- `src/evaluation/report.py`: matriz de confusión compartida por todos los modelos.
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
