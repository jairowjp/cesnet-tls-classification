# Notas para retomar: Semana 5 y Proyecto Final

Proyecto Integrador en IA (MIAR0545), UEES · Jairo Wladimir Jhayya Perlaza · Docente: Gladys María Villegas Rugel
Guardado el 20/10/2026. Estado: **plan aprobado, ejecución pendiente** (Jairo pidió guardar y ejecutar más adelante).

---

## A. SEMANA 5 · Validación en contextos reales o simulados

**Entrega: antes del 28/10/2026.** Materiales: presentación de 26 diapositivas y notebook de apoyo `Semana5_Workshops_semana5.ipynb`
(caso de ejemplo: tumores de mama, binario, 569 muestras, Random Forest).

### A.1 Decisiones de Jairo (20/10/2026)
1. **Modelo a optimizar en el Workshop 2: XGBoost** (el de producción).
2. **Entregables (los dos):**
   - El **notebook de la docente adaptado al proyecto y ejecutado**, con las TAREAS 1, 2 y 3 respondidas.
   - Un **informe breve en formato UEES** con las 8 secciones de la fase (cada una con decisión, evidencia y limitaciones);
     será la base de `docs/optimizacion.md` del proyecto final.
3. **Incluir el comentario de la docente de la semana 3** (parada temprana en las redes), con una ejecución nocturna más.

### A.2 Estructura del notebook de la docente (a respetar)
| Sección | Contenido | Tareas |
|---|---|---|
| 0. Baseline | DummyClassifier(strategy="stratified") vs. modelo; 6 métricas; CV 5×3 + IC95 % por bootstrap (2 000) | — |
| Workshop 1 | Escenarios A ruido, B faltantes, C desbalance, D cambio de distribución, E variable caída; cada uno con 10 semillas (`evaluar_perturbado`); UMBRAL definido ANTES; tabla con veredicto OK/FALLA; gráficos | **TAREA 1** (4 preguntas) y **TAREA 2** (escenario propio) |
| Workshop 2 | Optuna TPE (semilla 42), CV solo sobre train, 60 pruebas; historial, `plot_param_importances`, `plot_slice`, dependencia parcial con RandomForestRegressor sustituto, `plot_contour` | **TAREA 3** (4 preguntas) |
| Cierre | Test una sola vez: optimizado vs. original; prueba de estrés del optimizado | — |

Preguntas de las tareas:
- **TAREA 1:** (1) ¿qué escenario degrada más el recall y cuál la precisión? (2) ¿por qué la accuracy engaña en el desbalance? (3) diferencia entre subgrupo de referencia y distinto, y qué harías en producción; (4) ¿qué escenario falla el criterio y qué mitigación propones?
- **TAREA 2:** diseñar un escenario propio y agregarlo al resumen.
- **TAREA 3:** (1) ¿qué hiperparámetros importan y cuáles no?, ¿las diferencias de F1 son grandes?; (2) ¿hay interacción entre los dos del contorno?; (3) ¿qué rangos ampliar y qué fijar?; (4) ¿por qué una diferencia de 0,002 en F1 no justifica elegir? (conectar con el IC de la parte 0).

Lección del notebook de referencia: el optimizado (F1 0,942) NO superó al original (0,959) y la diferencia cae en la variabilidad.
Se valora la honestidad del análisis: si nos pasa, se reporta así.

### A.3 Adaptaciones necesarias al proyecto (multiclase, datos grandes)
| Notebook | Problema | Adaptación |
|---|---|---|
| Recall de la clase positiva | 23 clases | F1 macro + **F1 de la peor categoría**; recall y precisión macro |
| AUC binaria (`[:, 1]`) | Solo dos clases | AUC multiclase uno contra el resto; AUC-PR macro |
| CV 5×3 sobre todo el train | XGBoost: 90 min por entrenamiento completo | CV sobre submuestra estratificada |
| Pliegues aleatorios | 30,3 % de secuencias repetidas → fuga entre pliegues | **Pliegues agrupados** (StratifiedGroupKFold) por huella de la secuencia de paquetes |
| Estrés con 10 semillas sobre el test completo | ~5 h de predicción | Submuestra estratificada del test (~100 000 flujos), ~35 min |
| Optuna 60 pruebas × CV 5 | 300 entrenamientos de XGBoost | 3 pliegues sobre submuestra + poda (MedianPruner); ~3 h de noche |
| "Test una sola vez" | Octubre ya se consultó en la semana 3 y en la v2 | Declararlo como limitación; usar octubre solo en el cierre |

Escenarios del Workshop 1 adaptados al tráfico de red:
- **A Ruido:** ruido en tamaños y tiempos de paquetes (latencia, jitter).
- **B Faltantes:** enmascarar variables (imputar con la mediana) y además **flujos truncados** a 20, 10 y 5 paquetes.
- **C Desbalance:** una categoría domina el tráfico del test → la exactitud engaña, el F1 macro no.
- **D Cambio de distribución:** entrenar con flujos cortos y probar con largos; además la deriva real hasta diciembre.
- **E Variable caída:** las 3 más importantes de XGBoost llegan constantes (mediana de entrenamiento).
- **TAREA 2 (propio): relleno de paquetes (padding), técnica de evasión real** → cierra el riesgo R5 del informe de ética.
- **Umbral (antes de ejecutar):** caída máxima de **5 puntos de F1 macro** (justificar en la TAREA 1).

Workshop 2 (Optuna sobre XGBoost): profundidad, tasa de aprendizaje (escala log), subsample, colsample_bytree,
reg_lambda (log), min_child_weight, con parada temprana; ~50–60 pruebas en submuestra.

Redes (comentario de la semana 3): programador **ReduceLROnPlateau** (OneCycle impide que la parada temprana actúe porque
fija la duración y baja la tasa hasta casi cero al final), límite de épocas alto (~150), parada temprana real; si al
detenerse siguen en subajuste, **aumentar capacidad** (opcional: Optuna con poda Hyperband sobre la capacidad).
Probar primero en la muestra de 200 000 flujos del diagnóstico.

### A.4 Las 8 secciones del informe (presentación de la fase)
1 Métricas básicas · 2 Evaluación estadística robusta · 3 Métricas del dominio · 4 Diagnóstico de sobreajuste y subajuste
· 5 Detección y mitigación de sesgo (Fairlearn `MetricFrame` por categoría de servicio) · 6 Problemas de generalización
(fuga, cambio de covariables, deriva de concepto, autocorrelación) · 7 Optimización de hiperparámetros · 8 Validación en
contextos reales (offline, simulación, sombra, piloto, monitoreo; umbrales y condición de reversión definidos de antemano).
En cada sección: **decisión, evidencia y limitaciones.**

### A.5 Calendario propuesto
| Día | Tarea | Quién |
|---|---|---|
| 20–21/10 | Módulos y scripts (baseline, IC, estrés, Optuna) con pruebas | Claude |
| 21/10 | Parte 0 y Workshop 1 (~1 h); Optuna de noche (~3 h) | Jairo |
| 22/10 | Workshop 2 y cierre; reentrenamiento de redes de noche | Claude y Jairo |
| 23–24/10 | Notebook con tareas respondidas e informe de 8 secciones | Claude |
| 25–26/10 | Tres revisiones completas | Claude |
| 27/10 | Revisión de Jairo y entrega (un día de margen) | Jairo |

Todo el código va en la estructura del proyecto final: `notebooks/04_optimizacion.ipynb`, `notebooks/05_evaluacion.ipynb`, `src/`.

---

## B. PROYECTO FINAL (documento PROYECTO_FINAL_descripcion_2026.pdf, 13 páginas)

Componentes: repositorio GitHub (60 %), interfaz (20 %), docs (10 %), código (10 %), videos (requisito obligatorio).
**Inconsistencia a consultar:** la rúbrica detallada da docs 15 %, interfaz 10 + 5 % y suma 95 %, no 100.

### B.1 Brechas principales detectadas
- **Estructura de carpetas** distinta: faltan los 6 `docs/*.md` (planificacion, analisis_datos, arquitectura, optimizacion,
  consideraciones_eticas, manual_usuario), `data/raw` y `data/README.md`, notebooks `01_exploracion` a `05_evaluacion`,
  `src/` con `data_processing.py`, `model.py`, `train.py`, `evaluate.py`, `utils.py` (propuesta: fachadas sobre el código
  actual), `models/` con `best_model.pkl`, `model_v1.pkl` y README (el RF v2 de 173 MB no cabe en GitHub), `app/app.py`
  (hoy `principal.py`: cambiar también la ruta en Streamlit Cloud), `tests/test_data_processing.py`, `test_model.py`,
  `test_app.py`, `results/figures`, `metrics`, `reports`.
- **README:** faltan insignias, tabla de contenidos, capturas, comparación con baseline, autores y roles (proyecto individual),
  agradecimientos y enlaces a los dos videos.
- **Código:** falta `logging` (hoy `print`), uso de **ramas** (todo en main) y versiones fijas en `requirements.txt`.
- **Modelo y evaluación:** comentario de la semana 3; **baseline explícito**; análisis de errores (qué categorías se confunden).
- **Interfaz:** verificar indicador de carga y botón "Probar con un ejemplo"; agregar sección "Acerca de"; mitigaciones
  prometidas en ética (SHAP, abstención, F1 de la peor categoría); valorados: exportación, historial, inglés, tema oscuro.
- **Ética:** `consideraciones_eticas.md` con 8 secciones (incluye uso dual y limitaciones); exige técnicas de explicabilidad
  implementadas. Penalización de 20 % si es superficial. La encuesta SUS sirve como prueba con usuarios reales.
- No pide informe técnico final separado: la documentación de `docs/` y el README lo reemplazan.

### B.2 Videos (ver la explicación en la conversación del 20/10/2026)
- **Video pitch:** máximo 5 min, 7 secciones con tiempos fijos, grabación de la interfaz real, 8–10 diapositivas, QR.
- **Video de preguntas:** preguntas entregadas la última semana de clases; responder en orden y numeradas.
- Fechas límite: en blanco en el documento (consultar).

---

## C. ESTADO AL 21/10/2026: PASO 1 PROGRAMADO Y VERIFICADO (paquete semana5_paso1.zip)
- Archivos nuevos: src/evaluate.py, src/train.py, src/utils.py, scripts/12_semana5_robustez.py (parte0, workshop1,
  errores, cierre), scripts/13_optuna_xgboost.py (optimizar, comparar, reentrenar), scripts/utilidades/09_correr_semana5.sh
  (noche1, noche2), tests/test_evaluate.py (10 pruebas), bloque semana5_config.yaml (se AGREGA a configs/experiment.yaml).
- Verificado de punta a punta con datos sintéticos (los 7 pasos). Correcciones hechas durante la verificación:
  Wilcoxon con 15 pliegues (con 5 el p mínimo es 0,0625); reentrenamiento con un solo entrenamiento; categorías ausentes
  al entrenar (escenario D y pliegues pequeños); escenario D comparado con su propia referencia; figuras en español.
- Tiempos estimados con las mediciones reales (XGBoost v2: 89,6 min con 512 000 flujos): noche1 ≈ 7–7,5 h
  (parte0 ~2 h, workshop1 ~40 min, errores ~10 min, Optuna tope 4,5 h); noche2 ≈ 4–5 h (comparar ~2–3 h, reentrenar ~1,5 h,
  cierre ~30 min).
- Pendiente después: redes con parada temprana real (comentario de la semana 3), notebook adaptado con TAREAS 1–3,
  informe UEES de 8 secciones, y el resto del proyecto final.
