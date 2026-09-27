# Bitácora del proyecto

Registro cronológico de lo que se hizo, los problemas encontrados y cómo se resolvieron.
Las fechas y horas son las reales de cada ejecución (hora de Ecuador, UTC−5).

## Semana 1 (hasta el 22/09/2026)
- Participación en el foro con tres problemas candidatos y selección del problema de tráfico cifrado.
- Workshop de Metodología SMART y Presentación del Proyecto (`docs/semana1/`).

## Semana 2

### 26/09/2026 — Obtención de los datos
| Hora | Acción | Resultado |
|---|---|---|
| Mañana | Intento con Google Colab y la librería DataZoo | Instalación correcta, pero el servidor de DataZoo (liberouter.org) respondió con error 503 y luego no respondió |
| 14:12 | Descarga desde Zenodo en Colab (28,4 GB) | Terminó en 3 h 56 min, pero la sesión se desconectó y Colab asignó una máquina nueva: el archivo se perdió |
| 14:54 | Descarga desde Zenodo en el equipo local (Kali), dentro de `tmux` | Terminó a las 18:38 (3 h 43 min). MD5 verificado: `d0dd7c84e2140bba362f6bd23de5cab7` |
| Noche | Inspección del contenido real del zip | 1 203 archivos, 45 columnas por flujo, 122 días entre septiembre y diciembre (9,7 GB) |

Script derivado: `scripts/utilidades/01_descargar_dataset.sh` y `02_inspeccionar_dataset.sh`.

### 26–27/09/2026 — Entorno de trabajo
| Problema | Causa | Solución |
|---|---|---|
| Falló la instalación de `pydantic-core` | Kali trae Python 3.14 y esa versión no tiene paquete compilado | Entorno con Python 3.13 instalado con `uv`, sin modificar el sistema |
| Riesgo de errores sutiles | Se instaló pandas 3, más nuevo que la versión probada | Se fijó `pandas < 3` (2.3.3) |
| `/tmp` de solo 408 MB | Partición pequeña del sistema | Temporales redirigidos a `~/tmp` con `TMPDIR` |

Versiones exactas guardadas en `requirements-lock.txt`. Script derivado: `scripts/utilidades/00_configurar_entorno.sh`.

### 27/09/2026 — Muestra y EDA
| Hora | Acción | Resultado |
|---|---|---|
| Madrugada | Prueba de extracción con un día | 27 485 de 1 369 296 flujos (2,007 %) en 6,6 s |
| Madrugada | Extracción completa | Falló: un día de diciembre no tiene flujos en el dataset original |
| Madrugada | Versión 2 del script: registra días vacíos y no se detiene por un día con error | 117 días con datos y 5 vacíos (12, 13, 29, 30 y 31 de diciembre) |
| Madrugada | Reparación de 18 días sin registrar en el manifiesto | Script derivado: `scripts/utilidades/03_limpiar_parquet_huerfanos.py` |
| 07:36 | Ejecución completa del notebook del EDA | 3 133 138 flujos; hallazgos en `results/eda/` y en `docs/semana2/03_eda_cesnet_tls_year22.pdf` |
| Mañana | Actualización de la Ficha de Decisión Técnica con las cifras reales | Corrección de 24 a 23 categorías; riesgo de secuencias repetidas (30,3 %) incorporado |
| 10:25 | Publicación del repositorio en GitHub | https://github.com/jairowjp/cesnet-tls-classification |

## Sprint 1 (28/09 – 04/10/2026)
- Script de entrenamiento de la línea base (`scripts/03_entrenar_clasicos.py`), probado con datos sintéticos.
- Random Forest acotado para no superar los 16 GB de RAM del equipo (con 23 clases, sin límites no cabría).
- 27/09/2026 — Prueba rápida con 50 000 flujos: F1 macro 0,635 (confirmó el pipeline con datos reales).
- 27/09/2026 — **Línea base oficial, Random Forest (hito H2):** 640 102 flujos de entrenamiento, 0,6 min.
  F1 macro en octubre 0,804 (mínimo de 0,70 cumplido); sin secuencias repetidas 0,775;
  deriva: noviembre 0,776 y diciembre 0,769; latencia 0,044 ms por flujo; modelo de 142 MB.
- Hallazgo: las clases pequeñas tienen recall alto (≈ 0,9) y precisión baja (0,46–0,63) por los pesos
  "balanced" (precisión macro 0,775 frente a recall macro 0,861). Se evaluarán pesos más suaves y ajuste de
  umbrales en el Sprint 3 (US-12). Mejores clases: Notifications (F1 0,969) y Analytics & Telemetry (0,953);
  más difíciles entre las grandes: Media (0,749) y Search (0,754).
- 27/09/2026 — **XGBoost (hito H3, adelantado al 11/10):** 23,7 min de entrenamiento con parada temprana.
  F1 macro en octubre 0,886 (umbral de excelencia de 0,85 cumplido); sin secuencias repetidas 0,863;
  deriva: noviembre 0,849 y diciembre 0,835 (cae 5,2 puntos, frente a 3,5 de Random Forest);
  latencia 0,042 ms por flujo; modelo de 36 MB (4 veces más liviano que Random Forest).
- 27/09/2026 — **McNemar con corrección de Holm, Random Forest frente a XGBoost:** en los flujos donde discrepan,
  XGBoost acierta 63 820 veces y Random Forest 16 315; diferencia significativa (p de Holm < 0,05).
  XGBoost pasa a ser la línea base que deben superar la CNN 1D y el Transformer.
- 27/09/2026 — **Redes profundas en CPU (hito H4, adelantado al 18/10):** 15 épocas cada una.
  CNN 1D: F1 macro 0,700 (mínimo cumplido al límite), sin repetidos 0,683, 28 695 parámetros (0,12 MB), 12,4 min.
  Transformer: F1 macro 0,747, sin repetidos 0,743, 70 743 parámetros (0,29 MB), 62,6 min.
  Ninguna red convergió: ambas lograron su mejor F1 de validación en la última época (limitación a abordar en US-12).
- 27/09/2026 — **Comparación final con McNemar y Holm (hito H5, adelantado al 25/10):** los 6 pares difieren con
  significancia. Orden: XGBoost (0,886) > Random Forest (0,804) > Transformer (0,747) > CNN 1D (0,700).
  **Modelo recomendado: XGBoost.** Matices: el Transformer es el que menos memoriza (pierde 0,5 puntos sin repetidos,
  frente a 2,3 de XGBoost) y supera a la CNN con significancia; las redes son hasta 300 veces más livianas.

## Sprint 3: aplicación web (27/09/2026)
- Aplicación Streamlit de ocho páginas con sistema de diseño propio: la firma del flujo como elemento central,
  cobalto y ámbar reservados para la dirección de los paquetes, tipografía Archivo y portada con el logo UEES.
- Participación guiada: recorrido de siete pasos, sugerencias en contexto, reto "Tú contra los modelos" y
  encuesta SUS con código de respuesta (el sitio no guarda datos personales).
- Seguridad desde el diseño: verificación SHA-256 del modelo, validación estricta de CSV, protección contra XSS e
  inyección de fórmulas, configuración endurecida, contenedor sin privilegios y página de Seguridad para el usuario.
- Hallazgos corregidos durante el desarrollo: función obsoleta de Streamlit (components.html), st.iframe no aísla
  el HTML (se protegió y se probó contra inyección), barras superpuestas en la firma, texto justificado en columnas
  y la app escuchando en todas las interfaces de red (ahora solo localhost en desarrollo).
- 53 pruebas automáticas: 14 de seguridad, 12 de interfaz y el resto de datos, modelos y encuesta.

## Herramientas utilizadas
Python 3.13, pandas, scikit-learn, XGBoost, PyTorch, Jupyter, VS Code, Git y GitHub en Kali Linux;
Google Colab para pruebas. El código se desarrolló con apoyo de un asistente de IA (Claude, de Anthropic);
cada script fue revisado, ejecutado y validado por el autor, como consta en esta bitácora.
