# Planificación del proyecto

**Proyecto:** Análisis comparativo de modelos de aprendizaje automático y arquitecturas Transformer para la
clasificación de tráfico de red cifrado sobre el dataset CESNET-TLS-Year22
**Autor:** Jairo Wladimir Jhayya Perlaza · **Docente:** Gladys María Villegas Rugel
**Universidad de Especialidades Espíritu Santo (UEES)** · Maestría en Inteligencia Artificial · Proyecto Integrador (MIAR0545)

Este documento resume cómo se planificó el proyecto y cómo se ejecutó realmente. El plan original completo, con el
estado del arte, el análisis de datasets y el presupuesto, está en la Presentación del Proyecto
(`docs/semana1/02_presentacion_del_proyecto.pdf`).

## 1. El problema

Hoy casi todo lo que viaja por Internet va cifrado. TLS 1.3 y QUIC protegen la privacidad de las personas, pero también
dejaron sin utilidad a las herramientas que los administradores de red usaban para saber qué servicios circulaban por
su infraestructura. Si el contenido ya no se puede leer, lo único que queda es observar cómo se comporta cada conexión:
cuánto pesan sus paquetes, en qué dirección van y cada cuánto aparecen.

La literatura reporta exactitudes cercanas al 100 % con ese enfoque, pero muchas de esas cifras salen de datasets
antiguos, divididos al azar y con acceso al contenido de los paquetes, lo que infla los resultados. Además, el tráfico
real cambia de un mes a otro y los modelos pierden precisión con rapidez.

**Pregunta que guía el proyecto:** si solo se cuenta con los metadatos de cada flujo, ¿qué conviene más, un modelo
clásico y barato como Random Forest o una red profunda con mecanismos de atención?

**Por qué es relevante:** un clasificador así permite gestionar una red (planificar su capacidad, detectar anomalías)
sin descifrar las comunicaciones, que es la alternativa más respetuosa con la privacidad. Al mismo tiempo plantea
riesgos de uso indebido que el proyecto analiza en `docs/consideraciones_eticas.md`.

## 2. Objetivos

**Objetivo general.** Desarrollar, en seis semanas, un estudio comparativo entre modelos clásicos de aprendizaje
automático y arquitecturas profundas basadas en atención para clasificar tráfico de red cifrado en categorías de
servicio sobre CESNET-TLS-Year22, evaluando desempeño (F1 macro) y costo computacional con partición temporal, y
poniendo los resultados a disposición en una aplicación web interactiva.

| Objetivo específico | Métrica comprometida | Resultado |
|---|---|---|
| **OE1.** Construir el conjunto experimental: muestra, partición por meses, exclusión de campos que filtran la etiqueta y análisis exploratorio | Calidad de los datos de al menos 95 %; duplicados reportados | Cumplido: completitud del 100 %, secuencias repetidas medidas (30,3 %) y ASN de destino excluido por fuga |
| **OE2.** Comparar Random Forest, XGBoost, una CNN 1D y un Transformer en desempeño y costo | F1 macro de al menos 0,70; latencia de hasta 10 ms por flujo; prueba de McNemar | Cumplido: XGBoost alcanza 0,903 (0,911 optimizado) con 0,149 ms; los seis pares difieren con significancia |
| **OE3.** Desarrollar y publicar una aplicación web y validar su usabilidad | SUS de al menos 68 con cinco evaluadores; éxito de tareas de al menos 80 % | Aplicación publicada y funcionando; la prueba SUS quedó en curso al cierre |

## 3. Alcance

| Incluido | Excluido |
|---|---|
| Metadatos de flujos TLS: secuencia de los primeros 30 paquetes, estadísticas e histogramas | Contenido de los paquetes, direcciones IP, nombre del servidor y huella del cliente |
| Cuatro modelos con las mismas particiones y semillas | Procesar capturas PCAP o tráfico propio |
| Prueba de McNemar y métricas de costo | Integración con colectores de red en producción |
| Aplicación web pública con clasificación de flujos y carga de CSV | Reentrenamiento automático y detección de intrusiones |

Limitaciones conocidas desde el inicio: los resultados provienen de una red académica checa y no se trasladan de forma
automática a otras redes, y sin el contenido de los paquetes no se pueden reproducir los modelos que lo usan.

## 4. Metodología de trabajo

El proyecto es individual, así que los tres roles de Scrum (Product Owner, Scrum Master y equipo de desarrollo)
recayeron en el autor, y la docente revisó cada incremento. Se planificaron cuatro sprints en seis semanas, del 28 de
septiembre al 8 de noviembre de 2026, con 15 horas semanales. El trabajo se registró en la bitácora del proyecto
(`docs/bitacora.md`) y en el historial del repositorio, usando ramas para cada bloque grande de trabajo (por ejemplo,
`semana5` y `proyecto-final`).

## 5. Cronograma: planificado frente a ejecutado

| Etapa | Planificado | Ejecutado | Comentario |
|---|---|---|---|
| Planificación y estado del arte | Semana 1 (hasta el 22/09) | 22/09 | Según lo previsto |
| Datos, entorno y análisis exploratorio | Sprint 1 (28/09 – 04/10) | 26/09 – 27/09 | Adelantado |
| Línea base y los cuatro modelos | Sprints 1 a 3 | 27/09 | Muy adelantado: los cuatro modelos quedaron entrenados en el primer fin de semana |
| Aplicación web | Sprint 3 | 27/09 – 05/10 | Adelantado; creció de 5 a 10 páginas |
| Seguridad de la aplicación | No planificado | 06/10 | Agregado: OWASP ZAP y pip-audit |
| Diagnóstico de sobreajuste y subajuste | No planificado | 06/10 | Actividad de la semana 3 |
| Versión 2 de los modelos | No planificado | 07/10 | Consecuencia del diagnóstico: los cuatro modelos mejoraron |
| Análisis ético | No planificado | 07/10 – 09/10 | Actividad de la semana 4 |
| Validación y optimización | Sprint 3 (ajuste de hiperparámetros) | 20/10 – 24/10 | Optuna, escenarios críticos y redes con parada temprana |
| Estructura final, interfaz y documentación | Sprint 4 (hasta el 08/11) | 24/10 – 28/10 | La entrega final se adelantó al 28/10 |
| Prueba de usabilidad (SUS) | Semana 5 | En curso | Desplazada por las actividades semanales |

## 6. Riesgos: previstos y ocurridos

| Riesgo previsto | Nivel | ¿Ocurrió? | Respuesta |
|---|---|---|---|
| Descarga fallida o lenta del dataset | Medio | No | Se descargó el primer día y se verificó su suma MD5 |
| El Transformer no converge a tiempo | Medio | En parte | Convergió, pero quedó en subajuste; se diagnosticó y se corrigió en dos rondas |
| Límites de la GPU gratuita | Medio | No | El diagnóstico, la versión 2 y la optimización se ejecutaron en el equipo local, en CPU y con ejecuciones nocturnas |
| Ningún modelo alcanza F1 macro de 0,70 | Bajo | No | Los cuatro superaron 0,80 en la versión 2 |
| Falla del despliegue en la nube | Bajo | Sí | Hugging Face dejó de ofrecer contenedores gratuitos; se migró a Streamlit Community Cloud |
| Pocos evaluadores para la prueba SUS | Bajo | Sí, por plazo | La prueba quedó en curso al cierre |
| Sobrecarga del autor | Medio | Sí | Se priorizaron los entregables; la red LSTM opcional se descartó |

Aparecieron además dos riesgos no previstos: la **memoria del equipo** (16 GB), que obligó a acotar Random Forest tras
una prueba piloto, y el **tamaño del modelo optimizado** (154 MB), que supera el límite de GitHub y se resolvió
publicándolo como archivo adjunto de una versión del repositorio.

## 7. Recursos

| Tipo | Recurso |
|---|---|
| Humanos | El autor (alrededor de 90 horas previstas) y la docente en la revisión de cada sprint |
| Hardware | Computador personal con 16 GB de RAM, sin GPU |
| Software | Python 3.13, scikit-learn, XGBoost, PyTorch, Optuna, Fairlearn, MLflow y Streamlit, todos de código abierto (versiones exactas en `requirements.txt`) |
| Plataformas | GitHub para el código y Streamlit Community Cloud para la aplicación, ambos gratuitos |
| Datos | CESNET-TLS-Year22, licencia CC BY 4.0 |

## 8. Lecciones de la planificación

1. **Planificar de menos lo desconocido.** El entrenamiento de los modelos fue mucho más rápido de lo previsto, pero
   el diagnóstico y la optimización, que no estaban en el plan, se volvieron la parte más valiosa del proyecto.
2. **Medir antes de comprometerse.** Las pruebas piloto (memoria de Random Forest, tiempos de XGBoost) evitaron
   ejecuciones nocturnas fallidas.
3. **Dejar margen para lo que depende de otros.** La prueba de usabilidad necesitaba evaluadores externos y fue lo
   primero que se desplazó cuando el calendario se apretó.
