# Consideraciones éticas

Este documento analiza los riesgos éticos del clasificador de tráfico cifrado y cómo se abordaron. Parte del análisis
de impacto social y responsabilidad de la semana 4 (`docs/semana4/Workshop_Etica_Impacto_Social_Jairo_Jhayya.pdf`),
pero va más allá: varias mitigaciones que entonces eran propuestas ya están **implementadas y medidas** en el modelo y
en la aplicación web.

**Resumen de las mitigaciones implementadas**

| Riesgo | Mitigación | Dónde está | Evidencia |
|---|---|---|---|
| Desempeño desigual entre servicios | F1 de la peor categoría reportado junto al promedio; métricas por categoría con Fairlearn | Páginas Modelos, Conclusiones y Acerca de; `results/semana5/errores/` | Peor categoría: Media, 0,830 |
| Respuestas poco fiables | Abstención cuando la confianza es menor que 45 % | Página Clasificar; `app/core/explicacion.py` | En validación, el acierto sube de 91,3 % a 94,6 % y el modelo clasifica el 94,2 % de los flujos |
| Opacidad del modelo | Explicación de cada predicción con valores SHAP | Página Clasificar | Las contribuciones suman exactamente la decisión del modelo |
| Uso indebido | Usos prohibidos documentados en la aplicación y en la ficha del modelo | Página Acerca de; `docs/ficha_del_modelo.md` | |
| Privacidad | Sin identificadores; solo metadatos; nunca el contenido | `src/data_processing.py` | Prueba automática que lo verifica |
| Evasión | Escenario de relleno de paquetes evaluado | `results/semana5/workshop1/` | Con el 25 % de flujos rellenados, el F1 cae 14,8 puntos |

## 1. Análisis de sesgos

**¿El dataset tiene sesgos demográficos, culturales o de otro tipo?** No tiene sesgos demográficos en el sentido
habitual, porque no contiene datos de personas: no hay edad, género ni ubicación. Pero sí tiene tres sesgos
relevantes:

| Sesgo | Descripción |
|---|---|
| **Geográfico y cultural** | Todo el tráfico proviene de la red académica de la República Checa en 2022. Los servicios propios de otros países, como los bancos o las aplicaciones de gobierno de Ecuador, no están representados |
| **De población** | Es tráfico de una red universitaria: estudiantes, docentes e investigadores. No representa el uso de un hogar, una empresa o una red móvil |
| **De representación** | Hay un desbalance de 144,9 a 1 entre la categoría más frecuente (Advertising) y la menos frecuente (Virtual assistant). Las categorías pequeñas tienen pocos ejemplos de los que aprender |

**¿Cómo podrían afectar estos sesgos las predicciones?** Un servicio que el modelo no conoce se clasificará como el más
parecido que sí conoce, con una confianza que puede ser engañosa. Y las categorías con pocos ejemplos se clasifican
peor: en las redes neuronales, las tres categorías más pequeñas quedaron por debajo de 0,70 de F1.

**¿Qué grupos podrían ser perjudicados?** Los usuarios de servicios minoritarios o locales, cuyo tráfico se
clasificaría peor, y los proveedores de esos servicios. Si el modelo se usara para gestionar la red, podrían recibir un
trato peor sin que nadie lo notara.

## 2. Equidad

**¿El modelo trata a todos los grupos de forma equitativa?** Como el sistema no toma decisiones sobre personas, el
grupo pertinente es el servicio. El modelo recomendado es razonablemente equitativo, pero no perfecto: ninguna
categoría queda por debajo de 0,80 de acierto, aunque entre la mejor y la peor hay 16 puntos.

**Métricas de equidad evaluadas.** La tasa de acierto por categoría se calculó con `MetricFrame` de Fairlearn (Weerts
et al., 2023) sobre los 872 731 flujos de octubre.

| Métrica | Valor |
|---|---|
| Categoría con menor acierto | Media (0,828) |
| Categoría con mayor acierto | Notifications (0,988) |
| Diferencia máxima entre categorías | 0,160 |
| Razón entre la peor y la mejor | 0,838 |
| F1 de la peor categoría, versión 2 frente a optimizada | 0,830 frente a 0,840 |

**Estrategias implementadas para mitigar inequidades.**

1. **Pesos por clase** durante el entrenamiento, inversos a la frecuencia de cada categoría, para que equivocarse en un
   servicio pequeño cueste tanto como en uno grande.
2. **F1 macro como métrica principal**, que da el mismo peso a las 23 categorías.
3. **F1 de la peor categoría** reportado siempre junto al promedio, en la aplicación y en los informes.
4. **Elección del modelo con criterio de equidad:** XGBoost no solo es el más preciso, sino el más parejo entre
   categorías (brecha de 15,6 puntos, frente a 39,9 del Transformer).

## 3. Privacidad

**¿Se utilizan datos personales o sensibles?** El modelo no usa datos personales. El dataset original incluye
direcciones IP, que la jurisprudencia europea considera datos personales incluso cuando son dinámicas (Tribunal de
Justicia de la Unión Europea, 2016), pero el proyecto las excluye junto con el nombre del servidor (SNI) y la huella del
cliente (JA3). El dataset fue publicado anonimizado y con aprobación de un comité de ética.

**¿Cómo se protege la privacidad de los usuarios?**

1. **Minimización:** solo se usan metadatos de los flujos; el contenido cifrado nunca se lee.
2. **Exclusión de identificadores**, verificada por una prueba automática (`tests/test_data_processing.py`).
3. **La aplicación no guarda nada:** los archivos que sube un usuario se procesan en memoria y no se almacenan.

**Cumplimiento con regulaciones.**

| Norma | Qué exige | Situación |
|---|---|---|
| Reglamento General de Protección de Datos (RGPD) de la Unión Europea | Base legal y minimización para tratar datos personales | Cumple: datos anonimizados en origen |
| Ley Orgánica de Protección de Datos Personales del Ecuador (2021) | Finalidad, minimización, evaluación de impacto y derecho a no ser objeto de decisiones automatizadas | Cumple en el uso académico; un despliegue real requeriría evaluación de impacto y supervisión humana |
| Ley Orgánica de Telecomunicaciones del Ecuador (2015) | Secreto de las comunicaciones y neutralidad de red (no priorizar ni degradar servicios) | El modelo no lee comunicaciones; la priorización comercial está prohibida en su uso |
| Constitución del Ecuador (2008), art. 66 | Inviolabilidad de la correspondencia y protección de datos | No se examina el contenido de ninguna comunicación |
| Reglamento de Inteligencia Artificial de la Unión Europea (2024) | Obligaciones según el riesgo del sistema | Como herramienta de gestión agregada, probablemente no sería de alto riesgo; sí lo sería si perfilara personas |

La CCPA de California no aplica, porque no hay datos de residentes de ese estado ni actividad comercial.

## 4. Transparencia y explicabilidad

**¿El modelo es interpretable?** Parcialmente. XGBoost combina cientos de árboles y no puede leerse como una regla
simple, pero sí permite explicar cada decisión con exactitud.

**¿Los usuarios entienden cómo funciona?** La aplicación lo explica en lenguaje no técnico: la página Inicio describe
el problema, la página Acerca de resume qué hace el modelo y sus límites, y cada clasificación viene acompañada de sus
razones.

**Técnicas de explicabilidad implementadas.**

| Técnica | Qué muestra | Dónde |
|---|---|---|
| **Valores SHAP por predicción** (Lundberg y Lee, 2017), calculados por el propio XGBoost | Las cinco variables que más pesaron en cada decisión, con nombres comprensibles | Página Clasificar, sección "¿Por qué…?" |
| Importancia de variables y de hiperparámetros | Qué información usa el modelo y qué decisiones de diseño importan | `docs/optimizacion.md` |
| Confianza calibrada | Si la confianza es fiable: error de calibración de 0,019 | Página Clasificar y `notebooks/05_evaluacion.ipynb` |
| Matrices de confusión y análisis de errores | Con qué categorías se confunde cada una | Páginas Modelos y Conclusiones |

Se comprobó que las contribuciones SHAP de las 60 variables, sumadas, reproducen exactamente la puntuación que el modelo
asigna a cada categoría: la explicación no es una aproximación, sino una descomposición fiel de la decisión.

## 5. Impacto social

**¿Qué impacto positivo puede tener el proyecto?**

1. Gestionar redes **sin vulnerar la confidencialidad**: es una alternativa menos invasiva que la inspección profunda
   de paquetes o la interceptación del cifrado.
2. **Acceso para instituciones con pocos recursos:** el modelo funciona en un procesador común (0,149 ms por flujo) y el
   código es abierto.
3. Mejor diagnóstico y planificación de la capacidad de una red.
4. **Conocimiento abierto y verificable:** código, resultados y aplicación son públicos.

**¿Qué impactos negativos podrían surgir?** Vigilancia de los hábitos de las personas, discriminación entre servicios
(priorizar unos y degradar otros), errores desiguales entre categorías y degradación silenciosa del modelo con el paso
del tiempo.

**¿Quiénes se benefician y quiénes podrían ser perjudicados?**

| Se benefician | Podrían ser perjudicados |
|---|---|
| Administradores de redes universitarias y de proveedores | Personas usuarias, si el modelo se usara para vigilarlas |
| Instituciones con pocos recursos | Usuarios y proveedores de servicios minoritarios o locales |
| La comunidad académica | Periodistas y activistas, más expuestos a la vigilancia |

## 6. Responsabilidad

**¿Quién es responsable si el modelo falla?**

| Rol | Responsabilidad |
|---|---|
| Desarrollador y científico de datos (el autor) | Implementación correcta, evaluación honesta, documentación de límites y control de la deriva |
| Organización que lo despliegue | Decidir para qué se usa, garantizar la supervisión humana y cumplir la normativa |
| Responsable del servicio de red | Aprobar cada decisión que se tome con base en el modelo |
| Reguladores | Supervisar el secreto de las comunicaciones y la protección de datos |

**Mecanismos de accountability.** El repositorio guarda cada versión del modelo con su huella SHA-256, sus métricas y
la decisión que llevó a adoptarla; las versiones se comparan siempre con datos de validación (`scripts/11_comparar_versiones.py`);
las 103 pruebas automáticas protegen el comportamiento del sistema; y cualquier persona puede reportar un problema en
los *issues* del repositorio.

**Plan de monitoreo y actualización.**

| Qué se vigila | Cada cuánto | Qué se hace si falla |
|---|---|---|
| Deriva del tráfico (distancia de Jensen-Shannon) y F1 en una muestra etiquetada | Mensual | Si el F1 cae más de 3 puntos, reentrenar |
| F1 de la peor categoría | En cada versión | No publicar una versión con una categoría por debajo de 0,80 |
| Proporción de abstenciones | Mensual | Si crece, revisar si apareció tráfico nuevo |
| Reentrenamiento | Trimestral | Adoptar la nueva versión solo si mejora en validación; si no, conservar la anterior |

## 7. Uso dual y mal uso

**¿Podría el modelo usarse con fines maliciosos?** Sí. El mismo modelo que sirve para gestionar una red podría usarse
para saber qué servicios usa una persona y cuándo, si se combina con su dirección IP, o para degradar servicios de la
competencia. Es un caso típico de uso dual.

**¿Qué salvaguardas se han implementado?**

1. **Usos prohibidos documentados** en la página Acerca de y en la ficha del modelo: vigilancia o perfilado de
   personas, priorización comercial de servicios y decisiones automáticas sin revisión humana.
2. **Diseño orientado a lo agregado:** el sistema clasifica flujos, no personas, y no conserva ninguna asociación entre
   un flujo y su origen.
3. **Abstención:** el modelo no da respuestas cuando duda, lo que reduce las decisiones basadas en errores.
4. **Transparencia sobre la evasión:** se documenta que el relleno de paquetes vuelve inútil al modelo, lo que también
   indica cómo protegerse de él.

**Limitaciones de uso claramente documentadas** en la página Acerca de, en `docs/ficha_del_modelo.md` y en la sección
siguiente.

## 8. Limitaciones reconocidas

**¿En qué casos no debe usarse el modelo?**

1. Para vigilar, perfilar o tomar decisiones sobre personas.
2. Para priorizar, degradar o bloquear servicios con fines comerciales.
3. En redes muy distintas de la de origen (hogares, empresas o redes móviles, u otros países) sin reentrenarlo con
   datos locales.
4. Como única fuente para decisiones automáticas sobre el tráfico, sin supervisión humana.

**¿Qué advertencias deben darse a los usuarios?** Que el modelo aprendió del tráfico checo de 2022; que su desempeño baja
con el tiempo (unos 4,7 puntos de octubre a diciembre); que confunde servicios que comparten infraestructura, como
Media, Social y Advertising; y que la confianza que muestra es fiable, pero no es una garantía.

**Casos límite donde el modelo no es confiable**, medidos en las pruebas de estrés de la semana 5:

| Caso | Efecto medido |
|---|---|
| Faltan datos del flujo (10 %) | El F1 cae 7,9 puntos |
| Un servicio domina el tráfico (80 %) | El F1 cae 11,1 puntos, aunque la exactitud apenas cambia |
| Flujos de una población que no vio al entrenar | El F1 cae 53,6 puntos |
| Paquetes rellenados hasta un tamaño fijo | Con todos los flujos rellenados, el F1 cae a 0,070 |
| Variaciones de red mayores del 20 % en tiempos y bytes | El F1 cae 7,8 puntos |

En cambio, el modelo sí es confiable frente a variaciones habituales de hasta el 10 % y a dos meses de deriva.

## Referencias

Lundberg, S. M., y Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural
Information Processing Systems* (Vol. 30, pp. 4765-4774). Curran Associates.

Tribunal de Justicia de la Unión Europea. (2016). Sentencia de 19 de octubre de 2016, *Patrick Breyer c. Bundesrepublik
Deutschland*, asunto C-582/14.

Weerts, H., Dudík, M., Edgar, R., Jalali, A., Lutz, R., y Madaio, M. (2023). Fairlearn: Assessing and improving
fairness of AI systems. *Journal of Machine Learning Research, 24*(257), 1-8.

Las normas ecuatorianas y europeas citadas, con sus artículos y fuentes oficiales, están en el informe de la semana 4.
