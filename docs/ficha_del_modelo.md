# Ficha del modelo: clasificador de tráfico TLS cifrado

Esta ficha sigue la estructura propuesta por Mitchell et al. (2019) para documentar modelos de aprendizaje automático.
Describe el modelo publicado en `models/best_model.pkl` y usado por la aplicación web.

## Detalles del modelo

| Campo | Valor |
|---|---|
| Desarrollador | Jairo Wladimir Jhayya Perlaza, Maestría en Inteligencia Artificial, UEES |
| Fecha | Octubre de 2026 |
| Versión | 2 (la versión optimizada, de 154 MB, se distribuye como archivo adjunto de la versión 1.0 del repositorio) |
| Tipo | XGBoost multiclase: árboles de decisión construidos en secuencia |
| Entrada | 60 variables de metadatos de un flujo TLS (estadísticas, banderas TCP, motivo de cierre e histogramas) |
| Salida | Una de 23 categorías de servicio, con su probabilidad |
| Licencia | MIT (código y modelo); los datos de entrenamiento tienen licencia CC BY 4.0 |
| Integridad | Huella SHA-256 registrada en `models/README.md`; la aplicación no carga un modelo cuya huella no coincida |

## Uso previsto

| Previsto | Fuera de alcance |
|---|---|
| Investigación y docencia sobre clasificación de tráfico cifrado | Vigilar o perfilar a personas |
| Estadísticas agregadas de una red para planificar capacidad o diagnosticar problemas, con supervisión humana | Priorizar, degradar o bloquear servicios con fines comerciales |
| Demostraciones en la aplicación web del proyecto | Decisiones automáticas sobre personas o sobre el tráfico sin revisión humana |
| | Redes muy distintas de la de origen sin reentrenar con datos locales |

## Factores relevantes

El desempeño varía según la **categoría de servicio**, la **longitud del flujo** (peor en los muy cortos y muy largos),
si la **secuencia ya apareció** en el entrenamiento, el **paso del tiempo** y la **red de origen** de los datos.

## Métricas

F1 macro como métrica principal, porque pesa igual a las 23 categorías; F1 de la peor categoría como métrica de
equidad; F1 sin secuencias repetidas, para medir generalización; error de calibración esperado; y tiempo por flujo.

## Datos de entrenamiento y de evaluación

| Conjunto | Datos | Flujos |
|---|---|---|
| Entrenamiento | Septiembre de 2022 (80 %) | 640 102 |
| Validación | Septiembre de 2022 (20 %) | 160 026 |
| Prueba | Octubre de 2022 | 872 731 |
| Deriva | Noviembre y diciembre de 2022 | 1 460 279 |

Fuente: muestra del 2 % de CESNET-TLS-Year22 (Hynek et al., 2024), tráfico real de la red académica checa. Sin
direcciones IP, nombre de servidor, huella del cliente ni ASN de destino. Detalles en `docs/analisis_datos.md`.

## Resultados

| Métrica | Valor |
|---|---|
| F1 macro en octubre | 0,903 (IC 95 % de 0,902 a 0,904) |
| F1 sin secuencias repetidas | 0,885 |
| F1 en diciembre | 0,856 |
| Peor categoría | Media, F1 de 0,830 |
| Área bajo la curva ROC (macro) | 0,996 |
| Error de calibración esperado | 0,019 |
| Tiempo por flujo | 0,149 ms |
| Con abstención (umbral de 45 %), en validación | Clasifica el 94,2 % de los flujos, con un acierto del 94,6 % |

## Consideraciones éticas

El modelo puede usarse para inferir hábitos de las personas si se combina con su dirección IP; por eso su uso para
vigilancia está prohibido. No lee el contenido de las comunicaciones. El análisis completo está en
`docs/consideraciones_eticas.md`.

## Advertencias y recomendaciones

1. Aprendió del tráfico checo de 2022; los servicios de otros países no están representados.
2. Pierde unos 4,7 puntos de F1 en dos meses: debe vigilarse la deriva y reentrenarse periódicamente.
3. Confunde servicios que comparten infraestructura, como Media, Social y Advertising.
4. El relleno de paquetes lo vuelve inútil; no debe usarse en contextos donde se espere evasión.
5. Usar siempre la abstención y revisar las explicaciones antes de actuar sobre una clasificación.

## Referencias

Hynek, K., Luxemburk, J., Pešek, J., Čejka, T., y Šiška, P. (2024). CESNET-TLS-Year22: A year-spanning TLS network
traffic dataset from backbone lines. *Scientific Data, 11*, 1156. https://doi.org/10.1038/s41597-024-03927-4

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., y Gebru, T.
(2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and
Transparency* (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596
