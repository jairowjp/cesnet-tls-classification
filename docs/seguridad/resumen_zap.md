# Resumen comparativo de OWASP ZAP

Generado por `scripts/utilidades/05_resumen_zap.py` a partir de los informes de ZAP de esta carpeta.
Columnas *Local* y *Pública*: número de apariciones de cada alerta (— = no apareció).

| Alerta | Riesgo | Local (activo) | Pública (pasivo) | Responsable | Tratamiento |
|---|---|---|---|---|---|
| Missing Anti-clickjacking Header | Medio | 2 | 1 | Plataforma | Riesgo aceptado: sin cabecera configurable; la app no tiene sesiones ni acciones sensibles que un marco malicioso pueda explotar |
| Content Security Policy (CSP) Header Not Set | Medio | 3 | 2 | Plataforma | Riesgo aceptado: Streamlit no permite configurar CSP; se mitiga en el código (sin HTML del usuario, textos con textContent) |
| HTTP Only Site | Medio | 1 | — | Plataforma | Resuelto en producción: la plataforma sirve todo por HTTPS |
| X-Content-Type-Options Header Missing | Bajo | 5 | 5 | Plataforma | Riesgo aceptado: cabecera no configurable; la app solo sirve archivos propios |
| Strict-Transport-Security Header Not Set | Bajo | — | 5 | Plataforma | Riesgo aceptado: sin HSTS configurable; HTTP redirige de inmediato a HTTPS y ningún dato de la app viaja sin cifrar |
| Server Leaks Version Information via "Server" HTTP Response Header Field | Bajo | — | 4 | Plataforma | Riesgo aceptado: la plataforma administra y actualiza su servidor |
| Timestamp Disclosure - Unix | Bajo | 2 | 2 | Falso positivo | Descartado: constantes numéricas en librerías compiladas de Streamlit, no marcas de tiempo |
| Re-examine Cache-control Directives | Informativo | — | 1 | Informativa | Sin acción: no-cache en el HTML es la práctica correcta para contenido que cambia |
| Information Disclosure - Suspicious Comments | Informativo | 5 | 5 | Falso positivo | Descartado: palabras comunes en el código de terceros de Streamlit; no revelan información |
| Modern Web Application | Informativo | 3 | 1 | Informativa | Sin acción: confirma una aplicación de una sola página; la lógica interna la cubren las pruebas propias |
| Session Management Response Identified | Informativo | — | 1 | Informativa | Sin acción: identifica la cookie anti-XSRF de Streamlit, un control de seguridad esperado |

- Alertas distintas: 11 · de riesgo alto: 0 · sin clasificar: 0
