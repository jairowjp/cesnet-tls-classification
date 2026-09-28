# Informe de seguridad de la aplicación web

**Proyecto:** Análisis comparativo de modelos de aprendizaje automático y arquitecturas Transformer para la
clasificación de tráfico de red cifrado sobre el dataset CESNET-TLS-Year22
**Aplicación evaluada:** https://trafico-cifrado-uees.streamlit.app (Streamlit Community Cloud) y su copia local
**Fecha de la evaluación:** 27 de septiembre de 2026 · **Autor:** Jairo Wladimir Jhayya Perlaza

## 1. Alcance y método

La seguridad se incorporó desde el diseño (ver `SECURITY.md`) y se verificó con tres técnicas complementarias:

| Técnica | Herramienta | Objetivo | Alcance ético |
|---|---|---|---|
| Auditoría de dependencias | `pip-audit` | Vulnerabilidades conocidas (CVE) en las librerías de la app (OWASP A06) | Consulta de bases públicas de avisos |
| Escaneo **activo** | OWASP ZAP 2.17.0, escaneo rápido | Ataques simulados: inyección, XSS, recorrido de rutas y otros | **Solo contra la copia local** (`http://localhost:8501`), que es del autor |
| Escaneo **pasivo** | OWASP ZAP 2.17.0, plan de automatización | Cabeceras, cookies, HTTPS y contenido de la app publicada | Solo navegación normal, sin ataques: la infraestructura pertenece a un tercero |
| Pruebas propias | `pytest` (14 pruebas de seguridad) | La lógica interna que ZAP no alcanza en una app de una sola página | Datos sintéticos |

El escaneo activo no se ejecutó contra la URL pública porque atacaría la infraestructura de Streamlit Community
Cloud sin autorización. El plan del escaneo pasivo (`zap_plan_pasivo.yaml`) documenta que no incluye ataques.

## 2. Resultados

| Evidencia | Archivo | Resultado |
|---|---|---|
| Dependencias | `01_pip_audit.txt` | **Sin vulnerabilidades conocidas** |
| Escaneo activo local | `02_zap_local.json` | **Ninguna vulnerabilidad explotable.** Solo alertas de configuración y contenido |
| Cabeceras públicas | `03_cabeceras_publicas.txt` | HTTPS con HTTP/2 y HTTP/3; cookie de sesión `HttpOnly; Secure; SameSite=Lax`; HTTP redirige a HTTPS |
| Escaneo pasivo público | `04_zap_publica_pasivo.json` y `.html` | Sin alertas de riesgo alto; cabeceras de seguridad ausentes (controladas por la plataforma) |
| Comparación y clasificación | `resumen_zap.md` | Tabla generada automáticamente con cada alerta, su responsable y su tratamiento |

**Ninguno de los escaneos encontró alertas de riesgo alto ni vulnerabilidades en el código del proyecto.**

## 3. Qué cambió entre la copia local y la publicada

- **Resuelto por la plataforma:** el sitio solo HTTP. En producción todo se sirve cifrado, y la cookie de sesión
  lleva los atributos `Secure`, `HttpOnly` y `SameSite=Lax`.
- **Nuevas en la publicación**, ambas de riesgo bajo y controladas por la plataforma: la ausencia de HSTS y la
  divulgación de la versión del servidor web (`nginx`).
- **Presentes en ambas:** la ausencia de cabeceras contra *clickjacking* y de `X-Content-Type-Options`. Streamlit no
  permite configurarlas y la plataforma no las agrega.

## 4. Riesgo residual aceptado

Las alertas pendientes dependen de cabeceras HTTP que ni Streamlit ni Community Cloud permiten configurar. Se
aceptan con esta justificación:

| Riesgo | Por qué su impacto es bajo en esta aplicación |
|---|---|
| Sin CSP | El código nunca inserta HTML del usuario; los textos se insertan con `textContent`; los datos se escapan antes de llegar al navegador (verificado con pruebas de inyección) |
| Sin protección contra *clickjacking* | La app no tiene inicio de sesión, pagos ni acciones que cambien datos: un marco malicioso no puede inducir al usuario a nada perjudicial |
| Sin HSTS | La petición HTTP redirige de inmediato a HTTPS y ningún dato de la app viaja sin cifrar; el riesgo se limita al primer salto en una red hostil |
| Sin `X-Content-Type-Options` | La app solo sirve archivos propios, no contenido subido por usuarios |
| Versión del servidor visible | El servidor lo administra y actualiza la plataforma |

## 5. Recomendación para un despliegue propio

Si la aplicación se alojara en infraestructura propia (el `Dockerfile` del repositorio es el plan B), un proxy
inverso como nginx delante de Streamlit permitiría agregar `Content-Security-Policy` (con `frame-ancestors`),
`Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `Referrer-Policy` y `Permissions-Policy`, y ocultar
la versión del servidor. Así se cerrarían todas las alertas de riesgo medio y bajo de este informe.

## 6. Conclusión

La aplicación no presenta vulnerabilidades explotables ni dependencias con vulnerabilidades conocidas. Las alertas
restantes son de configuración de la plataforma de alojamiento, están identificadas, clasificadas y justificadas, y
su impacto se reduce con los controles implementados en el propio código.

## Cómo reproducir esta evaluación

```bash
pip-audit -r requirements-app.txt --desc
bash scripts/utilidades/04_ejecutar_app_local.sh          # en otra terminal, para el escaneo activo local
zaproxy -cmd -quickurl http://localhost:8501 -quickprogress -quickout "$PWD/docs/seguridad/02_zap_local.json"
zaproxy -cmd -autorun "$PWD/docs/seguridad/zap_plan_pasivo.yaml"
python scripts/utilidades/05_resumen_zap.py
```
