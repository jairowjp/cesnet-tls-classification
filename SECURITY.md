# Seguridad

Este documento explica cómo se protege el proyecto y cómo se verificará la aplicación web
con el **OWASP Top 10 (2021)** y **OWASP ZAP** antes de publicarla.

## Principios que ya se aplican

| Principio | Cómo se aplica |
|---|---|
| Minimización de datos | El modelo nunca recibe direcciones IP, SNI, huella JA3 ni ASN de destino. La muestra publicada tampoco los contiene |
| Sin secretos en el código | Las claves, si alguna vez se usan, van como secretos de Streamlit Community Cloud o variables de entorno. `.gitignore` bloquea `.env`, `*.key` y `.streamlit/secrets.toml` |
| Integridad de datos | El dataset se verificó con su MD5 oficial (`d0dd7c84e2140bba362f6bd23de5cab7`) y la configuración se lee con `yaml.safe_load` |
| Dependencias controladas | Versiones exactas en `requirements-lock.txt` y auditoría con `pip-audit` (tarea de VS Code) |
| Revisión estática | Ruff con las reglas de seguridad de Bandit (`S`) activadas en `pyproject.toml` |

## Controles implementados en la aplicación web

Cada control tiene al menos una prueba automática que lo verifica (`python -m pytest -q`).

| Riesgo OWASP Top 10 (2021) | Control implementado | Dónde está | Prueba |
|---|---|---|---|
| A03 Inyección (CSV malicioso) | Validación de tamaño (2 MB), filas (1 000), columnas exactas, tipos numéricos, valores finitos y rangos físicos | `app/core/seguridad.py` → `validar_csv` | `tests/test_app_seguridad.py` (9 pruebas) |
| A03 Inyección (XSS en la firma del flujo) | JSON con `<`, `>` y `&` escapados; textos insertados con `textContent`, nunca `innerHTML`; nunca HTML del usuario | `app/core/firma.py` | `test_firma_no_permite_inyectar_codigo_en_textos` |
| A03 Inyección de fórmulas en descargas | Textos que empiezan con `= + - @` se neutralizan con apóstrofo | `app/core/seguridad.py` → `neutralizar_formulas` | `test_formulas_neutralizadas_en_descargas` |
| A04 Diseño inseguro | Límites de subida coherentes en Streamlit (`maxUploadSize = 2`) y en la validación; mensajes de error que no repiten el contenido del archivo | `.streamlit/config.toml`, `validar_csv` | `test_columnas_faltantes_se_rechaza_sin_listar_nombres` |
| A05 Configuración insegura | XSRF y CORS activos, errores sin trazas (`showErrorDetails = "none"`), sin opciones de desarrollador, sin estadísticas de uso, sin recarga de código; en desarrollo la app se lanza solo en `localhost` (`scripts/utilidades/04_ejecutar_app_local.sh`) | `.streamlit/config.toml` | Revisión con OWASP ZAP (pendiente) |
| A06 Componentes vulnerables | Dependencias mínimas de la app (sin PyTorch ni Jupyter) y versiones fijadas | `requirements-app.txt` | `pip-audit -r requirements-app.txt` |
| A08 Integridad de software y datos | El modelo solo se carga si su SHA-256 coincide con `app/models/manifest.json`; nunca se cargan modelos del usuario | `app/core/seguridad.py` → `verificar_integridad` | `test_modelo_alterado_no_se_carga` |
| Mínimo privilegio | El contenedor corre con un usuario sin privilegios (id 1000), nunca como root | `Dockerfile` | Inspección de la imagen |
| Privacidad | Tipografía servida desde la app (sin Google Fonts); la muestra publicada no contiene IP, SNI, JA3 ni ASN | `app/static/fonts`, `scripts/02_crear_muestra_app.py` | — |
| Privacidad de la encuesta | El sitio no guarda respuestas ni pide datos personales: el evaluador recibe un código con sus respuestas y su perfil, y decide si enviarlo; los códigos individuales quedan fuera de GitHub | `app/paginas/encuesta.py`, `app/core/sus.py`, `.gitignore` | `test_codigo_ida_y_vuelta` |
| Integridad de la evaluación SUS | Caracteres de control que detectan errores de copia; el puntaje se recalcula desde las respuestas | `app/core/sus.py`, `scripts/07_consolidar_sus.py` | `test_codigo_detecta_todo_error_de_un_caracter`, `test_consolidacion_con_codigos` |
| Transparencia | Página **Seguridad** que explica al usuario, en lenguaje claro, cómo se tratan sus datos | `app/paginas/seguridad.py` | Prueba de interfaz |

## Plan de verificación de la aplicación web (Sprint 3)

| Riesgo OWASP Top 10 | Dónde aplica en este proyecto | Control previsto |
|---|---|---|
| A01 Control de acceso roto | La app es pública y de solo lectura; no hay cuentas | Sin funciones de escritura ni de administración expuestas |
| A02 Fallas criptográficas | Tráfico entre el navegador y la app | HTTPS obligatorio en Streamlit Community Cloud; ningún secreto en el cliente |
| A03 Inyección (incluye XSS) | CSV subido por el usuario y textos mostrados en pantalla | Esquema de columnas y tipos validado; nunca se inserta HTML con contenido del usuario |
| A04 Diseño inseguro | Carga de archivos | Tamaño máximo de archivo, número máximo de filas y rechazo de formatos distintos a CSV |
| A05 Configuración insegura | Servidor de Streamlit | Protección XSRF activa, modo de desarrollo desactivado, mensajes de error sin trazas internas |
| A06 Componentes vulnerables | Librerías de Python | `pip-audit` antes de cada publicación; versiones fijadas |
| A08 Integridad de software y datos | Modelos serializados | Solo se cargan modelos propios del repositorio; nunca archivos `pickle` o `joblib` enviados por el usuario, porque pueden ejecutar código |
| A09 Registro y monitoreo | Errores de la aplicación | Registro de errores sin datos del usuario |

## Escaneo con OWASP ZAP

| Tipo de escaneo | Contra qué | Motivo |
|---|---|---|
| **Activo** (ataques simulados) | Copia local de la app en Docker (`http://localhost:7860`) | Se prueba a fondo sin afectar infraestructura ajena |
| **Pasivo** (solo observa respuestas) | URL pública en Streamlit Community Cloud | Verifica cabeceras y configuración reales sin atacar servidores de terceros |

Los reportes de ZAP y las correcciones aplicadas se guardarán en `docs/seguridad/`.

**Limitación conocida:** Streamlit no permite configurar todas las cabeceras HTTP de seguridad (por ejemplo, una
Content-Security-Policy estricta), y en Streamlit Community Cloud la plataforma controla parte de ellas.
Los hallazgos que dependan de la plataforma se documentarán como riesgo aceptado, con su justificación.

## Reportar un problema

Si encuentras una vulnerabilidad, abre un *issue* en este repositorio sin incluir datos sensibles.
