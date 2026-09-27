# Seguridad

Este documento explica cómo se protege el proyecto y cómo se verificará la aplicación web
con el **OWASP Top 10 (2021)** y **OWASP ZAP** antes de publicarla.

## Principios que ya se aplican

| Principio | Cómo se aplica |
|---|---|
| Minimización de datos | El modelo nunca recibe direcciones IP, SNI, huella JA3 ni ASN de destino. La muestra publicada tampoco los contiene |
| Sin secretos en el código | Las claves, si alguna vez se usan, van como secretos de Hugging Face o variables de entorno. `.gitignore` bloquea `.env`, `*.key` y `.streamlit/secrets.toml` |
| Integridad de datos | El dataset se verificó con su MD5 oficial (`d0dd7c84e2140bba362f6bd23de5cab7`) y la configuración se lee con `yaml.safe_load` |
| Dependencias controladas | Versiones exactas en `requirements-lock.txt` y auditoría con `pip-audit` (tarea de VS Code) |
| Revisión estática | Ruff con las reglas de seguridad de Bandit (`S`) activadas en `pyproject.toml` |

## Plan de verificación de la aplicación web (Sprint 3)

| Riesgo OWASP Top 10 | Dónde aplica en este proyecto | Control previsto |
|---|---|---|
| A01 Control de acceso roto | La app es pública y de solo lectura; no hay cuentas | Sin funciones de escritura ni de administración expuestas |
| A02 Fallas criptográficas | Tráfico entre el navegador y la app | HTTPS obligatorio en Hugging Face Spaces; ningún secreto en el cliente |
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
| **Pasivo** (solo observa respuestas) | URL pública en Hugging Face Spaces | Verifica cabeceras y configuración reales sin atacar servidores de terceros |

Los reportes de ZAP y las correcciones aplicadas se guardarán en `docs/seguridad/`.

**Limitación conocida:** Streamlit no permite configurar todas las cabeceras HTTP de seguridad (por ejemplo, una
Content-Security-Policy estricta), y en Hugging Face Spaces el proxy de la plataforma controla parte de ellas.
Los hallazgos que dependan de la plataforma se documentarán como riesgo aceptado, con su justificación.

## Reportar un problema

Si encuentras una vulnerabilidad, abre un *issue* en este repositorio sin incluir datos sensibles.
