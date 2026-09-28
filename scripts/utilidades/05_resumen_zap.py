"""
Compara los dos escaneos de OWASP ZAP y clasifica cada alerta.

Entrada:
  docs/seguridad/02_zap_local.json            escaneo ACTIVO de la copia local (http://localhost:8501)
  docs/seguridad/04_zap_publica_pasivo.json   escaneo PASIVO de la app publicada (Streamlit Community Cloud)
Salida:
  docs/seguridad/resumen_zap.md               tabla comparativa con clasificación y tratamiento

Las cifras se leen de los informes de ZAP: nada se transcribe a mano. Cada alerta se clasifica por su
identificador de regla (plugin) según quién puede corregirla: nuestro código, la plataforma o nadie
(falso positivo). Una alerta con un plugin no clasificado aparece como "Revisar" para no pasar inadvertida.

Uso:  python scripts/utilidades/05_resumen_zap.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "docs/seguridad"
RIESGO = {"3": "Alto", "2": "Medio", "1": "Bajo", "0": "Informativo"}

# plugin de ZAP → (responsable, tratamiento)
CLASIFICACION = {
    "10038": (
        "Plataforma",
        "Riesgo aceptado: Streamlit no permite configurar CSP; se mitiga en el código "
        "(sin HTML del usuario, textos con textContent)",
    ),
    "10020": (
        "Plataforma",
        "Riesgo aceptado: sin cabecera configurable; la app no tiene sesiones ni acciones "
        "sensibles que un marco malicioso pueda explotar",
    ),
    "10106": ("Plataforma", "Resuelto en producción: la plataforma sirve todo por HTTPS"),
    "10021": ("Plataforma", "Riesgo aceptado: cabecera no configurable; la app solo sirve archivos propios"),
    "10035": (
        "Plataforma",
        "Riesgo aceptado: sin HSTS configurable; HTTP redirige de inmediato a HTTPS y "
        "ningún dato de la app viaja sin cifrar",
    ),
    "10036": ("Plataforma", "Riesgo aceptado: la plataforma administra y actualiza su servidor"),
    "10096": (
        "Falso positivo",
        "Descartado: constantes numéricas en librerías compiladas de Streamlit, no marcas de tiempo",
    ),
    "10027": (
        "Falso positivo",
        "Descartado: palabras comunes en el código de terceros de Streamlit; no revelan información",
    ),
    "10109": (
        "Informativa",
        "Sin acción: confirma una aplicación de una sola página; la lógica interna la cubren las pruebas propias",
    ),
    "10015": ("Informativa", "Sin acción: no-cache en el HTML es la práctica correcta para contenido que cambia"),
    "10112": (
        "Informativa",
        "Sin acción: identifica la cookie anti-XSRF de Streamlit, un control de seguridad esperado",
    ),
}


def alertas(archivo: Path) -> dict:
    """{plugin: (nombre, riesgo, apariciones)} de un informe de ZAP; vacío si el archivo no existe."""
    if not archivo.exists():
        return {}
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    res = {}
    for sitio in datos.get("site", []):
        for a in sitio.get("alerts", []):
            pid = str(a.get("pluginid"))
            previo = res.get(pid, (a.get("name"), a.get("riskcode"), 0))
            res[pid] = (a.get("name"), str(a.get("riskcode")), previo[2] + int(a.get("count", 0)))
    return res


def main():
    local = alertas(SEG / "02_zap_local.json")
    publica = alertas(SEG / "04_zap_publica_pasivo.json")
    if not local and not publica:
        raise SystemExit("No se encontraron los informes de ZAP en docs/seguridad/")
    plugins = sorted(set(local) | set(publica), key=lambda p: (-int((local.get(p) or publica.get(p))[1]), p))
    filas = []
    for p in plugins:
        nombre, riesgo, _ = local.get(p) or publica.get(p)
        pendiente = ("Revisar", "Plugin no clasificado: analizar antes de cerrar el informe")
        responsable, tratamiento = CLASIFICACION.get(p, pendiente)
        filas.append(
            f"| {nombre} | {RIESGO.get(riesgo, riesgo)} | {local[p][2] if p in local else '—'} | "
            f"{publica[p][2] if p in publica else '—'} | {responsable} | {tratamiento} |"
        )
    altos = [p for p in plugins if (local.get(p) or publica.get(p))[1] == "3"]
    revisar = [p for p in plugins if p not in CLASIFICACION]
    texto = (
        "# Resumen comparativo de OWASP ZAP\n\n"
        "Generado por `scripts/utilidades/05_resumen_zap.py` a partir de los informes de ZAP de esta carpeta.\n"
        "Columnas *Local* y *Pública*: número de apariciones de cada alerta (— = no apareció).\n\n"
        "| Alerta | Riesgo | Local (activo) | Pública (pasivo) | Responsable | Tratamiento |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(filas) + "\n\n"
        f"- Alertas distintas: {len(plugins)} · de riesgo alto: {len(altos)} · sin clasificar: {len(revisar)}\n"
    )
    (SEG / "resumen_zap.md").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
