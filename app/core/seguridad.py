"""
Controles de seguridad de la aplicación web.

Este módulo concentra las defensas frente a los riesgos del OWASP Top 10 (2021) que aplican
a esta aplicación. Ver SECURITY.md para la matriz completa.

  * A08 — Integridad de software y datos:
        los modelos se guardan con joblib, que internamente usa pickle. Cargar un pickle
        alterado puede EJECUTAR CÓDIGO. Por eso, antes de cargar un modelo se compara su
        huella SHA-256 con la registrada en app/models/manifest.json al prepararlo.
        La aplicación nunca carga modelos ni archivos binarios subidos por el usuario.

  * A03 — Inyección / A04 — Diseño inseguro:
        el único dato que entra desde afuera es el CSV del clasificador. Se valida ANTES de
        usarlo: tamaño, número de filas, columnas exactas, tipos numéricos, valores finitos
        y rangos físicamente posibles. Todo lo que no cumpla se rechaza con un mensaje claro.

  * Inyección de fórmulas en CSV (CSV injection):
        al descargar resultados, cualquier texto que empiece con = + - @ se neutraliza,
        para que una hoja de cálculo no lo ejecute como fórmula.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Límites del CSV subido: pequeños a propósito. El clasificador es una demostración, no un
# servicio de procesamiento masivo, y límites bajos reducen el riesgo de agotar la memoria.
MAX_BYTES = 2 * 1024 * 1024  # 2 MB (coincide con server.maxUploadSize en .streamlit/config.toml)
MAX_FILAS = 1000
SEQ = 30

FLAGS = [f"FLAG_{f}{s}" for f in ["CWR", "ECE", "URG", "ACK", "PSH", "RST", "SYN", "FIN"] for s in ["", "_REV"]]
END_REASONS = ["FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_END", "FLOW_ENDREASON_OTHER"]
FLOW_STATS = ["DURATION", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS"]
PHIST = [f"{h}_{i}" for h in ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"] for i in range(8)]
SECUENCIA = [f"{c}_{i}" for c in ["SIZE", "DIR", "IPT"] for i in range(SEQ)]
# Columnas obligatorias: las 60 que usa XGBoost más la secuencia que dibuja la firma del flujo
REQUERIDAS = FLOW_STATS + FLAGS + END_REASONS + PHIST + SECUENCIA

# Rangos válidos: cualquier valor fuera de ellos no puede provenir de un flujo real
RANGOS = {
    **{c: (0, 1) for c in FLAGS},
    **{c: (-1, 1) for c in [f"DIR_{i}" for i in range(SEQ)]},
    **{c: (0, 65535) for c in [f"SIZE_{i}" for i in range(SEQ)]},  # tamaño máximo de un paquete IP
    **{c: (0, 30) for c in ["PPI_LEN"]},
}


class ModeloAlterado(Exception):
    """La huella del archivo del modelo no coincide con la registrada: no se debe cargar."""


def sha256_de(path: Path, bloque: int = 1 << 20) -> str:
    """Huella SHA-256 de un archivo, leída por bloques para no cargarlo entero en memoria."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def verificar_integridad(modelo: Path, manifiesto: Path) -> dict:
    """Comprueba que el modelo es exactamente el que se preparó; devuelve su entrada del manifiesto.

    Lanza ModeloAlterado si el archivo fue modificado, reemplazado o no figura en el manifiesto.
    """
    registro = json.loads(manifiesto.read_text(encoding="utf-8"))
    entrada = registro.get(modelo.name)
    if entrada is None:
        raise ModeloAlterado(f"{modelo.name} no está registrado en el manifiesto")
    if sha256_de(modelo) != entrada["sha256"]:
        raise ModeloAlterado(f"La huella SHA-256 de {modelo.name} no coincide con la registrada")
    return entrada


@dataclass
class ResultadoValidacion:
    """Resultado de validar un CSV: los datos limpios o la lista de problemas encontrados."""

    datos: pd.DataFrame | None = None
    errores: list[str] = field(default_factory=list)

    @property
    def valido(self) -> bool:
        return self.datos is not None and not self.errores


def validar_csv(contenido: bytes) -> ResultadoValidacion:
    """Valida un CSV subido por el usuario antes de clasificarlo.

    Los mensajes de error describen el problema sin repetir el contenido del archivo, para
    que un texto malicioso nunca termine mostrado en la interfaz.
    """
    r = ResultadoValidacion()
    if len(contenido) == 0:
        r.errores.append("El archivo está vacío.")
        return r
    if len(contenido) > MAX_BYTES:
        r.errores.append(f"El archivo supera el máximo de {MAX_BYTES // (1024 * 1024)} MB.")
        return r
    try:
        texto = contenido.decode("utf-8")
    except UnicodeDecodeError:
        r.errores.append("El archivo no está codificado en UTF-8. Guárdalo como «CSV UTF-8».")
        return r
    try:
        # nrows = MAX_FILAS + 1: basta leer una fila de más para saber si se excede el límite
        df = pd.read_csv(io.StringIO(texto), nrows=MAX_FILAS + 1)
    except Exception:  # cualquier error de formato se informa sin exponer detalles internos
        r.errores.append("No se pudo leer el archivo como CSV. Usa la plantilla como referencia.")
        return r

    if len(df) == 0:
        r.errores.append("El archivo no contiene filas de datos.")
    if len(df) > MAX_FILAS:
        r.errores.append(f"El archivo tiene más de {MAX_FILAS} filas. Divídelo en partes más pequeñas.")
    faltantes = [c for c in REQUERIDAS if c not in df.columns]
    if faltantes:
        r.errores.append(
            f"Faltan {len(faltantes)} columnas obligatorias. Descarga la plantilla y conserva sus columnas."
        )
    if r.errores:
        return r

    datos = df[REQUERIDAS].apply(pd.to_numeric, errors="coerce")  # texto no numérico → NaN
    no_numericas = int(datos.isna().any(axis=0).sum())
    if no_numericas:
        r.errores.append(f"{no_numericas} columnas contienen valores vacíos o no numéricos.")
    elif not np.isfinite(datos.to_numpy(dtype=float)).all():
        r.errores.append("Hay valores infinitos en el archivo.")
    else:
        if (datos[FLOW_STATS + PHIST] < 0).any().any():
            r.errores.append("Hay estadísticas del flujo con valores negativos, que no son posibles.")
        fuera = [c for c, (lo, hi) in RANGOS.items() if ((datos[c] < lo) | (datos[c] > hi)).any()]
        if fuera:
            r.errores.append(
                f"{len(fuera)} columnas tienen valores fuera del rango válido "
                "(por ejemplo, direcciones distintas de −1, 0 o 1)."
            )
    if not r.errores:
        r.datos = datos.astype("float32")
    return r


def neutralizar_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Evita la inyección de fórmulas al abrir el CSV descargado en Excel o LibreOffice.

    Todo texto que empiece con = + - @ (o tabulador/retorno) se antepone con un apóstrofo,
    que las hojas de cálculo interpretan como "esto es texto, no una fórmula".
    """
    out = df.copy()
    for c in out.select_dtypes(include=["object", "string", "category"]).columns:
        out[c] = out[c].astype(str).map(lambda s: "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s)
    return out
