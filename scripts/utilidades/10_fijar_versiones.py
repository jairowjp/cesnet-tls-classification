"""
Fija las versiones exactas de las dependencias a partir del entorno donde el proyecto funciona.

  requirements.txt       Cada dependencia directa pasa de un rango (>=) a la versión instalada (==),
                         conservando los comentarios. Es lo que exige el proyecto final.
  requirements-lock.txt  Todas las dependencias, directas e indirectas, con su versión exacta (uv pip freeze).

Ejecútalo con el entorno del proyecto activado:  python scripts/utilidades/10_fijar_versiones.py
"""

import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ruta = ROOT / "requirements.txt"
    salida, faltan = [], []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*[<>=!~]", linea)
        if not m:  # comentario o línea vacía: se conserva tal cual
            salida.append(linea)
            continue
        nombre = m.group(1)
        comentario = linea[linea.index("#") :] if "#" in linea else ""
        try:
            fija = f"{nombre}=={version(nombre)}"
        except PackageNotFoundError:
            faltan.append(nombre)
            fija = linea.split("#")[0].strip()
        salida.append(f"{fija:<28}{comentario}".rstrip())
    texto = "\n".join(salida) + "\n"
    texto = texto.replace(
        "# Dependencias directas del proyecto (Python 3.13).",
        f"# Dependencias directas del proyecto, con las versiones exactas probadas (Python {sys.version.split()[0]}).",
    )
    ruta.write_text(texto, encoding="utf-8")
    print(texto)
    if faltan:
        print("No instalados en este entorno (se dejaron sin fijar):", ", ".join(faltan))
    uv = shutil.which("uv")
    orden = [uv, "pip", "freeze"] if uv else [sys.executable, "-m", "pip", "freeze"]
    lock = subprocess.run(orden, capture_output=True, text=True, check=True).stdout  # noqa: S603
    (ROOT / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    print(f"requirements-lock.txt: {len(lock.splitlines())} paquetes con versión exacta")


if __name__ == "__main__":
    main()
