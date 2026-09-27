"""
Configuración central del proyecto.

Por qué existe este módulo
--------------------------
Todos los parámetros del experimento (meses, semilla, umbrales, hiperparámetros)
viven en un único archivo: configs/experiment.yaml. Este módulo lo lee y resuelve
las rutas respecto de la raíz del repositorio, de modo que el mismo código funciona
igual desde la terminal, desde VS Code o desde un notebook, sin rutas escritas a mano.
"""
from functools import lru_cache
from pathlib import Path

import yaml

# La raíz del repositorio es la carpeta que contiene a src/ (dos niveles arriba de este archivo).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lee configs/experiment.yaml una sola vez y lo devuelve como diccionario.

    lru_cache evita volver a leer el archivo en cada llamada. Se usa safe_load y no
    load porque este último puede construir objetos arbitrarios de Python a partir
    del YAML, un riesgo innecesario (OWASP A08: integridad de datos y software).
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Devuelve una ruta de la sección 'datos' como ruta absoluta dentro del repositorio.

    Ejemplo: path("muestra") -> /…/repositorio/data/processed/muestra_2pct
    """
    return ROOT / load_config()["datos"][key]
