"""
Carga de la muestra y construcción de las entradas de los modelos.

Qué resuelve
------------
El EDA (docs/semana2/03_eda_cesnet_tls_year22.pdf) fijó qué recibe cada rama del
pipeline. Este módulo convierte esas decisiones en código, para que los cuatro
modelos usen exactamente los mismos datos:

  * Rama A (Random Forest y XGBoost): una tabla de 60 variables por flujo.
  * Rama B (CNN 1D y Transformer): una matriz de 30 × 3 por flujo
    (tamaño, dirección y tiempo entre paquetes).

Los campos que filtran la etiqueta o identifican personas (SNI, JA3, IP, ASN)
nunca se devuelven como entrada. Esa exclusión está en un solo lugar, aquí,
para que ningún modelo la omita por error.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, path

SEQ = 30  # longitud máxima de la secuencia de paquetes en el dataset

# --- Variables de la rama A (tabular) -------------------------------------------------
FLOW_STATS = ["DURATION", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
              "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS"]
FLAGS = [f"FLAG_{f}{s}" for f in ["CWR", "ECE", "URG", "ACK", "PSH", "RST", "SYN", "FIN"] for s in ["", "_REV"]]
END_REASONS = ["FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_END", "FLOW_ENDREASON_OTHER"]
PHIST = [f"{h}_{i}" for h in ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"] for i in range(8)]
TABULAR = FLOW_STATS + FLAGS + END_REASONS + PHIST  # 8 + 16 + 4 + 32 = 60 variables

# Variables con asimetría > 2 en el EDA: se transforman con log10(1 + x)
SKEWED = ["DURATION", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "PPI_DURATION"]

# --- Variables de la rama B (secuencia) -----------------------------------------------
SIZE = [f"SIZE_{i}" for i in range(SEQ)]
DIR = [f"DIR_{i}" for i in range(SEQ)]
IPT = [f"IPT_{i}" for i in range(SEQ)]


def load_months(months: list[str], sample_dir: Path | None = None) -> pd.DataFrame:
    """Carga los días de la muestra que pertenecen a los meses indicados ("AAAA-MM").

    Solo lee los archivos Parquet de esos meses (el nombre de cada archivo es su fecha),
    así no se carga en memoria toda la muestra cuando basta con un mes.
    """
    sample_dir = Path(sample_dir or path("muestra"))
    prefixes = tuple(m.replace("-", "") for m in months)
    files = sorted(p for p in sample_dir.glob("*.parquet") if p.stem.startswith(prefixes))
    if not files:
        raise FileNotFoundError(f"No hay archivos de {months} en {sample_dir}. Ejecuta scripts/01_extraer_muestra.py")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["CATEGORY"] = df["CATEGORY"].astype("category")
    return df


def load_split(name: str, sample_dir: Path | None = None) -> pd.DataFrame:
    """Carga una partición del diseño experimental: 'entrenamiento', 'prueba' o 'deriva'."""
    months = load_config()["datos"]["meses"]
    mapping = {"entrenamiento": [months["entrenamiento"]], "prueba": [months["prueba"]], "deriva": months["deriva"]}
    if name not in mapping:
        raise ValueError(f"Partición desconocida: {name!r}. Usa una de {list(mapping)}")
    return load_months(mapping[name], sample_dir)


def tabular_features(df: pd.DataFrame, log_transform: bool = True) -> np.ndarray:
    """Construye la matriz de la rama A (n_flujos × 60).

    log10(1 + x) comprime las colas largas detectadas en el EDA (asimetría de hasta 936
    en BYTES_REV); sin ella, unos pocos flujos enormes dominarían el modelo.
    El escalado estándar no se hace aquí: debe ajustarse solo con entrenamiento
    (dentro de un Pipeline de scikit-learn) para no filtrar información de la prueba.
    """
    X = df[TABULAR].astype("float32").copy()
    if log_transform:
        X[SKEWED] = np.log10(1.0 + X[SKEWED])
    return X.to_numpy()


def sequence_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Construye la entrada de la rama B.

    Devuelve:
      X    -> arreglo (n_flujos, 30, 3) con [tamaño/1500, dirección, log10(1 + tiempo)]
      mask -> arreglo booleano (n_flujos, 30), True en las posiciones de relleno.

    El EDA mostró que el 51,6 % de las posiciones son relleno; la máscara permite que el
    Transformer las ignore en lugar de tratarlas como paquetes reales.
    """
    size = df[SIZE].to_numpy(dtype=np.float32) / 1500.0          # 1500 bytes = tamaño máximo en Ethernet
    direction = df[DIR].to_numpy(dtype=np.float32)               # +1 cliente→servidor, −1 servidor→cliente, 0 relleno
    ipt = np.log10(1.0 + df[IPT].to_numpy(dtype=np.float32))     # tiempos en ms, muy asimétricos
    X = np.stack([size, direction, ipt], axis=-1)
    mask = direction == 0
    return X, mask


def labels(df: pd.DataFrame, classes: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Convierte CATEGORY en enteros con un orden de clases fijo y reproducible.

    Si se pasan las clases de entrenamiento, se reutilizan en prueba para que el
    entero 5 signifique siempre la misma categoría en todas las particiones.
    """
    classes = classes or sorted(df["CATEGORY"].astype(str).unique())
    index = {c: i for i, c in enumerate(classes)}
    y = df["CATEGORY"].astype(str).map(index)
    if y.isna().any():
        raise ValueError("Hay categorías en estos datos que no existen en entrenamiento")
    return y.to_numpy(dtype=np.int64), classes
