"""
Extrae una muestra aleatoria y reproducible de CESNET-TLS-Year22 para los meses del
proyecto (septiembre a diciembre de 2022), leyendo directamente del zip de Zenodo,
sin descomprimirlo en disco.

Por cada día:
  1. Lee el CSV comprimido (xz) en bloques y cuenta el total de flujos del día.
  2. Conserva cada flujo con probabilidad FRAC (2 % por defecto, la misma proporción
     de la muestra XS de DataZoo: 10 M de 507,7 M flujos).
  3. Convierte la secuencia de paquetes (PPI) y los histogramas (PHIST) en columnas
     numéricas de tamaño fijo y guarda el día en Parquet.

Es reanudable: si se interrumpe, al volver a ejecutarlo salta los días ya procesados.

Uso:
  python scripts/01_extraer_muestra.py --zip datasets/CESNET-TLS-Year22.zip
  python scripts/01_extraer_muestra.py --zip datasets/CESNET-TLS-Year22.zip --limit-days 1   # prueba
"""
import argparse
import io
import json
import lzma
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SEQ = 30  # longitud máxima de la secuencia de paquetes en el dataset
FLAGS = [f"FLAG_{f}{s}" for f in ["CWR", "ECE", "URG", "ACK", "PSH", "RST", "SYN", "FIN"] for s in ["", "_REV"]]
ENDREASONS = ["FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_END", "FLOW_ENDREASON_OTHER"]
SCALARS = ["TIME_FIRST", "DURATION", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
           "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS", "APP", "CATEGORY",
           "DST_ASN", "DST_PORT", "PROTOCOL"] + FLAGS + ENDREASONS
PHIST = ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]
USECOLS = SCALARS + ["PPI"] + PHIST
# SRC_IP, DST_IP, TLS_SNI y TLS_JA3 no se cargan: identifican usuarios o servidores y el SNI
# es la fuente de la etiqueta. DST_ASN y DST_PORT se conservan solo para demostrar la fuga en el EDA.

DAY_RE = re.compile(r"flows-(\d{8})\.csv\.xz$")


def list_days(zip_path, months):
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if DAY_RE.search(n)]
    days = []
    for m in members:
        d = DAY_RE.search(m).group(1)
        if f"{d[:4]}-{d[4:6]}" in months:
            days.append((d, m))
    return sorted(days)


def parse_ppi(series):
    n = len(series)
    ipt = np.zeros((n, SEQ), dtype=np.int32)
    dirs = np.zeros((n, SEQ), dtype=np.int8)
    size = np.zeros((n, SEQ), dtype=np.int16)
    push = np.zeros((n, SEQ), dtype=np.int8)
    for i, s in enumerate(series):
        a = json.loads(s)  # [tiempos entre paquetes (ms), direcciones, tamaños, flags PUSH]
        k = min(len(a[0]), SEQ)
        ipt[i, :k] = a[0][:k]
        dirs[i, :k] = a[1][:k]
        size[i, :k] = a[2][:k]
        push[i, :k] = a[3][:k]
    return ipt, dirs, size, push


def parse_hist(series):
    return np.array([json.loads(s) for s in series], dtype=np.int32).reshape(len(series), -1)


def process_day(zip_path, member, day, frac, seed, out_dir):
    out = Path(out_dir) / f"{day}.parquet"
    empty_mark = Path(out_dir) / f"{day}.vacio"
    if out.exists() or empty_mark.exists():
        return {"day": day, "skipped": True}
    t0 = time.time()
    rng = np.random.default_rng(seed + int(day))
    kept, total = [], 0
    with zipfile.ZipFile(zip_path) as z, z.open(member) as raw, lzma.open(raw) as f:
        header = f.readline()
        if not header.strip():
            # día sin datos en el dataset (captura vacía): se registra y se continúa
            empty_mark.write_text("sin flujos en el archivo original\n")
            return {"day": day, "total_flows": 0, "sampled": 0, "empty": True, "seconds": round(time.time() - t0, 1)}
        while True:
            block = f.readlines(1 << 24)
            if not block:
                break
            total += len(block)
            idx = np.flatnonzero(rng.random(len(block)) < frac)
            kept.extend(block[i] for i in idx)
    if total == 0:
        empty_mark.write_text("solo encabezado, sin flujos\n")
        return {"day": day, "total_flows": 0, "sampled": 0, "empty": True, "seconds": round(time.time() - t0, 1)}
    df = pd.read_csv(io.BytesIO(header + b"".join(kept)), usecols=USECOLS)
    ipt, dirs, size, push = parse_ppi(df["PPI"])
    parts = [df[SCALARS].reset_index(drop=True)]
    for name, arr in [("IPT", ipt), ("DIR", dirs), ("SIZE", size), ("PUSH", push)]:
        parts.append(pd.DataFrame(arr, columns=[f"{name}_{i}" for i in range(SEQ)]))
    for col in PHIST:
        h = parse_hist(df[col])
        parts.append(pd.DataFrame(h, columns=[f"{col}_{i}" for i in range(h.shape[1])]))
    res = pd.concat(parts, axis=1)
    res.insert(0, "DATE", pd.to_datetime(day, format="%Y%m%d"))
    # huella de la secuencia (dirección × tamaño) para medir flujos repetidos entre particiones
    res["SEQ_HASH"] = pd.util.hash_pandas_object(pd.DataFrame(size.astype(np.int32) * dirs), index=False).values
    tmp = out.with_suffix(".tmp")
    res.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return {"day": day, "total_flows": total, "sampled": len(res), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/processed/muestra_2pct")
    ap.add_argument("--months", nargs="+", default=["2022-09", "2022-10", "2022-11", "2022-12"])
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--limit-days", type=int, default=0, help="procesar solo N días (prueba)")
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    days = list_days(a.zip, set(a.months))
    if a.limit_days:
        days = days[:a.limit_days]
    print(f"Días a procesar: {len(days)} · fracción {a.frac:.1%} · procesos {a.workers}", flush=True)

    manifest_path = Path(a.out) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    t0 = time.time()
    fallidos = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_day, a.zip, m, d, a.frac, a.seed, a.out): d for d, m in days}
        for n, fu in enumerate(as_completed(futs), 1):
            d = futs[fu]
            try:
                r = fu.result()
            except Exception as e:  # un día con problemas no detiene a los demás
                fallidos[d] = f"{type(e).__name__}: {e}"
                print(f"[{n}/{len(days)}] {d}: ERROR {fallidos[d]}", flush=True)
                continue
            if not r.get("skipped"):
                manifest[r["day"]] = r
                manifest_path.write_text(json.dumps(manifest, indent=1))
            if r.get("skipped"):
                estado = "ya existía"
            elif r.get("empty"):
                estado = "día sin flujos en el dataset (registrado como vacío)"
            else:
                estado = f"{r['sampled']:,} de {r['total_flows']:,} flujos en {r['seconds']} s"
            print(f"[{n}/{len(days)}] {r['day']}: {estado}", flush=True)
    meta = {"frac": a.frac, "seed": a.seed, "months": a.months, "zip": os.path.basename(a.zip)}
    (Path(a.out) / "config.json").write_text(json.dumps(meta, indent=1))
    vacios = sorted(k for k, v in manifest.items() if v.get("empty"))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min. Muestra en {a.out}")
    print(f"Días con datos: {len(manifest) - len(vacios)} · días vacíos: {len(vacios)} {vacios if vacios else ''}")
    if fallidos:
        (Path(a.out) / "errores.json").write_text(json.dumps(fallidos, indent=1))
        print(f"ATENCIÓN: {len(fallidos)} días con error (detalle en errores.json): {sorted(fallidos)}")


if __name__ == "__main__":
    main()
