"""
Consolida los códigos de respuesta de la encuesta SUS.

Cada evaluador responde la encuesta en la página "Tu opinión" de la aplicación y recibe un código
corto (por ejemplo, SUS-10F3-HA1P-ME7Y3) que envía al autor por mensaje o correo. Pega los códigos
recibidos en results/sus/codigos.txt, uno por línea, y ejecuta este script.

Controles:
  * Cada código se verifica con sus caracteres de control: los mal copiados se informan y se descartan.
  * El puntaje se RECALCULA a partir de las 10 respuestas del código.
  * Los códigos repetidos se informan (pueden ser dos personas con respuestas idénticas o un envío duplicado).

Meta del proyecto (hito H6): al menos 5 evaluadores, SUS promedio de 68 o más y al menos 80 % de las
tareas logradas sin ayuda.

Uso:  python scripts/07_consolidar_sus.py
Salida: results/sus/resumen_sus.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))  # core/sus.py

import pandas as pd  # noqa: E402
from core.sus import TAREAS, CodigoInvalido, decodificar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARPETA = ROOT / "results/sus"
ARCHIVO = CARPETA / "codigos.txt"


def main():
    if not ARCHIVO.exists():
        sys.exit(f"Crea {ARCHIVO.relative_to(ROOT)} y pega ahí los códigos recibidos, uno por línea.")
    lineas = [ln.strip() for ln in ARCHIVO.read_text(encoding="utf-8").splitlines()]
    codigos = [ln for ln in lineas if ln and not ln.startswith("#")]

    validas, invalidos = [], []
    for n, codigo in enumerate(codigos, start=1):
        try:
            validas.append({"codigo": codigo.upper(), **decodificar(codigo)})
        except (CodigoInvalido, ValueError) as e:
            invalidos.append({"linea": n, "motivo": str(e)})
    for inv in invalidos:
        print(f"Descartado (línea {inv['linea']}): {inv['motivo']}")
    if not validas:
        sys.exit("Ningún código es válido.")
    repetidos = [c for c, k in Counter(v["codigo"] for v in validas).items() if k > 1]
    if repetidos:
        print(f"Aviso: {len(repetidos)} códigos aparecen más de una vez. Revisa si son envíos duplicados.")

    ok = pd.DataFrame(validas)
    tareas = {
        t: round(100 * float(ok["tareas"].map(lambda d, t=t: d.get(t) == "Sí, sin ayuda").mean()), 1) for t in TAREAS
    }
    exito = round(sum(tareas.values()) / len(tareas), 1)
    resumen = {
        "evaluadores": int(len(ok)),
        "sus_promedio": round(float(ok["puntaje_sus"].mean()), 1),
        "sus_desviacion": round(float(ok["puntaje_sus"].std(ddof=1)), 1) if len(ok) > 1 else None,
        "sus_minimo": float(ok["puntaje_sus"].min()),
        "sus_maximo": float(ok["puntaje_sus"].max()),
        "evaluadores_con_68_o_mas_%": round(100 * float((ok["puntaje_sus"] >= 68).mean()), 1),
        "exito_sin_ayuda_por_tarea_%": tareas,
        "exito_sin_ayuda_promedio_%": exito,
        "perfiles": ok["perfil"].value_counts().to_dict(),
        "codigos_descartados": len(invalidos),
        "codigos_repetidos": len(repetidos),
        "meta_h6": {
            "al_menos_5_evaluadores": bool(len(ok) >= 5),
            "sus_promedio_68_o_mas": bool(ok["puntaje_sus"].mean() >= 68),
            "exito_sin_ayuda_80_o_mas": bool(exito >= 80),
        },
    }
    (CARPETA / "resumen_sus.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluadores válidos: {resumen['evaluadores']}  (descartados: {len(invalidos)})")
    print(f"SUS promedio: {resumen['sus_promedio']}  (mín. {resumen['sus_minimo']}, máx. {resumen['sus_maximo']})")
    print(f"Tareas logradas sin ayuda: {exito} % en promedio")
    for t, v in tareas.items():
        print(f"   {t}: {v} %")
    cumple = all(resumen["meta_h6"].values())
    print(f"Meta del hito H6: {'CUMPLIDA' if cumple else 'NO cumplida todavía'} {resumen['meta_h6']}")
    print(f"Resumen en {(CARPETA / 'resumen_sus.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
