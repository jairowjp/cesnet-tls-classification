"""Prueba de scripts/11_comparar_versiones.py: la decisión se toma SOLO con validación, y la v1 se restaura bien."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metricas(val, prueba, mb=10.0, modo_prueba=False):
    return {
        "modo_prueba": modo_prueba,
        "validacion": {"f1_macro": val},
        "prueba_octubre": {"f1_macro": prueba},
        "costo": {"tamano_modelo_MB": mb, "memoria_pico_GB": 1.0},
    }


def test_decide_por_validacion_y_restaura_v1(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/11_comparar_versiones.py", tmp_path / "scripts")
    casos = {  # modelo: (v1, v2)
        "random_forest": (_metricas(0.70, 0.70), _metricas(0.80, 0.78)),  # mejora en validación → v2
        "xgboost": (_metricas(0.88, 0.86), _metricas(0.87, 0.90)),  # empeora en validación → v1, aunque la prueba suba
    }
    for modelo, (v1, v2) in casos.items():
        for carpeta, datos in (("results/modelos_v1", v1), ("results/modelos", v2)):
            (tmp_path / carpeta / modelo).mkdir(parents=True)
            (tmp_path / carpeta / modelo / "metricas.json").write_text(json.dumps(datos))
    (tmp_path / "models/v1").mkdir(parents=True)
    (tmp_path / "models/v1/xgboost.joblib").write_text("modelo v1")
    (tmp_path / "models/xgboost.joblib").write_text("modelo v2")
    # Seguro: se ejecuta el propio intérprete con una copia del script del repositorio, sin datos externos
    subprocess.run(  # noqa: S603
        [sys.executable, str(tmp_path / "scripts/11_comparar_versiones.py"), "--restaurar"],
        check=True,
        capture_output=True,
        text=True,
    )
    decisiones = {
        f["modelo"]: f["decision"] for f in json.loads((tmp_path / "results/comparativa/versiones.json").read_text())
    }
    assert decisiones == {"random_forest": "v2", "xgboost": "v1"}
    restaurado = json.loads((tmp_path / "results/modelos/xgboost/metricas.json").read_text())
    assert restaurado["validacion"]["f1_macro"] == 0.88  # resultados de la v1 de vuelta
    assert (tmp_path / "models/xgboost.joblib").read_text() == "modelo v1"  # y también el modelo entrenado
