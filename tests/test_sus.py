"""Pruebas del cálculo SUS, del código de respuesta y de la consolidación."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from core.sus import (  # noqa: E402
    ALFABETO,
    RESULTADOS_TAREA,
    TAREAS,
    CodigoInvalido,
    codificar,
    decodificar,
    interpretar,
    puntaje_sus,
)

ROOT = Path(__file__).resolve().parents[1]


def test_puntajes_de_referencia():
    assert puntaje_sus([3] * 10) == 50.0  # respuestas neutras
    assert puntaje_sus([5, 1] * 5) == 100.0  # la mejor evaluación posible
    assert puntaje_sus([1, 5] * 5) == 0.0  # la peor evaluación posible


def test_respuestas_invalidas_se_rechazan():
    with pytest.raises(ValueError):
        puntaje_sus([3] * 9)
    with pytest.raises(ValueError):
        puntaje_sus([3] * 9 + [7])


def test_interpretacion_segun_bangor():
    assert interpretar(90) == "excelente"
    assert interpretar(68).startswith("aceptable")
    assert interpretar(40) == "deficiente"


def test_codigo_ida_y_vuelta():
    tareas = {TAREAS[0]: RESULTADOS_TAREA[0], TAREAS[2]: RESULTADOS_TAREA[2]}
    codigo = codificar("Docente", [4, 2, 5, 1, 4, 2, 5, 1, 4, 2], tareas)
    d = decodificar(codigo)
    assert codigo.startswith("SUS-") and d["perfil"] == "Docente" and d["puntaje_sus"] == 85.0
    assert d["tareas"][TAREAS[0]] == RESULTADOS_TAREA[0] and d["tareas"][TAREAS[1]] == ""


def test_codigo_tolera_minusculas_espacios_y_letras_confundibles():
    codigo = codificar("Estudiante", [3] * 10, {})
    variante = codigo.lower().replace("-", " ").replace("0", "o").replace("1", "l")
    assert decodificar(variante)["puntaje_sus"] == 50.0


def test_codigo_detecta_todo_error_de_un_caracter():
    codigo = codificar("Otro", [5, 1] * 5, {}).replace("SUS-", "").replace("-", "")
    for pos in range(len(codigo)):
        for ch in ALFABETO:
            if ch != codigo[pos]:
                with pytest.raises(CodigoInvalido):
                    decodificar(codigo[:pos] + ch + codigo[pos + 1 :])


def test_consolidacion_con_codigos():
    carpeta = ROOT / "results/sus"
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / "codigos.txt"
    respaldo = archivo.read_text(encoding="utf-8") if archivo.exists() else None
    tareas_ok = {t: RESULTADOS_TAREA[0] for t in TAREAS}
    codigos = [codificar("Estudiante", [4, 2] * 5, tareas_ok) for _ in range(5)]  # 5 evaluadores, SUS 75
    try:
        archivo.write_text("# códigos de prueba\n" + "\n".join(codigos) + "\nSUS-MAL-COPIADO\n", encoding="utf-8")
        # Seguro: se ejecuta el propio intérprete con un script del repositorio, sin datos externos
        salida = subprocess.run(  # noqa: S603
            [sys.executable, str(ROOT / "scripts/07_consolidar_sus.py")], capture_output=True, text=True, check=True
        ).stdout
        assert "Evaluadores válidos: 5" in salida and "SUS promedio: 75.0" in salida
        assert "descartados: 1" in salida and "CUMPLIDA" in salida
    finally:
        if respaldo is None:
            archivo.unlink(missing_ok=True)
        else:
            archivo.write_text(respaldo, encoding="utf-8")
        (carpeta / "resumen_sus.json").unlink(missing_ok=True)
