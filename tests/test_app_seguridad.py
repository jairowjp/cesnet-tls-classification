"""Pruebas de los controles de seguridad de la aplicación web (core/seguridad.py).

Cada prueba simula un intento concreto de abuso y comprueba que se rechaza con un
mensaje que no repite el contenido del archivo.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from core.seguridad import (  # noqa: E402
    MAX_BYTES,
    MAX_FILAS,
    REQUERIDAS,
    ModeloAlterado,
    neutralizar_formulas,
    sha256_de,
    validar_csv,
    verificar_integridad,
)


def _csv_valido(filas: int = 3) -> pd.DataFrame:
    df = pd.DataFrame(np.ones((filas, len(REQUERIDAS))), columns=REQUERIDAS)
    for i in range(30):
        df[f"DIR_{i}"] = 1 if i < 5 else 0
        df[f"SIZE_{i}"] = 500 if i < 5 else 0
    df["PPI_LEN"] = 5
    return df


def _bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def test_csv_valido_se_acepta():
    r = validar_csv(_bytes(_csv_valido()))
    assert r.valido and r.datos.shape == (3, len(REQUERIDAS))


def test_archivo_vacio_se_rechaza():
    assert not validar_csv(b"").valido


def test_archivo_demasiado_grande_se_rechaza():
    r = validar_csv(b"x" * (MAX_BYTES + 1))
    assert not r.valido and "MB" in r.errores[0]


def test_demasiadas_filas_se_rechaza():
    assert not validar_csv(_bytes(_csv_valido(MAX_FILAS + 1))).valido


def test_columnas_faltantes_se_rechaza_sin_listar_nombres():
    df = _csv_valido().drop(columns=["BYTES"])
    r = validar_csv(_bytes(df))
    assert not r.valido and "BYTES" not in " ".join(r.errores)


def test_texto_en_columna_numerica_se_rechaza():
    df = _csv_valido().astype(object)
    df.loc[0, "BYTES"] = '=HYPERLINK("http://malicioso")'  # intento de inyección
    r = validar_csv(_bytes(df))
    assert not r.valido and "malicioso" not in " ".join(r.errores)


def test_valores_fuera_de_rango_se_rechazan():
    df = _csv_valido()
    df["DIR_0"] = 7  # dirección imposible
    assert not validar_csv(_bytes(df)).valido


def test_valores_negativos_se_rechazan():
    df = _csv_valido()
    df["BYTES"] = -10
    assert not validar_csv(_bytes(df)).valido


def test_binario_no_utf8_se_rechaza():
    assert not validar_csv(b"\xff\xfe\x00\x01" * 50).valido


def test_modelo_alterado_no_se_carga(tmp_path):
    modelo = tmp_path / "xgboost.joblib"
    modelo.write_bytes(b"modelo original")
    (tmp_path / "manifest.json").write_text(json.dumps({"xgboost.joblib": {"sha256": sha256_de(modelo)}}))
    assert verificar_integridad(modelo, tmp_path / "manifest.json")
    modelo.write_bytes(b"modelo reemplazado por un atacante")
    with pytest.raises(ModeloAlterado):
        verificar_integridad(modelo, tmp_path / "manifest.json")


def test_modelo_no_registrado_no_se_carga(tmp_path):
    otro = tmp_path / "otro.joblib"
    otro.write_bytes(b"x")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ModeloAlterado):
        verificar_integridad(otro, tmp_path / "manifest.json")


def test_formulas_neutralizadas_en_descargas():
    df = pd.DataFrame({"Categoría": ["=1+1", "Media", "@SUMA(A1)", "-2"]})
    out = neutralizar_formulas(df)["Categoría"].tolist()
    assert out == ["'=1+1", "Media", "'@SUMA(A1)", "'-2"]


def test_firma_no_permite_inyectar_codigo_en_textos():
    """Un nombre de categoría malicioso no debe poder cerrar el <script> ni inyectar etiquetas."""
    from core.firma import construir_html

    ataque = "</script><script>alert(document.cookie)</script><img src=x onerror=alert(1)>"
    html = construir_html(
        np.full(30, 500.0),
        np.r_[np.ones(5), np.zeros(25)],
        np.full(30, 10.0),
        veredicto={"titulo": ataque, "detalle": ataque},
    )
    assert html.count("<script>") == 1 and html.count("</script>") == 1  # solo el script propio
    assert "<img" not in html and "alert(document.cookie)</script>" not in html
    assert "textContent" in html and "innerHTML" not in html


def test_firma_solo_incluye_paquetes_reales():
    from core.firma import construir_html

    html = construir_html(np.r_[np.full(3, 300.0), np.zeros(27)], np.r_[np.ones(3), np.zeros(27)], np.zeros(30))
    assert '"tam": [300.0, 300.0, 300.0]' in html  # el relleno no se envía al navegador
