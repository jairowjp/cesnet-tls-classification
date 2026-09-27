"""
Escala de Usabilidad del Sistema (SUS), de J. Brooke (1996).

Diez afirmaciones que se responden de 1 (totalmente en desacuerdo) a 5 (totalmente de acuerdo).
Las impares son positivas y las pares, negativas; por eso se puntúan distinto:
  * impares: respuesta − 1        * pares: 5 − respuesta
La suma (0 a 40) se multiplica por 2,5 para obtener un puntaje de 0 a 100.
Un puntaje de 68 es el promedio de referencia: el proyecto se comprometió a alcanzar al menos 68.
"""

import hashlib

PREGUNTAS = [
    "Creo que me gustaría usar este sitio con frecuencia.",
    "Encontré el sitio innecesariamente complejo.",
    "Pensé que el sitio era fácil de usar.",
    "Creo que necesitaría el apoyo de una persona técnica para poder usar este sitio.",
    "Encontré que las distintas funciones del sitio estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en el sitio.",
    "Imagino que la mayoría de las personas aprendería a usar este sitio muy rápidamente.",
    "Encontré el sitio muy incómodo de usar.",
    "Me sentí con confianza usando el sitio.",
    "Necesité aprender muchas cosas antes de poder usar el sitio.",
]

TAREAS = [
    "Reproducir un flujo y entender su firma",
    "Comparar dos modelos en la página Modelos",
    "Clasificar un flujo en la página Clasificar",
    "Completar el reto contra los modelos",
]
RESULTADOS_TAREA = ["Sí, sin ayuda", "Sí, con ayuda", "No lo logré"]


def puntaje_sus(respuestas: list[int]) -> float:
    """Puntaje SUS de 0 a 100 a partir de 10 respuestas entre 1 y 5."""
    if len(respuestas) != 10 or any(r not in (1, 2, 3, 4, 5) for r in respuestas):
        raise ValueError("Se necesitan 10 respuestas entre 1 y 5")
    suma = sum((r - 1) if i % 2 == 0 else (5 - r) for i, r in enumerate(respuestas))
    return suma * 2.5


def interpretar(puntaje: float) -> str:
    """Lectura del puntaje según la escala de adjetivos de Bangor, Kortum y Miller (2009)."""
    if puntaje >= 85:
        return "excelente"
    if puntaje >= 73:
        return "buena"
    if puntaje >= 68:
        return "aceptable, por encima del promedio de referencia"
    if puntaje >= 52:
        return "regular, por debajo del promedio de referencia"
    return "deficiente"


# --------------------------------------------------------------------------------------------------
# Código de respuesta: el evaluador copia un texto corto y lo envía por mensaje o correo
# --------------------------------------------------------------------------------------------------
# Contenido (16 dígitos): versión (1) + perfil (0–3) + 10 respuestas (1–5) + 4 tareas (0–2, 3 = sin responder).
# Se expresa en base 32 de Crockford (sin letras confundibles: I, L, O, U) y se agregan dos caracteres de
# control tomados de una huella SHA-256, que detectan errores de copia. No incluye nombre ni alias.

ALFABETO = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
PERFILES = ["Estudiante", "Docente", "Profesional de redes o seguridad", "Otro"]
VERSION = "1"


class CodigoInvalido(ValueError):
    """El código está mal copiado, incompleto o no corresponde a una respuesta válida."""


def _base32(n: int, largo: int) -> str:
    s = ""
    for _ in range(largo):
        n, r = divmod(n, 32)
        s = ALFABETO[r] + s
    return s


def _control(datos: str) -> str:
    return _base32(int(hashlib.sha256(datos.encode()).hexdigest(), 16) % (32 * 32), 2)


def codificar(perfil: str, respuestas: list[int], tareas: dict[str, str]) -> str:
    """Código de respuesta, por ejemplo SUS-2F9K-7QXM-4BN8T. Valida las respuestas antes de codificar."""
    puntaje_sus(respuestas)  # lanza ValueError si hay respuestas inválidas
    t = "".join(str(RESULTADOS_TAREA.index(tareas[x])) if tareas.get(x) in RESULTADOS_TAREA else "3" for x in TAREAS)
    digitos = VERSION + str(PERFILES.index(perfil)) + "".join(map(str, respuestas)) + t
    datos = _base32(int(digitos), 11)
    c = datos + _control(datos)
    return f"SUS-{c[:4]}-{c[4:8]}-{c[8:]}"


def decodificar(codigo: str) -> dict:
    """Recupera perfil, respuestas y tareas de un código. Tolera minúsculas, espacios y letras confundibles."""
    limpio = (codigo or "").upper().replace(" ", "").replace("-", "")
    limpio = limpio.removeprefix("SUS").translate(str.maketrans("OIL", "011"))
    if len(limpio) != 13 or any(ch not in ALFABETO for ch in limpio):
        raise CodigoInvalido("el código no tiene el formato esperado")
    datos, control = limpio[:11], limpio[11:]
    if _control(datos) != control:
        raise CodigoInvalido("los caracteres de control no coinciden: el código está mal copiado")
    n = 0
    for ch in datos:
        n = n * 32 + ALFABETO.index(ch)
    digitos = str(n)
    if len(digitos) != 16 or digitos[0] != VERSION:
        raise CodigoInvalido("versión de código desconocida")
    perfil, respuestas, tareas = int(digitos[1]), [int(d) for d in digitos[2:12]], digitos[12:]
    if perfil >= len(PERFILES) or any(t not in "0123" for t in tareas):
        raise CodigoInvalido("el código contiene valores fuera de rango")
    return {
        "perfil": PERFILES[perfil],
        "respuestas": respuestas,
        "puntaje_sus": puntaje_sus(respuestas),  # valida también que cada respuesta esté entre 1 y 5
        "tareas": {t: (RESULTADOS_TAREA[int(v)] if v != "3" else "") for t, v in zip(TAREAS, tareas, strict=True)},
    }
