"""
Lanzador de compatibilidad con Streamlit Community Cloud.

La aplicación vive en app/app.py, el nombre que exige la estructura del proyecto final. La app pública se
publicó con este archivo como punto de entrada, y Community Cloud no permite cambiarlo sin borrar y volver
a publicar la app (con riesgo de perder su dirección). Por eso este archivo solo ejecuta app/app.py.

En el equipo local se usa directamente:  streamlit run app/app.py
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
