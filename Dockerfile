# =============================================================================
# Imagen de la aplicación web para Hugging Face Spaces (y para ejecutarla en local con Docker)
# -----------------------------------------------------------------------------
# Construir:  docker build -t cesnet-app .
# Ejecutar:   docker run --rm -p 7860:7860 cesnet-app      →  http://localhost:7860
# =============================================================================
FROM python:3.13-slim

# libgomp1: biblioteca de paralelismo (OpenMP) que usa XGBoost al predecir
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Seguridad: la app corre con un usuario sin privilegios, nunca como root.
# Hugging Face Spaces exige además que ese usuario tenga el id 1000.
RUN useradd --create-home --uid 1000 usuario
USER usuario
ENV HOME=/home/usuario PATH=/home/usuario/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /home/usuario/proyecto

# Primero solo las dependencias: Docker reutiliza esta capa si el código cambia pero ellas no
COPY --chown=usuario requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

# Luego el código y los datos que usa la app (lo demás lo excluye .dockerignore)
COPY --chown=usuario . .

EXPOSE 7860
# Hugging Face y Docker comprueban que la app responde en el endpoint de salud de Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app/principal.py", "--server.port=7860", "--server.address=0.0.0.0"]
