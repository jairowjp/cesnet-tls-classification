# Clasificación de tráfico de red cifrado con aprendizaje automático y Transformers

**Análisis comparativo de modelos de aprendizaje automático y arquitecturas Transformer para la clasificación de tráfico de red cifrado sobre el dataset CESNET-TLS-Year22**

Proyecto Integrador en Inteligencia Artificial · Maestría en Inteligencia Artificial · Universidad de Especialidades Espíritu Santo (UEES)
Autor: **Jairo Wladimir Jhayya Perlaza** · Docente: Gladys María Villegas Rugel · 2026

| Enlace | Estado |
|---|---|
| Repositorio: https://github.com/jairowjp/cesnet-tls-classification | Activo |
| Aplicación web: `https://huggingface.co/spaces/jairowjp/cesnet-tls-classification` | Se publica en el Sprint 3 (hito H6, 01/11/2026) |
| Tablero Scrum: pestaña *Projects* de este repositorio | Se habilita en el Sprint 1 |

---

## El problema

Casi todo el tráfico de Internet viaja cifrado con TLS 1.3 y QUIC. Eso protege la privacidad, pero deja sin efecto a las
herramientas que inspeccionaban el contenido de los paquetes para gestionar y proteger las redes. La alternativa es
clasificar cada conexión por su comportamiento observable (tamaños, direcciones y tiempos de sus paquetes) con
aprendizaje automático.

Este proyecto responde una pregunta concreta: **usando solo metadatos, ¿conviene un modelo clásico y barato o una red
profunda con atención?** Para responderla compara cuatro modelos con el mismo protocolo, sobre un año de tráfico real, y
mide tanto el desempeño como el costo de ejecutarlos.

| Rol | Modelos | Entrada |
|---|---|---|
| Línea base | Random Forest, XGBoost | 60 variables del flujo |
| Comparación | Red convolucional 1D, Transformer encoder ligero | Secuencia de 30 paquetes (30 × 3) |
| Opcional | LSTM | Secuencia de 30 paquetes |

La red profunda solo se recomendará si supera a la línea base con significancia estadística (prueba de McNemar, p < 0,05)
y con una latencia de hasta 10 ms por flujo.

## Estado del proyecto

| Sprint | Fechas | Contenido | Estado |
|---|---|---|---|
| 1 | 28/09 – 04/10/2026 | Datos, EDA y línea base Random Forest | EDA terminado; línea base en curso |
| 2 | 05/10 – 18/10/2026 | XGBoost, CNN 1D y Transformer | Pendiente |
| 3 | 19/10 – 01/11/2026 | Ajuste, McNemar, aplicación web, OWASP y prueba de usabilidad | Pendiente |
| 4 | 02/11 – 08/11/2026 | Pruebas integrales, informe y presentación final | Pendiente |

## Resultados del EDA

Sobre una muestra aleatoria del 2 % de septiembre a diciembre de 2022 (3 133 138 flujos). El detalle está en
`docs/semana2/03_eda_cesnet_tls_year22.pdf` y en el notebook `notebooks/01_eda.ipynb`.

| Hallazgo | Valor | Consecuencia |
|---|---|---|
| Categorías | 23 | Clasificación multiclase de 23 clases |
| Calidad | 100 % completo, sin valores imposibles | No hace falta imputar |
| Desbalance | 145 a 1 | F1 macro y pesos por clase |
| Variables asimétricas | 6 de 8 | Transformación log10(1 + x) |
| Relleno en la secuencia | 51,6 % de las posiciones | Máscara de relleno en el Transformer |
| Fuga de información | El ASN de destino filtra la categoría (NMI 0,430) | Se excluye |
| Secuencias repetidas | 30,3 % de la prueba repite una secuencia de entrenamiento | F1 también sin repetidos |
| Deriva | Jensen-Shannon de 0,05 a 0,06 frente a septiembre | Se mide en noviembre y diciembre |

## Estructura

```
├── configs/experiment.yaml     configuración única: meses, semilla, exclusiones, modelos y umbrales
├── scripts/                    pipeline numerado (01–03) y utilidades/; índice y origen en scripts/README.md
├── notebooks/01_eda.ipynb      EDA ejecutado, con todas sus salidas
├── src/                        código reutilizable: datos (src/data), modelos (src/models) y evaluación (src/evaluation)
├── tests/                      pruebas unitarias (pytest)
├── results/eda/                figuras, tablas y resumen del EDA
├── data/samples/               muestra pequeña publicada (CC BY 4.0)
├── app/                        aplicación web (Sprint 3)
├── docs/                       entregables del curso por semana (PDF)
├── .vscode/                    espacio de trabajo: intérprete, tareas, depuración y extensiones
├── SECURITY.md                 controles de seguridad y plan OWASP Top 10 / OWASP ZAP
├── CHANGELOG.md                cambios por entrega, con fechas reales
└── docs/bitacora.md            registro cronológico: qué se ejecutó, qué falló y cómo se resolvió
```

## Cómo reproducir el proyecto

Requisitos: Linux (probado en Kali), Python 3.13 y unos 40 GB libres para el dataset.

```bash
# 1. Clonar el repositorio
git clone https://github.com/jairowjp/cesnet-tls-classification.git
cd cesnet-tls-classification

# 2. Entorno con Python 3.13 (uv descarga esa versión sin tocar el Python del sistema)
uv python install 3.13
uv venv --python 3.13 ../.venv && source ../.venv/bin/activate
uv pip install -r requirements-lock.txt

# 3. Dataset oficial desde Zenodo (28,4 GB) y verificación de integridad
mkdir -p datasets && cd datasets
wget -c "https://zenodo.org/records/10608607/files/CESNET-TLS-Year22.zip?download=1" -O CESNET-TLS-Year22.zip
wget "https://zenodo.org/records/10608607/files/servicemap.csv?download=1" -O servicemap.csv
md5sum CESNET-TLS-Year22.zip        # debe dar d0dd7c84e2140bba362f6bd23de5cab7
cd ..

# 4. Muestra, EDA, pruebas y línea base
python scripts/01_extraer_muestra.py --zip datasets/CESNET-TLS-Year22.zip
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m pytest -q
python scripts/03_entrenar_clasicos.py --modelo random_forest
python scripts/03_entrenar_clasicos.py --modelo xgboost
```

En VS Code, los mismos pasos están disponibles como tareas: **Ctrl + Shift + P → Tasks: Run Task**.

## Datos, licencias y citas

- **Dataset:** CESNET-TLS-Year22 © CESNET, licencia **CC BY 4.0**.
  K. Hynek, J. Luxemburk, J. Pešek, T. Čejka y P. Šiška, "CESNET-TLS-Year22: A year-spanning TLS network traffic
  dataset from backbone lines," *Scientific Data*, vol. 11, 2024, doi: 10.1038/s41597-024-03927-4.
- **Código de este repositorio:** licencia MIT (`LICENSE`).
- **Privacidad:** los modelos y la muestra publicada no contienen direcciones IP, dominios (SNI), huellas JA3 ni ASN.

## Documentos del curso

Cada documento se puede ver en GitHub o descargar con el botón de descarga de su página.

| Semana | Documento | Enlace |
|---|---|---|
| 1 | Workshop de Metodología SMART | [01_workshop_smart.xlsx](docs/semana1/01_workshop_smart.xlsx) |
| 1 | Presentación del Proyecto | [02_presentacion_del_proyecto.pdf](docs/semana1/02_presentacion_del_proyecto.pdf) |
| 2 | Ficha de Decisión Técnica | [01_ficha_decision_tecnica.pdf](docs/semana2/01_ficha_decision_tecnica.pdf) |
| 2 | Análisis comparativo de algoritmos | [02_analisis_comparativo_algoritmos.pdf](docs/semana2/02_analisis_comparativo_algoritmos.pdf) |
| 2 | Análisis exploratorio de datos (EDA) | [03_eda_cesnet_tls_year22.pdf](docs/semana2/03_eda_cesnet_tls_year22.pdf) |
