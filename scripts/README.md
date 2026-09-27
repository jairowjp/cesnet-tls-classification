# Scripts del proyecto

Todos los scripts se ejecutan **desde la raíz del repositorio** con el entorno activado
(`source ../.venv/bin/activate`). Cada uno explica al inicio de su código qué hace, por qué
existe y cómo se usa. El orden de los números es el orden en que se ejecutan.

## Pipeline principal

| Script | Qué hace | Entrada | Salida | Se usó por primera vez |
|---|---|---|---|---|
| `01_extraer_muestra.py` | Muestra aleatoria y reproducible del 2 % de septiembre a diciembre de 2022, leída directo del zip | `datasets/CESNET-TLS-Year22.zip` | `data/processed/muestra_2pct/` (117 días en Parquet, 5 días vacíos, manifiesto) | 27/09/2026 |
| `02_crear_muestra_app.py` | Muestra pequeña del mes de prueba (hasta 300 flujos por categoría) y plantilla CSV para la aplicación | Muestra del 2 % | `data/samples/` | 27/09/2026 |
| `03_entrenar_clasicos.py` | Entrena y evalúa Random Forest o XGBoost con el protocolo del proyecto | Muestra del 2 % | `results/modelos/<modelo>/` y `models/<modelo>.joblib` | 27/09/2026 |
| `05_entrenar_profundos.py` | Entrena y evalúa la CNN 1D, el Transformer o la LSTM con el mismo protocolo y formato de resultados; latencia siempre medida en CPU | Muestra del 2 % | `results/modelos/<modelo>/` y `models/<modelo>.pt` | 27/09/2026 |
| `04_comparar_modelos.py` | Tabla comparativa, prueba de McNemar por pares con corrección de Holm, gráfica F1 frente a latencia y recomendación | `results/modelos/*/` | `results/comparativa/` | 27/09/2026 |

El EDA no es un script sino un notebook: `notebooks/01_eda.ipynb`. Las redes profundas también se pueden entrenar
con GPU gratuita en Google Colab con `notebooks/02_entrenar_profundos_colab.ipynb`, que usa el mismo script 05.

**Nota sobre la numeración:** `05` se ejecuta antes que `04`, porque la comparación necesita todos los modelos entrenados.
Se conservó el número original de `04` para no romper las referencias de los commits anteriores.

## Utilidades (`utilidades/`)

Comandos que se ejecutaron a mano durante el proyecto, reunidos en scripts documentados
para que cualquiera pueda repetir exactamente los mismos pasos.

| Script | Qué hace | Por qué existe |
|---|---|---|
| `00_configurar_entorno.sh` | Crea `../.venv` con Python 3.13 y las versiones exactas de `requirements-lock.txt` | Kali trae Python 3.14, incompatible con la versión de pydantic que exige cesnet-datazoo |
| `01_descargar_dataset.sh` | Descarga el dataset desde Zenodo y verifica su MD5 oficial | El servidor de DataZoo estaba caído (error 503); Zenodo es el registro oficial |
| `02_inspeccionar_dataset.sh` | Muestra la estructura del zip, las columnas y los días disponibles | Se usó para diseñar el EDA sobre el formato real, sin suposiciones |
| `03_limpiar_parquet_huerfanos.py` | Borra archivos de la muestra que no están en el manifiesto (simula por defecto) | Reparó los 18 días que quedaron sin registrar tras el primer fallo de la extracción |

## Orden para reproducir todo desde cero

```bash
bash scripts/utilidades/00_configurar_entorno.sh
bash scripts/utilidades/01_descargar_dataset.sh
python scripts/01_extraer_muestra.py --zip datasets/CESNET-TLS-Year22.zip
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python scripts/02_crear_muestra_app.py
python scripts/03_entrenar_clasicos.py --modelo random_forest
python scripts/03_entrenar_clasicos.py --modelo xgboost
python scripts/05_entrenar_profundos.py --modelo cnn1d
python scripts/05_entrenar_profundos.py --modelo transformer
python scripts/04_comparar_modelos.py
```

El registro de cuándo y por qué se ejecutó cada paso está en `docs/bitacora.md`.
