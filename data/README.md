# Datos

El proyecto usa **CESNET-TLS-Year22** (Hynek et al., 2024), un dataset público de flujos TLS capturados durante
todo 2022 en la red académica nacional de la República Checa. Licencia **CC BY 4.0**, © CESNET.
DOI del dataset: [10.5281/zenodo.10608607](https://doi.org/10.5281/zenodo.10608607).

| Carpeta | Contenido | ¿Está en GitHub? |
|---|---|---|
| `raw/` | El archivo original del dataset (`CESNET-TLS-Year22.zip`) | No: pesa 28,4 GB, demasiado para GitHub |
| `processed/` | La muestra del 2 % de septiembre a diciembre (3 133 138 flujos, un parquet por día) | No: se regenera con un script |
| `samples/` | La muestra pequeña que usa la aplicación web y la plantilla del clasificador | Sí |

## Cómo obtener los datos

1. Descargar el dataset desde Zenodo en la carpeta `datasets/` y verificar su suma MD5, que debe ser
   `d0dd7c84e2140bba362f6bd23de5cab7` (los comandos exactos están en el README principal).
2. Extraer la muestra del 2 % (reproducible: semilla fija en `configs/experiment.yaml`):
   ```bash
   python scripts/01_extraer_muestra.py --zip datasets/CESNET-TLS-Year22.zip
   ```
   El resultado queda en `data/processed/muestra_2pct/`. La carpeta `raw/` queda disponible si se prefiere guardar
   ahí el archivo; en ese caso, indica su ruta en `--zip`.

## Partición

| Uso | Mes de 2022 | Flujos |
|---|---|---|
| Entrenamiento (80 %) y validación (20 %) | Septiembre | 640 102 y 160 026 |
| Prueba | Octubre | 872 731 |
| Deriva | Noviembre y diciembre | Ver `results/eda/` |

## Variables excluidas

Las direcciones IP, el nombre de servidor (SNI) y la huella JA3 no se usan para proteger la privacidad. El ASN
de destino se excluyó porque filtraba la respuesta (información mutua normalizada de 0,430 con la categoría).

Hynek, K., Luxemburk, J., Pešek, J., Čejka, T., y Šiška, P. (2024). CESNET-TLS-Year22: A year-spanning TLS network
traffic dataset from backbone lines. *Scientific Data, 11*, 1156. https://doi.org/10.1038/s41597-024-03927-4
