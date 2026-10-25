# Modelos entrenados

| Archivo | Modelo | F1 macro (validación) | F1 macro (octubre) | Tamaño | SHA-256 |
|---|---|---|---|---|---|
| `best_model.pkl` | XGBoost versión 2 | 0.916 | 0.903 | 58.9 MB | `ed3fcaab0d9765ae…` |
| `model_v1.pkl` | XGBoost versión 1 | 0.903 | 0.886 | 36.0 MB | `4a7662b492a083ec…` |

- **Validación:** el 20 % de septiembre de 2022. **Octubre:** 872 731 flujos que ningún modelo vio al entrenar.
- Cada archivo contiene un diccionario `{"modelo", "clases", "variables"}`: el clasificador de XGBoost, las 23
  categorías en orden y las 60 variables de entrada (ver `src/data_processing.py`).
- Para cargarlo con verificación de integridad:
  ```python
  from src.model import cargar_mejor_modelo
  paquete = cargar_mejor_modelo(sha256="<huella completa de la tabla>")
  ```
  Un archivo de este tipo puede ejecutar código al cargarse: no cargues nunca uno de origen desconocido.
- Los modelos de Random Forest, CNN 1D y Transformer no se publican aquí por su tamaño o porque no se recomiendan;
  sus métricas están en `results/modelos/` y se regeneran con los scripts 03 y 05.
- Generado con `python scripts/14_exportar_modelos.py --version 2`.
