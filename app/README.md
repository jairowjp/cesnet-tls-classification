# Aplicación web

Aplicación Streamlit de ocho páginas que demuestra el proyecto de forma interactiva.

| Página | Qué permite hacer |
|---|---|
| Inicio | Ver la firma de un flujo real de octubre reproducida paquete a paquete y clasificada en vivo por XGBoost |
| Datos | Comparar la forma típica de dos categorías, el desbalance y la deriva durante el año |
| Modelos | Explorar acierto frente a costo, la prueba de McNemar por pares, el F1 por categoría y la deriva |
| Reto | Adivinar la categoría de flujos reales y competir contra los modelos en cinco rondas |
| Clasificar | Clasificar flujos reales de octubre o un CSV propio con el formato de la plantilla |
| Conclusiones | Recomendación, matices, limitaciones y créditos |
| Tu opinión | Encuesta SUS con puntaje inmediato y un código de respuesta para enviar al autor |
| Seguridad | Qué datos usa el sitio, qué hace con lo que sube el usuario y cómo verificar cada control |

## Estructura

```
app/
├── principal.py      punto de entrada y navegación superior
├── paginas/          una página por archivo
├── core/
│   ├── seguridad.py  integridad del modelo (SHA-256) y validación de CSV
│   ├── datos.py      carga en caché de datos y del modelo verificado
│   ├── firma.py      componente de la firma del flujo (seguro frente a XSS)
│   ├── estilo.py     paleta, tipografía y plantilla de gráficos
│   ├── guia.py       recorrido guiado y sugerencias en contexto
│   ├── sus.py        cálculo de la escala de usabilidad SUS
│   └── pie.py        atribución del dataset (CC BY 4.0)
├── static/fonts/     tipografía Archivo (SIL OFL), servida desde la propia app
├── models/           modelo XGBoost y su manifiesto de huellas (generados por scripts/06_preparar_app.py)
└── assets/           predicciones precalculadas de los 4 modelos (generadas por scripts/06_preparar_app.py)
```

## Ejecutarla

Desde la raíz del repositorio:

```bash
python scripts/06_preparar_app.py
streamlit run app/principal.py
```

La configuración de seguridad y del tema está en `.streamlit/config.toml`; los controles, en `SECURITY.md`.
