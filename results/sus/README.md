# Encuesta de usabilidad (SUS)

1. Cada evaluador responde en la página **Tu opinión** de la aplicación y recibe un código corto,
   por ejemplo `SUS-10F3-HA1P-ME7Y3`, que te envía por mensaje o correo.
2. Pega los códigos recibidos en `results/sus/codigos.txt`, uno por línea. Las líneas que empiezan con `#` se ignoran.
3. Ejecuta `python scripts/07_consolidar_sus.py`.

El script verifica cada código con sus caracteres de control (descarta los mal copiados), recalcula cada
puntaje y genera `resumen_sus.json` con el promedio SUS, el éxito de las tareas y el cumplimiento de la meta
del hito H6: al menos 5 evaluadores, SUS de 68 o más y 80 % de tareas sin ayuda. Los códigos individuales no
se suben a GitHub (`.gitignore`); solo se publica el resumen.
