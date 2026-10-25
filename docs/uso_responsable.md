# Declaración de uso responsable

Este proyecto publica en abierto su código, sus resultados y un modelo capaz de identificar qué tipo de servicio hay
detrás de una conexión cifrada. Esa apertura permite verificar y reutilizar el trabajo, pero también pone una
herramienta al alcance de quien quiera usarla mal. Esta declaración fija los límites de su uso.

## Usos permitidos

1. Investigación y docencia sobre clasificación de tráfico cifrado.
2. Gestión agregada de una red (planificar su capacidad, detectar anomalías, diagnosticar problemas de calidad),
   siempre con supervisión humana.
3. Reproducción y verificación de los resultados del proyecto.

## Usos prohibidos

1. **Vigilar o perfilar a personas**, es decir, usar el modelo para saber qué servicios usa alguien y cuándo.
2. **Priorizar, degradar o bloquear servicios con fines comerciales**, lo que además contraviene la neutralidad de red
   que establece la Ley Orgánica de Telecomunicaciones del Ecuador.
3. **Tomar decisiones automáticas sobre personas** a partir de sus clasificaciones, sin revisión humana.
4. Presentar sus clasificaciones como certezas: el modelo se equivoca y su confianza debe considerarse.

## Condiciones para un despliegue real

Antes de usar el modelo fuera del ámbito académico debe hacerse una evaluación de impacto en protección de datos,
informar a las personas usuarias de la red, conservar solo resultados agregados (nunca la asociación entre un flujo y
su origen), reentrenarlo con datos de la red donde se usará y vigilar su desempeño cada mes.

## Reportar un mal uso o un problema

Si detectas un uso indebido o un error del modelo, abre un *issue* en el repositorio del proyecto:
https://github.com/jairowjp/cesnet-tls-classification/issues

Jairo Wladimir Jhayya Perlaza · Octubre de 2026
