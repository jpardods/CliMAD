# CliMAD 
Aplicación para la predicción y visualización del clima en Madrid por estación de control  
App online: https://climad-o3b0.onrender.com/

## Descripción del proyecto  
CliMAD es una aplicación web orientada a la predicción y visualización detallada de variables meteorológicas en el municipio de Madrid, por día y por estación de control (asociadas a los distintos distritos).  

La aplicación se entrenará con datos históricos de 2023 y 2024, validará el modelo con datos de 2025 y ofrecerá predicciones para 2026 en adelante. El producto final será una página web interactiva con un mapa de estaciones, un selector de fecha y un panel de visualización dinámico.  

Se mostrarán predicciones de las siguientes variables:  
- Radiación ultravioleta  
- Velocidad y dirección del viento  
- Temperatura  
- Humedad relativa  
- Presión barométrica  
- Radiación solar  
- Precipitación  

Además, la aplicación permitirá comparar las predicciones con series históricas, visualizar métricas de calidad del modelo y descargar datos en formato CSV para su análisis externo. El objetivo principal es facilitar la toma de decisiones y el análisis climático a usuarios académicos, técnicos municipales y ciudadanos interesados en información local del clima.

## Objetivos principales  
- **Modelo predictivo local**: Entrenar y validar un modelo con datos reales de estaciones meteorológicas de Madrid para obtener estimaciones fiables y ajustadas a microclimas urbanos.  
- **Visualización interactiva**: Desarrollar un cuadro de mando claro y accesible que muestre predicciones y comparaciones históricas mediante gráficos dinámicos.  
- **Medidas de incertidumbre**: Incorporar rangos de confianza y métricas de validación para que las decisiones puedan considerar distintos escenarios y niveles de riesgo.  
- **Accesibilidad y usabilidad**: Ofrecer una interfaz web intuitiva que pueda ser utilizada tanto por perfiles técnicos como por usuarios no especializados.  
- **Reproducibilidad y escalabilidad**: Diseñar una aplicación con repositorio abierto y código estructurado que permita replicar el proyecto en otros municipios o ampliarlo con nuevas funcionalidades.   

## Plan inicial de trabajo  
1. **Fase 1 – Preparación de datos (octubre 2025)**  
   - Recopilación de datos históricos de 2023 y 2024.  
   - Limpieza, estandarización de formatos y creación de la base de datos unificada.  
   - Identificación de las variables objetivo a predecir y preparación de series temporales por estación de control.  

2. **Fase 2 – Desarrollo del modelo (octubre 2025)**  
   - Entrenamiento del modelo con los datos históricos.  
   - Validación con los registros de 2025 para ajustar la precisión.  
   - Implementación de métricas de calidad y estimación de incertidumbre en las predicciones.  

3. **Fase 3 – Desarrollo de la aplicación (octubre-noviembre 2025)**  
   - Construcción de la interfaz web interactiva con mapa dinámico de estaciones.  
   - Integración de selector de fecha y panel de resultados con gráficos y tablas.  
   - Implementación de funcionalidades de descarga de datos en CSV.  
   - Diseño del cuadro de mando con énfasis en claridad y accesibilidad.  

4. **Fase 4 – Despliegue y presentación (noviembre-diciembre 2025)**  
   - Despliegue en un servicio en la nube accesible mediante URL pública.  
   - Pruebas de rendimiento y accesibilidad para asegurar el correcto funcionamiento.  
   - Redacción de la documentación del proyecto en el repositorio.  
   - Preparación de la presentación final y demostración de la aplicación. 

Autor: Juan Pardo de Santayana Navarro  
Curso 2025, 5ºGITT-BA – Asignatura: Desarrollo de Aplicaciones para la Visualización de Datos  