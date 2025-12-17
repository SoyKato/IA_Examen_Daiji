# IA_Examen_Daiji

**Objetivo del proyecto**

El objetivo fue crear un modelo para predecir las proximas condiciones del clima en Ensenada basandose en la temperatura maxima y minima.

**Modelo(s) utilizados**

Arbol de decision

**Descripción del dataset y preprocesamiento**

El dataset fue creado a partir de los datos recabados de The Weather Channel que van desde el 1ro de octubre hasta el 15 de diciembre del 2025.

La tecnica de preprocesamiento fue normalizacion y limpieza de valores nulos


**Instrucciones para ejecutar el proyecto**

1. Instalar Python 3.8 o superiro
2. Instalar numpy, pandas, scikit-learn matplotlib seaborn:
   ``pip install numpy pandas scikit-learn matplotlib seaborn``
3. Ejecutar el proyecto:
   ```
   cd C:..\ExamenIA
   python main.py

   ```

**Explicacion del programa**

Se predice codiciones del clima y se compara con los datos reales

Cuando la diagonal tiene > 80% de precision entonces es un modelo confiable por que sobre pasa la posibilidad del 50% de errar la prediccion

Menor a 60% es un modelo poco confiable por que considerando que usamos un programa que realiza sus propias predicciones a base de estadisticas, es casi lo mismo que adivinar en un 50/50

Si usaramos la temperatura maxima y minima del dia actual podriamos ponerlas en el modelo y asi que nos genere una prediccion de condicion de clima.
**Total de datos:** *77*
**Distribucion de datos de entrenamiento:** *70 Muestra / 30 Testeo*
**Arboles usados:** *100*
**Niveles de profundidad:** *10*
**Reproducibilidad:** *42*
**Clasificador de tipo Random Forest**
**Precision aproximada:** *72.55%*
**DATASET TOMADO DESDE:** ```https://weather.com/es-US/tiempo/mensual/l/b3419305f250d733f1fe0a95f542d09924efef133a416afdc603e12ece8c26fa```

