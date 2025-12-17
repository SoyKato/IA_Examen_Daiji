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

Cuando la diagonal tiene > 80% de precision entonces es un modelo confiable

Menor a 60% es un modelo poco confiable

Si usaramos la temperatura maxima y minima del dia actual podriamos ponerlas en el modelo y asi que nos genere una prediccion de condicion de clima.


DATASET TOMADO DESDE: ```https://weather.com/es-US/tiempo/mensual/l/b3419305f250d733f1fe0a95f542d09924efef133a416afdc603e12ece8c26fa```
