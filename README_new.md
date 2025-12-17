# Modelo de Clasificación - Predicción de Compra de Cliente

## Descripción del Proyecto
Este proyecto implementa un modelo de Machine Learning para predecir si un cliente realizará una compra basándose en características como edad, ingresos anuales, puntuación de crédito, género y estado civil.

## Dataset
- **Nombre**: dataset.csv
- **Dimensiones**: 50 muestras, 6 características
- **Variable objetivo**: compra (0 = No Compra, 1 = Compra)
- **Características**:
  - edad: edad del cliente
  - ingresos_anuales: ingresos anuales en dólares
  - puntuacion_credito: puntuación crediticia
  - genero: género del cliente (M/F)
  - estado_civil: estado civil del cliente (S/C)

## Preprocesamiento de Datos

### Técnicas Aplicadas:

1. **Limpieza de valores nulos**: Se eliminaron filas con valores faltantes (0 valores encontrados)

2. **Codificación de variables categóricas**: 
   - Género: M→1, F→0
   - Estado civil: C→0, S→1
   - Se utilizó LabelEncoder de scikit-learn

3. **Normalización/Estandarización**:
   - Se aplicó StandardScaler a las variables numéricas
   - Variables normalizadas: edad, ingresos_anuales, puntuacion_credito
   - Esto centra los datos en media 0 y desviación estándar 1

## Modelo Seleccionado: **Regresión Logística**

### Justificación
Se eligió **Regresión Logística** por las siguientes razones:
- **Simplicidad e interpretabilidad**: Es un modelo lineal fácil de comprender y explicar
- **Adecuado para clasificación binaria**: El problema es una clasificación de 2 clases (compra/no compra)
- **Probabilidades calibradas**: Proporciona probabilidades que son fáciles de interpretar
- **Eficiencia**: Entrena rápidamente incluso con pocos datos
- **Estabilidad**: Es robusto con datos pequeños
- **Sin requerimientos adicionales**: No necesita ajuste de hiperparámetros complejos

## Resultados y Métricas

### Rendimiento del Modelo:
- **Accuracy**: 0.8667 (86.67%) - Proporción de predicciones correctas
- **Precisión**: 0.8571 (85.71%) - De las predicciones positivas, cuántas fueron correctas
- **Recall**: 1.0000 (100%) - De los casos positivos reales, cuántos fueron detectados

### Matriz de Confusión:
```
                 Predicho No Compra  Predicho Compra
Real No Compra         8                1
Real Compra            0               6
```

Interpretación:
- Verdaderos Negativos (TN): 8 - Clientes que no compran y se predijeron correctamente
- Falsos Positivos (FP): 1 - Cliente que no compra pero se predijo que compraría
- Falsos Negativos (FN): 0 - Ningún cliente que compra fue predicho como no comprador
- Verdaderos Positivos (TP): 6 - Clientes que compran y se predijeron correctamente

## Problemas Encontrados Durante el Desarrollo

1. **Desbalance de datos inicial**: El dataset tenía más ejemplos de compra que no compra
   - **Solución**: Se mantuvieron los datos originales para simular un escenario realista

2. **Pequeño tamaño de dataset**: Solo 50 muestras
   - **Solución**: Se utilizó estratificación en train_test_split para mantener proporciones

3. **Variables categóricas**: Género y estado civil necesitaban conversión numérica
   - **Solución**: Se aplicó LabelEncoder de scikit-learn

4. **Escalas diferentes**: Las variables tenían diferentes rangos de valores
   - **Solución**: Se normalizaron con StandardScaler

## Mejoras Futuras

Si se contara con más información o tiempo:

1. **Ampliar el dataset**:
   - Recopilar más muestras de clientes (preferiblemente 1000+)
   - Agregar más características: historial de compras, recencia, frecuencia, etc.

2. **Ingeniería de características**:
   - Crear variables derivadas (ratio ingresos/edad, etc.)
   - Agregar interacciones entre variables

3. **Explorar otros modelos**:
   - Comparar con Árbol de Decisión para mayor interpretabilidad
   - Probar KNN para capturar patrones locales
   - Usar Random Forest para combinar múltiples árboles

4. **Validación y ajuste**:
   - Implementar validación cruzada (K-Fold)
   - Ajustar hiperparámetros con GridSearchCV
   - Analizar la curva de aprendizaje

5. **Tratamiento de desbalance**:
   - Si el dataset fuera más grande y desbalanceado, usar SMOTE
   - Ajustar los pesos de las clases

6. **Monitoreo en producción**:
   - Realizar predicciones en tiempo real
   - Monitorear degradación del modelo
   - Reentrenar periódicamente con nuevos datos

## Dependencias

```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Instalación

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Cómo Ejecutar

```bash
python main.py
```

El script generará:
- Salida detallada en consola con todas las métricas
- Gráfico guardado como `resultados_modelo.png` con:
  - Matriz de confusión
  - Comparación de métricas
  - Distribución de la variable objetivo
  - Importancia de características

## Estructura del Proyecto

```
.
├── main.py              # Script principal del modelo
├── dataset.csv          # Dataset de entrenamiento
├── README.md            # Este archivo
└── resultados_modelo.png # Gráficos de resultados (generado al ejecutar)
```

## Autor
Proyecto de Examen de Inteligencia Artificial - Diciembre 2025

---

**Nota**: Este proyecto utiliza Git para control de versiones y scikit-learn para el desarrollo del modelo de Machine Learning, como se requiere en las especificaciones del examen.