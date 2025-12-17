# ========================================
# MODELO DE CLASIFICACIÓN - PREDICCIÓN DE COMPRA
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. CARGAR DATASET
# ========================================
print("=" * 60)
print("PASO 1: CARGAR DATASET")
print("=" * 60)

df = pd.read_csv('dataset.csv')
print(f"\n✓ Dataset cargado correctamente")
print(f"\nDimensiones del dataset: {df.shape}")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# ========================================
# 2. EXPLORAR DATASET
# ========================================
print("\n" + "=" * 60)
print("PASO 2: EXPLORACIÓN DEL DATASET")
print("=" * 60)

print("\nTipos de datos:")
print(df.dtypes)

print("\nPrimeras filas del dataset:")
print(df.head())

print("\nEstadísticas descriptivas:")
print(df.describe())

print("\nValores nulos:")
print(df.isnull().sum())

# ========================================
# 3. PREPROCESAMIENTO DE DATOS
# ========================================
print("\n" + "=" * 60)
print("PASO 3: PREPROCESAMIENTO DE DATOS")
print("=" * 60)

# 3.1 Limpieza de valores nulos
print("\n[3.1] Limpieza de valores nulos:")
print(f"Filas con valores nulos antes: {df.isnull().sum().sum()}")
df = df.dropna()
print(f"Filas con valores nulos después: {df.isnull().sum().sum()}")
print(f"Dimensiones después de limpieza: {df.shape}")

# 3.2 Codificación de variables categóricas
print("\n[3.2] Codificación de variables categóricas:")
df_procesado = df.copy()

# Codificar 'genero'
label_encoder_genero = LabelEncoder()
df_procesado['genero'] = label_encoder_genero.fit_transform(df_procesado['genero'])
print(f"✓ 'genero' codificado: {dict(zip(label_encoder_genero.classes_, label_encoder_genero.transform(label_encoder_genero.classes_)))}")

# Codificar 'estado_civil'
label_encoder_estado = LabelEncoder()
df_procesado['estado_civil'] = label_encoder_estado.fit_transform(df_procesado['estado_civil'])
print(f"✓ 'estado_civil' codificado: {dict(zip(label_encoder_estado.classes_, label_encoder_estado.transform(label_encoder_estado.classes_)))}")

# 3.3 Normalización/Estandarización
print("\n[3.3] Normalización/Estandarización:")
scaler = StandardScaler()
columnas_numericas = ['edad', 'ingresos_anuales', 'puntuacion_credito']
df_procesado[columnas_numericas] = scaler.fit_transform(df_procesado[columnas_numericas])
print(f"✓ Variables numéricas estandarizadas usando StandardScaler")

print("\nDataset después del preprocesamiento:")
print(df_procesado.head())

# ========================================
# 4. SEPARAR DATOS EN ENTRENAMIENTO Y PRUEBA
# ========================================
print("\n" + "=" * 60)
print("PASO 4: SEPARACIÓN DE DATOS")
print("=" * 60)

X = df_procesado.drop('compra', axis=1)
y = df_procesado['compra']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
print(f"Proporción: {X_train.shape[0] / len(X) * 100:.1f}% entrenamiento, {X_test.shape[0] / len(X) * 100:.1f}% prueba")

# ========================================
# 5. ENTRENAR MODELO
# ========================================
print("\n" + "=" * 60)
print("PASO 5: ENTRENAR MODELO")
print("=" * 60)

modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X_train, y_train)

print("\n✓ Modelo de Regresión Logística entrenado correctamente")
print(f"Coeficientes del modelo: {modelo.coef_[0]}")
print(f"Intercepto: {modelo.intercept_[0]:.4f}")

# ========================================
# 6. REALIZAR PREDICCIONES
# ========================================
print("\n" + "=" * 60)
print("PASO 6: PREDICCIONES")
print("=" * 60)

y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)

print(f"\n✓ Predicciones realizadas en conjunto de prueba")
print(f"Primeras 10 predicciones: {y_pred[:10]}")

# ========================================
# 7. EVALUAR MODELO
# ========================================
print("\n" + "=" * 60)
print("PASO 7: EVALUACIÓN DEL MODELO")
print("=" * 60)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 MÉTRICAS DE RENDIMIENTO:")
print(f"├── Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"├── Precisión: {precision:.4f} ({precision*100:.2f}%)")
print(f"└── Recall: {recall:.4f} ({recall*100:.2f}%)")

print(f"\n📋 MATRIZ DE CONFUSIÓN:")
print(cm)
print(f"\n├── Verdaderos Negativos (TN): {cm[0, 0]}")
print(f"├── Falsos Positivos (FP): {cm[0, 1]}")
print(f"├── Falsos Negativos (FN): {cm[1, 0]}")
print(f"└── Verdaderos Positivos (TP): {cm[1, 1]}")

print(f"\n📄 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=['No Compra', 'Compra']))

# ========================================
# 8. VISUALIZACIONES
# ========================================
print("\n" + "=" * 60)
print("PASO 8: GENERANDO VISUALIZACIONES")
print("=" * 60)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
            xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'])
axes[0, 0].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Valor Real')
axes[0, 0].set_xlabel('Valor Predicho')

# Gráfico 2: Métricas
metricas = ['Accuracy', 'Precisión', 'Recall']
valores = [accuracy, precision, recall]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = axes[0, 1].bar(metricas, valores, color=colors)
axes[0, 1].set_ylim([0, 1])
axes[0, 1].set_title('Métricas de Rendimiento', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Valor')
for bar, valor in zip(bars, valores):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{valor:.3f}', ha='center', va='bottom')

# Gráfico 3: Distribución de la variable objetivo
df['compra'].value_counts().plot(kind='bar', ax=axes[1, 0], color=['#d62728', '#2ca02c'])
axes[1, 0].set_title('Distribución de Compras en Dataset', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Compra')
axes[1, 0].set_ylabel('Cantidad')
axes[1, 0].set_xticklabels(['No Compra', 'Compra'], rotation=0)

# Gráfico 4: Importancia de características (coeficientes)
features = X.columns
coef = modelo.coef_[0]
indices = np.argsort(np.abs(coef))[::-1]
axes[1, 1].barh(features[indices], np.abs(coef[indices]), color='steelblue')
axes[1, 1].set_title('Importancia de Características (Valor Absoluto)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Valor Absoluto del Coeficiente')

plt.tight_layout()
plt.savefig('resultados_modelo.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado como 'resultados_modelo.png'")
plt.show()

# ========================================
# 9. RESUMEN FINAL
# ========================================
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"""
✓ Modelo entrenado: Regresión Logística
✓ Exactitud (Accuracy): {accuracy*100:.2f}%
✓ Precisión: {precision*100:.2f}%
✓ Recall: {recall*100:.2f}%
✓ Dataset: {df.shape[0]} muestras, {df.shape[1]} características
✓ División: 70% entrenamiento ({X_train.shape[0]} muestras), 30% prueba ({X_test.shape[0]} muestras)
""")

print("=" * 60)
print("Proceso completado exitosamente ✓")
print("=" * 60)
