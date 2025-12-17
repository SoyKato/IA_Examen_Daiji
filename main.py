import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix

# Carga de dataset
df = pd.read_csv('dataset_clima_ensenada.csv')

# Eliminacion de filas con valores nulos
df = df.dropna()

# Codificar variable objetivo (condicion_clima)
le = LabelEncoder()
df['condicion_clima_encoded'] = le.fit_transform(df['condicion_clima'])

# Caracteristicas y variables importantes
X = df[['temp_max_f', 'temp_min_f']]
y = df['condicion_clima_encoded']

# Datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nDatos de entrenamiento: {len(X_train)} muestras")
print(f"Datos de prueba: {len(X_test)} muestras")

# Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
modelo.fit(X_train, y_train)

# Generar la prediccion
y_pred = modelo.predict(X_test)

# Calculo de metricas
precision = precision_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# 8. Mostrar resultados en consola
print("Prediccion del modelo")
print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
print(f"\nMatriz de Confusion:")
print(cm)

# 9. Crear gráfico de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_,
            cbar_kws={'label': 'Cantidad'})
plt.title('Matriz de Confusion - Prediccion del Clima\nEnsenada Baja California', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Valor Predicho', fontsize=12)
plt.tight_layout()
plt.savefig('matriz_confusion_clima.png', dpi=300, bbox_inches='tight')
plt.show()

