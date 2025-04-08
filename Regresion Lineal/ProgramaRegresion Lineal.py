# =============================================================================
# PASO 1: IMPORTAR LIBRERÍAS
# =============================================================================
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones matemáticas
import matplotlib.pyplot as plt  # Para visualización de datos
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento/prueba
import statsmodels.api as sm  # Para análisis estadístico más detallado
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# PASO 2: CARGA Y VERIFICACIÓN DE DATOS
# =============================================================================
try:
    # Cargamos el dataset con codificación Windows-1252
    # Nota: 'r' antes del string evita problemas con barras invertidas
    ruta_archivo = r"CO2 Emissions_Canada.csv"
    df = pd.read_csv(ruta_archivo, encoding='1252')
    
    # Verificación crítica de columnas necesarias
    columnas_requeridas = ['Engine Size(L)', 'CO2 Emissions(g/km)']
    if not all(col in df.columns for col in columnas_requeridas):
        print("❌ Error: El archivo no contiene las columnas requeridas.")
        print("Columnas disponibles:", df.columns.tolist())
        exit()
        
    print("✅ Dataset cargado correctamente. Primeras filas:")
    print(df[['Make', 'Model'] + columnas_requeridas].head())

except Exception as e:
    print(f"❌ Error al cargar el archivo: {str(e)}")
    exit()

# =============================================================================
# PASO 3: LIMPIEZA Y PREPARACIÓN DE DATOS
# =============================================================================
# Eliminamos filas con valores nulos en las columnas clave
df_clean = df.dropna(subset=columnas_requeridas)

# Convertimos a numérico y eliminamos posibles valores no numéricos
for col in columnas_requeridas:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=columnas_requeridas)

# Verificamos valores atípicos (outliers)
print("\n📊 Estadísticas descriptivas de los datos limpios:")
print(df_clean[columnas_requeridas].describe())

# =============================================================================
# PASO 4: ANÁLISIS EXPLORATORIO (EDA)
# =============================================================================
plt.figure(figsize=(12, 7))

# Scatter plot con personalización
plt.scatter(
    df_clean['Engine Size(L)'], 
    df_clean['CO2 Emissions(g/km)'], 
    s=100,                  # Tamaño de puntos
    alpha=0.7,              # Transparencia
    color='#1f77b4',        # Color azul estándar
    edgecolor='black',      # Borde negro
    linewidth=0.5,          # Grosor del borde
    label='Datos reales'    # Etiqueta para leyenda
)

# Personalización de ejes
plt.xlim(0, df_clean['Engine Size(L)'].max() * 1.1)
plt.ylim(0, df_clean['CO2 Emissions(g/km)'].max() * 1.1)
plt.xticks(np.arange(0, df_clean['Engine Size(L)'].max() + 2, 1))
plt.yticks(np.arange(0, df_clean['CO2 Emissions(g/km)'].max() + 50, 50))

# Títulos y etiquetas
plt.title('Relación entre Tamaño del Motor y Emisiones de CO2\n(Datos de Vehículos Canadienses)', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Tamaño del Motor (Litros)', fontsize=12)
plt.ylabel('Emisiones de CO2 (g/km)', fontsize=12)

# Grid y leyenda
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300)  # Guardamos con alta resolución
plt.show()

# =============================================================================
# PASO 5: MODELADO DE REGRESIÓN LINEAL
# =============================================================================
# Preparamos datos para el modelo
X = df_clean[['Engine Size(L)']].values  # Variable independiente (2D array)
y = df_clean['CO2 Emissions(g/km)'].values  # Variable dependiente

# Dividimos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  # Semilla para reproducibilidad
)

# Creamos y entrenamos el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# =============================================================================
# PASO 6: EVALUACIÓN DEL MODELO
# =============================================================================
# Hacemos predicciones
y_pred = model.predict(X_test)

# Calculamos métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Raíz cuadrada del MSE
r2 = r2_score(y_test, y_pred)

# Mostramos resultados
print("\n" + "="*50)
print("RESULTADOS DEL MODELO DE REGRESIÓN LINEAL")
print("="*50)
print(f"Intercepto (β0): {model.intercept_:.2f}")
print(f"Pendiente (β1): {model.coef_[0]:.2f}")
print(f"\nMétricas de evaluación:")
print(f"- MSE (Error Cuadrático Medio): {mse:.2f}")
print(f"- RMSE (Raíz del MSE): {rmse:.2f}")
print(f"- R² (Coeficiente de Determinación): {r2:.4f}")

# Interpretación
print("\n📝 Interpretación:")
print(f"Por cada litro adicional de tamaño de motor, las emisiones aumentan en promedio {model.coef_[0]:.2f} g/km")
print(f"El modelo explica el {r2*100:.1f}% de la variabilidad en las emisiones de CO2")

# =============================================================================
# PASO 7: VISUALIZACIÓN DEL MODELO
# =============================================================================
plt.figure(figsize=(12, 7))

# Scatter plot de los datos reales
plt.scatter(
    X_test, y_test, 
    color='blue', 
    alpha=0.6, 
    label='Datos reales (test)'
)

# Línea de regresión
plt.plot(
    X_test, y_pred, 
    color='red', 
    linewidth=2, 
    label=f'Regresión lineal: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x'
)

# Personalización
plt.title('Ajuste del Modelo de Regresión Lineal', fontsize=14)
plt.xlabel('Tamaño del Motor (Litros)', fontsize=12)
plt.ylabel('Emisiones de CO2 (g/km)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Añadimos anotación con métricas
metrics_text = f"""Métricas:
- MSE = {mse:.2f}
- RMSE = {rmse:.2f}
- R² = {r2:.4f}"""
plt.annotate(
    metrics_text, 
    xy=(0.65, 0.15), 
    xycoords='axes fraction',
    bbox=dict(boxstyle='round', alpha=0.2)
)

plt.tight_layout()
plt.savefig('regression_line.png', dpi=300)
plt.show()

# =============================================================================
# PASO 8: ANÁLISIS ESTADÍSTICO AVANZADO (OPCIONAL)
# =============================================================================
print("\n" + "="*50)
print("ANÁLISIS ESTADÍSTICO AVANZADO CON STATSMODELS")
print("="*50)

# Añadimos una columna de unos para el intercepto
X_sm = sm.add_constant(X_train)

# Creamos y ajustamos el modelo
model_sm = sm.OLS(y_train, X_sm).fit()

# Mostramos el resumen estadístico
print(model_sm.summary())

# Interpretación de resultados estadísticos
print("\n🔍 Interpretación estadística:")
print("- Valores p (P>|t|) < 0.05 indican que la variable es estadísticamente significativa")
print("- Intervalos de confianza muestran el rango probable para los coeficientes")
