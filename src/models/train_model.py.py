import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Definir rutas de archivos
data_path = "C:/Users/Jesus/Documents/DS_Online_Octubre24_Exercises/05_Deep_Learning/Sprint_17/Team_Challenge/bitcoin-pipeline/src/data"
model_path = "C:/Users/Jesus/Documents/DS_Online_Octubre24_Exercises/05_Deep_Learning/Sprint_17/Team_Challenge/bitcoin-pipeline/src/models/bitcoin_pipeline.pkl"

# Cargar el modelo entrenado
pipeline = joblib.load(model_path)
print("\n✅ Modelo cargado correctamente.")

# Cargar los datos de prueba
test_file = os.path.join(data_path, "bitcoin_test.csv")
df_test = pd.read_csv(test_file)

# Selección de características y variable objetivo
X_test = df_test.drop(columns=['Close'])
y_test = df_test['Close']

# Realizar predicciones
y_pred = pipeline.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
print("\n🔹 Resultados del modelo:")
print(f"📉 MAE (Error Absoluto Medio): {mae:.4f}")
print(f"📉 MSE (Error Cuadrático Medio): {mse:.4f}")
print(f"📈 R² (Coeficiente de determinación): {r2:.4f}")

# Guardar las predicciones en un archivo CSV
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_file = f"{data_path}/bitcoin_predictions.csv"
results_df.to_csv(results_file, index=False)

print(f"\n✅ Predicciones guardadas en: {results_file}")
