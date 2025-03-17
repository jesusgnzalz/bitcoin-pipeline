import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Definir rutas
data_path = "C:/Users/Jesus/Documents/DS_Online_Octubre24_Exercises/05_Deep_Learning/Sprint_17/Team_Challenge/bitcoin-pipeline/src/data"
model_dir = "C:/Users/Jesus/Documents/DS_Online_Octubre24_Exercises/05_Deep_Learning/Sprint_17/Team_Challenge/bitcoin-pipeline/src/models"
model_path = os.path.join(model_dir, "bitcoin_pipeline.pkl")

# Crear la carpeta models si no existe
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Cargar los datos de entrenamiento
train_file = os.path.join(data_path, "bitcoin_train.csv")
df_train = pd.read_csv(train_file)

# Seleccionar características y variable objetivo
X_train = df_train.drop(columns=['Close'])
y_train = df_train['Close']

# Identificar columnas numéricas y categóricas
num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns

# Crear transformadores para preprocesamiento
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Crear preprocesador
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Definir el pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(pipeline, model_path)

print(f"\n✅ Modelo guardado en: {model_path}")
