import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

# Leer datos
train = pd.read_csv("../data/train_100k_clean.csv")

y = train["win_rate"]
X = train.drop(columns=["win_rate", "id", "name", "team"])

# Detectar columnas numéricas y categóricas
num_cols = ["age", "kda", "games_played", "avg_kills", "avg_deaths", "avg_assists"]
cat_cols = ["role", "region"]

# Preprocesamiento para convertir texto en números
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Modelo
model = RandomForestRegressor(n_estimators=150)

# Pipeline = procesamiento + modelo
pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# Entrenar/validar
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Predecir validación
preds = pipeline.predict(X_val)

# Error
mse = mean_squared_error(y_val, preds)
print("MSE:", mse)
print("RMSE:", mse**0.5)

# Guardar modelo
joblib.dump(pipeline, "../results/modelo_winrate.pkl")

print("Modelo guardado.")
