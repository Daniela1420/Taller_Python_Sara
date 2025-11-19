import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Cargar modelo
model = joblib.load("../results/modelo_winrate.pkl")

# Cargar datos nuevos
test = pd.read_csv("../data/test_50k.csv")

y_test = test["win_rate"]
X_test = test.drop(columns=["win_rate", "id", "name", "team"])

# Hacer predicciones
preds = model.predict(X_test)

# Guardar predicciones en archivo
test["predicted_win_rate"] = preds
test.to_csv("../results/predicciones_test_50k.csv", index=False)

# Métricas
mse = mean_squared_error(y_test, preds)
print("MSE Test:", mse)
print("RMSE Test:", mse**0.5)

print("Predicciones guardadas correctamente.")
