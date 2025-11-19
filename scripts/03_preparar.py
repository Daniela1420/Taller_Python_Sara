import pandas as pd

train = pd.read_csv("../data/train_100k_clean.csv")

# Objetivo
y = train["win_rate"]

# Variables que usamos para predecir
X = train.drop(columns=["win_rate", "id", "name", "team"])
# Quitamos id, name y team porque no ayudan a predecir

print("X listo con columnas:")
print(X.head())

print("y listo:")
print(y.head())
