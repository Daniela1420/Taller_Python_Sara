import pandas as pd

# Leer la base de 100.000 datos
train = pd.read_csv("../data/train_100k.csv")

print("Primeras filas:")
print(train.head())

print("\nInformación del dataset:")
print(train.info())
