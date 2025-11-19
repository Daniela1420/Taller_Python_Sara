import pandas as pd

train = pd.read_csv("../data/train_100k.csv")

# Llenar vacíos en numéricos
train = train.fillna(train.mean(numeric_only=True))

# Llenar vacíos en texto con "Unknown"
train = train.fillna("Unknown")

# Guardar datos limpios
train.to_csv("../data/train_100k_clean.csv", index=False)

print("Datos limpios guardados.")
