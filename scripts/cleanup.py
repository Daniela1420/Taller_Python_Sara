import os

paths = [
    "../data/train_100k_clean.csv",
    "../results/modelo_winrate.pkl",
    "../results/predicciones_test_50k.csv"
]

for p in paths:
    try:
        os.remove(p)
        print(f"Eliminado: {p}")
    except FileNotFoundError:
        pass

print("Proyecto limpio.")
