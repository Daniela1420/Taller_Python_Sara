import joblib
import pandas as pd

modelo = joblib.load("../results/modelo_winrate.pkl")

# Crear un jugador ejemplo
jugador = pd.DataFrame([{
    "role": "Jungle",
    "region": "LAN",
    "age": 22,
    "kda": 3.2,
    "games_played": 500,
    "avg_kills": 5.1,
    "avg_deaths": 2.9,
    "avg_assists": 7.3
}])

pred = modelo.predict(jugador)
print("Predicción de win_rate:", pred[0])
