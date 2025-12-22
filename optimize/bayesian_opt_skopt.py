# optimize/bayesian_skopt.py
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from src.train.trainer import train_lstm  # ← ta fonction d'entraînement
import joblib
import json

# --- Charge config ---
with open("config.json", 'r') as f:
    config = json.load(f)

# --- Espace de recherche bayésien ---
dimensions = [
    Real(10, 200, name='lstm_units'),
    Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
    Integer(20, 100, name='seq_len'),
    Real(10, 50, name='renko_size'),
]

@use_named_args(dimensions)
def objective(**params):
    # Met à jour config
    for key, value in params.items():
        if key in config:
            config[key] = value
        elif key == 'lstm_units':
            config['model_params'] = {'units': int(value)}
        elif key == 'learning_rate':
            config['optimizer'] = {'lr': value}

    # Entraîne et retourne -accuracy (minimisation)
    try:
        acc = train_lstm(config, return_accuracy=True)
        print(f"Test: {params} → Accuracy: {acc:.4f}")
        return -acc
    except:
        return 0.0  # pénalité

# --- Lancement optimisation bayésienne ---
res = gp_minimize(
    func=objective,
    dimensions=dimensions,
    n_calls=50,
    n_random_starts=10,
    noise=1e-5,
    acq_func='gp_hedge',
    random_state=83  # Pierre 83 ans
)

print("MEILLEUR RÉSULTAT BAYÉSIEN:")
print(f"Params: {res.x}")
print(f"Accuracy: {-res.fun:.4f}")

# Sauvegarde
joblib.dump(res, "models/bayesian_best.pkl")
with open("config_best.json", "w") as f:
    best_config = config.copy()
    for dim, val in zip(dimensions, res.x):
        best_config[dim.name] = val
    json.dump(best_config, f, indent=2)
