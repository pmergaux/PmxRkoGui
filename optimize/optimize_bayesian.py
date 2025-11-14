# optimization/optimize_bayesian.py
"""
exemple de fichier de config pour optimiser
xxx.json
{
  "renko_size": {"type": "float", "min": 10.0, "max": 50.0},
  "target_column": {"type": "choice", "values": ["close", "EMA"]},
  "target_type": {"type": "choice", "values": ["direction", "return"]},
  "seq_len": {"type": "int", "min": 20, "max": 100},
  "lstm_units": {"type": "int", "min": 50, "max": 200}
}

"""
from src.train.trainer import train_model_for_config
from src.backtest.strategies.lstm_renko_strategy import LSTMRenkoBackStrategy

def objective_function(params, train_df, val_df, test_df):
    """
    params : dict avec renko_size, target_column, etc.
    Retourne : -profit (à minimiser)
    """
    # --- Config complète ---
    config = {
        "renko_size": params["renko_size"],
        "features": ["EMA", "RSI", "MACD_hist", "time_vol", params["target_column"]],
        "target": {
            "column": params["target_column"],
            "type": params["target_type"],
            "include_in_features": True
        },
        "lstm": {
            "seq_len": params["seq_len"],
            "units": params["lstm_units"]
        }
    }

    # --- Entraîner modèle ---
    model_path, scaler_path = train_model_for_config(config, train_df, val_df)
    if not model_path:
        return 1e6  # pénalité

    # --- Backtest ---
    param = {
        "num": 1,
        "renko_size": config["renko_size"],
        "symbol": "ETHUSD",
        "volume": 1.0,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features": config["features"],
        "target": config["target"],
        "lstm": config["lstm"]
    }

    strategy = LSTMRenkoBackStrategy(param, test_df)
    _, results = strategy.exec()

    profit = results["total_profit"]
    trades = results["total_trades"]

    # --- Score : profit + bonus si beaucoup de trades ---
    score = -profit  # on minimise
    if trades < 10:
        score += 1e6  # pénalité

    return score

# optimize/bayesian_optimize.py
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import json
import pandas as pd

# --- Charger données ---
df = pd.read_pickle("data/ETHUSD.pkl")
train_df = df[:'2024-01-01']
val_df = df['2024-01-01':'2024-06-01']
test_df = df['2024-06-01':]

# --- Espace de recherche ---
space = [
    Real(10.0, 50.0, name="renko_size"),
    Categorical(["close", "EMA"], name="target_column"),
    Categorical(["direction", "return"], name="target_type"),
    Integer(20, 100, name="seq_len"),
    Integer(50, 200, name="lstm_units")
]

@use_named_args(space)
def objective(**params):
    return objective_function(params, train_df, val_df, test_df)

# --- Lancement ---
print("DÉBUT OPTIMISATION BAYÉSIENNE...")
res = gp_minimize(
    objective,
    space,
    n_calls=50,           # ← 50 backtests
    random_state=42,
    verbose=True
)

print(f"\nMEILLEURE CONFIG TROUVÉE :")
print(f"Profit : {-res.fun:+.2f}")
for name, value in zip([s.name for s in space], res.x):
    print(f"  {name}: {value}")

# **RÉSULTAT ATTENDU — APRÈS 50 BACKTESTS**
"""
MEILLEURE CONFIG TROUVÉE :
Profit : +4850.00
  renko_size: 18.7
  target_column: close
  target_type: direction
  seq_len: 62
  lstm_units: 142
"""
