# objective/optuna_objective.py
from optuna import TrialPruned
from src.train.trainer import train_model_for_config
from src.backtest.strategies.lstm_renko_strategy import LSTMRenkoBackStrategy

def optuna_objective(trial, train_df, val_df, test_df):
    # --- Paramètres ---
    renko_size = trial.suggest_float("renko_size", 10.0, 50.0)
    target_column = trial.suggest_categorical("target_column", ["close", "EMA"])
    target_type = trial.suggest_categorical("target_type", ["direction", "return"])
    seq_len = trial.suggest_int("seq_len", 20, 100)
    lstm_units = trial.suggest_int("lstm_units", 50, 200)
    threshold_buy = trial.suggest_float("threshold_buy", 0.5, 0.8)
    threshold_sell = trial.suggest_float("threshold_sell", 0.2, 0.5)

    # --- Config ---
    config = {
        "renko_size": renko_size,
        "features": ["EMA", "RSI", "MACD_hist", "time_vol", target_column],
        "target": {"column": target_column, "type": target_type, "include_in_features": True},
        "lstm": {"seq_len": seq_len, "units": lstm_units},
        "threshold_buy": threshold_buy,
        "threshold_sell": threshold_sell
    }

    # --- Entraîner ---
    model_path, scaler_path = train_model_for_config(config, train_df, val_df)
    if not model_path:
        raise TrialPruned()

    # --- Backtest ---
    param = {**config, "model_path": model_path, "scaler_path": scaler_path}
    strategy = LSTMRenkoBackStrategy(param, test_df)
    _, results = strategy.exec()

    # --- Pruning : si peu de trades ---
    if results["total_trades"] < 10:
        raise TrialPruned()

    # --- Score ---
    score = results["total_profit"] + 100 * results["win_rate"]
    trial.report(score, step=results["total_trades"])

    return score

## **2. LANCEMENT — `optuna_run.py`**

# optimize/optuna_run.py
import optuna
import pandas as pd

df = pd.read_pickle("data/ETHUSD.pkl")
train_df = df[:'2024-01-01']
val_df = df['2024-01-01':'2024-06-01']
test_df = df['2024-06-01':]

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(
    lambda trial: optuna_objective(trial, train_df, val_df, test_df),
    n_trials=100,
    timeout=None
)

print("MEILLEURE CONFIG OPTUNA :")
print(study.best_params)
print(f"Score : {study.best_value}")

## **3. DASHBOARD — `optuna-dashboard`**
"""
---bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db

→ Ouvre `http://localhost:8080` → **visualisation en temps réel**
"""
