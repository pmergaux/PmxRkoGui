## **2. FONCTION OBJECTIF — `hyperopt_objective.py`**

# objective/hyperopt_objective.py
from src.train.trainer import train_model_for_config
from src.backtest.strategies.lstm_renko_strategy import LSTMRenkoBackStrategy
import numpy as np

def hyperopt_objective(params, train_df, val_df, test_df):
    # --- Arrondir les entiers ---
    params['seq_len'] = int(params['seq_len'])
    params['lstm_units'] = int(params['lstm_units'])

    # --- Config ---
    config = {
        "renko_size": params["renko_size"],
        "features": ["EMA", "RSI", "MACD_hist", "time_vol", params["target_column"]],
        "target": {
            "column": params["target_column"],
            "type": params["target_type"],
            "include_in_features": True
        },
        "lstm": {"seq_len": params["seq_len"], "units": params["lstm_units"]},
        "threshold_buy": params["threshold_buy"],
        "threshold_sell": params["threshold_sell"]
    }

    # --- Entraîner modèle ---
    model_path, scaler_path = train_model_for_config(config, train_df, val_df)
    if not model_path:
        return {'loss': 1e6, 'status': 'fail'}

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
        "lstm": config["lstm"],
        "threshold_buy": config["threshold_buy"],
        "threshold_sell": config["threshold_sell"]
    }

    strategy = LSTMRenkoBackStrategy(param, test_df)
    _, results = strategy.exec()

    profit = results["total_profit"]
    trades = results["total_trades"]
    winrate = results["win_rate"]

    # --- Score : maximiser profit + winrate ---
    loss = - (profit + 100 * winrate)  # on minimise
    if trades < 10:
        loss += 1e6

    return {'loss': loss, 'status': 'ok', 'profit': profit, 'winrate': winrate}

