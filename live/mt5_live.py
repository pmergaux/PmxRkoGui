# live/mt5_live.py
from mt5linux import MetaTrader5
import zmq
import json
import time
import pandas as pd
from src.utils.lstm_utils import create_sequences
import numpy as np
import tensorflow as tf
import joblib

def start_live(config):
    # --- Connexion MT5 ---
    if not mt5.initialize():
        print("MT5 init failed")
        return

    login = config["live"]["mt5_login"]
    password = config["live"]["mt5_password"]
    server = config["live"]["mt5_server"]

    if not mt5.login(login, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"MT5 connecté: {login}@{server}")

    # --- ZeroMQ ---
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    # --- Charger modèle ---
    model_path = "models/best_model.keras"
    scaler_path = "models/scaler.pkl"
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    symbol = config["symbol"]
    seq_len = config["seq_len"]
    features = config["features"]
    target_cfg = config["target"]

    buffer = []

    while True:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, seq_len + 1)
        if rates is None or len(rates) < seq_len + 1:
            time.sleep(1)
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # --- Calcul indicateurs ---
        df['EMA'] = df['close'].ewm(span=14).mean()
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_hist'] = df['MACD'].diff()
        df['time_vol'] = df.index.hour * 100 + df.index.minute

        df_feat = df[features].copy()
        df_feat = df_feat.dropna()

        if len(df_feat) < seq_len:
            time.sleep(1)
            continue

        seq_data = df_feat.tail(seq_len).values
        if not target_cfg["include_in_features"]:
            target_idx = features.index(target_cfg["column"])
            seq_data = np.delete(seq_data, target_idx, axis=1)

        X_input = scaler.transform(seq_data).reshape(1, seq_len, -1)
        proba = model.predict(X_input, verbose=0).item()

        signal = "BUY" if proba > 0.6 else "SELL" if proba < 0.4 else "HOLD"
        msg = json.dumps({"symbol": symbol, "signal": signal, "proba": proba})
        socket.send_string(msg)

        print(f"[{df.index[-1]}] {signal} (proba={proba:.3f})")
        time.sleep(60)
