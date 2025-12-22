# optimize/hyperopt_full.py
# HYPEROPT TPE COMPLET EN UN SEUL FICHIER — RIEN D'AUTRE À CONNAÎTRE
# Auteur : Pierre (83 ans)

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
from hyperopt import fmin, tpe, Trials, hp
from datetime import datetime

# ==================================================================
# 1. DONNÉES (à adapter à ton chemin)
# ==================================================================
df = pd.read_pickle("data/ETHUSD.pkl")  # ← change si besoin
train_df = df[:'2024-01-01']
val_df = df['2024-01-01':'2024-06-01']
test_df = df['2024-06-01':]

# ==================================================================
# 2. ESPACE DE RECHERCHE HYPEROPT (TPE)
# ==================================================================
space = {
    'renko_size': hp.uniform('renko_size', 10.0, 50.0),
    'target_column_idx': hp.choice('target_column_idx', [0, 1]),  # 0=close, 1=EMA
    'target_type_idx': hp.choice('target_type_idx', [0, 1]),      # 0=direction, 1=return
    'seq_len': hp.quniform('seq_len', 20, 100, 1),
    'lstm_units': hp.quniform('lstm_units', 50, 200, 1),
    'threshold_buy': hp.uniform('threshold_buy', 0.5, 0.7),
    'threshold_sell': hp.uniform('threshold_sell', 0.3, 0.5)
}

TARGET_COLUMNS = ['close', 'EMA']
TARGET_TYPES = ['direction', 'return']

# ==================================================================
# 3. FONCTIONS UTILITAIRES (création séquences, indicateurs)
# ==================================================================
def add_indicators(df):
    df = df.copy()
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
    return df

def create_sequences(data, seq_len, target_col, target_type, include_target=True):
    X, y = [], []
    data = data.dropna()
    values = data[target_col].values
    feat_cols = ["EMA", "RSI", "MACD_hist", "time_vol", target_col] if include_target else ["EMA", "RSI", "MACD_hist", "time_vol"]
    features = data[feat_cols].values

    for i in range(len(data) - seq_len):
        X.append(features[i:i+seq_len])
        if target_type == 'direction':
            y.append(1 if values[i+seq_len] > values[i+seq_len-1] else 0)
        else:  # return
            y.append((values[i+seq_len] - values[i+seq_len-1]) / values[i+seq_len-1])
    return np.array(X), np.array(y)

# ==================================================================
# 4. ENTRAÎNEMENT LSTM
# ==================================================================
def train_lstm_model(X_train, y_train, X_val, y_val, units, seq_len, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(seq_len, n_features)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)
    return model

# ==================================================================
# 5. FONCTION OBJECTIF HYPEROPT
# ==================================================================
def objective(params):
    # --- Conversion ---
    target_column = TARGET_COLUMNS[int(params['target_column_idx'])]
    target_type = TARGET_TYPES[int(params['target_type_idx'])]
    seq_len = int(params['seq_len'])
    units = int(params['lstm_units'])

    # --- Préparation données ---
    try:
        train_prep = add_indicators(train_df)
        val_prep = add_indicators(val_df)
        test_prep = add_indicators(test_df)

        X_train, y_train = create_sequences(train_prep, seq_len, target_column, target_type)
        X_val, y_val = create_sequences(val_prep, seq_len, target_column, target_type)
        X_test, y_test = create_sequences(test_prep, seq_len, target_column, target_type)

        if len(X_train) == 0 or len(X_val) == 0:
            return {'loss': 1e6, 'status': 'fail'}

        # --- Normalisation ---
        scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
            joblib.dump(scaler, "models/scaler.pkl")

        X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # --- Entraînement ---
        model = train_lstm_model(X_train, y_train, X_val, y_val, units, seq_len, X_train.shape[2])
        model.save("models/temp_model.keras")

        # --- Prédiction ---
        proba = model.predict(X_test, verbose=0).flatten()
        buy = proba > params['threshold_buy']
        sell = proba < params['threshold_sell']

        # --- Backtest simple ---
        returns = np.diff(test_prep[target_column].values[-len(proba):])
        profit = 0
        trades = 0
        wins = 0

        for i in range(len(proba)):
            if buy[i] and i < len(returns):
                profit += returns[i]
                trades += 1
                if returns[i] > 0: wins += 1
            elif sell[i] and i < len(returns):
                profit -= returns[i]
                trades += 1
                if returns[i] < 0: wins += 1

        winrate = wins / trades if trades > 0 else 0
        if trades < 10:
            return {'loss': 1e6, 'status': 'fail'}

        loss = - (profit * 1000 + winrate * 100)
        print(f"Profit: {profit:.2f} | Winrate: {winrate:.1%} | Trades: {trades}")

        return {
            'loss': loss,
            'status': 'ok',
            'profit': profit,
            'winrate': winrate,
            'trades': trades
        }

    except Exception as e:
        print(f"ERREUR: {e}")
        return {'loss': 1e6, 'status': 'fail'}

# ==================================================================
# 6. LANCEMENT HYPEROPT
# ==================================================================
if __name__ == "__main__":
    print("DÉBUT HYPEROPT TPE — 100 ESSAIS")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.RandomState(83)
    )

    # --- Meilleur résultat ---
    best_config = {
        "renko_size": best["renko_size"],
        "target_column": TARGET_COLUMNS[int(best["target_column_idx"])],
        "target_type": TARGET_TYPES[int(best["target_type_idx"])],
        "seq_len": int(best["seq_len"]),
        "lstm_units": int(best["lstm_units"]),
        "threshold_buy": best["threshold_buy"],
        "threshold_sell": best["threshold_sell"]
    }

    print("\nMEILLEURE CONFIGURATION TROUVÉE :")
    print(json.dumps(best_config, indent=2))

    # --- Sauvegarde ---
    os.makedirs("models/hyperopt", exist_ok=True)
    with open("models/hyperopt/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print("Sauvegarde → models/hyperopt/best_config.json")
