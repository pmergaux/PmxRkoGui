# optimize/simple_optimize.py
# OPTIMISATION SIMPLE SANS HYPEROPT — RIEN D'INCONNU
# Auteur : Pierre (83 ans)

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
import random
from datetime import datetime

# ==================================================================
# 1. DONNÉES
# ==================================================================
df = pd.read_pickle("data/ETHUSD.pkl")
train_df = df[:'2024-01-01']
val_df = df['2024-01-01':'2024-06-01']
test_df = df['2024-06-01':]

# ==================================================================
# 2. FONCTIONS UTILITAIRES
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

def create_sequences(data, seq_len, target_col, target_type):
    X, y = [], []
    data = data.dropna()
    values = data[target_col].values
    features = data[["EMA", "RSI", "MACD_hist", "time_vol", target_col]].values

    for i in range(len(data) - seq_len):
        X.append(features[i:i+seq_len])
        if target_type == 'direction':
            y.append(1 if values[i+seq_len] > values[i+seq_len-1] else 0)
        else:
            y.append((values[i+seq_len] - values[i+seq_len-1]) / values[i+seq_len-1])
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_val, y_val, units, seq_len):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(seq_len, 5)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)
    return model

# ==================================================================
# 3. FONCTION D'ÉVALUATION
# ==================================================================
def evaluate_config(config):
    try:
        # Préparer données
        train_p = add_indicators(train_df)
        val_p = add_indicators(val_df)
        test_p = add_indicators(test_df)

        X_train, y_train = create_sequences(train_p, config['seq_len'], config['target_column'], config['target_type'])
        X_val, y_val = create_sequences(val_p, config['seq_len'], config['target_column'], config['target_type'])
        X_test, y_test = create_sequences(test_p, config['seq_len'], config['target_column'], config['target_type'])

        if len(X_train) < 100: return -1e9

        # Normaliser
        scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
        if not scaler:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train.reshape(-1, 5))
            joblib.dump(scaler, "models/scaler.pkl")

        X_train = scaler.transform(X_train.reshape(-1, 5)).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, 5)).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)

        # Entraîner
        model = train_model(X_train, y_train, X_val, y_val, config['lstm_units'], config['seq_len'])
        model.save("models/temp.keras")

        # Prédire
        proba = model.predict(X_test, verbose=0).flatten()
        buy = proba > config['threshold_buy']
        sell = proba < config['threshold_sell']

        # Backtest
        returns = np.diff(test_p[config['target_column']].values[-len(proba):])
        profit = 0
        trades = 0
        wins = 0

        for i in range(len(proba)):
            if i >= len(returns): break
            if buy[i]:
                profit += returns[i]
                trades += 1
                if returns[i] > 0: wins += 1
            elif sell[i]:
                profit -= returns[i]
                trades += 1
                if returns[i] < 0: wins += 1

        winrate = wins / trades if trades > 0 else 0
        if trades < 10: return -1e9

        score = profit * 1000 + winrate * 100
        print(f"Config: {config['renko_size']:.1f}, {config['seq_len']}, {config['lstm_units']} → Score: {score:.1f}")
        return score

    except:
        return -1e9

# ==================================================================
# 4. OPTIMISATION PAR RECHERCHE ALÉATOIRE (100 ESSAIS)
# ==================================================================
if __name__ == "__main__":
    print("DÉBUT OPTIMISATION SIMPLE — 100 ESSAIS")

    best_score = -1e9
    best_config = None

    for i in range(100):
        config = {
            'renko_size': round(random.uniform(10, 50), 1),
            'target_column': random.choice(['close', 'EMA']),
            'target_type': random.choice(['direction', 'return']),
            'seq_len': random.randint(20, 100),
            'lstm_units': random.randint(50, 200),
            'threshold_buy': round(random.uniform(0.5, 0.7), 3),
            'threshold_sell': round(random.uniform(0.3, 0.5), 3)
        }

        score = evaluate_config(config)
        if score > best_score:
            best_score = score
            best_config = config

    print("\nMEILLEURE CONFIGURATION TROUVÉE :")
    print(json.dumps(best_config, indent=2))
    print(f"Score final: {best_score:.1f}")

    # Sauvegarde
    os.makedirs("models/simple_opt", exist_ok=True)
    with open("models/simple_opt/best.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print("Sauvegarde → models/simple_opt/best.json")
