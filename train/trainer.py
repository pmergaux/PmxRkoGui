import os

import tensorflow as tf
import joblib
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Sequential

from src.utils.config_utils import config_to_hash
from src.utils.lstm_utils import create_sequences
import pandas as pd

def train_lstm(config):
    df = pd.read_pickle("data/ETHUSD_real.pkl")
    train_df = df[:'2024-01-01']
    val_df = df['2024-01-01':'2024-06-01']

    scaler = joblib.load("models/scaler.pkl")
    X_train, y_train = create_sequences(train_df, config["seq_len"], config["target"], config["features"])
    X_val, y_val = create_sequences(val_df, config["seq_len"], config["target"], config["features"])

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(config["seq_len"], X_train.shape[2])),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=1)
    model.save("models/best_model.keras")

# train/train_for_config.py
def train_model_for_config(config, train_df, val_df):
    hash_id = config_to_hash(config)
    model_path = f"models/lstm_{hash_id}.keras"
    scaler_path = f"models/scaler_{hash_id}.pkl"

    if os.path.exists(model_path):
        print(f"Modèle existant : {model_path}")
        return model_path, scaler_path

    # --- Préparer données ---
    features = config["features"]
    target_cfg = config["target"]
    seq_len = config["lstm"]["seq_len"]

    # Scaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    val_scaled = scaler.transform(val_df[features])

    # Séquence
    X_train, y_train = create_sequences(train_scaled, seq_len, target_cfg, features)
    X_val, y_val = create_sequences(val_scaled, seq_len, target_cfg, features)

    if len(X_train) == 0:
        return None, None

    # --- Modèle ---
    n_features = X_train.shape[2]
    output_units = 1
    activation = 'sigmoid' if target_cfg["type"] == "direction" else 'linear'
    loss = 'binary_crossentropy' if target_cfg["type"] == "direction" else 'mse'

    model = Sequential([
        LSTM(config["lstm"]["units"], input_shape=(seq_len, n_features)),
        Dense(output_units, activation=activation)
    ])
    model.compile(optimizer='adam', loss=loss)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, verbose=0)

    # --- Sauvegarde ---
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modèle entraîné : {model_path}")
    return model_path, scaler_path
