# utils/lstm_utils.py
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from numba import njit, prange
from typing import List, Tuple, Dict


def load_model(model_path: str):
    """
    Charge un modèle Keras (.h5 ou .keras).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    return tf.keras.models.load_model(model_path)

def load_scaler(scaler_path: str):
    """
    Charge un scaler (StandardScaler, MinMaxScaler, etc.).
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler non trouvé : {scaler_path}")
    return joblib.load(scaler_path)

def prepare_sequence(df: pd.DataFrame, feature_cols: list, seq_len: int) -> np.ndarray:
    """
    Transforme un DataFrame en séquence pour LSTM.
    """
    data = df[feature_cols].values
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)

def predict_sequence(model, scaler, sequence: np.ndarray) -> float:
    """
    Prédit une seule séquence.
    """
    seq = scaler.transform(sequence)
    seq = seq.reshape(1, seq.shape[0], seq.shape[1])
    pred = model.predict(seq, verbose=0)[0][0]
    return float(pred)

def predict_lstm(df, seq_len, model):
    df['pred_signal'] = 0.0
    X, _ = create_sequences(df, seq_len)
    pred = model.predict(X[-1:], verbose=0)[0][0]
    df.loc[df.index[-2], 'pred_signal'] = float(pred)
    print(f"Prédiction LSTM : {pred:.4f}, Signal : {df.iloc[-2]['pred_signal']:.4f}")

def get_feature_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Retourne les colonnes numériques utilisables comme features.
    """
    exclude = exclude or ['time', 'open_renko', 'close_renko', 'direction', 'sigo', 'sigc']
    return [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

# ================================
# 1. FONCTIONS UTILITAIRES
# ================================
def clean_features(df, cols):
    dfc = df[cols].copy()
    dfc = dfc.replace([np.inf, -np.inf], np.nan).fillna(0)
    return dfc.values

# ===============================
# x. FUNCTIONS CREATE_SEQUENCES
# ===============================
@njit
def _create_sequences_opt_numba(data, seq_len):
    n = len(data)
    X = np.empty((n - seq_len, seq_len, data.shape[1]), dtype=np.float64)
    y = np.empty((n - seq_len, data.shape[1]), dtype=np.float64)
    for i in range(n - seq_len):
        X[i] = data[i:i + seq_len]
        y[i] = data[i + seq_len]
    return X, y

def create_sequences_opt(data, seq_len):
    if len(data) <= seq_len:
        return np.array([]), np.array([])
    return _create_sequences_numba(data, seq_len)

def create_sequences_target(data, seq_len):
    X, y = [], []
    # On s'arrête à len(data) - seq_len pour avoir une cible
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        # Cible : close[i + seq_len] vs close[i + seq_len - 1]
        close_n = data[i + seq_len - 1, 0]      # close à t
        close_n1 = data[i + seq_len, 0]         # close à t+1
        y.append( 1 if close_n1 > close_n else 0)
    return np.array(X), np.array(y)

@njit(parallel=True, cache=True)
def _create_sequences_numba(data: np.ndarray, seq_len: int, target_col_idx: int, include_target_in_features: bool,
                            target_type: int ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Version NUMBA ultra-rapide de create_sequences.
    target_type   # 0: direction, 1: value, 2: return
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    if include_target_in_features:
        seq_features = n_features
    else:
        seq_features = n_features - 1
    n_seq = n_samples - seq_len
    if n_seq <= 0:
        return np.empty((0, seq_len, seq_features)), np.empty((0,))
    X = np.empty((n_seq, seq_len, seq_features), dtype=np.float64)
    y = np.empty((n_seq,), dtype=np.float64)
    for i in prange(n_seq):
        # --- Copie séquence ---
        start_idx = i
        end_idx = i + seq_len
        seq = data[start_idx:end_idx]
        if not include_target_in_features:
            # Supprime la cible de chaque ligne
            seq = np.delete(seq, target_col_idx, axis=1)
        X[i] = seq
        # --- Cible à t+1 ---
        val_t = data[i + seq_len - 1, target_col_idx]
        val_t1 = data[i + seq_len, target_col_idx]
        if target_type == 0:  # direction
            y[i] = 1.0 if val_t1 > val_t else 0.0
        elif target_type == 1:  # value
            y[i] = val_t1
        elif target_type == 2:  # return
            y[i] = (val_t1 - val_t) / val_t if val_t != 0.0 else 0.0
    return X, y


def create_sequences(data: np.ndarray, seq_len: int, target_config: Dict, feature_names: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Système ouvert : crée X et y selon la configuration.
    Parameters:
    -----------
    data : np.ndarray
        Données normalisées (n_samples, n_features)
    seq_len : int
        Longueur de la séquence
    target_config : dict
        {
            "column": "close" | "EMA" | "RSI",
            "type": "direction" | "value" | "return",
            "include_in_features": True | False
        }
    feature_names : list
        Noms des colonnes (ex: ["EMA", "RSI", "MACD_hist", "close"])

    Returns:
    --------
    X : (n_samples, seq_len, n_features)
    y : (n_samples,) ou (n_samples, 1)
    """
    target_col = target_config["column"]
    try:
        target_idx = feature_names.index(target_col)
    except ValueError:
        raise ValueError(f"Colonne cible '{target_col}' non trouvée dans features")
    target_type_map = {
        "direction": 0,
        "value": 1,
        "return": 2
    }
    # -- converti le type en int
    target_type = target_type_map[target_config["type"]]
    # -- la cible est elle dans les données d'établissement du modèle
    include_target = target_config.get("include_in_features", True)

    return _create_sequences_numba(
        data=data,
        seq_len=seq_len,
        target_col_idx=target_idx,
        include_target_in_features=include_target,
        target_type=target_type
    )

# ================================
# 3. MODÈLE LSTM (avec ou sans hyperparamètres)
# ================================
def build_lstm(params, seq_len,  hp=None):
    """
    Construit le modèle LSTM
    - Si hp is None → utilise params['lstm_units']
    - Si hp est fourni → utilise hp.Int / hp.Choice
    """

    if 'features' not in params:
        raise KeyError("'features' manquant dans params")

    if hp is None:
        # === MODE FIXE (optimisation classique) ===
        model = Sequential([
            Input(shape=(seq_len, len(params['features']))),
            LSTM(params['lstm_units'], return_sequences=True),
            Dropout(0.2),
            LSTM(params['lstm_units'] // 2),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # === MODE HYPERPARAMÈTRES (Keras Tuner) ===
        model = Sequential([
            Input(shape=(seq_len, len(params['features']))),
            LSTM(hp.Int('units1', 32, 128, step=32), return_sequences=True),
            Dropout(hp.Float('drop1', 0.1, 0.4, step=0.1)),
            LSTM(hp.Int('units2', 32, 128, step=32)),
            Dropout(hp.Float('drop2', 0.1, 0.4, step=0.1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(hp.Choice('lr', [1e-3, 5e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    return model
# ================================
# 5. TRANSFORMER (avec ou sans hyperparamètres)
# ================================
def build_transformer(seq_len, features_len, hp=None):
    """
    Construit un modèle Transformer
    - hp=None → mode fixe (optimisation classique)
    - hp fourni → mode Keras Tuner
    """
    inputs = Input(shape=(seq_len, features_len))
    if hp is None:
        # === MODE FIXE (votre optimisation actuelle) ===
        x = Dense(64)(inputs)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        attn = Dropout(0.1)(attn)
        attn = LayerNormalization()(x + attn)
        ff = Dense(128, activation='relu')(attn)
        ff = Dense(x.shape[-1])(ff)
        x = LayerNormalization()(attn + ff)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # === MODE HYPERPARAMÈTRES (Keras Tuner) ===
        x = Dense(hp.Int('dense_units', 32, 128, step=32))(inputs)
        attn = MultiHeadAttention(
            num_heads=hp.Choice('num_heads', [2, 4, 8]),
            key_dim=hp.Int('key_dim', 16, 64, step=16)
        )(x, x)
        attn = Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1))(attn)
        attn = LayerNormalization()(x + attn)
        ff = Dense(hp.Int('ff_units', 64, 256, step=64), activation='relu')(attn)
        ff = Dense(x.shape[-1])(ff)
        x = LayerNormalization()(attn + ff)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(hp.Choice('lr', [1e-3, 5e-4, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    return model

def build_transformer_tunable(seq_len, n_features, hp):
    inputs = Input(shape=(seq_len, n_features))
    x = MultiHeadAttention(
        num_heads=hp.Int('heads', min_value=2, max_value=8, step=2),
        key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32)
    )(inputs, inputs)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ================================
# 6. ÉVALUATION
# ================================
def evaluate_model(model, X, y_true, name):
    pred_proba = model.predict(X, verbose=0).flatten()
    pred_class = (pred_proba > 0.5).astype(int)
    print(f"\n{name}:")
    print(f"Accuracy:  {accuracy_score(y_true, pred_class):.4f}")
    print(f"F1-Score:  {f1_score(y_true, pred_class):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, pred_proba):.4f}")
    print(f"Pred 0/1:  {np.bincount(pred_class)}")
    return pred_proba, pred_class
# ================================
# 7. ALIGNEMENT + BACKTEST
# ================================
def backtest(pred_signal, df_test, df_full_with_rules, seq_len):
    """
    pred_signal: LSTM signals (1/-1)
    df_test: test_df (OHLC + features)
    df_full_with_rules: df_bricks avec sigo, sigc (même index)
    """
    start_idx = seq_len
    end_idx = start_idx + len(pred_signal)

    df_bt = df_test.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    df_rules = df_full_with_rules.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    if len(df_bt) != len(pred_signal):
        raise ValueError(f"Mismatch: {len(df_bt)} vs {len(pred_signal)}")

    df_bt['signal'] = pred_signal
    df_bt['sigo'] = df_rules['sigo'].values
    df_bt['sigc'] = df_rules['sigc'].values
    df_bt['time'] = df_rules['time'].values
    df_bt['open_renko'] = df_rules['open_renko'].values
    df_bt['close_renko'] = df_rules['close_renko'].values
    df_bt['EMA'] = df_rules['EMA'].values
    df_bt['pred_signal'] = pred_signal  # pour visu

    # --- PNL ---
    df_bt['next_close'] = df_bt['close'].shift(-1)
    df_bt['return'] = np.where(
        df_bt['signal'] == 1,
        (df_bt['next_close'] - df_bt['close']) / df_bt['close'],
        (df_bt['close'] - df_bt['next_close']) / df_bt['close']
    )
    df_bt['return'] = df_bt['return'].fillna(0)
    df_bt['cum_pnl'] = (1 + df_bt['return']).cumprod() - 1
    sharpe = df_bt['return'].mean() / df_bt['return'].std() * np.sqrt(252) if df_bt['return'].std() > 0 else 0
    print(f"PNL: {df_bt['cum_pnl'].iloc[-1]:+.2%} | Sharpe: {sharpe:.2f}")
    df_bt['final_signal'] = 0
    df_bt.loc[(df_bt['sigo'] == 1) & (df_bt['pred_signal'] == 1), 'final_signal'] = 1
    df_bt.loc[(df_bt['sigo'] == -1) & (df_bt['pred_signal'] == -1), 'final_signal'] = -1
    return df_bt

def recalculate_pnl(df_bt, signal_col='final_signal'):
    """
    Recalcule le PNL sur un signal existant dans df_bt
    """
    df = df_bt.copy()
    df['signal'] = df[signal_col]
    df['next_close'] = df['close'].shift(-1)

    df['return'] = np.where(
        df['signal'] == 1,
        (df['next_close'] - df['close']) / df['close'],
        (df['close'] - df['next_close']) / df['close']
    )
    df['return'] = df['return'].fillna(0)
    df['cum_pnl'] = (1 + df['return']).cumprod() - 1

    sharpe = df['return'].mean() / df['return'].std() * np.sqrt(252) if df['return'].std() > 0 else 0
    print(f"\n=== SIGNAL HYBRIDE ===")
    print(f"PNL: {df['cum_pnl'].iloc[-1]:+.2%} | Sharpe: {sharpe:.2f}")
    return df

