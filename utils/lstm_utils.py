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
from typing import List, Tuple, Dict
from numba import njit, prange
from sklearn.preprocessing import MinMaxScaler

from utils.scaler_utils import train_save_live_scaler, load_and_transform

def generate_param_combinations(grid):
    import itertools
    keys = grid.keys()
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

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
    return dfc

def config_to_features(config:dict):
    features = config["features"]
    target = config["target"]
    features_cols = []
    utils_cols = []
    for col in features:
        if len(col) > 1:
            features_cols.append(col)
            utils_cols.append(col)
    target_cols = target["target_col"]
    if not isinstance(target_cols, list):
        target_cols = [target_cols]
    for col in target_cols:
        if col not in features_cols:
            if target["target_include"]:
                features_cols.append(col)
            utils_cols.append(col)
    return features_cols, target_cols, utils_cols

# =============================================
# 1. SCALING (features seulement)
# =============================================
def scale_features_only(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    path: str
):
    X_train = train_save_live_scaler(train_df[feature_cols].to_numpy(dtype=np.float32), feature_cols, path)
    X_val   = load_and_transform(val_df[feature_cols].to_numpy(dtype=np.float32), path)
    X_test  = load_and_transform(test_df[feature_cols].to_numpy(dtype=np.float32), path)
    return X_train, X_val, X_test

# =============================================
# 2. ASSEMBLAGE + TARGETS BRUTS
# =============================================
def assemble_with_targets(
    X_train_scaled, X_val_scaled, X_test_scaled,
    train_df, val_df, test_df,
    target_cols: List[str]
):
    y_train = train_df[target_cols].to_numpy(dtype=np.float32)
    y_val   = val_df[target_cols].to_numpy(dtype=np.float32)
    y_test  = test_df[target_cols].to_numpy(dtype=np.float32)

    train_ready = np.hstack([X_train_scaled, y_train])
    val_ready   = np.hstack([X_val_scaled,   y_val])
    test_ready  = np.hstack([X_test_scaled,  y_test])

    return train_ready, val_ready, test_ready

# =============================================
# 3. CREATE SEQUENCES — NUMBA ULTRA-RAPIDE
# =============================================
"""
pour seq = 5 et len = 11
n_samples = 6
i = 0,1,2,3,4,5
X 0-4 à 5-10
y 4,5,6,7,8,9,10 
soit y avec -1 si on veut symchroniser sinon on anticipe y par rapport à X
"""
@njit(fastmath=True)
def create_sequences_numba(data: np.ndarray, seq_len: int, n_features: int, horizon: int = 0):
    n_samples = len(data) - seq_len - horizon + 1   # nombre d'échantillons
    n_targets = data.shape[1] - n_features  # nombre total de colonnes - celles des features = nombre colonnes target

    X = np.empty((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.empty((n_samples, n_targets), dtype=np.float32)

    # Attn normalement la dernière ligne aurait dû être enlevée car on ne veut que des bougies closes (valides)
    for i in range(n_samples):
        X[i] = data[i:i + seq_len, :n_features]
        y[i] = data[i + horizon + seq_len-1, n_features:]

    return X, y

# ========================================================
# x. FUNCTIONS CREATE_SEQUENCES anciennes a revoir .......
# ========================================================
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

@njit(parallel=True, cache=True)
def _create_sequences_numba_old(data: np.ndarray, seq_len: int, input_idx: list, target_idx: list, target_type: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Version NUMBA ultra-rapide de create_sequences.
    target_type   # 0: direction, 1: value, 2: return
    """
    n_samples = data.shape[0]       # taille # de lignes
    n_features = data.shape[1]      # nb de col
    n_seq = n_samples - seq_len     # nb de target possible
    if n_seq <= 0:
        return np.empty((0, seq_len, n_features)), np.empty((0,))
    X = np.empty((n_seq, seq_len, n_features), dtype=np.float64)
    y = np.empty((n_seq,), dtype=np.float64)
    for i in prange(n_seq):
        # --- Copie séquence ---
        start_idx = i
        end_idx = i + seq_len
        seq = data[start_idx:end_idx, input_idx]
        X[i] = seq
        # --- Cible à t+1 ---
        val_t = data[i + seq_len - 1, target_idx]
        val_t1 = data[i + seq_len, target_idx]
        if target_type == 0:  # direction
            y[i] = 1.0 if val_t1 > val_t else 0.0
        elif target_type == 1:  # value
            y[i] = val_t1
        elif target_type == 2:  # return
            y[i] = (val_t1 - val_t) / val_t if val_t != 0.0 else 0.0
    return X, y

@njit(parallel=True, cache=True)
def _create_sequences_numba(
    data: np.ndarray,           # shape (n_samples, n_features) – doit être float64 et C-contiguous
    seq_len: int,
    input_idx: np.ndarray,      # ← CHANGEMENT : np.ndarray int32/int64, PAS list[]
    target_idx: np.ndarray,     # ← même chose
    target_type: int            # 0=dir, 1=value, 2=return
) -> Tuple[np.ndarray, np.ndarray]:

    n_samples, n_features = data.shape
    n_seq = n_samples - seq_len
    if n_seq <= 0:
        return np.empty((0, seq_len, len(input_idx)), dtype=np.float64), \
               np.empty((0,), dtype=np.float64)

    # Préallocation
    X = np.empty((n_seq, seq_len, len(input_idx)), dtype=np.float64)
    y = np.empty((n_seq,), dtype=np.float64)

    for i in prange(n_seq):
        # Copie de la séquence (on boucle manuellement sur les features sélectionnées)
        for s in range(seq_len):
            for j in range(len(input_idx)):
                X[i, s, j] = data[i + s, input_idx[j]]

        # Cible à t+1
        val_t  = data[i + seq_len - 1, target_idx[0]]   # on prend le premier (normalement un seul)
        val_t1 = data[i + seq_len,     target_idx[0]]

        if target_type == 0:      # direction
            y[i] = 1.0 if val_t1 > val_t else 0.0
        elif target_type == 1:    # valeur absolue
            y[i] = val_t1
        else:                     # return
            y[i] = (val_t1 - val_t) / val_t if val_t != 0.0 else 0.0

    return X, y

def create_sequences_classic(data, seq_len, features_cols, target_col, target_type):
    X, y = [], []
    data = data.dropna()
    values = data[target_col].values
    features = data[features_cols].values

    for i in range(len(data) - seq_len):
        X.append(features[i:i+seq_len])
        if target_type == 'direction':
            y.append(1 if values[i+seq_len] > values[i+seq_len-1] else 0)
        else:
            y.append((values[i+seq_len] - values[i+seq_len-1]) / values[i+seq_len-1])
    return np.array(X), np.array(y)

def create_sequences(data: np.ndarray, seq_len: int, input_idx: list,  target_idx: list, target_type: int, include: bool) -> Tuple[np.ndarray, np.ndarray]:
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
    n_features = data.shape[1]      # nb de col
    n_col = len(input_idx)+len(target_idx)
    if include:
        n_col -=1
    if n_features != n_col:
        raise "nb colonnes data # nb de features"
    # Avant d’appeler la fonction
    input_idx_np = np.array(input_idx, dtype=np.int32)  # ← important
    target_idx_np = np.array(target_idx, dtype=np.int32)

    # Assure-toi que data est float64 et C-contiguous
    data = np.ascontiguousarray(data, dtype=np.float64)

    # Appel
    X, y = _create_sequences_numba(
        data, seq_len, input_idx_np, target_idx_np, target_type
    )
    return X, y
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

# ========================================================================
### ALTERNATIVE SI TU VEUX RESTER EN KERAS/TF (plus simple)
# 4 Si tu veux rester en TensorFlow → le meilleur compromis 2025
# =====================================================================
def build_informer_like(n_features, seq_len=120, d_model=64, n_heads=8):
    inputs = tf.keras.Input(shape=(seq_len, n_features))

    # ProbSparse Attention (Informer style)
    x = tf.keras.layers.MultiHeadAttention(n_heads, d_model // n_heads)(inputs, inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization()(x + inputs)

    x = tf.keras.layers.Conv1D(d_model, 3, padding='causal')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4), loss='huber')
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
# ======================================================================
# tft_model.py — LE MODÈLE QUE PIERRE UTILISE EN 2025
# =====================================================================
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
import torch

def build_tft(train_df, feature_cols, target_cols=['future_return'], max_encoder_length=120):
    # 1. Dataset TFT
    training = TimeSeriesDataSet(
        train_df.assign(time_idx=train_df.index),
        time_idx="time_idx",
        target=target_cols[0],
        group_ids=["symbol"],  # ou ["regime"] si tu veux multi-séries
        max_encoder_length=max_encoder_length,
        max_prediction_length=1,
        static_categoricals=["volatility_regime", "trend_strength"],
        static_reals=["avg_brick_size"],
        time_varying_known_reals=feature_cols,
        time_varying_unknown_reals=target_cols,
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 2. Modèle TFT
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-4,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 3. Entraînement rapide
    trainer = pl.Trainer(max_epochs=40, gpus=1 if torch.cuda.is_available() else 0,
                         gradient_clip_val=0.1, limit_train_batches=30)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=6)

    trainer.fit(tft, train_dataloaders=train_loader)

    return tft, training

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

