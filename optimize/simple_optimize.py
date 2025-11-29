# optimize/simple_optimize.py
# OPTIMISATION SIMPLE SANS HYPEROPT — RIEN D'INCONNU
# Auteur : Pierre (83 ans)
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import os
import random
from datetime import datetime

from decision.candle_decision import add_indicators
from utils.config_utils import config_to_hash, prepare_to_hashcode
from utils.lstm_utils import scale_features_only, config_to_features, assemble_with_targets, create_sequences_numba
from utils.renko_utils import tick21renko
from utils.utils import reload_ticks_from_pickle

# ==================================================================
# 1. DONNÉES
# ==================================================================
df =  None   # pd.read_pickle("../data/ETHUSD_2025_09_29_00_00_00_2025_11_13_00_00_00.pkl")
start = datetime(2025, 9, 1, 0, 0, 0)
end = datetime(2025, 11, 20, 23, 59, 59)
base_name = f"../data/ETHUSD_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
fbrick = os.path.join('models/', 'simple_model_*.keras')
nbrick = glob.glob(fbrick)
if nbrick:
    try:
        for fic in nbrick:
            os.remove(fic)
    except OSError as e:
        print(f"err supr fichiers {e}")
fbrick = os.path.join('models/', 'simple_scaler_*.pkl')
nbrick = glob.glob(fbrick)
if nbrick:
    try:
        for fic in nbrick:
            os.remove(fic)
    except OSError as e:
        print(f"err supr fichiers {e}")
df = reload_ticks_from_pickle(base_name, 'ETHUSD', None, start, end)
if df is None or df.empty:
    exit()
df['time'] = pd.to_datetime(df['time'])
# ==========================================================================
feature_cols = ["EMA", "RSI", "MACD_hist", "time_live"]
params = {"ema_period": 9, "rsi_period": 14, "rsi_high": 70, "rsi_low": 30, "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}}
# ==================================================================
# 2. FONCTIONS UTILITAIRES
# ==================================================================
# indicators_numba.py — VERSION ULTIME 2025 (Pierre-approved)
def add_indicators_old(df):
    df = df.copy()

    df['EMA'] = df['close'].ewm(span=14).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi_min = df['RSI'].rolling(14).min()
    rsi_max = df['RSI'].rolling(14).max()
    df['STK_RSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min)

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_hist'] = df['MACD'].diff()

    highest = df['high'].rolling(14).max()
    lowest = df['low'].rolling(14).min()
    df['Williams-R'] =  -100 * (highest - df['close']) / (highest - lowest)

    df['time_vol'] = df.index.hour * 100 + df.index.minute

    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['CCI'] = (tp - ma) / (0.015 * md)

    df['time_diff'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['time_live'] = df['volatility'] / (df['time_diff'] + 1e-6)  # évite / par zéro
    df['time_live'] = df['time_live'].replace([np.inf, -np.inf], 0).fillna(0)

    return df.dropna().reset_index(drop=True)


def prepare_targets_final(
        df,
        price_col='target_co',  # ← TON COLONNE DE PRIX RÉELLE (plus jamais 'close')
        horizon=5,
        methods=None,  # ← CHOIX INTELLIGENT : tu choisis ce que tu veux
        prefix='target'
):
    """
    - Utilise price_col (target_co, close, whatever)
    - Ne calcule QUE ce que tu demandes
    - method = =['raw', 'mean', 'sign', 'sum', 'strength']
    - on peut mixer valeur eton ne garde que sign par ex.
    - Zéro look-ahead
    - 100% compatible Numba sequences
    """
    if methods is None:
        methods = ['raw', 'sign']
    df = df.copy()
    price = df[price_col]

    return_cols = []
    # === 1. On calcule UNIQUEMENT les retours nécessaires (pas de colonnes inutiles) ===
    if 'raw' in methods:
        # Ancienne méthode — mais avec ta colonne prix
        raw_col = f'{prefix}_return_t{horizon}'
        return_cols =[raw_col]
        df[raw_col] = price.pct_change(horizon).shift(-horizon)
        if 'sign' in methods:
            df[f'{prefix}_direction_raw'] = np.sign(df[raw_col])
            return_cols.append(f'{prefix}_direction_raw')
    else:
        if 'mean' in methods or 'sum' in methods:
            for i in range(1, horizon + 1):
                col_name = f'{prefix}_return_t{i}'
                df[col_name] = price.pct_change(i).shift(-i)
                return_cols.append(col_name)

        if 'strength' in methods:
            if 'mean' in methods:
                df[f'{prefix}_strength'] = df[return_cols].mean(axis=1).abs()
            elif 'sum' in methods:
                df[f'{prefix}_sum_horizon'] = df[return_cols].sum(axis=1).abs()
        else:
            if 'mean' in methods:
                df[f'{prefix}_mean_horizon'] = df[return_cols].mean(axis=1)
            elif 'sum' in methods:
                df[f'{prefix}_sum_horizon'] = df[return_cols].sum(axis=1)
            if 'sign' in methods:
                df[f'{prefix}_sign_mean'] = np.sign(df[return_cols].mean(axis=1))
                # 0.0 si exactement 0 (rare), sinon +1 / -1

    # === 3. Nettoyage final : on supprime seulement les lignes où la cible demandée est NaN ===
    if 'raw' in methods:
        return_cols.append(f'{prefix}_direction_raw')

    df = df.dropna(subset=return_cols).reset_index(drop=True)

    print(f"Targets prêtes → {len(df):,} bricks | méthodes: {methods}")
    return df

# ==================================================================================
# --- les divers créations de modèles
# ==================================================================================
def lstm_train_model(X_train, y_train, X_val, y_val, units, seq_len, features_len):
    # print("------ lstm train ---------")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, features_len)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # print("train model", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)
    return model

# --------------------------------------------------------- tft
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

def build_tft_dataset(df, feature_cols, target="future_return", seq=120):
    df = df.copy()
    df["time_idx"] = range(len(df))
    df["symbol"] = "SYMB"  # ou ton ticker réel

    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target,
        group_ids=["symbol"],
        max_encoder_length=seq,
        max_prediction_length=1,
        static_reals=["avg_brick_size"] if "avg_brick_size" in df.columns else [],
        time_varying_known_reals=feature_cols,
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return training

def train_tft(train_df, val_df, feature_cols, epochs=30):
    training = build_tft_dataset(train_df, feature_cols)
    validation = build_tft_dataset(val_df, feature_cols)

    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-4,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        output_size=7,
        loss=QuantileLoss(),
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return tft, training
# ---------------------------------------------------------------------
from darts import TimeSeries
from darts.models import NBEATSModel

def nbeat_train(train_series):
    model = NBEATSModel(
        input_chunk_length=120,
        output_chunk_length=1,
        n_neurons=256,
        num_blocks=4,
        num_layers=4,
        layer_widths=512,
        generic_architecture=True,
        loss_fn=torch.nn.MSELoss(),
        optimizer_kwargs={"lr": 1e-4},
    )

    model.fit(train_series, epochs=50)
    pred = model.predict(n=1, series=train_series)
# ----------------------------------------------------------------
import lightgbm as lgb

params_lgb = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l2": 1.0,
    "verbose": -1
}

def lgb_train(X_train, y_train, X_val, y_val):
    model = lgb.train(
        params_lgb, lgb.Dataset(X_train, y_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        early_stopping_rounds=50,    # inconnu ?
        verbose_eval=False    # inconnu ?
    )
# ----------------------------------------------------
#### 3. **CatBoost avec categorical features** (quand tu as des régimes, brick_size, etc.)
from catboost import CatBoostRegressor

def catboost_train(X_train, y_train):
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        cat_features=['volatility_regime', 'trend_type'],
        verbose=False
    )
    model.fit(X_train, y_train)
# -------------------------------------------------------
#### 4. **DeepAR (AWS-style)** — Pour du multi-actifs simultané
## Prédit 50 cryptos en même temps → 4.8 Sharpe global**
from gluonts.torch.model.deepar import DeepAREstimator

def gluonts_train(list_of_series):
    estimator = DeepAREstimator(
        freq="1min",
        prediction_length=1,
        num_layers=4,
        hidden_size=64,
        trainer_kwargs={"max_epochs": 50}
    )
    predictor = estimator.train(training_data=list_of_series)


#### 5. **Ensemble final (ce que font les meilleurs fonds)**
# La vraie recette 2025
#pred_tft     = tft.predict(...)
#pred_nbeats  = nbeats.predict(...)
#pred_lgb     = lgb.predict(X_test)


# =========================================================================
# --- utilities
# =========================================================================
def to_config_std(config):
    param = {"renko_size": config["renko_size"]} | params
    config_std = {"parameters": param, "features": feature_cols,
                  "target": {"target_col": config["target_col"], "target_type": config["target_type"],
                             "target_include": config["target_include"]},
                  "lstm": {"lstm_seq_len": config["seq_len"], "lstm_units": config["lstm_units"],
                           "lstm_threshold_buy": config["threshold_buy"],
                           "lstm_threshold_sell": config["threshold_sell"]},
                  "live": {"symbol": "ETHUSD"}}
    return config_std

# ==================================================================
# 3. FONCTION D'ÉVALUATION
# ==================================================================
def evaluate_config(config):
    try:
        # ------------- standardiser les paramétrages
        config_std = to_config_std(config)
        # print("c std", config_std)
        hcode = config_to_hash(prepare_to_hashcode(config_std))
        # print("hcode", hcode)
        config["hcode"] = hcode
        # ------------- Préparer données
        df_bricks = tick21renko(df, None, config['renko_size'], 'bid')
        print("taille max ", len(df_bricks))
        try:
            df_renko = add_indicators(df_bricks)        # un calcul ici local -> cancles_decision ?
        except BaseException as b:
            print(f"calcul indicateur {b}")
            return -1e9
        # les colonnes data, cible...
        features_cols, target_cols, utils_cols = config_to_features(config_std)
        # ----------- controles
        for col in features_cols:
            if col not in df_renko.columns:
                print(f"features {col} not in data")
                return -1e9
        for col in target_cols:
            if col not in df_renko.columns:
                print(f"target {col} not in data")
                return -1e9
        features_len = len(features_cols)
        # print("colonnes", features_cols, target_cols)
        # ------------- calcul target
        type = config_std["target"]["target_type"]
        new_cols = []
        try:
            if type == 'direction':
                for col in target_cols:
                    current = df_renko[col]
                    previous = df_renko[col].shift(1)
                    df_renko[col + 'd'] = np.where(current > previous, 1, np.where(current < previous, -1, 0))
                    new_cols.append(col+'d')
            elif type == 'return':
                new_cols = []
                for col in target_cols:
                    current = df_renko[col]
                    previous = df_renko[col].shift(1)
                    df_renko[col + 'r'] = np.where(previous != 0, (current - previous) / previous, 0.0)
                    new_cols.append(col+'r')
            else:
                new_cols = target_cols
        except BaseException as e:
            print("calcul target col", e)
            return -1e9
        config_std['target']['target_cols'] = new_cols
        target_cols = new_cols
        # print("target", target_cols)
        df_renko = df_renko.dropna(subset=new_cols).reset_index(drop=True)
        # ----------- repartition data, val, test
        train_len = int(len(df_renko)*0.65)
        val_len = int(len(df_renko)*0.15)
        train_p = df_renko.iloc[:train_len]
        val_p = df_renko.iloc[train_len:train_len+val_len]
        test_p = df_renko[train_len+val_len:]
        # print("shape", train_p.shape, val_p.shape, test_p.shape)
        seq_len = config["seq_len"]
        # print("seq ", seq_len)
        if len(train_p) < seq_len*3 or len(val_p) < seq_len or len(test_p) < seq_len:
             print(f"pas assez pour sequences {seq_len}")
             return -1e9
        # ---------- Normaliser
        scaler_path = f"models/simple_scaler_{hcode}.pkl"
        X_train, X_val, X_test = scale_features_only(train_p, val_p, test_p, features_cols, scaler_path)
        train_r, val_r, test_r = assemble_with_targets(X_train, X_val, X_test, train_p, val_p, test_p, target_cols)
        # ----------- sequencer
        X_train_seq, y_train_seq = create_sequences_numba(train_r, seq_len, features_len)
        X_val_seq, y_val_seq = create_sequences_numba(val_r, seq_len, features_len)
        X_test_seq, y_test_seq = create_sequences_numba(test_r, seq_len, features_len)
        # print("seq shape", X_train_seq.shape, X_val_seq.shape, X_test_seq.shape)
        # ------------ Entraîner
        proba = pd.DataFrame([0]*len(X_test_seq), columns=["proba"])
        try:
            model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, config['lstm_units'], seq_len, features_len)
            model.save(f"models/simple_model_{hcode}.keras")
            # Prédire
            proba = model.predict(X_test_seq, verbose=0).flatten()
        except BaseException as b:
            print(f"err dans model {b}")
        buy = proba > config['threshold_buy']
        sell = proba < config['threshold_sell']
        # ------------ Backtest
        try:
            returns = np.diff(test_p['close'].values[-len(proba):])
        except BaseException as b:
            print("calcul BT return", b)
        profit = 0
        trades = 0
        wins = 0
        for i in range(len(proba)):
            if i >= len(returns): break
            if buy[i]:
                profit += returns[i]-2.5   #spread
                trades += 1
                if returns[i] > 0: wins += 1
            elif sell[i]:
                profit -= returns[i]+2.5
                trades += 1
                if returns[i] < 0: wins += 1

        winrate = wins / trades if trades > 0 else 0
        if trades < 10: return -1e9

        score = profit * 1000 + winrate * 100
        print(f"Config: {config['renko_size']:.1f}, {config['seq_len']}, {config['lstm_units']} → Score: {score:.1f}")
        return score

    except BaseException as e:
        print("err générale ", e)
        return -1e9

# ==================================================================
# 4. OPTIMISATION PAR RECHERCHE ALÉATOIRE (100 ESSAIS)
# ==================================================================
if __name__ == "__main__":
    print("DÉBUT OPTIMISATION SIMPLE — 100 ESSAIS")

    results = []
    best_score = -1e9
    best_config = None
    for i in range(100):
        config = {
            'renko_size': round(random.uniform(10, 50), 1),
            'target_col': random.choice(['close', 'EMA']),
            'target_type': random.choice(['direction', 'return']),
            'target_include': random.choice([True, False]),
            'seq_len': random.randint(20, 100),
            'lstm_units': random.randint(50, 200),
            'threshold_buy': round(random.uniform(0.5, 0.7), 3),
            'threshold_sell': round(random.uniform(0.3, 0.5), 3),
            'hcode': ""
        }

        score = evaluate_config(config)
        if score > best_score:
            best_score = score
            best_config = config
        results.append([score, config])

    print("\nMEILLEURE CONFIGURATION TROUVÉE :")
    print(json.dumps(best_config, indent=2))
    print(f"Score final: {best_score:.1f}")
    top5 = sorted(results, key=lambda x: x[0], reverse=True)[:10]
    for top in top5:
        print(top)

    # Sauvegarde
    scaler_name = f"models/scaler_{best_config['hcode']}.pkl"
    scaler = joblib.load(scaler_name) if os.path.exists(scaler_name) else None
    os.makedirs("models/simple_opt", exist_ok=True)
    with open(f"models/simple_opt/best_{best_config['hcode']}.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print("Sauvegarde → models/simple_opt/best.json")
